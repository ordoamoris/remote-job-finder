"""
Remote Job Finder - Simple daily job search with probabilistic ranking
"""
import streamlit as st
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import re
from dataclasses import dataclass
import hashlib
import time
import feedparser

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class Job:
    """Single job posting with normalized fields"""
    title: str
    company: str
    location: str
    posted_date: Optional[datetime]
    source: str
    url: str
    description: str
    salary_text: Optional[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
    
    def dedup_key(self) -> str:
        """Generate unique key for deduplication"""
        normalized = f"{self.company.lower()}|{self.title.lower()}".replace(" ", "")
        return hashlib.md5(normalized.encode()).hexdigest()


@dataclass
class ScoredJob:
    """Job with score and explanation"""
    job: Job
    score: float
    explanation: Dict[str, any]
    is_new: bool = False


# ============================================================================
# JOB FETCHERS
# ============================================================================

def fetch_remotive(timeout: int = 10) -> List[Job]:
    """Fetch from Remotive API"""
    jobs = []
    try:
        url = "https://remotive.com/api/remote-jobs"
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        
        for item in data.get('jobs', []):
            try:
                posted = datetime.fromisoformat(item.get('publication_date', '').replace('Z', '+00:00'))
            except:
                posted = None
            
            jobs.append(Job(
                title=item.get('title', 'Unknown'),
                company=item.get('company_name', 'Unknown'),
                location=item.get('candidate_required_location', 'Remote'),
                posted_date=posted,
                source='Remotive',
                url=item.get('url', ''),
                description=item.get('description', ''),
                salary_text=item.get('salary', None),
                tags=item.get('tags', [])
            ))
    except Exception as e:
        st.warning(f"Remotive fetch failed: {str(e)}")
    
    return jobs


def fetch_remoteok(timeout: int = 10) -> List[Job]:
    """Fetch from RemoteOK"""
    jobs = []
    try:
        url = "https://remoteok.com/api"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        
        # First item is metadata, skip it
        for item in data[1:]:
            try:
                posted = datetime.fromtimestamp(item.get('date', 0))
            except:
                posted = None
            
            tags = item.get('tags', [])
            salary = None
            if item.get('salary_min') or item.get('salary_max'):
                salary = f"${item.get('salary_min', 0)}-${item.get('salary_max', 0)}"
            
            jobs.append(Job(
                title=item.get('position', 'Unknown'),
                company=item.get('company', 'Unknown'),
                location=item.get('location', 'Remote'),
                posted_date=posted,
                source='RemoteOK',
                url=item.get('url', ''),
                description=item.get('description', ''),
                salary_text=salary,
                tags=tags
            ))
    except Exception as e:
        st.warning(f"RemoteOK fetch failed: {str(e)}")
    
    return jobs


def fetch_wwr(timeout: int = 10) -> List[Job]:
    """Fetch from We Work Remotely RSS (optional, fail gracefully)"""
    jobs = []
    try:
        url = "https://weworkremotely.com/categories/remote-full-stack-programming-jobs.rss"
        feed = feedparser.parse(url)
        
        for entry in feed.entries[:50]:  # Limit to recent
            try:
                posted = datetime(*entry.published_parsed[:6])
            except:
                posted = None
            
            # Extract company from title if possible
            title_parts = entry.title.split(':')
            company = title_parts[0].strip() if len(title_parts) > 1 else 'Unknown'
            title = title_parts[1].strip() if len(title_parts) > 1 else entry.title
            
            jobs.append(Job(
                title=title,
                company=company,
                location='Remote',
                posted_date=posted,
                source='WWR',
                url=entry.link,
                description=entry.get('summary', ''),
                tags=[]
            ))
    except Exception as e:
        # Silently fail for WWR as it's optional
        pass
    
    return jobs


def fetch_all_jobs() -> Tuple[List[Job], Dict[str, int]]:
    """Fetch from all sources with error handling"""
    all_jobs = []
    stats = {}
    
    # Fetch from each source
    remotive_jobs = fetch_remotive()
    remoteok_jobs = fetch_remoteok()
    wwr_jobs = fetch_wwr()
    
    stats['remotive'] = len(remotive_jobs)
    stats['remoteok'] = len(remoteok_jobs)
    stats['wwr'] = len(wwr_jobs)
    
    all_jobs.extend(remotive_jobs)
    all_jobs.extend(remoteok_jobs)
    all_jobs.extend(wwr_jobs)
    
    # Deduplicate
    seen = set()
    deduped = []
    for job in all_jobs:
        key = job.dedup_key()
        if key not in seen:
            seen.add(key)
            deduped.append(job)
    
    stats['total_fetched'] = len(all_jobs)
    stats['after_dedup'] = len(deduped)
    
    return deduped, stats


# ============================================================================
# PROBABILISTIC SCORING SYSTEM
# ============================================================================

class JobScorer:
    """Probabilistic job scoring with uncertainty handling"""
    
    def __init__(self, 
                 include_keywords: List[str],
                 exclude_keywords: List[str],
                 min_salary: float = 40000):
        self.include_keywords = [k.lower() for k in include_keywords]
        self.exclude_keywords = [k.lower() for k in exclude_keywords]
        self.min_salary = min_salary
    
    def score_job(self, job: Job) -> Tuple[float, Dict]:
        """
        Compute Edge Score = P(viable) * expected_comp_uplift * (1 - time_trap_risk) - app_cost
        
        Returns: (score, explanation_dict)
        """
        features = self._extract_features(job)
        explanation = {'features': features}
        
        # Core probabilities
        p_viable = self._compute_viability(features)
        expected_comp = self._estimate_compensation(features)
        time_trap_risk = self._estimate_time_trap_risk(features)
        app_cost = 0.5  # Normalized units (assume 30min per app)
        
        # Edge score computation
        comp_uplift = max(0, (expected_comp - self.min_salary) / 10000)  # Normalize
        edge_score = p_viable * comp_uplift * (1 - time_trap_risk) - app_cost
        
        # Scale to 0-100 for readability
        edge_score = max(0, (edge_score + 1) * 50)
        
        explanation.update({
            'p_viable': p_viable,
            'expected_comp': expected_comp,
            'comp_uplift': comp_uplift,
            'time_trap_risk': time_trap_risk,
            'edge_score': edge_score
        })
        
        return edge_score, explanation
    
    def _extract_features(self, job: Job) -> Dict:
        """Extract all relevant features from job posting"""
        text = f"{job.title} {job.description} {' '.join(job.tags)}".lower()
        
        # Remote certainty
        remote_indicators = ['remote', 'work from home', 'wfh', 'distributed', 'anywhere']
        remote_score = sum(1 for ind in remote_indicators if ind in text) / len(remote_indicators)
        
        # Keyword matching
        include_matches = [kw for kw in self.include_keywords if kw in text]
        exclude_matches = [kw for kw in self.exclude_keywords if kw in text]
        
        # Salary extraction
        salary_min, salary_max, salary_certainty = self._extract_salary(job)
        
        # Seniority detection
        seniority = self._detect_seniority(text)
        
        # Sales risk
        sales_risk = self._detect_sales_risk(text)
        
        # Slack time indicators
        slack_score = self._estimate_slack_potential(text)
        
        # Freshness
        days_old = self._get_age_days(job)
        freshness = max(0, 1 - days_old / 30)  # Decay over 30 days
        
        return {
            'remote_score': remote_score,
            'include_matches': include_matches,
            'exclude_matches': exclude_matches,
            'salary_min': salary_min,
            'salary_max': salary_max,
            'salary_certainty': salary_certainty,
            'seniority': seniority,
            'sales_risk': sales_risk,
            'slack_score': slack_score,
            'freshness': freshness,
            'days_old': days_old
        }
    
    def _extract_salary(self, job: Job) -> Tuple[float, float, float]:
        """Extract salary with uncertainty estimate"""
        if not job.salary_text:
            # No salary info - use prior based on role type
            text = f"{job.title} {job.description}".lower()
            if any(kw in text for kw in ['senior', 'lead', 'principal', 'staff']):
                return 70000, 100000, 0.3  # Wide range, low certainty
            elif any(kw in text for kw in ['junior', 'entry', 'associate']):
                return 40000, 60000, 0.3
            else:
                return 50000, 80000, 0.2  # Mid-level prior
        
        # Try to parse salary
        text = job.salary_text.lower()
        numbers = re.findall(r'\$?(\d+)[,\s]*(\d*)', text)
        
        if numbers:
            try:
                amounts = []
                for num, decimal in numbers:
                    amount = int(num.replace(',', ''))
                    if decimal:
                        amount = int(f"{num}{decimal}")
                    
                    # Handle k notation
                    if 'k' in text:
                        amount *= 1000
                    
                    # Filter reasonable salary range
                    if 10000 <= amount <= 500000:
                        amounts.append(amount)
                
                if amounts:
                    return min(amounts), max(amounts), 0.8
            except:
                pass
        
        # Fallback
        return 50000, 80000, 0.2
    
    def _detect_seniority(self, text: str) -> str:
        """Detect seniority level"""
        if any(kw in text for kw in ['senior', 'sr.', 'lead', 'principal', 'staff', 'architect']):
            return 'senior'
        elif any(kw in text for kw in ['junior', 'jr.', 'entry', 'associate', 'intern']):
            return 'junior'
        else:
            return 'mid'
    
    def _detect_sales_risk(self, text: str) -> float:
        """Estimate probability this is a sales/commission role"""
        sales_indicators = [
            'commission', '1099', 'door-to-door', 'high ticket',
            'business development rep', 'bdr', 'sdr', 'sales development',
            'insurance agent', 'financial advisor', 'real estate agent',
            'cold calling', 'prospecting', 'quota', 'pipeline'
        ]
        
        matches = sum(1 for ind in sales_indicators if ind in text)
        return min(1.0, matches * 0.3)  # Cap at 1.0
    
    def _estimate_slack_potential(self, text: str) -> float:
        """
        Estimate likelihood of consistent slack time (async, ops, low meetings)
        Higher = more likely to have slack
        """
        # Positive indicators
        slack_positive = [
            'operations', 'ops', 'compliance', 'regulatory', 'surveillance',
            'reconciliation', 'reporting', 'monitoring', 'kyc', 'aml',
            'back office', 'middle office', 'support', 'async',
            'flexible', 'autonomous', 'independent'
        ]
        
        # Negative indicators (high meeting load)
        slack_negative = [
            'client-facing', 'customer success', 'account management',
            'project manager', 'scrum master', 'agile coach',
            'stakeholder management', 'cross-functional',
            'fast-paced', 'startup', 'move fast'
        ]
        
        positive_score = sum(1 for ind in slack_positive if ind in text)
        negative_score = sum(1 for ind in slack_negative if ind in text)
        
        # Normalize to 0-1
        return min(1.0, max(0, (positive_score - negative_score * 0.5) / 5))
    
    def _get_age_days(self, job: Job) -> int:
        """Get job age in days"""
        if not job.posted_date:
            return 7  # Assume 1 week if unknown
        
        age = datetime.now() - job.posted_date.replace(tzinfo=None)
        return age.days
    
    def _compute_viability(self, features: Dict) -> float:
        """Compute P(viable role for my constraints)"""
        # Start with base probability
        p = 0.5
        
        # Remote requirement (must be remote)
        if features['remote_score'] < 0.5:
            p *= 0.1  # Strong penalty if not clearly remote
        else:
            p *= (0.5 + features['remote_score'] * 0.5)
        
        # Keyword matching
        if features['include_matches']:
            p *= min(1.5, 1 + len(features['include_matches']) * 0.1)
        else:
            p *= 0.5  # Penalty if no include matches
        
        if features['exclude_matches']:
            p *= max(0.1, 1 - len(features['exclude_matches']) * 0.3)
        
        # Sales risk (must not be sales)
        p *= (1 - features['sales_risk'])
        
        # Salary floor
        if features['salary_min'] < self.min_salary:
            p *= 0.5  # Penalty for low salary
        
        # Cap at 1.0
        return min(1.0, p)
    
    def _estimate_compensation(self, features: Dict) -> float:
        """Estimate expected compensation with uncertainty"""
        salary_min = features['salary_min']
        salary_max = features['salary_max']
        certainty = features['salary_certainty']
        
        # Expected value with uncertainty penalty
        expected = (salary_min + salary_max) / 2
        
        # Uncertainty penalty (wider range = more uncertain)
        range_width = salary_max - salary_min
        uncertainty_penalty = (1 - certainty) * range_width * 0.1
        
        return max(self.min_salary, expected - uncertainty_penalty)
    
    def _estimate_time_trap_risk(self, features: Dict) -> float:
        """Estimate P(role will be time trap with no slack)"""
        # Lower slack score = higher time trap risk
        base_risk = 1 - features['slack_score']
        
        # Sales roles are time traps
        base_risk = max(base_risk, features['sales_risk'])
        
        # Very fresh jobs might be rushed hiring (slight risk)
        if features['days_old'] < 2:
            base_risk *= 1.1
        
        return min(1.0, base_risk)


# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    st.set_page_config(
        page_title="Remote Job Finder",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 Remote Job Finder")
    st.caption("Daily one-click job search with probabilistic ranking")
    
    # Initialize session state
    if 'jobs' not in st.session_state:
        st.session_state.jobs = []
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = None
    if 'previous_job_keys' not in st.session_state:
        st.session_state.previous_job_keys = set()
    if 'stats' not in st.session_state:
        st.session_state.stats = {}
    
    # Sidebar - Controls
    with st.sidebar:
        st.header("Controls")
        
        # Main refresh button
        if st.button("🔄 Refresh Jobs", type="primary", use_container_width=True):
            with st.spinner("Fetching jobs..."):
                jobs, stats = fetch_all_jobs()
                
                # Track which jobs are new
                current_keys = {job.dedup_key() for job in jobs}
                
                st.session_state.jobs = jobs
                st.session_state.stats = stats
                st.session_state.last_refresh = datetime.now()
                
                # Update previous keys for next refresh
                st.session_state.previous_job_keys = current_keys
                
                st.success(f"✅ Found {len(jobs)} jobs")
        
        st.divider()
        
        # Filters
        st.header("Filters")
        
        # Include keywords
        default_include = [
            "operations", "ops", "trade ops", "brokerage ops", "middle office",
            "back office", "client onboarding", "KYC", "AML", "compliance",
            "regulatory", "surveillance", "risk operations", "payments ops",
            "reconciliations", "reporting", "account services", 
            "client service associate", "analyst"
        ]
        include_kw = st.text_area(
            "Include Keywords (one per line)",
            value="\n".join(default_include),
            height=150
        )
        include_keywords = [k.strip() for k in include_kw.split('\n') if k.strip()]
        
        # Exclude keywords
        default_exclude = [
            "financial advisor", "insurance", "commission", "1099",
            "door-to-door", "high ticket sales", "business development",
            "real estate agent"
        ]
        exclude_kw = st.text_area(
            "Exclude Keywords (one per line)",
            value="\n".join(default_exclude),
            height=100
        )
        exclude_keywords = [k.strip() for k in exclude_kw.split('\n') if k.strip()]
        
        # Other filters
        max_age_days = st.slider("Max Age (days)", 1, 90, 30)
        min_salary = st.number_input("Min Salary ($)", 0, 200000, 40000, step=5000)
        top_n = st.slider("Show Top N Jobs", 10, 100, 30)
        
        st.divider()
        
        # Stats
        if st.session_state.stats:
            st.header("Source Stats")
            stats = st.session_state.stats
            st.metric("Remotive", stats.get('remotive', 0))
            st.metric("RemoteOK", stats.get('remoteok', 0))
            st.metric("WWR", stats.get('wwr', 0))
            st.metric("Total Fetched", stats.get('total_fetched', 0))
            st.metric("After Dedup", stats.get('after_dedup', 0))
        
        if st.session_state.last_refresh:
            st.caption(f"Last refresh: {st.session_state.last_refresh.strftime('%I:%M %p')}")
    
    # Main content
    if not st.session_state.jobs:
        st.info("👆 Click 'Refresh Jobs' to start")
        
        # Verification section
        st.subheader("How it works")
        st.markdown("""
        1. **One click**: Fetches from multiple sources (Remotive, RemoteOK, WWR)
        2. **Smart scoring**: Probabilistic ranking based on your constraints
        3. **Uncertainty-aware**: Handles missing salary data with Bayesian priors
        4. **Explainable**: See why each job ranked where it did
        
        **Your constraints**:
        - Must be remote
        - Must pay >$40k (or uncertain enough to be worth checking)
        - Must not be sales/commission heavy
        - Should have slack time potential (ops, compliance, async roles)
        """)
    else:
        # Score all jobs
        scorer = JobScorer(include_keywords, exclude_keywords, min_salary)
        scored_jobs = []
        
        for job in st.session_state.jobs:
            # Filter by age
            if job.posted_date:
                age_days = (datetime.now() - job.posted_date.replace(tzinfo=None)).days
                if age_days > max_age_days:
                    continue
            
            score, explanation = scorer.score_job(job)
            
            # Check if new
            is_new = job.dedup_key() not in st.session_state.previous_job_keys
            
            scored_jobs.append(ScoredJob(job, score, explanation, is_new))
        
        # Sort by score
        scored_jobs.sort(key=lambda x: x.score, reverse=True)
        
        # Display top N
        st.subheader(f"Top {min(top_n, len(scored_jobs))} Jobs")
        st.caption(f"Showing {min(top_n, len(scored_jobs))} of {len(scored_jobs)} jobs matching filters")
        
        for i, sjob in enumerate(scored_jobs[:top_n], 1):
            job = sjob.job
            exp = sjob.explanation
            feats = exp['features']
            
            # Card layout
            with st.expander(
                f"{'🆕 ' if sjob.is_new else ''}{i}. {job.title} @ {job.company} | Score: {sjob.score:.1f}",
                expanded=(i <= 3)  # Auto-expand top 3
            ):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Company:** {job.company}")
                    st.markdown(f"**Location:** {job.location}")
                    st.markdown(f"**Source:** {job.source}")
                    if job.posted_date:
                        st.markdown(f"**Posted:** {job.posted_date.strftime('%Y-%m-%d')} ({feats['days_old']} days ago)")
                    st.markdown(f"**Link:** [Apply Here]({job.url})")
                
                with col2:
                    st.metric("Edge Score", f"{sjob.score:.1f}")
                    st.metric("Viability", f"{exp['p_viable']:.0%}")
                    st.metric("Comp Est.", f"${exp['expected_comp']:,.0f}")
                
                # Explanation
                st.markdown("**Why this score?**")
                
                reasons = []
                
                # Positive factors
                if feats['include_matches']:
                    reasons.append(f"✅ Matches keywords: {', '.join(feats['include_matches'][:3])}")
                if feats['slack_score'] > 0.5:
                    reasons.append(f"✅ High slack potential ({feats['slack_score']:.0%})")
                if feats['remote_score'] > 0.7:
                    reasons.append(f"✅ Strong remote indicators ({feats['remote_score']:.0%})")
                if feats['salary_certainty'] > 0.6:
                    reasons.append(f"✅ Salary info available (${feats['salary_min']:,.0f}-${feats['salary_max']:,.0f})")
                if feats['days_old'] < 7:
                    reasons.append(f"✅ Fresh posting ({feats['days_old']} days)")
                
                # Negative factors
                if feats['exclude_matches']:
                    reasons.append(f"⚠️ Exclude keywords: {', '.join(feats['exclude_matches'][:3])}")
                if feats['sales_risk'] > 0.3:
                    reasons.append(f"⚠️ Sales risk detected ({feats['sales_risk']:.0%})")
                if exp['time_trap_risk'] > 0.5:
                    reasons.append(f"⚠️ Time trap risk ({exp['time_trap_risk']:.0%})")
                if feats['salary_certainty'] < 0.5:
                    reasons.append(f"⚠️ Salary uncertain (prior: ${feats['salary_min']:,.0f}-${feats['salary_max']:,.0f})")
                if feats['remote_score'] < 0.5:
                    reasons.append(f"⚠️ Remote unclear ({feats['remote_score']:.0%})")
                
                if reasons:
                    for reason in reasons:
                        st.markdown(reason)
                else:
                    st.markdown("No strong signals")
                
                # Technical details (collapsed)
                with st.expander("🔍 Technical Details"):
                    st.json({
                        'edge_score': sjob.score,
                        'p_viable': exp['p_viable'],
                        'expected_comp': exp['expected_comp'],
                        'comp_uplift': exp['comp_uplift'],
                        'time_trap_risk': exp['time_trap_risk'],
                        'features': {k: v for k, v in feats.items() if k not in ['include_matches', 'exclude_matches']}
                    })


if __name__ == "__main__":
    main()
