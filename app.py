"""
Remote Job Finder - Analyst Dashboard
Hedge-fund style scoring with decomposed components
"""
import streamlit as st
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import re
import hashlib
import feedparser
import pandas as pd

# ============================================================================
# DATA FETCHING
# ============================================================================

def fetch_remotive(timeout=10):
    """Fetch from Remotive API"""
    jobs = []
    try:
        response = requests.get("https://remotive.com/api/remote-jobs", timeout=timeout)
        response.raise_for_status()
        data = response.json()
        
        for item in data.get('jobs', []):
            try:
                posted = datetime.fromisoformat(item.get('publication_date', '').replace('Z', '+00:00'))
            except:
                posted = None
            
            jobs.append({
                'title': item.get('title', 'Unknown'),
                'company': item.get('company_name', 'Unknown'),
                'location': item.get('candidate_required_location', 'Remote'),
                'posted_date': posted,
                'source': 'Remotive',
                'url': item.get('url', ''),
                'description': item.get('description', ''),
                'salary_text': item.get('salary', None),
                'tags': item.get('tags', [])
            })
    except Exception as e:
        st.sidebar.warning(f"Remotive failed: {str(e)[:50]}")
    return jobs


def fetch_remoteok(timeout=10):
    """Fetch from RemoteOK"""
    jobs = []
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get("https://remoteok.com/api", headers=headers, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        
        for item in data[1:]:
            try:
                posted = datetime.fromtimestamp(item.get('date', 0))
            except:
                posted = None
            
            salary = None
            if item.get('salary_min') or item.get('salary_max'):
                salary = f"${item.get('salary_min', 0)}-${item.get('salary_max', 0)}"
            
            jobs.append({
                'title': item.get('position', 'Unknown'),
                'company': item.get('company', 'Unknown'),
                'location': item.get('location', 'Remote'),
                'posted_date': posted,
                'source': 'RemoteOK',
                'url': item.get('url', ''),
                'description': item.get('description', ''),
                'salary_text': salary,
                'tags': item.get('tags', [])
            })
    except Exception as e:
        st.sidebar.warning(f"RemoteOK failed: {str(e)[:50]}")
    return jobs


def fetch_wwr(timeout=10):
    """Fetch from We Work Remotely RSS"""
    jobs = []
    try:
        feed = feedparser.parse("https://weworkremotely.com/categories/remote-full-stack-programming-jobs.rss")
        
        for entry in feed.entries[:50]:
            try:
                posted = datetime(*entry.published_parsed[:6])
            except:
                posted = None
            
            title_parts = entry.title.split(':')
            company = title_parts[0].strip() if len(title_parts) > 1 else 'Unknown'
            title = title_parts[1].strip() if len(title_parts) > 1 else entry.title
            
            jobs.append({
                'title': title,
                'company': company,
                'location': 'Remote',
                'posted_date': posted,
                'source': 'WWR',
                'url': entry.link,
                'description': entry.get('summary', ''),
                'salary_text': None,
                'tags': []
            })
    except:
        pass  # Fail silently for WWR
    return jobs


def dedup_key(job):
    """Generate dedup key"""
    normalized = f"{job['company'].lower()}|{job['title'].lower()}".replace(" ", "")
    return hashlib.md5(normalized.encode()).hexdigest()


def fetch_all_jobs():
    """Fetch and deduplicate from all sources"""
    all_jobs = []
    all_jobs.extend(fetch_remotive())
    all_jobs.extend(fetch_remoteok())
    all_jobs.extend(fetch_wwr())
    
    seen = set()
    deduped = []
    for job in all_jobs:
        key = dedup_key(job)
        if key not in seen:
            seen.add(key)
            deduped.append(job)
    
    return deduped


# ============================================================================
# SCORING ENGINE
# ============================================================================

class JobScorer:
    """Decomposed scoring with qualifications fit"""
    
    def __init__(self, include_kw, exclude_kw, exp_level, certs, strict_mode):
        self.include_kw = [k.lower().strip() for k in include_kw if k.strip()]
        self.exclude_kw = [k.lower().strip() for k in exclude_kw if k.strip()]
        self.exp_years = self._exp_to_years(exp_level)
        self.certs = [c.lower().strip() for c in certs if c.strip()]
        self.strict_mode = strict_mode
    
    def _exp_to_years(self, level):
        """Convert experience level to years"""
        mapping = {
            "0-2 years": 1,
            "2-4 years": 3,
            "4-7 years": 5.5,
            "7+ years": 10
        }
        return mapping.get(level, 1)
    
    def score_job(self, job):
        """Score a single job with decomposed components"""
        text = f"{job['title']} {job['description']} {' '.join(job.get('tags', []))}".lower()
        
        # Extract signals
        signals = self._extract_signals(job, text)
        
        # Component scores (0-1)
        role_fit = self._score_role_family(text)
        remote_cert = self._score_remote_certainty(text)
        comp_signal = self._score_compensation(job, text)
        sales_risk = self._score_sales_risk(text)
        time_trap = self._score_time_trap_risk(text)
        uncertainty = self._score_uncertainty(job, text)
        freshness = self._score_freshness(job)
        
        # P_fit (qualifications match)
        p_fit = self._score_fit_probability(signals)
        
        # Edge score
        edge = self._compute_edge_score(
            p_fit, role_fit, remote_cert, comp_signal, 
            sales_risk, time_trap, uncertainty, freshness
        )
        
        return {
            'P_fit': p_fit,
            'RoleFamilyFit': role_fit,
            'RemoteCertainty': remote_cert,
            'CompensationSignal': comp_signal,
            'SalesRisk': sales_risk,
            'TimeTrapRisk': time_trap,
            'UncertaintyPenalty': uncertainty,
            'Freshness': freshness,
            'EdgeScore': edge,
            'signals': signals
        }
    
    def _extract_signals(self, job, text):
        """Extract qualification signals"""
        signals = {}
        
        # Seniority detection
        senior_keywords = ['senior', 'sr.', 'lead', 'principal', 'manager', 'director', 'vp', 'head of', 'chief']
        signals['is_senior'] = any(kw in text for kw in senior_keywords)
        
        junior_keywords = ['junior', 'jr.', 'entry', 'associate', 'coordinator', 'specialist', 'analyst']
        signals['is_junior'] = any(kw in text for kw in junior_keywords)
        
        # Years requirement
        years_pattern = r'(\d+)\+?\s*years?'
        years_matches = re.findall(years_pattern, text)
        if years_matches:
            signals['years_required'] = max([int(y) for y in years_matches])
        else:
            signals['years_required'] = None
        
        # Credentials required
        cred_keywords = ['cpa', 'jd', 'cfa', 'cams', 'series 24', 'series 63', 'finra', 'mba required']
        signals['creds_required'] = [cred for cred in cred_keywords if cred in text]
        
        # Training/early career boost
        boost_keywords = ['training provided', 'will train', 'early career', 'new grad', 'recent graduate']
        signals['training_offered'] = any(kw in text for kw in boost_keywords)
        
        # Salary presence
        signals['has_salary'] = job.get('salary_text') is not None
        
        return signals
    
    def _score_fit_probability(self, signals):
        """Compute P(qualified for this role)"""
        p = 1.0
        
        # Senior role penalty
        if signals['is_senior']:
            p *= 0.3
        
        # Junior role boost
        if signals['is_junior']:
            p *= 1.5
        
        # Years requirement penalty
        if signals['years_required'] is not None:
            years_gap = signals['years_required'] - self.exp_years
            if years_gap > 0:
                p *= max(0.2, 1 - (years_gap * 0.15))
        
        # Credentials penalty
        if signals['creds_required']:
            missing_creds = [c for c in signals['creds_required'] if c not in ' '.join(self.certs)]
            if missing_creds:
                p *= max(0.3, 1 - (len(missing_creds) * 0.25))
        
        # Training boost
        if signals['training_offered']:
            p *= 1.3
        
        return min(1.0, p)
    
    def _score_role_family(self, text):
        """Score match to target role families"""
        if not self.include_kw:
            return 0.5
        
        matches = sum(1 for kw in self.include_kw if kw in text)
        return min(1.0, matches / max(3, len(self.include_kw) * 0.3))
    
    def _score_remote_certainty(self, text):
        """Score remote work certainty"""
        remote_indicators = ['remote', 'work from home', 'wfh', 'distributed', 'anywhere', 'location independent']
        onsite_indicators = ['onsite', 'on-site', 'in-office', 'hybrid required', 'must be located in']
        
        remote_count = sum(1 for ind in remote_indicators if ind in text)
        onsite_count = sum(1 for ind in onsite_indicators if ind in text)
        
        score = (remote_count - onsite_count * 2) / len(remote_indicators)
        return max(0, min(1.0, score))
    
    def _score_compensation(self, job, text):
        """Score compensation signal"""
        if not job.get('salary_text'):
            # Conservative prior for missing salary
            return 0.3
        
        salary_text = job['salary_text'].lower()
        numbers = re.findall(r'\$?(\d+)[,\s]*(\d*)', salary_text)
        
        if numbers:
            try:
                amounts = []
                for num, decimal in numbers:
                    amount = int(num.replace(',', ''))
                    if decimal:
                        amount = int(f"{num}{decimal}")
                    if 'k' in salary_text:
                        amount *= 1000
                    if 10000 <= amount <= 500000:
                        amounts.append(amount)
                
                if amounts:
                    avg = sum(amounts) / len(amounts)
                    # Normalize to 0-1 (40k = 0, 100k+ = 1)
                    return min(1.0, max(0, (avg - 40000) / 60000))
            except:
                pass
        
        return 0.3
    
    def _score_sales_risk(self, text):
        """Score sales/commission risk"""
        sales_indicators = [
            'commission', '1099', 'door-to-door', 'high ticket',
            'bdr', 'sdr', 'sales development', 'business development rep',
            'cold calling', 'prospecting', 'quota', 'pipeline'
        ]
        matches = sum(1 for ind in sales_indicators if ind in text)
        return min(1.0, matches * 0.25)
    
    def _score_time_trap_risk(self, text):
        """Score time trap risk (meeting-heavy, high-touch)"""
        trap_indicators = [
            'client-facing', 'customer success', 'account management',
            'fast-paced', 'startup environment', 'move fast',
            'on-call', 'nights and weekends', 'urgent', 'tight deadline',
            'scrum master', 'agile coach', 'stakeholder management'
        ]
        
        low_trap_indicators = [
            'async', 'flexible schedule', 'autonomous', 'independent',
            'back office', 'operations', 'compliance', 'monitoring'
        ]
        
        trap_count = sum(1 for ind in trap_indicators if ind in text)
        low_count = sum(1 for ind in low_trap_indicators if ind in text)
        
        score = (trap_count - low_count * 0.7) / len(trap_indicators)
        return max(0, min(1.0, score))
    
    def _score_uncertainty(self, job, text):
        """Score posting uncertainty/vagueness"""
        penalty = 0.0
        
        # No salary
        if not job.get('salary_text'):
            penalty += 0.4
        
        # Short description
        if len(job.get('description', '')) < 200:
            penalty += 0.2
        
        # Vague language
        vague_keywords = ['exciting opportunity', 'dynamic team', 'fast-growing', 'competitive salary', 'tbd']
        vague_count = sum(1 for kw in vague_keywords if kw in text)
        penalty += min(0.3, vague_count * 0.1)
        
        return min(1.0, penalty)
    
    def _score_freshness(self, job):
        """Score posting freshness"""
        if not job.get('posted_date'):
            return 0.5
        
        age_days = (datetime.now() - job['posted_date'].replace(tzinfo=None)).days
        return max(0, 1 - (age_days / 60))  # Decay over 60 days
    
    def _compute_edge_score(self, p_fit, role_fit, remote_cert, comp_signal, 
                           sales_risk, time_trap, uncertainty, freshness):
        """Compute final edge score"""
        # Core viability
        viability = p_fit * role_fit * remote_cert * (1 - sales_risk) * (1 - time_trap * 0.5)
        
        # Value signal
        value = comp_signal * (1 - uncertainty)
        
        # Freshness weight
        recency = 0.5 + freshness * 0.5
        
        # Final edge
        edge = viability * value * recency
        
        # Strict mode penalty for low fit
        if self.strict_mode and p_fit < 0.4:
            edge *= 0.3
        
        # Scale to 0-100
        return edge * 100


# ============================================================================
# STREAMLIT APP
# ============================================================================

def main():
    st.set_page_config(page_title="Job Finder", layout="wide")
    
    st.title("Remote Job Finder - Analyst Dashboard")
    
    # Session state
    if 'jobs_df' not in st.session_state:
        st.session_state.jobs_df = None
    if 'selected_row' not in st.session_state:
        st.session_state.selected_row = None
    
    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        
        if st.button("🔄 Refresh Jobs", type="primary"):
            with st.spinner("Fetching..."):
                jobs = fetch_all_jobs()
                if jobs:
                    st.session_state.raw_jobs = jobs
                    st.success(f"Fetched {len(jobs)} jobs")
                else:
                    st.error("No jobs fetched")
        
        st.divider()
        
        # Filters
        st.subheader("Qualifications")
        
        exp_level = st.selectbox(
            "Experience Level",
            ["0-2 years", "2-4 years", "4-7 years", "7+ years"],
            index=0
        )
        
        certs_input = st.text_input(
            "Certifications (comma-separated)",
            value="Series 7 (pending)"
        )
        certs = [c.strip() for c in certs_input.split(',') if c.strip()]
        
        st.divider()
        st.subheader("Keywords")
        
        default_include = """operations, ops, trade ops, brokerage ops
middle office, back office, client onboarding
KYC, AML, compliance, regulatory, surveillance
reconciliations, reporting, analyst, associate"""
        
        include_input = st.text_area(
            "Include Keywords",
            value=default_include,
            height=100
        )
        include_kw = [k.strip() for k in include_input.replace('\n', ',').split(',') if k.strip()]
        
        default_exclude = """financial advisor, insurance agent, commission
1099, door-to-door, high ticket sales
business development rep, real estate agent"""
        
        exclude_input = st.text_area(
            "Exclude Keywords",
            value=default_exclude,
            height=80
        )
        exclude_kw = [k.strip() for k in exclude_input.replace('\n', ',').split(',') if k.strip()]
        
        st.divider()
        st.subheader("Options")
        
        strict_mode = st.checkbox("Strict Realism Mode", value=False)
        fit_threshold = st.checkbox("Show Only Above Fit Threshold", value=False)
    
    # Main content
    if 'raw_jobs' in st.session_state:
        # Score jobs
        scorer = JobScorer(include_kw, exclude_kw, exp_level, certs, strict_mode)
        
        scored_data = []
        for job in st.session_state.raw_jobs:
            scores = scorer.score_job(job)
            
            # Filter by fit threshold
            if fit_threshold and scores['P_fit'] < 0.4:
                continue
            
            # Exclude keywords filter
            text = f"{job['title']} {job['description']}".lower()
            if any(kw in text for kw in exclude_kw):
                continue
            
            row = {
                'Title': job['title'],
                'Company': job['company'],
                'Source': job['source'],
                'Posted': job['posted_date'].strftime('%Y-%m-%d') if job.get('posted_date') else 'Unknown',
                'URL': job['url'],
                'P_fit': round(scores['P_fit'], 3),
                'RoleFamilyFit': round(scores['RoleFamilyFit'], 3),
                'RemoteCertainty': round(scores['RemoteCertainty'], 3),
                'CompensationSignal': round(scores['CompensationSignal'], 3),
                'SalesRisk': round(scores['SalesRisk'], 3),
                'TimeTrapRisk': round(scores['TimeTrapRisk'], 3),
                'UncertaintyPenalty': round(scores['UncertaintyPenalty'], 3),
                'Freshness': round(scores['Freshness'], 3),
                'EdgeScore': round(scores['EdgeScore'], 2),
                '_signals': scores['signals'],
                '_description': job['description'][:500]
            }
            scored_data.append(row)
        
        if scored_data:
            df = pd.DataFrame(scored_data)
            
            # Display summary
            st.metric("Jobs Displayed", len(df))
            
            # Main table
            st.dataframe(
                df.drop(['_signals', '_description'], axis=1),
                use_container_width=True,
                height=600,
                hide_index=True
            )
            
            # Row selection for details
            st.divider()
            st.subheader("Job Details")
            
            selected_idx = st.number_input(
                "Select row index for details",
                min_value=0,
                max_value=len(df)-1,
                value=0,
                step=1
            )
            
            if selected_idx < len(df):
                row = df.iloc[selected_idx]
                signals = row['_signals']
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**{row['Title']}** @ {row['Company']}")
                    st.markdown(f"[Apply Here]({row['URL']})")
                    st.text(row['_description'])
                
                with col2:
                    st.markdown("**Extracted Signals**")
                    st.json({
                        'is_senior': signals.get('is_senior', False),
                        'is_junior': signals.get('is_junior', False),
                        'years_required': signals.get('years_required'),
                        'creds_required': signals.get('creds_required', []),
                        'training_offered': signals.get('training_offered', False),
                        'has_salary': signals.get('has_salary', False)
                    })
                    
                    st.markdown("**Component Scores**")
                    st.json({
                        'P_fit': float(row['P_fit']),
                        'RoleFamilyFit': float(row['RoleFamilyFit']),
                        'RemoteCertainty': float(row['RemoteCertainty']),
                        'CompensationSignal': float(row['CompensationSignal']),
                        'SalesRisk': float(row['SalesRisk']),
                        'TimeTrapRisk': float(row['TimeTrapRisk']),
                        'UncertaintyPenalty': float(row['UncertaintyPenalty']),
                        'Freshness': float(row['Freshness']),
                        'EdgeScore': float(row['EdgeScore'])
                    })
        else:
            st.info("No jobs match current filters")
    else:
        st.info("Click 'Refresh Jobs' to start")


if __name__ == "__main__":
    main()
