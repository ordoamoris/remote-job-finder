# Remote Job Finder

A simple Streamlit app that fetches remote jobs from multiple sources and ranks them based on your constraints.

## Deploy to Streamlit Community Cloud

1. **Create a GitHub repository**
   - Go to github.com → New repository
   - Name it `remote-job-finder` (public)
   - Click "uploading an existing file"
   - Upload `app.py`, `requirements.txt`, and this `README.md`
   - Commit changes

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository, branch `main`, file `app.py`
   - Click "Deploy"

3. **Bookmark your URL**
   - You'll get a URL like `https://remote-job-finder-xxxxx.streamlit.app`
   - Bookmark it for daily use

## Daily Use

Open your Streamlit URL, click the "🔄 Refresh Jobs" button in the sidebar, and review the ranked list of remote jobs. Jobs are scored based on remote certainty, keyword matches (ops/compliance/back-office roles), salary estimates, and likelihood of having consistent slack time. Adjust the sidebar filters (keywords, salary, age) to customize results. Click "Apply Here" on jobs that interest you. That's it - come back tomorrow and repeat.

## Local Testing (Optional)

```bash
pip install -r requirements.txt
streamlit run app.py
```
