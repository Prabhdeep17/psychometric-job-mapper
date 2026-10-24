# Psychometric Job Mapper

Type a job title. The app lists the psychometric domains that matter most for that job, with a match score and a short reason.

## Files

- `app.py` — the website
- `rank_domains.py` — finds the job and ranks the domains
- `explain_domains.py` — writes the "why it fits" line
- `data/processed/` — the four data tables the app reads

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Optional: copy `.env.example` to `.env` and add a Groq key so the "why it fits" lines are written by the model. Without a key, the app still works and uses a simple built-in sentence. On Streamlit Cloud, put the same values in **Settings → Secrets**.

Scores are preliminary. The domain mappings are still waiting for expert review and must not be used for hiring.
