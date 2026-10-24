# Psychometric Job Mapper

Enter a job title and get the psychometric domains that matter most for
that role — ranked, with a plain-English explanation of what each domain
means and why it fits the job.

## How it works

1. The job title is matched to an O*NET occupation. Ambiguous titles
   (e.g. "Data Analyst") are never guessed — the user picks from a list.
2. The occupation's domain scores are read from a precomputed mapping of
   1,016 occupations × 50 psychometric domains, built from O*NET 31.0 data.
3. Domains are ranked and, if a Groq API key is configured, the LLM writes
   a one-sentence explanation for each. The LLM cannot change scores or
   rankings, and every explanation is checked against local evidence.
   Without a key, a simple built-in explanation is used instead.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit chat interface |
| `api.py` | FastAPI backend (`/health`, `/roles/search`, `/analyze`) |
| `analysis.py` | Data loading, title matching, evidence, ranking, analysis |
| `explanations.py` | Groq requests, response validation, preview fallback |
| `data/processed/` | The four CSV datasets the app reads (see below) |
| `tests/` | Automated tests (`test_core.py`, `test_api.py`) |

Runtime data: `occupation_domain_mapping.csv` (50,800 rows),
`occupation_domain_evidence.csv` (120,770 evidence records),
`psychometric_domains.csv` (the 50 domains), `job_title_lookup.csv`
(63,474 job titles). Source: O*NET 31.0, U.S. Department of Labor (CC BY 4.0).

## Run

```bash
pip install -r requirements.txt

streamlit run app.py            # interface → http://localhost:8501
uvicorn api:app --port 8000     # backend (separate terminal) → docs at /docs
python -m pytest tests/ -q      # tests
```

## Groq (optional)

Copy `.env.example` to `.env` and add your key:

```
LLM_PROVIDER=groq
LLM_MODEL=openai/gpt-oss-20b
GROQ_API_KEY=your-key-here
```

`.env` is excluded from Git — never commit it. The key that was previously
in `.env.example` is exposed: revoke it at console.groq.com and use a new one.
On Streamlit Cloud, add the same values under **Settings → Secrets**.

## Limitations

Scores are preliminary, occupation-relative indicators, not validated
psychometric scores. The domain mappings are still pending qualified human
review, occupational ratings describe jobs rather than people, and the
output must not be used for hiring decisions.

Research material (original O*NET data, build scripts, reports and the
review workbook) is kept outside this project in
`../psychometric_job_mapper_research_archive/`.
