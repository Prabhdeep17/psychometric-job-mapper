# Psychometric Job Mapper

Enter a job title and get the psychometric domains that matter most for
that role — ranked, with a short explanation of what each one means and
why it fits the job.

The title is matched to an O*NET occupation. If more than one occupation
could match, the app asks which one you mean instead of guessing. Scores
come from a precomputed mapping of 1,016 occupations × 50 domains, built
from O*NET 31.0 data. If a Groq key is set, the LLM writes a one-sentence
explanation; it cannot change scores or rankings. Without a key, a simple
built-in explanation is used.

## Files

```
app.py              Streamlit chat interface
api.py              FastAPI backend
analysis.py         Data loading, title matching, evidence, ranking
llm.py              Groq requests and response checks
data/processed/     Four CSV files the app reads
tests/              Automated tests
```

The four CSVs are `occupation_domain_mapping.csv` (50,800 rows),
`occupation_domain_evidence.csv` (120,770 records),
`psychometric_domains.csv` (50 domains) and `job_title_lookup.csv`
(63,474 titles). Source: O*NET 31.0, U.S. Department of Labor (CC BY 4.0).

## Run

```bash
pip install -r requirements.txt

streamlit run app.py            # http://localhost:8501
uvicorn api:app --port 8000     # docs at http://127.0.0.1:8000/docs
python -m pytest tests/ -q
```

## Groq (optional)

```bash
cp .env.example .env
```

Then put your key in `.env`:

```
LLM_PROVIDER=groq
LLM_MODEL=openai/gpt-oss-20b
GROQ_API_KEY=
```

`.env` is git-ignored. The key that was previously in `.env.example` is
exposed — revoke it at console.groq.com and use a new one. On Streamlit
Cloud, add the same values under Settings → Secrets.

## Limitations

Scores are preliminary, occupation-relative indicators, not validated
psychometric scores. The domain mappings still need qualified human
review. Occupational ratings describe jobs, not people. Do not use this
output for hiring decisions.

Research material (original O*NET data, build scripts, reports and the
review workbook) lives in `../psychometric_job_mapper_research_archive/`.
