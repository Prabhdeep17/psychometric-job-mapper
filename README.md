# Psychometric Job Mapper

Type a job title or pick an occupation. The app returns the psychometric domains most relevant to that job, an estimated relevance percentage, and a clear explanation of why each domain matters.

It recommends assessment areas for a job. It does not measure a person, and it is not a hiring test.

## How it works

1. **Title resolution** — The title is matched to an O\*NET occupation (1,016 occupations). If several fit, the app lists them for you to pick.
2. **Domain scoring** — For each of 50 psychometric domains, the app calculates an estimated relevance percentage from O\*NET evidence. Direct links count as 1.0, partial as 0.6, proxy as 0.3. Removed and no-match links are excluded.
3. **Display** — Top 10 domains are shown with a relevance %, definition, and an explanation of why that domain fits the job. You can expand to see all domains.
4. **LLM explanations** — Groq writes the "Why it fits" text. The app auto-switches between available Groq models on rate limits. If all models are unavailable, a deterministic fallback is used. Scores, ranks, and data stay local.

## Project structure

| File | Purpose |
|------|---------|
| `app.py` | Streamlit web interface |
| `api.py` | FastAPI backend (`/health`, `/roles/search`, `/analyze`) |
| `rank_domains.py` | Title matching, evidence loading, scoring, and ranking |
| `explain_domains.py` | Groq LLM explanations with auto model switching and fallback |
| `data/processed/` | Four runtime CSV tables |
| `tests/` | 42 automated tests |

## Data

| Table | Rows | Description |
|-------|------|-------------|
| `psychometric_domains.csv` | 50 | Domains P001–P050 across 6 construct families |
| `occupation_domain_mapping.csv` | 50,800 | Complete grid: 1,016 occupations × 50 domains |
| `occupation_domain_evidence.csv` | 117,130 | O\*NET ratings backing each mapping |
| `job_title_lookup.csv` | 63,474 | Job title → occupation lookup |

All domain-to-O\*NET crosswalk links have been reviewed against established psychometric frameworks (CHC, Big Five, HEXACO, Holland RIASEC, Mayer-Salovey, SDT, Duckworth Grit). See the psychometric review report for details.

## Install

```bash
pip install -r requirements.txt
```

## Configure Groq

```bash
cp .env.example .env
```

Open `.env` and set `GROQ_API_KEY` to your key from [console.groq.com](https://console.groq.com/keys).

Leave `LLM_MODEL` blank for automatic model switching (recommended), or pin a specific model.

Without a key, the app still runs using deterministic fallback explanations.

## Run

```bash
streamlit run app.py
uvicorn api:app --port 8000
python3 -m pytest tests/ -q
```

API docs: http://127.0.0.1:8000/docs

## Limitations

- Estimated domain relevance is a project heuristic — not a calibrated probability, not an individual test score, and not a validated hiring result.
- Domain-to-O\*NET crosswalk links have been reviewed against psychometric theory but are not independently validated by a licensed psychometrician.
- Work-value domains P041–P044 have no O\*NET element and remain unscored.
- Do not use this output to hire or reject anyone.
