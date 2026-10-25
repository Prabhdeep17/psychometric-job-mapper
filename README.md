# Psychometric Job Mapper

Type a job title. The app returns the psychometric domains most relevant to that job, an estimated relevance percentage, and a short explanation of why each domain matters.

It recommends assessment areas for a job. The percentage describes the occupation. It does not measure a person, and it is not a hiring test.

Live app: https://prabhdeep17-psychometric-job-mapper-app-0oq174.streamlit.app/

## What a result contains

For a resolved occupation the page shows the top 10 domains:

- Domain name and construct family
- Relevance percentage and a bar
- What it means: the first sentence of the domain definition
- Why it fits: two plain sentences tied to that job

"Show all domains" opens the rest of the eligible set. "Show detailed evidence" adds the matching job tasks and whether the strongest link is direct, partial, or proxy. Domains with no usable rating are labelled "Insufficient evidence" and have no percentage.

## How a title becomes an occupation

Job titles are matched against O\*NET 31.0 (U.S. Department of Labor): **1,016 occupations** and **63,474 titles** (1,016 official, 54,269 alternate, 8,189 reported).

1. The typed title is normalised. Spacing and a trailing plural are ignored, so "datascientist" and "Data Scientists" match "Data Scientist". "HR" is read as "human resource".
2. An exact match to one occupation is accepted.
3. If the same title belongs to more than one occupation, the app lists them and waits. It does not merge occupations or pick one silently.
4. A near-match is offered only when the titles are at least 85% similar, and the user still has to confirm it.
5. After that, the O\*NET-SOC code is the key for scoring. Data Scientists are `15-2051.00`.

An optional second paragraph can be a job description. It can help choose among ambiguous titles, and it is passed to the explanation step. It does not change the percentage.

## How the percentage is calculated

The score is computed in code from O\*NET ratings. The language model does not choose the domains or the number.

Each domain is linked to real O\*NET elements. Each link has a weight:

| Link | Weight | Meaning |
|------|--------|---------|
| Direct | 1.0 | The O\*NET element is the same idea as the domain |
| Partial | 0.6 | The element covers only part of the domain |
| Proxy | 0.3 | The element is only a neighbouring clue |

Removed links and no-match links are excluded. Two links were removed in review: Inductive Reasoning does not count for Pattern Recognition (P013), and Time Sharing does not count for Working Memory (P005). If two domains claim the same element, only one receives the score: the stronger link, then an exact name match, then the lower domain id.

Only these ratings enter the percentage:

- Abilities: Importance and Level. The two percentiles are averaged so one ability is not counted twice.
- Work styles: Importance
- Interests: Occupational Interest

Work-activity rows are stored as background evidence and do not move the percentage. A missing rating stays missing. It is never stored as zero.

For each occupation, a rating is turned into a percentile against the other occupations on that same element and scale. Then, for one occupation and one domain:

- **Signal** = weighted average of those percentiles
- **Mapping strength** = average of the link weights
- **Relevance %** = signal × mapping strength, rounded to one decimal

A single direct link shows a number close to that percentile. A single proxy link shows about 30% of the percentile, because the connection itself is weak.

The app keeps domains marked `provisional_evidence` or `requires_review`. It drops `no_match`, `no_evidence`, and `missing_source_data`. Inside each family it keeps up to five domains using the rank stored in the mapping file, then sorts that set by the relevance percentage and shows ten.

## The 50 domains

`psychometric_domains.csv` holds domains P001–P050 in six families. Abilities, personality, interests, and work values stay separate.

| Family | Domains |
|--------|---------|
| Cognitive abilities and reasoning | P001–P014. Includes Fluid Reasoning, Quantitative Reasoning, Verbal Comprehension, Working Memory, Pattern Recognition |
| Personality traits and behavioural work styles | P015–P025. Big Five and HEXACO traits, plus Dependability, Achievement, Initiative, Adaptability, Integrity |
| Emotional regulation and self-management | P026–P032. Emotion regulation, emotional intelligence, stress tolerance, self-control, grit |
| Interpersonal orientations and social behaviour | P033–P038. Empathy, social perceptiveness, cooperation, leadership, conflict, service |
| Motivation and work values | P039–P044 |
| Vocational interests | P045–P050. Holland RIASEC: Realistic, Investigative, Artistic, Social, Enterprising, Conventional |

P041–P044 (Altruism, Autonomy, Recognition, Structure and Security) have no O\*NET element. They are `no_match` for every occupation and never receive a percentage.

The links were checked against CHC, the Big Five, HEXACO, Holland RIASEC, Mayer-Salovey emotional intelligence, self-determination theory, and Duckworth grit. A licensed psychometrician has not independently validated the crosswalk.

## What the language model does

Groq writes only the "Why it fits" sentences. The prompt receives the occupation, the domain definition, the linked element names, the link strength, and up to two real task statements. The reply must be JSON. Unknown domains, invented evidence ids, and hiring claims such as "must possess" are rejected.

If `LLM_MODEL` is empty, the app tries `openai/gpt-oss-20b`, then `openai/gpt-oss-120b`, then `qwen/qwen3.8-27b`. A rate limit moves to the next model. Requests go out in batches of six domains.

With no key, a timeout, or a rejected reply, that domain gets a fixed sentence instead. The sentence still names the stored occupational requirement and says when the link is only provisional. The percentage stays the same.

## Data

O\*NET 31.0 ratings describe job requirements. They come from job analysts and incumbents.

| Table | Rows | Description |
|-------|------|-------------|
| `psychometric_domains.csv` | 50 | Domain name, family, definition, theory, and limits |
| `job_title_lookup.csv` | 63,474 | Typed title to an occupation code |
| `occupation_domain_mapping.csv` | 50,800 | Full grid: 1,016 occupations × 50 domains |
| `occupation_domain_evidence.csv` | 117,130 | O\*NET ratings behind each cell of that grid |

## Project structure

| File | Purpose |
|------|---------|
| `app.py` | Streamlit chat interface. This is what Streamlit Cloud runs |
| `api.py` | FastAPI: `GET /health`, `GET /roles/search`, `POST /analyze` |
| `rank_domains.py` | Title matching, evidence, scoring, and ranking |
| `explain_domains.py` | Groq explanations and the deterministic fallback |
| `data/processed/` | The four CSV tables above |
| `tests/` | 42 automated tests. Groq is mocked |

## Install

```bash
pip install -r requirements.txt
```

## Configure Groq

```bash
cp .env.example .env
```

Set `GROQ_API_KEY` in `.env` to a key from [console.groq.com](https://console.groq.com/keys). Leave `LLM_MODEL` blank for automatic model switching, or set it to pin one model.

On Streamlit Cloud, put the same values in **Settings → Secrets**:

```toml
GROQ_API_KEY = "your-new-key"
LLM_MODEL = ""
```

The key is read from the environment or from Streamlit secrets. It is not written into the page or the API response. Without a key, the app still runs and uses the fallback sentences.

## Run

```bash
streamlit run app.py
uvicorn api:app --port 8000
python3 -m pytest tests/ -q
```

API docs: http://127.0.0.1:8000/docs

## Limitations

- The relevance percentage is a project heuristic. It is not a calibrated probability of job success and not a validated psychometric score.
- Percentiles from different O\*NET instruments are not scientifically comparable across construct families.
- The crosswalk was reviewed against published frameworks. It is not an independent validation by a licensed psychometrician.
- Work-value domains P041–P044 stay unscored.
- Do not use this output to hire, reject, or rank a person.
