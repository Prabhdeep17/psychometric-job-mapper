# Psychometric domain mapping

This app shows which of 50 work-related domains are stored as most relevant for a reference job role. Rankings come from the dataset. A Groq model can explain those rows or read a job description. The model cannot change the stored 1–5 scores.

The scores are **provisional estimates**. They are not a personality test, not a mental-health assessment, and not a score of whether a person should be hired.

## What works without an API key

You can use these as soon as the app starts:

- Role mapping for the 20 stored roles
- Alias and closest-title proxies, labeled as proxies
- Full 50-domain profile, filters, and CSV download
- Domain explorer, including construct-family overlap and O*NET crosswalk notes
- Dataset and evidence page

These need `GROQ_API_KEY`:

- Written explanation of a stored ranking
- Job-description analysis
- Model catalogue and health prompt

If Groq is down, the stored ranking still appears.

## Install

Use Python 3.10 or newer.

```bash
cd /path/to/this/project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`.

## Groq API key

1. Open [https://console.groq.com](https://console.groq.com) and create an API key.
2. Copy `.env.example` to `.env` in this folder.
3. Set `GROQ_API_KEY` to your key.
4. Leave the model lines as they are, or change them after you see the live model list (below).
5. Start or restart the app. The key stays in `.env`. It is not written into the pages.

## Run the app

```bash
streamlit run app.py
```

Open the local URL Streamlit prints, usually [http://localhost:8501](http://localhost:8501).

Try **Data Scientist** or **Data Analyst** on the Role mapping tab. The score on each card is the stored `project_relevance_score_1_to_5` value. The importance label is the dataset’s `score_interpretation`. Domains with the same score share a rank. If you ask for the top 10 and several domains tie at the cutoff, they are all shown.

An unknown title is not treated as a verified role. Pick one of the closest stored roles. That choice is a **proxy**: the scores belong to the stored role, not to the title you typed.

Aliases such as “ML Engineer” are convenience shortcuts in the app. They are not extra occupations in the dataset, and they are labeled as proxies.

## Run the tests

```bash
python3 -m pytest tests -q
```

Dataset checks:

```bash
python3 scripts/validate_dataset.py
python3 scripts/generate_sample_output.py
python3 scripts/build_evaluation_report.py
```

Sample rankings are written to `reports/`. The SME file `docs/sme_review_template.csv` is a blank review form. Empty reviewer columns are not validation results.

## How model fallback works

The app calls Groq at `https://api.groq.com/openai/v1`.

1. It loads `GROQ_MODEL_PREFERENCE`, then `GROQ_FALLBACK_MODELS`, and drops duplicates.
2. It calls `GET /models` and keeps only configured IDs that are active text-chat models. Whisper, speech, and guard models are skipped.
3. It calls the first remaining model.
4. Timeouts, HTTP 5xx, and a **model-specific** HTTP 429 are retried up to `GROQ_MAX_RETRIES`. The app waits for `Retry-After` only up to `GROQ_MAX_RETRY_WAIT_SECONDS`.
5. If the model is missing, retired, or decommissioned, the catalogue is refreshed and the next model is tried.
6. After the retries are used up on a model-specific failure, the next model is tried, up to `GROQ_MAX_FALLBACKS` extra models.
7. HTTP 401 or 403 stops immediately. Check the key.
8. A malformed request (HTTP 400 that is not a retired model or a too-long input) stops immediately and shows the error.
9. If the input is too long, the app shortens it once. If it is still too long, you need a shorter job description.
10. If the rate limit is for the whole account or organization, the app stops and tells you to wait. Switching models does not raise that kind of quota.
11. If every allowed model fails, the page shows an error and a Retry button. The dataset scores stay as they were.

The Model status tab can pin one model. A pinned model is not swapped for a fallback.

The catalogue is cached for `GROQ_MODEL_CATALOG_TTL_SECONDS`. Use **Refresh model catalogue** after Groq adds or removes a model.

## Change models without editing code

1. Open Model status and refresh the catalogue.
2. Copy a model ID that is marked eligible for text.
3. Edit `.env`:

```bash
GROQ_MODEL_PREFERENCE=the-id-you-copied,llama-3.1-8b-instant
GROQ_FALLBACK_MODELS=llama-3.1-8b-instant
```

4. Restart the app.

Do not put a model ID in `app.py`. If none of the configured IDs are in your account’s catalogue, the Model status page says so and ranking still works.

## Replace the dataset

1. Put the new CSV and workbook in `data/` using the same file names, or point the app at new paths by editing `src/config.py` (`CSV_FILENAME` and `XLSX_FILENAME`) only if the names must change.
2. Keep the column `project_relevance_score_1_to_5` as the 1–5 estimate. Do not overwrite it with an O*NET 0–100 number.
3. Run `python3 scripts/validate_dataset.py`.
4. Read `reports/dataset_reconciliation.md`. If scores differ between the CSV and the workbook sheet `All 1000 mappings`, the report lists the fields. Do not silently pick one.
5. Run `python3 -m pytest tests -q`. Update tests that hard-code the old row counts if the new file is a different size.
6. Restart Streamlit. The app caches the file until the process restarts.

The shipped workbook has a sheet title, `O*NET source benchmarks`, that contains `*`. A normal openpyxl load rejects that title. The app reads an in-memory copy with a legal title. It does not edit the workbook you downloaded.

On this version, the CSV and the workbook’s 1,000 mapping rows agree, including the 1–5 scores. Extra workbook columns such as `selection_note` and `review_reason` are reference notes, not a second score.

## Deploy

The interface is a Streamlit app (`streamlit run app.py`).

- Install the same `requirements.txt` on the server.
- Set `GROQ_API_KEY` and the model variables in the host’s secret store, not in git.
- On Streamlit Community Cloud, add those names in the app’s Secrets. Streamlit exposes secret keys as environment variables to the process.
- Do not commit `.env`, `.streamlit/secrets.toml`, or a real key inside a zip.
- The CSV and workbook can be deployed with the app. They contain occupational reference data, not your API key.
- Local review notes, if you use them, are stored in `data/review_flags.sqlite` on that machine only.

## Project layout

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit interface |
| `src/data_loader.py` | CSV load and checks |
| `src/workbook_loader.py` | Workbook read and CSV comparison |
| `src/role_matcher.py` | Exact, alias, and fuzzy title matching |
| `src/ranking_engine.py` | Sort by the stored score; keep ties |
| `src/groq_router.py` | Model catalogue, retries, and fallback |
| `src/prompts.py` | Instructions sent to the model |
| `src/output_validation.py` | JSON and domain-ID checks |
| `src/llm_service.py` | Optional explanations |
| `src/review_flags.py` | Local SQLite notes |
| `data/` | Reference CSV and workbook |
| `tests/` | Automated tests |
| `docs/` | Architecture, validation method, SME template |
| `reports/` | Reconciliation, samples, evaluation write-up |

## Limits you should keep in view

- Every row’s evidence status is “Structured expert-style estimate; not empirically validated.”
- O*NET links are occupational references. Several roles are partial, composite, or weak proxies. The Evidence tab lists them.
- Domains that share a construct family, or the same O*NET work style, are not independent measures.
- Model text can be wrong. It is labeled as model-generated and is not written back to the dataset.
