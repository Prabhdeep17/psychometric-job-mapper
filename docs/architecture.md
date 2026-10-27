# Architecture Documentation

## System Overview

The Psychometric Domain Mapping AI is a modular Python application that maps job roles to psychometric domains using a curated reference dataset of 20 roles × 50 domains (1,000 mappings), with optional LLM-powered explanations via Groq.

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────┐
│                    Streamlit UI (app.py)                  │
│  ┌──────────┬──────────┬──────────┬──────────┬────────┐  │
│  │ Role Map │  Profile │ Domain   │ JD       │Evidence│  │
│  │          │          │ Explorer │ Analysis │ Review │  │
│  └────┬─────┴────┬─────┴────┬─────┴────┬─────┴───┬────┘  │
└───────┼──────────┼──────────┼──────────┼─────────┼───────┘
        │          │          │          │         │
┌───────▼──────────▼──────────▼──────────▼─────────▼───────┐
│                    Application Layer (src/)               │
│                                                          │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │ role_matcher │  │ranking_engine│  │   prompts.py    │  │
│  │  (alias +   │  │(deterministic│  │ (LLM templates) │  │
│  │   fuzzy)    │  │  sort+ties)  │  │                 │  │
│  └──────┬──────┘  └──────┬───────┘  └────────┬────────┘  │
│         │               │                    │           │
│  ┌──────▼───────────────▼────────────────────▼────────┐  │
│  │              data_loader.py                        │  │
│  │   (schema validation, caching, typed accessors)    │  │
│  └─────────────────────┬──────────────────────────────┘  │
│                        │                                 │
│  ┌─────────────────────▼──────────────────────────────┐  │
│  │              groq_router.py                        │  │
│  │ (catalogue, fallback, retry, error classification) │  │
│  └─────────────────────┬──────────────────────────────┘  │
│                        │                                 │
│  ┌─────────────────────▼──────────────────────────────┐  │
│  │           output_validation.py                     │  │
│  │   (domain ID checks, JSON parsing, sanitization)  │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │              config.py                             │  │
│  │  (env vars, typed dataclasses, defaults)           │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
        │                              │
        ▼                              ▼
┌───────────────┐            ┌──────────────────┐
│  data/*.csv   │            │  Groq API        │
│  (reference   │            │  /v1/models      │
│   dataset)    │            │  /v1/chat/...    │
└───────────────┘            └──────────────────┘
```

## Module Responsibilities

### config.py
- Loads all settings from environment variables and `.env` files
- Typed dataclasses (`GroqConfig`, `AppConfig`) with validation
- Centralises model preference lists, TTLs, and retry limits

### data_loader.py
- Reads the corrected-master CSV with BOM-aware encoding
- Validates all 25 expected columns, score range (1–5), and uniqueness
- Caches the loaded DataFrame (singleton pattern)
- Typed accessors: `get_roles()`, `get_domains()`, `get_role_domains(role)`, etc.
- `DatasetInfo` summary for the evidence-review page

### role_matcher.py
- Three-tier matching: exact (case-insensitive) → alias table → fuzzy (SequenceMatcher)
- 80+ aliases mapping common job titles to the 20 reference roles
- Returns `RoleMatch` with match type, confidence, proxy warning, and alternatives
- Never silently pretends an unknown title is a verified role

### ranking_engine.py
- Deterministic retrieval: sorts by `project_relevance_score_1_to_5` descending
- Dense ranking with ties respected at the top-N cutoff boundary
- Never modifies stored scores
- `filter_domains()` for family/score/evidence/review filtering
- `domains_to_dataframe()` for CSV export

### workbook_loader.py
- Reads every sheet in the corrected-master workbook
- Works around the sheet title `O*NET source benchmarks`, which contains `*` and is rejected by a normal openpyxl load
- Compares shared fields with the CSV and records differences explicitly
- Does not copy O*NET 0–100 values into the 1–5 score

### groq_router.py
- Queries `GET https://api.groq.com/openai/v1/models`
- Caches the catalogue with `GROQ_MODEL_CATALOG_TTL_SECONDS`
- Keeps configured IDs that are present, active, and not audio/speech/guard models
- Routes chat completions through that ordered list, capped by `GROQ_MAX_FALLBACKS`
- Error actions: ABORT (auth or malformed input), ACCOUNT_LIMIT (organization/account 429), RETRY_SAME (timeout, 5xx, model-specific 429), TRY_NEXT (missing or decommissioned model), REDUCE_CONTEXT
- Refreshes the catalogue before switching away from a missing or retired model
- Does not treat a model switch as a way around an account-wide quota
- Model pinning disables automatic switching
- API keys are redacted from stored error text

### llm_service.py
- Calls the router for explanations and job-description analysis
- Drops unknown domain IDs into an unverified list
- Removes any numeric `score` field the model adds
- Returns a structured error when the key is missing, so ranking still renders

### prompts.py
- System prompts with strict rules (no inventing scores/citations/diagnoses)
- `build_explain_prompt()`: explains stored domain relevance
- `build_jd_analysis_prompt()`: analyses a job description against reference domains
- All LLM suggestions labelled as model-generated

### output_validation.py
- Validates domain ID references against the known set
- Parses structured JSON from LLM responses (with code-fence removal)
- Sanitises output for invented-score patterns

## Data Flow

### Deterministic Path (no LLM required)
1. User enters a job title
2. `role_matcher` resolves to a reference role (with proxy warning if needed)
3. `ranking_engine` retrieves and sorts all 50 domain mappings
4. Top-N domains (with ties) displayed in the UI

### LLM-Enhanced Path
1. Deterministic path completes first
2. User requests an AI explanation
3. `prompts.py` builds a structured message with domain context
4. `groq_router` routes to the first available model
5. `output_validation` checks the response
6. Explanation displayed with model attribution and caveats

## Key Design Decisions

1. **CSV as primary runtime source**: Fast, portable, no database needed for 1,000 rows
2. **httpx over openai SDK**: Lighter dependency, full control over error handling
3. **Dense ranks**: Equal scores get equal rank numbers (ties respected)
4. **Tie inclusion at cutoff**: Requesting top-10 may return 12 if scores tie at position 10
5. **Separate stored vs. LLM-generated**: Stored scores are never modified by LLM output
6. **Graceful degradation**: All dataset features work without an API key
