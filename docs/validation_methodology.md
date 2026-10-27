# Validation Methodology

## Scope

This document describes the validation approach for the Psychometric Domain Mapping AI project. It distinguishes between structural/technical validation (automated) and occupational/scientific validation (requires external experts).

## 1. Structural Validation (Automated)

These checks are implemented in the test suite (`tests/`).

### Dataset Integrity
- All 25 expected columns present
- 20 unique roles × 50 unique domains = 1,000 rows
- No duplicate role–domain pairs
- Relevance scores constrained to integers 1–5
- Domain IDs follow `D##` format (D01–D50)
- Each role has exactly 50 domain mappings
- Required fields (role, domain_id, score, rationale, evidence_status) are never null

### Ranking Correctness
- Domains sorted by score descending for every role
- Ties at the top-N cutoff boundary are included
- Dense ranking: equal scores → equal rank numbers
- Stored scores are never modified by the ranking engine
- All 20 roles can be retrieved and ranked

### Role Matching
- All 20 stored roles match exactly (case-insensitive)
- All aliases resolve to valid reference roles
- Unknown titles return alternatives with proxy warnings
- Empty/special-character inputs handled gracefully

### Model Router
- Error classification (401/403 → ABORT, 404 → TRY_NEXT, 429/5xx → RETRY_SAME, 400+context → REDUCE_CONTEXT)
- Retry count bounded by configuration
- Fallback sequence follows preference order
- Auth errors abort immediately (no model cycling)
- Malformed-input errors abort immediately
- All-fail scenario returns clear error message

### Output Validation
- LLM domain references checked against known IDs
- Structured JSON parsed correctly (including code-fenced JSON)
- Unknown domains flagged with warnings

## 2. Occupational Evidence Review (Requires Experts)

These items cannot be validated automatically and require subject-matter expert review.

### What Needs Expert Review
- Whether each domain's 1–5 score accurately reflects its importance to the role
- Whether the O*NET proxy occupations are appropriate
- Whether the rationale text correctly explains the score
- Whether any domains are over- or under-rated
- Whether construct definitions align with psychometric literature
- Whether construct overlap is appropriately noted

### SME Review Template

The file `docs/sme_review_template.csv` provides a structured template for expert evaluation. It includes:

| Field | Description |
|-------|-------------|
| role | The job role being evaluated |
| domain_id | Domain identifier (D01–D50) |
| domain_name | Domain name |
| stored_score | Current 1–5 score from the dataset |
| sme_independent_score | Expert's independent rating (1–5) |
| sme_agrees_with_rationale | Yes / Partially / No |
| sme_comments | Free-text expert commentary |
| sme_onet_proxy_appropriate | Yes / Partially / No |
| sme_review_date | Date of review |
| sme_reviewer_id | Anonymous reviewer identifier |

### Important Limitations

1. **No predictive validity data**: No scores have been validated against job performance outcomes.
2. **No inter-rater reliability**: Scores reflect one estimation process, not consensus among multiple experts.
3. **No convergent/discriminant validity**: The 50 domains have not been factor-analysed or correlated with established psychometric instruments.
4. **Provisional nature**: All scores are labelled as "Structured expert-style estimate; not empirically validated."

## 3. LLM Output Evaluation

### What Is Tested
- LLM references only known domain IDs
- LLM does not invent numerical scores
- LLM response parses as valid JSON (for structured tasks)
- LLM labels suggestions as model-generated

### What Cannot Be Automatically Validated
- Factual accuracy of LLM explanations
- Quality of reasoning
- Appropriateness of domain suggestions for job descriptions
- Whether the LLM inappropriately invents citations (requires human review)

## 4. Dataset file agreement

`scripts/validate_dataset.py` compares the CSV with the workbook sheet `All 1000 mappings`. Supplemental sheets (`Top domains by role`, `Expert review queue`, `50 domain definitions`) are checked on shared fields. Extra columns are reported as metadata. A difference is written into `reports/dataset_reconciliation.md` instead of being resolved by overwriting a score.

The workbook sheet title `O*NET source benchmarks` contains `*`. Stock openpyxl refuses that title. The loader reads a temporary in-memory copy. The file in `data/` is not rewritten.

## 5. Reporting

### Test Results
Run `pytest tests/ -v --tb=short` to generate a complete test report. Results cover:
- Dataset integrity: schema, scores, uniqueness
- Role matching: exact, alias, fuzzy, no-match
- Ranking: sorting, ties, all-role retrieval, score consistency
- Router: error handling, fallback, retries
- Validation: domain references, JSON parsing

### What the Report Does NOT Claim
- Scientific validation of domain relevance scores
- Predictive accuracy for job performance
- Psychometric validity of the 50-domain framework
- That simulated expert ratings represent real validation
