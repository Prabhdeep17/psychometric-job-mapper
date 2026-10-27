# Evaluation report

This file records structural checks against the shipped dataset.
It does **not** report predictive validity, inter-rater reliability, or job-performance accuracy.
Those have not been measured. Blank SME columns in `docs/sme_review_template.csv` are a review form, not results.

## Structural checks

- Roles: 20
- Domains: 50
- Mappings: 1000
- CSV/workbook score agreement: True
- Evidence label: Structured expert-style estimate; not empirically validated
- Weak or composite proxies: Entrepreneur, Healthcare Professional, Machine Learning Engineer

All structural checks in this script passed.

## What was not evaluated

- Whether a 1–5 score is the right importance judgment for a real job.
- Whether an O*NET proxy occupation is the right benchmark for a specific employer.
- Whether domain scores predict performance, hiring outcomes, or training needs.
- Whether model explanations are factually adequate. Automated tests only check shape, known domain IDs, and score-invention patterns.

See `docs/validation_methodology.md` for how a human reviewer would use the SME template.

