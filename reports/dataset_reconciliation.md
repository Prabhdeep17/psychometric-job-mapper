# Dataset reconciliation

This report compares the corrected-master workbook with the CSV used at runtime.
It is a structural check. It is not a psychometric validation.

- CSV: `/home/omen/Desktop/llm2/data/psychometric_role_domain_mapping_corrected_master.csv`
- CSV SHA-256: `525a995345f5b779790df023ffcec04a3a4b3a20de49eef3c4ae1ec0189cdbf1`
- Workbook: `/home/omen/Desktop/llm2/data/psychometric_role_domain_mapping_corrected_master.xlsx`
- Workbook SHA-256: `b4fd7f6cbd326ee0ee5a5706c1487a0c5d44b5160ad9c9869e7c8db0c0d87186`
- Roles: 20
- Domains: 50
- Mappings: 1000
- Duplicate role–domain rows: 0
- Score range: 1–5

## Reader note

Stock openpyxl cannot open this workbook because a sheet title contains '*'. The application reads an in-memory copy whose sheet title is made legal. Stored cell values, including the 1–5 scores, are not modified.

## Mapping sheet

- Sheet: All 1000 mappings
- Rows: 1000
- Rows matched to the CSV on role + domain_id: 1000
- Shared fields that differ: none
- Scores agree: True

No stored 1–5 score was changed to force a match.

## Supplemental sheets

- **Top domains by role**: 364 rows, matched 364, differences none.
  Extra columns kept as metadata: selection_note.
  Extra workbook columns are preserved as reference metadata and are not score conflicts: selection_note
- **Expert review queue**: 717 rows, matched 717, differences none.
  Extra columns kept as metadata: review_reason.
  Extra workbook columns are preserved as reference metadata and are not score conflicts: review_reason
- **50 domain definitions**: 50 rows, matched 50, differences none.

## Missing values in the CSV

- `O*NET_standardized_work_style_impact_0_to_100`: 753
- `O*NET_work_style_crosswalk`: 560

## Weak or composite occupational proxies

Entrepreneur, Healthcare Professional, Machine Learning Engineer

O*NET 0–100 benchmarks stay in their own columns and sheet. They were not converted into `project_relevance_score_1_to_5`.

