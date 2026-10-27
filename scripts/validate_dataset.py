#!/usr/bin/env python3
"""Compare the workbook with the CSV and write a reconciliation report.

Run from the project root:

    python3 scripts/validate_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config
from src.data_loader import DatasetLoader


def main() -> int:
    config = get_config()
    loader = DatasetLoader(config=config)
    loader.load()
    pack = loader.get_reference_pack()
    report = pack.reconciliation
    info = loader.get_dataset_info()

    lines = [
        "# Dataset reconciliation",
        "",
        "This report compares the corrected-master workbook with the CSV used at runtime.",
        "It is a structural check. It is not a psychometric validation.",
        "",
        f"- CSV: `{report.csv_path}`",
        f"- CSV SHA-256: `{report.csv_sha256}`",
        f"- Workbook: `{report.xlsx_path}`",
        f"- Workbook SHA-256: `{report.xlsx_sha256}`",
        f"- Roles: {info.num_roles}",
        f"- Domains: {info.num_domains}",
        f"- Mappings: {info.num_mappings}",
        f"- Duplicate role–domain rows: {info.duplicate_role_domain}",
        f"- Score range: {info.score_range[0]}–{info.score_range[1]}",
        "",
        "## Reader note",
        "",
        report.reader_note or "No reader note.",
        "",
        "## Mapping sheet",
        "",
    ]
    if report.load_error:
        lines.append(f"Workbook load error: {report.load_error}")
    elif report.mappings:
        item = report.mappings
        lines.extend(
            [
                f"- Sheet: {item.sheet_name}",
                f"- Rows: {item.rows}",
                f"- Rows matched to the CSV on role + domain_id: {item.matched_rows}",
                f"- Shared fields that differ: {item.differing_fields or 'none'}",
                f"- Scores agree: {report.scores_agree}",
                "",
                "No stored 1–5 score was changed to force a match.",
                "",
            ]
        )
    lines.append("## Supplemental sheets")
    lines.append("")
    if not report.supplemental:
        lines.append("No supplemental comparisons were produced.")
    for item in report.supplemental:
        lines.append(
            f"- **{item.sheet_name}**: {item.rows} rows, matched {item.matched_rows}, "
            f"differences {item.differing_fields or 'none'}."
        )
        if item.extra_columns:
            lines.append(f"  Extra columns kept as metadata: {', '.join(item.extra_columns)}.")
        if item.note:
            lines.append(f"  {item.note}")
    lines.extend(
        [
            "",
            "## Missing values in the CSV",
            "",
        ]
    )
    if not info.missing_values:
        lines.append("None.")
    else:
        for column, count in sorted(info.missing_values.items(), key=lambda pair: -pair[1]):
            lines.append(f"- `{column}`: {count}")
    lines.extend(
        [
            "",
            "## Weak or composite occupational proxies",
            "",
            ", ".join(info.weak_proxy_roles) or "None labeled weak or composite.",
            "",
            "O*NET 0–100 benchmarks stay in their own columns and sheet. "
            "They were not converted into `project_relevance_score_1_to_5`.",
            "",
        ]
    )

    output = Path(__file__).resolve().parent.parent / "reports" / "dataset_reconciliation.md"
    output.parent.mkdir(exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    print(info.reconciliation_summary)
    return 0 if report.scores_agree else 1


if __name__ == "__main__":
    raise SystemExit(main())
