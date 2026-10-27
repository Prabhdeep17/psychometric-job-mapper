#!/usr/bin/env python3
"""Write the structural evaluation report.

The report separates checks this repository can perform from occupational
evidence review and predictive validity, which have not been measured.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config
from src.data_loader import DatasetLoader
from src.ranking_engine import get_full_profile, rank_domains
from src.role_matcher import MatchType, match_role


def main() -> int:
    loader = DatasetLoader(config=get_config())
    loader.load()
    df = loader.df
    info = loader.get_dataset_info()
    roles = loader.get_roles()
    failures: list[str] = []

    def check(name: str, ok: bool) -> None:
        if not ok:
            failures.append(name)

    check("20 roles", info.num_roles == 20)
    check("50 domains", info.num_domains == 50)
    check("1000 mappings", info.num_mappings == 1000)
    check("no duplicate pairs", info.duplicate_role_domain == 0)
    check("scores 1-5", info.score_range == (1, 5))
    check("csv and workbook scores agree", info.csv_xlsx_scores_agree is True)
    for role in roles:
        profile = get_full_profile(role, loader)
        check(f"{role} has 50 domains", profile is not None and len(profile.domains) == 50)
        if profile:
            scores = [item.relevance_score for item in profile.domains]
            check(f"{role} sorted", scores == sorted(scores, reverse=True))
            raw = loader.get_role_domains(role)
            stored = dict(zip(raw["domain_id"], raw["within_role_rank_ties_respected"].astype(int)))
            raw_scores = dict(zip(raw["domain_id"], raw["project_relevance_score_1_to_5"].astype(int)))
            for domain in profile.domains:
                if domain.rank != stored[domain.domain_id] or domain.relevance_score != raw_scores[domain.domain_id]:
                    failures.append(f"{role} {domain.domain_id} rank or score drift")
                    break
    unknown = match_role("Astronaut Space Commander", roles)
    check("unknown title is not marked exact", unknown.match_type != MatchType.EXACT)
    analyst = rank_domains("Data Analyst", loader, top_n=5, include_ties=True)
    check("tie inclusion returns at least 5", analyst is not None and analyst.returned_count >= 5)

    lines = [
        "# Evaluation report",
        "",
        "This file records structural checks against the shipped dataset.",
        "It does **not** report predictive validity, inter-rater reliability, or job-performance accuracy.",
        "Those have not been measured. Blank SME columns in `docs/sme_review_template.csv` are a review form, not results.",
        "",
        "## Structural checks",
        "",
        f"- Roles: {info.num_roles}",
        f"- Domains: {info.num_domains}",
        f"- Mappings: {info.num_mappings}",
        f"- CSV/workbook score agreement: {info.csv_xlsx_scores_agree}",
        f"- Evidence label: {', '.join(info.evidence_statuses)}",
        f"- Weak or composite proxies: {', '.join(info.weak_proxy_roles)}",
        "",
    ]
    if failures:
        lines.append(f"Failures ({len(failures)}):")
        lines.extend(f"- {item}" for item in failures)
        status = 1
    else:
        lines.append("All structural checks in this script passed.")
        status = 0
    lines.extend(
        [
            "",
            "## What was not evaluated",
            "",
            "- Whether a 1–5 score is the right importance judgment for a real job.",
            "- Whether an O*NET proxy occupation is the right benchmark for a specific employer.",
            "- Whether domain scores predict performance, hiring outcomes, or training needs.",
            "- Whether model explanations are factually adequate. Automated tests only check shape, known domain IDs, and score-invention patterns.",
            "",
            "See `docs/validation_methodology.md` for how a human reviewer would use the SME template.",
            "",
        ]
    )
    output = Path(__file__).resolve().parent.parent / "reports" / "evaluation_report.md"
    output.parent.mkdir(exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
