#!/usr/bin/env python3
"""Write sample rankings for Data Scientist and Data Analyst.

Run from the project root:

    python3 scripts/generate_sample_output.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config
from src.data_loader import DatasetLoader
from src.ranking_engine import domains_to_dataframe, get_full_profile, rank_domains
from src.role_matcher import match_role


def write_role(loader: DatasetLoader, role_name: str, output_dir: Path, top_n: int = 10) -> None:
    roles = loader.get_roles()
    match = match_role(role_name, roles)
    if not match.matched_role:
        raise SystemExit(f"No match for {role_name}")
    result = rank_domains(match.matched_role, loader, top_n=top_n)
    full = get_full_profile(match.matched_role, loader)
    if result is None or full is None:
        raise SystemExit(f"No ranking for {role_name}")

    slug = match.matched_role.lower().replace(" ", "_")
    domains_to_dataframe(result.domains).to_csv(output_dir / f"sample_output_{slug}_top{top_n}.csv", index=False)
    domains_to_dataframe(full.domains).to_csv(output_dir / f"sample_output_{slug}_full.csv", index=False)

    lines = [
        f"# Sample output: {match.matched_role}",
        "",
        "These rows are copied from the stored dataset. Scores were not recomputed.",
        "They are provisional role-level estimates, not an individual profile or a hiring score.",
        "",
        f"- Input title: {role_name}",
        f"- Match type: {match.match_type.value}",
        f"- Reference role: {match.matched_role}",
        f"- Requested top {top_n}; returned {result.returned_count} (ties at the cutoff are included)",
        f"- Tie included: {result.tie_included}",
        "",
        "## Role scope",
        "",
        result.role_scope,
        "",
        "## Highest stored domains",
        "",
        "| Rank | ID | Domain | Score | Importance label | Family |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for domain in result.domains:
        lines.append(
            f"| {domain.rank} | {domain.domain_id} | {domain.domain_name} | "
            f"{domain.relevance_score} | {domain.score_interpretation} | {domain.construct_family} |"
        )
    lines.extend(
        [
            "",
            "O*NET impact values, when present, are 0–100 source benchmarks and are not the 1–5 score.",
            "Equal scores share a rank. Order inside a tie follows domain ID and is not a second ranking.",
            "",
        ]
    )
    (output_dir / f"sample_output_{slug}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote sample files for {match.matched_role}")


def write_sme_template(loader: DatasetLoader, docs_dir: Path) -> None:
    frame = loader.df[
        [
            "role",
            "domain_id",
            "domain_name",
            "project_relevance_score_1_to_5",
            "score_interpretation",
            "rationale",
            "O*NET_reference_occupation",
            "O*NET_profile_match_quality",
            "review_priority",
        ]
    ].copy()
    frame = frame.rename(
        columns={
            "project_relevance_score_1_to_5": "stored_score",
            "score_interpretation": "stored_importance_label",
        }
    )
    frame["template_status"] = "blank_template_not_a_validation_result"
    frame["sme_independent_score_1_to_5"] = ""
    frame["sme_agrees_with_rationale"] = ""
    frame["sme_onet_proxy_appropriate"] = ""
    frame["sme_comments"] = ""
    frame["sme_review_date"] = ""
    frame["sme_reviewer_id"] = ""
    frame.to_csv(docs_dir / "sme_review_template.csv", index=False)
    print(f"Wrote SME template with {len(frame)} blank review rows")


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    output_dir = root / "reports"
    output_dir.mkdir(exist_ok=True)
    loader = DatasetLoader(config=get_config())
    loader.load()
    write_role(loader, "Data Scientist", output_dir)
    write_role(loader, "Data Analyst", output_dir)
    write_sme_template(loader, root / "docs")


if __name__ == "__main__":
    main()
