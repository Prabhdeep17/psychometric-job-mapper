"""Deterministic ranking engine for psychometric domain relevance.

Retrieves a role's domain mappings from the dataset and returns them sorted
by stored relevance score with ties fully respected.

Key invariants
--------------
* Stored ``project_relevance_score_1_to_5`` values are never modified.
* Ties at the top-N cutoff boundary are *included*, so the returned list
  may exceed the requested N when equal scores exist at the boundary.
* The engine works entirely from the loaded dataset — no LLM call needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from src.data_loader import DatasetLoader

logger = logging.getLogger(__name__)


@dataclass
class RankedDomain:
    """A single domain result with all relevant metadata."""

    domain_id: str
    domain_name: str
    construct_family: str
    operational_definition: str
    relevance_score: int
    score_interpretation: str
    rank: int
    rationale: str
    confidence: str
    score_status: str
    score_review_note: str
    onet_reference_occupation: str
    onet_reference_soc: str
    onet_profile_match_quality: str
    onet_source_url: str
    onet_work_style_crosswalk: Optional[str]
    crosswalk_quality: str
    onet_impact_score: Optional[float]
    evidence_status: str
    evidence_alignment_review: str
    review_priority: str
    notes: str


@dataclass
class RankingResult:
    """Complete ranking result for a role query."""

    role: str
    role_scope: str
    requested_top_n: int
    returned_count: int
    total_domains: int
    tie_included: bool
    domains: List[RankedDomain]


def _row_to_ranked_domain(row: pd.Series, rank: int) -> RankedDomain:
    """Convert a DataFrame row to a RankedDomain dataclass."""
    return RankedDomain(
        domain_id=row["domain_id"],
        domain_name=row["domain_name"],
        construct_family=row["construct_family"],
        operational_definition=row["operational_definition"],
        relevance_score=int(row["project_relevance_score_1_to_5"]),
        score_interpretation=row["score_interpretation"],
        rank=rank,
        rationale=row["rationale"],
        confidence=row["confidence"],
        score_status=row["score_status"],
        score_review_note=row["score_review_note"],
        onet_reference_occupation=row["O*NET_reference_occupation"],
        onet_reference_soc=row["O*NET_reference_SOC"],
        onet_profile_match_quality=row["O*NET_profile_match_quality"],
        onet_source_url=row["O*NET_source_URL"],
        onet_work_style_crosswalk=(
            row["O*NET_work_style_crosswalk"]
            if pd.notna(row.get("O*NET_work_style_crosswalk"))
            else None
        ),
        crosswalk_quality=row["crosswalk_quality"],
        onet_impact_score=(
            float(row["O*NET_standardized_work_style_impact_0_to_100"])
            if pd.notna(row.get("O*NET_standardized_work_style_impact_0_to_100"))
            else None
        ),
        evidence_status=row["evidence_status"],
        evidence_alignment_review=row["evidence_alignment_review"],
        review_priority=row["review_priority"],
        notes=row["notes"],
    )


def rank_domains(
    role: str,
    loader: DatasetLoader,
    top_n: int = 10,
    include_ties: bool = True,
) -> Optional[RankingResult]:
    """
    Retrieve and rank domains for a given role.

    Parameters
    ----------
    role : str
        Exact reference role name (after matching).
    loader : DatasetLoader
        Loaded dataset accessor.
    top_n : int
        Number of top domains to return.  Ties at the boundary are
        included by default.
    include_ties : bool
        When True, all domains that share the score at position *top_n*
        are included, so the returned list may exceed *top_n*.

    Returns
    -------
    RankingResult or None if the role is not found.
    """
    role_df = loader.get_role_domains(role)
    if role_df.empty:
        return None

    role_scope = role_df.iloc[0]["role_scope"]
    total = len(role_df)

    # Assign dense ranks respecting ties
    role_df = role_df.copy()
    # Use the dataset's own tie-aware rank. It matches competition ranking
    # (pandas method="min"): equal scores share a rank, and the next score
    # continues at the next occupied position. Display order inside a tie
    # is domain_id, which does not create a second score.
    role_df["_rank"] = role_df["within_role_rank_ties_respected"].astype(int)

    # Top-N with tie inclusion
    if include_ties and top_n < total:
        cutoff_score = role_df.iloc[min(top_n - 1, total - 1)][
            "project_relevance_score_1_to_5"
        ]
        selected = role_df[
            role_df["project_relevance_score_1_to_5"] >= cutoff_score
        ]
    else:
        selected = role_df.head(top_n)

    tie_included = len(selected) > top_n

    ranked = [
        _row_to_ranked_domain(row, int(row["_rank"]))
        for _, row in selected.iterrows()
    ]

    return RankingResult(
        role=role,
        role_scope=role_scope,
        requested_top_n=top_n,
        returned_count=len(ranked),
        total_domains=total,
        tie_included=tie_included,
        domains=ranked,
    )


def get_full_profile(
    role: str,
    loader: DatasetLoader,
) -> Optional[RankingResult]:
    """Return all 50 domains for a role (full profile view)."""
    return rank_domains(role, loader, top_n=50, include_ties=False)


def filter_domains(
    result: RankingResult,
    construct_family: Optional[str] = None,
    min_score: Optional[int] = None,
    max_score: Optional[int] = None,
    evidence_status: Optional[str] = None,
    review_priority: Optional[str] = None,
) -> List[RankedDomain]:
    """Filter a ranking result by various criteria."""
    domains = result.domains

    if construct_family:
        domains = [d for d in domains if d.construct_family == construct_family]
    if min_score is not None:
        domains = [d for d in domains if d.relevance_score >= min_score]
    if max_score is not None:
        domains = [d for d in domains if d.relevance_score <= max_score]
    if evidence_status:
        domains = [d for d in domains if d.evidence_status == evidence_status]
    if review_priority:
        domains = [d for d in domains if d.review_priority == review_priority]

    return domains


def domains_to_dataframe(domains: List[RankedDomain]) -> pd.DataFrame:
    """Convert a list of RankedDomain to a DataFrame for display/export."""
    if not domains:
        return pd.DataFrame()
    records = []
    for d in domains:
        records.append({
            "Rank": d.rank,
            "Domain ID": d.domain_id,
            "Domain": d.domain_name,
            "Construct Family": d.construct_family,
            "Relevance Score (1-5)": d.relevance_score,
            "Importance label": d.score_interpretation,
            "Score validity": (
                "Provisional estimate. Not a scientifically validated "
                "measurement, individual profile, or hiring score."
            ),
            "Definition": d.operational_definition,
            "Rationale": d.rationale,
            "Confidence": d.confidence,
            "Score Status": d.score_status,
            "Evidence Status": d.evidence_status,
            "Review Priority": d.review_priority,
            "O*NET Occupation": d.onet_reference_occupation,
            "O*NET SOC": d.onet_reference_soc,
            "O*NET Match Quality": d.onet_profile_match_quality,
            "O*NET Source URL": d.onet_source_url,
            "O*NET Work Style": d.onet_work_style_crosswalk or "",
            "O*NET Impact (0-100)": d.onet_impact_score if d.onet_impact_score else "",
            "Notes": d.notes,
        })
    return pd.DataFrame(records)
