"""Dataset loader for the psychometric role–domain mapping data.

Reads the corrected-master CSV, validates schema and integrity, and exposes
typed accessors for roles, domains, and mappings.

Design decisions
-----------------
* Uses the CSV as the primary runtime source (fast, portable).
* BOM-aware reading (``utf-8-sig``) because the master CSV includes a BOM.
* Validates all expected columns, score range, and duplicate rows on load.
* Caches the loaded DataFrame so repeated calls are cheap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from src.config import AppConfig, get_config
from src.workbook_loader import ReferencePack, reconcile, read_workbook_sheets

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Expected schema (all 25 columns)
# ---------------------------------------------------------------------------
EXPECTED_COLUMNS: list[str] = [
    "role",
    "role_scope",
    "domain_id",
    "domain_name",
    "construct_family",
    "operational_definition",
    "project_relevance_score_1_to_5",
    "score_interpretation",
    "within_role_rank_ties_respected",
    "rationale",
    "confidence",
    "score_status",
    "score_review_note",
    "O*NET_reference_occupation",
    "O*NET_reference_SOC",
    "O*NET_profile_match_quality",
    "O*NET_source_URL",
    "O*NET_work_style_crosswalk",
    "crosswalk_quality",
    "O*NET_standardized_work_style_impact_0_to_100",
    "score_change_vs_original_estimate",
    "notes",
    "evidence_status",
    "evidence_alignment_review",
    "review_priority",
]


class DataLoadError(Exception):
    """Raised when the dataset fails validation on load."""


@dataclass
class DatasetInfo:
    """Summary metadata about the loaded dataset."""

    csv_path: str
    num_roles: int
    num_domains: int
    num_mappings: int
    score_range: tuple[int, int]
    missing_values: Dict[str, int]
    duplicate_role_domain: int
    roles: List[str]
    domains: List[Dict[str, str]]
    construct_families: List[str]
    evidence_statuses: List[str]
    review_priorities: List[str]
    confidence_levels: List[str]
    score_distribution: Dict[int, int]
    onet_crosswalk_missing: int
    onet_impact_missing: int
    csv_sha256: str = ""
    xlsx_sha256: str = ""
    csv_xlsx_scores_agree: Optional[bool] = None
    xlsx_reader_note: str = ""
    weak_proxy_roles: List[str] = field(default_factory=list)
    reconciliation_summary: str = ""


@dataclass
class DatasetLoader:
    """Loads, validates, and provides access to the psychometric dataset."""

    config: AppConfig = field(default_factory=get_config)
    _df: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)
    _reference: Optional[ReferencePack] = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load(self, path: Optional[Path] = None) -> pd.DataFrame:
        """Load the CSV, validate schema, and cache the result."""
        csv_path = path or self.config.csv_path
        if self._df is not None:
            return self._df

        if not csv_path.exists():
            raise DataLoadError(f"Dataset file not found: {csv_path}")

        logger.info("Loading dataset from %s", csv_path)
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        self._validate(df, csv_path)
        self._df = df
        return df

    @property
    def df(self) -> pd.DataFrame:
        """Lazy-loaded DataFrame."""
        if self._df is None:
            self.load()
        assert self._df is not None
        return self._df

    def reload(self, path: Optional[Path] = None) -> pd.DataFrame:
        """Force-reload the dataset and the workbook reference."""
        self._df = None
        self._reference = None
        return self.load(path)

    def get_reference_pack(self) -> ReferencePack:
        """Load workbook sheets and the CSV reconciliation report.

        A workbook read failure does not discard the CSV. The report's
        ``load_error`` explains what could not be read.
        """
        if self._reference is not None:
            return self._reference
        df = self.df
        xlsx_path = self.config.xlsx_path
        report = reconcile(self.config.csv_path, xlsx_path, df)
        sheets: Dict[str, pd.DataFrame] = {}
        if report.load_error is None and xlsx_path.exists():
            try:
                sheets = read_workbook_sheets(xlsx_path)
            except Exception as exc:
                report.load_error = str(exc)
                logger.exception("Failed to cache workbook sheets")
        self._reference = ReferencePack(sheets=sheets, reconciliation=report)
        return self._reference

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate(self, df: pd.DataFrame, path: Path) -> None:
        missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
        if missing_cols:
            raise DataLoadError(f"Missing columns in {path}: {missing_cols}")

        scores = df["project_relevance_score_1_to_5"]
        if scores.min() < 1 or scores.max() > 5:
            raise DataLoadError(
                f"Scores out of 1–5 range: min={scores.min()}, max={scores.max()}"
            )

        dupes = df.duplicated(subset=["role", "domain_id"]).sum()
        if dupes > 0:
            logger.warning("Found %d duplicate role–domain rows", dupes)

        n_roles = df["role"].nunique()
        n_domains = df["domain_id"].nunique()
        logger.info(
            "Dataset loaded: %d rows, %d roles, %d domains, %d duplicates",
            len(df), n_roles, n_domains, dupes,
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get_roles(self) -> List[str]:
        """Return sorted list of unique role names."""
        return sorted(self.df["role"].unique().tolist())

    def get_role_scope(self, role: str) -> Optional[str]:
        """Return the scope description for a role."""
        subset = self.df[self.df["role"] == role]
        if subset.empty:
            return None
        return subset.iloc[0]["role_scope"]

    def get_domains(self) -> List[Dict[str, str]]:
        """Return unique domains with id, name, family, and definition."""
        cols = ["domain_id", "domain_name", "construct_family", "operational_definition"]
        return (
            self.df[cols]
            .drop_duplicates()
            .sort_values("domain_id")
            .to_dict(orient="records")
        )

    def get_construct_families(self) -> List[str]:
        """Return sorted unique construct families."""
        return sorted(self.df["construct_family"].unique().tolist())

    def get_role_domains(self, role: str) -> pd.DataFrame:
        """Return all 50 domain records for a role, sorted by score descending."""
        subset = self.df[self.df["role"] == role].copy()
        if subset.empty:
            return subset
        return subset.sort_values(
            by=["project_relevance_score_1_to_5", "domain_id"],
            ascending=[False, True],
        ).reset_index(drop=True)

    def get_domain_roles(self, domain_id: str) -> pd.DataFrame:
        """Return all role records for a specific domain."""
        return self.df[self.df["domain_id"] == domain_id].copy()

    def get_dataset_info(self) -> DatasetInfo:
        """Return a comprehensive summary of the loaded dataset."""
        df = self.df
        missing = df.isnull().sum()
        missing_dict = {col: int(v) for col, v in missing.items() if v > 0}

        score_dist = df["project_relevance_score_1_to_5"].value_counts().sort_index()
        reference = self.get_reference_pack()
        recon = reference.reconciliation
        weak = sorted(
            df.loc[
                df["O*NET_profile_match_quality"].astype(str).str.contains(
                    "Weak|Composite", case=False, na=False
                ),
                "role",
            ].unique().tolist()
        )
        if recon.mappings and recon.scores_agree:
            summary = (
                "CSV and the workbook sheet 'All 1000 mappings' agree on all "
                "shared fields, including project_relevance_score_1_to_5. "
                "Supplemental sheets add review metadata and do not change scores."
            )
        elif recon.load_error:
            summary = f"Workbook comparison incomplete: {recon.load_error}"
        else:
            summary = (
                "CSV and workbook differ on one or more shared fields. "
                "See the reconciliation detail before using either source."
            )

        domains = (
            df[["domain_id", "domain_name", "construct_family"]]
            .drop_duplicates()
            .sort_values("domain_id")
            .to_dict(orient="records")
        )

        return DatasetInfo(
            csv_path=str(self.config.csv_path),
            num_roles=df["role"].nunique(),
            num_domains=df["domain_id"].nunique(),
            num_mappings=len(df),
            score_range=(int(df["project_relevance_score_1_to_5"].min()),
                         int(df["project_relevance_score_1_to_5"].max())),
            missing_values=missing_dict,
            duplicate_role_domain=int(df.duplicated(subset=["role", "domain_id"]).sum()),
            roles=sorted(df["role"].unique().tolist()),
            domains=domains,
            construct_families=sorted(df["construct_family"].unique().tolist()),
            evidence_statuses=sorted(df["evidence_status"].unique().tolist()),
            review_priorities=sorted(df["review_priority"].unique().tolist()),
            confidence_levels=sorted(df["confidence"].unique().tolist()),
            score_distribution={int(k): int(v) for k, v in score_dist.items()},
            onet_crosswalk_missing=int(df["O*NET_work_style_crosswalk"].isnull().sum()),
            onet_impact_missing=int(
                df["O*NET_standardized_work_style_impact_0_to_100"].isnull().sum()
            ),
            csv_sha256=recon.csv_sha256,
            xlsx_sha256=recon.xlsx_sha256,
            csv_xlsx_scores_agree=recon.scores_agree,
            xlsx_reader_note=recon.reader_note,
            weak_proxy_roles=weak,
            reconciliation_summary=summary,
        )
