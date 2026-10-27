"""Read the corrected-master Excel workbook and compare it with the CSV.

The workbook is the reference document. The CSV is the runtime table.
This module does not change either file.

openpyxl rejects the workbook as shipped because the sheet title
``O*NET source benchmarks`` contains ``*``. The reader builds an
in-memory copy with that title character replaced, then reads the copy.
Cell values are not edited.
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

MAPPINGS_SHEET = "All 1000 mappings"
AUDIT_SHEET = "Role profile audit"
DEFINITIONS_SHEET = "50 domain definitions"
CROSSWALK_SHEET = "Domain crosswalk"
BENCHMARKS_SHEET = "O*NET source benchmarks"
REVIEW_QUEUE_SHEET = "Expert review queue"
SCORING_SHEET = "Scoring method"
QA_SHEET = "QA summary"
TOP_SHEET = "Top domains by role"

_ILLEGAL_SHEET_CHARS = re.compile(r"[\*\:/\\?\[\]]")


@dataclass
class SheetComparison:
    """Result of comparing one workbook sheet with the CSV on shared keys."""

    sheet_name: str
    rows: int
    matched_rows: int
    differing_fields: Dict[str, int] = field(default_factory=dict)
    extra_columns: List[str] = field(default_factory=list)
    note: str = ""

    @property
    def agrees_on_shared_fields(self) -> bool:
        return self.matched_rows == self.rows and not self.differing_fields


@dataclass
class ReconciliationReport:
    """Explicit record of CSV/workbook agreement and reader limitations."""

    csv_path: str
    xlsx_path: str
    csv_sha256: str
    xlsx_sha256: str
    xlsx_readable_by_openpyxl: bool
    reader_note: str
    sheet_names: List[str]
    mappings: Optional[SheetComparison] = None
    supplemental: List[SheetComparison] = field(default_factory=list)
    load_error: Optional[str] = None

    @property
    def scores_agree(self) -> bool:
        if self.mappings is None:
            return False
        score_diffs = self.mappings.differing_fields.get(
            "project_relevance_score_1_to_5", 0
        )
        return self.mappings.agrees_on_shared_fields and score_diffs == 0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def openpyxl_can_read(path: Path) -> bool:
    """Return whether a normal openpyxl load accepts the workbook.

    Read-only mode skips the sheet-title check that blocks pandas, so this
    uses the same style of load that ``pandas.read_excel`` depends on.
    """
    try:
        from openpyxl import load_workbook

        workbook = load_workbook(path, data_only=True)
        workbook.close()
        return True
    except Exception:
        return False


def _safe_sheet_title(name: str, used: set[str]) -> str:
    safe = _ILLEGAL_SHEET_CHARS.sub("x", name)[:31] or "Sheet"
    base = safe
    n = 2
    while safe in used:
        suffix = f"_{n}"
        safe = f"{base[:31 - len(suffix)]}{suffix}"
        n += 1
    used.add(safe)
    return safe


def read_workbook_sheets(path: Path) -> Dict[str, pd.DataFrame]:
    """Load every sheet, using original titles as dictionary keys.

    Raises FileNotFoundError or ValueError if the workbook cannot be read.
    """
    if not path.exists():
        raise FileNotFoundError(f"Workbook not found: {path}")

    original = path.read_bytes()
    renamed: Dict[str, str] = {}
    with zipfile.ZipFile(io.BytesIO(original)) as zin:
        workbook_xml = zin.read("xl/workbook.xml").decode("utf-8")
        names = re.findall(r'name="([^"]+)"', workbook_xml)
        used: set[str] = set()
        updated = workbook_xml
        for name in names:
            if _ILLEGAL_SHEET_CHARS.search(name):
                safe = _safe_sheet_title(name, used)
                renamed[safe] = name
                updated = updated.replace(f'name="{name}"', f'name="{safe}"', 1)
            else:
                used.add(name)

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for info in zin.infolist():
                payload = zin.read(info.filename)
                if info.filename == "xl/workbook.xml":
                    payload = updated.encode("utf-8")
                zout.writestr(info.filename, payload)
    buffer.seek(0)

    excel = pd.ExcelFile(buffer)
    sheets: Dict[str, pd.DataFrame] = {}
    for sheet_name in excel.sheet_names:
        original_name = renamed.get(sheet_name, sheet_name)
        sheets[original_name] = pd.read_excel(excel, sheet_name=sheet_name)
    if renamed:
        logger.info(
            "Read workbook with temporary sheet-title fix: %s",
            renamed,
        )
    return sheets


def _series_equal(left: pd.Series, right: pd.Series) -> pd.Series:
    """Element-wise equality that treats missing values as equal."""
    left_num = pd.to_numeric(left, errors="coerce")
    right_num = pd.to_numeric(right, errors="coerce")
    numeric_mask = left.notna() & right.notna() & left_num.notna() & right_num.notna()
    same = pd.Series(False, index=left.index)
    same.loc[numeric_mask] = (
        left_num.loc[numeric_mask] - right_num.loc[numeric_mask]
    ).abs() < 1e-6
    text_mask = ~numeric_mask
    left_text = left.fillna("").astype(str).str.strip().replace({"nan": "", "None": ""})
    right_text = right.fillna("").astype(str).str.strip().replace({"nan": "", "None": ""})
    same.loc[text_mask] = left_text.loc[text_mask] == right_text.loc[text_mask]
    both_missing = left.isna() & right.isna()
    same.loc[both_missing] = True
    return same


def _compare_on_keys(
    sheet_name: str,
    sheet: pd.DataFrame,
    csv_df: pd.DataFrame,
    keys: List[str],
    fields: List[str],
) -> SheetComparison:
    extra = sorted(set(sheet.columns) - set(csv_df.columns))
    merged = sheet.merge(csv_df, on=keys, how="left", suffixes=("_xlsx", "_csv"), indicator=True)
    matched = int((merged["_merge"] == "both").sum())
    differing: Dict[str, int] = {}
    both = merged[merged["_merge"] == "both"]
    for column in fields:
        xlsx_col = f"{column}_xlsx" if f"{column}_xlsx" in both.columns else column
        csv_col = f"{column}_csv" if f"{column}_csv" in both.columns else None
        if csv_col is None or xlsx_col not in both.columns:
            continue
        equal = _series_equal(both[xlsx_col], both[csv_col])
        n_diff = int((~equal).sum())
        if n_diff:
            differing[column] = n_diff
    note = ""
    if extra:
        note = "Extra workbook columns are preserved as reference metadata and are not score conflicts: " + ", ".join(extra)
    return SheetComparison(
        sheet_name=sheet_name,
        rows=len(sheet),
        matched_rows=matched,
        differing_fields=differing,
        extra_columns=extra,
        note=note,
    )


_SHARED_FIELDS = [
    "domain_name",
    "construct_family",
    "operational_definition",
    "project_relevance_score_1_to_5",
    "score_interpretation",
    "within_role_rank_ties_respected",
    "rationale",
    "confidence",
    "score_status",
    "evidence_status",
    "review_priority",
    "O*NET_reference_occupation",
    "O*NET_reference_SOC",
    "O*NET_profile_match_quality",
    "O*NET_source_URL",
    "O*NET_work_style_crosswalk",
    "crosswalk_quality",
    "O*NET_standardized_work_style_impact_0_to_100",
]


def reconcile(csv_path: Path, xlsx_path: Path, csv_df: pd.DataFrame) -> ReconciliationReport:
    """Compare the workbook with the loaded CSV and return an explicit report."""
    report = ReconciliationReport(
        csv_path=str(csv_path),
        xlsx_path=str(xlsx_path),
        csv_sha256=file_sha256(csv_path) if csv_path.exists() else "",
        xlsx_sha256=file_sha256(xlsx_path) if xlsx_path.exists() else "",
        xlsx_readable_by_openpyxl=openpyxl_can_read(xlsx_path) if xlsx_path.exists() else False,
        reader_note="",
        sheet_names=[],
    )
    if not xlsx_path.exists():
        report.load_error = f"Workbook not found: {xlsx_path}"
        report.reader_note = report.load_error
        return report

    if not report.xlsx_readable_by_openpyxl:
        report.reader_note = (
            "Stock openpyxl cannot open this workbook because a sheet title "
            "contains '*'. The application reads an in-memory copy whose sheet "
            "title is made legal. Stored cell values, including the 1–5 scores, "
            "are not modified."
        )
    else:
        report.reader_note = "Workbook opened with openpyxl without a sheet-title workaround."

    try:
        sheets = read_workbook_sheets(xlsx_path)
    except Exception as exc:
        report.load_error = f"Could not read workbook: {exc}"
        logger.exception("Workbook read failed")
        return report

    report.sheet_names = list(sheets)
    mappings = sheets.get(MAPPINGS_SHEET)
    if mappings is None:
        report.load_error = f"Workbook is missing the '{MAPPINGS_SHEET}' sheet."
        return report

    report.mappings = _compare_on_keys(
        MAPPINGS_SHEET,
        mappings,
        csv_df,
        ["role", "domain_id"],
        _SHARED_FIELDS,
    )

    if TOP_SHEET in sheets:
        report.supplemental.append(
            _compare_on_keys(TOP_SHEET, sheets[TOP_SHEET], csv_df, ["role", "domain_id"], _SHARED_FIELDS)
        )
    if REVIEW_QUEUE_SHEET in sheets:
        report.supplemental.append(
            _compare_on_keys(
                REVIEW_QUEUE_SHEET,
                sheets[REVIEW_QUEUE_SHEET],
                csv_df,
                ["role", "domain_id"],
                _SHARED_FIELDS,
            )
        )
    if DEFINITIONS_SHEET in sheets:
        definitions = (
            csv_df[["domain_id", "domain_name", "construct_family", "operational_definition"]]
            .drop_duplicates()
        )
        report.supplemental.append(
            _compare_on_keys(
                DEFINITIONS_SHEET,
                sheets[DEFINITIONS_SHEET],
                definitions,
                ["domain_id"],
                ["domain_name", "construct_family", "operational_definition"],
            )
        )
    return report


@dataclass
class ReferencePack:
    """Workbook sheets kept beside the CSV. None of these overwrite scores."""

    sheets: Dict[str, pd.DataFrame]
    reconciliation: ReconciliationReport

    def sheet(self, name: str) -> pd.DataFrame:
        return self.sheets.get(name, pd.DataFrame()).copy()
