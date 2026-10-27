"""Workbook reconciliation tests.

These check the reference Excel file against the runtime CSV. They do not
treat agreement as scientific validation.
"""

from __future__ import annotations

from src.config import get_config
from src.data_loader import DatasetLoader
from src.workbook_loader import MAPPINGS_SHEET, read_workbook_sheets


def test_mapping_sheet_matches_csv_scores():
    config = get_config()
    loader = DatasetLoader(config=config)
    loader.load()
    pack = loader.get_reference_pack()
    report = pack.reconciliation
    assert report.load_error is None
    assert report.mappings is not None
    assert report.mappings.sheet_name == MAPPINGS_SHEET
    assert report.mappings.rows == 1000
    assert report.mappings.matched_rows == 1000
    assert report.scores_agree is True
    assert report.mappings.differing_fields == {}
    assert report.xlsx_readable_by_openpyxl is False
    assert "*" in report.reader_note or "sheet title" in report.reader_note.lower() or "openpyxl" in report.reader_note.lower()


def test_supplemental_sheets_do_not_contradict_scores():
    config = get_config()
    loader = DatasetLoader(config=config)
    loader.load()
    pack = loader.get_reference_pack()
    names = {item.sheet_name for item in pack.reconciliation.supplemental}
    assert "Top domains by role" in names
    assert "Expert review queue" in names
    assert "50 domain definitions" in names
    for item in pack.reconciliation.supplemental:
        assert "project_relevance_score_1_to_5" not in item.differing_fields
        assert item.differing_fields == {}


def test_workbook_sheets_are_readable():
    sheets = read_workbook_sheets(get_config().xlsx_path)
    assert "O*NET source benchmarks" in sheets
    assert "Scoring method" in sheets
    applied = sheets["Role profile audit"]["numeric_O*NET_scores_applied_to_project_1_to_5"]
    assert set(applied.astype(str).str.strip()) == {"No"}
