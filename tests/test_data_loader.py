"""Tests for the dataset loader module.

Covers schema validation, data integrity, accessor methods,
and edge cases.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.config import AppConfig
from src.data_loader import DatasetLoader, DataLoadError, EXPECTED_COLUMNS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def loader() -> DatasetLoader:
    """Loader using the real project dataset."""
    config = AppConfig()
    ldr = DatasetLoader(config=config)
    ldr.load()
    return ldr


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """Create a small valid CSV for isolated tests."""
    rows = []
    for role in ["Test Role A", "Test Role B"]:
        for i in range(1, 4):
            rows.append({
                "role": role,
                "role_scope": f"Scope for {role}",
                "domain_id": f"D{i:02d}",
                "domain_name": f"Domain {i}",
                "construct_family": "Test family",
                "operational_definition": f"Def {i}",
                "project_relevance_score_1_to_5": (6 - i),
                "score_interpretation": "Core / highly salient",
                "within_role_rank_ties_respected": 1,
                "rationale": "Test rationale",
                "confidence": "Moderate",
                "score_status": "Provisional",
                "score_review_note": "Note",
                "O*NET_reference_occupation": "Test Occ",
                "O*NET_reference_SOC": "11-1111.00",
                "O*NET_profile_match_quality": "Partial proxy",
                "O*NET_source_URL": "https://example.com",
                "O*NET_work_style_crosswalk": None,
                "crosswalk_quality": "No direct crosswalk",
                "O*NET_standardized_work_style_impact_0_to_100": None,
                "score_change_vs_original_estimate": 0,
                "notes": "Test note",
                "evidence_status": "Structured expert-style estimate; not empirically validated",
                "evidence_alignment_review": "Test review",
                "review_priority": "Routine — still provisional",
            })
    df = pd.DataFrame(rows)
    path = tmp_path / "test_data.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


# ---------------------------------------------------------------------------
# Schema & integrity tests
# ---------------------------------------------------------------------------
class TestSchemaValidation:
    def test_all_expected_columns_present(self, loader: DatasetLoader):
        for col in EXPECTED_COLUMNS:
            assert col in loader.df.columns, f"Missing column: {col}"

    def test_no_extra_columns(self, loader: DatasetLoader):
        extra = set(loader.df.columns) - set(EXPECTED_COLUMNS)
        assert not extra, f"Unexpected columns: {extra}"

    def test_shape(self, loader: DatasetLoader):
        assert loader.df.shape == (1000, 25)

    def test_scores_in_range(self, loader: DatasetLoader):
        scores = loader.df["project_relevance_score_1_to_5"]
        assert scores.min() >= 1
        assert scores.max() <= 5

    def test_no_duplicate_role_domain(self, loader: DatasetLoader):
        dupes = loader.df.duplicated(subset=["role", "domain_id"]).sum()
        assert dupes == 0

    def test_20_roles(self, loader: DatasetLoader):
        assert loader.df["role"].nunique() == 20

    def test_50_domains(self, loader: DatasetLoader):
        assert loader.df["domain_id"].nunique() == 50

    def test_each_role_has_50_domains(self, loader: DatasetLoader):
        counts = loader.df.groupby("role")["domain_id"].nunique()
        for role, count in counts.items():
            assert count == 50, f"Role '{role}' has {count} domains, expected 50"

    def test_domain_ids_format(self, loader: DatasetLoader):
        for did in loader.df["domain_id"].unique():
            assert did.startswith("D"), f"Domain ID {did} doesn't start with D"
            assert len(did) == 3, f"Domain ID {did} is not 3 chars"

    def test_scores_are_integers(self, loader: DatasetLoader):
        assert loader.df["project_relevance_score_1_to_5"].dtype in ("int64", "int32")


class TestMissingData:
    def test_no_missing_required_fields(self, loader: DatasetLoader):
        required = ["role", "role_scope", "domain_id", "domain_name",
                     "project_relevance_score_1_to_5", "rationale",
                     "evidence_status"]
        for col in required:
            assert loader.df[col].isnull().sum() == 0, f"Missing values in {col}"

    def test_onet_crosswalk_missing_count(self, loader: DatasetLoader):
        missing = loader.df["O*NET_work_style_crosswalk"].isnull().sum()
        assert missing == 560  # Known from inspection

    def test_onet_impact_missing_count(self, loader: DatasetLoader):
        missing = loader.df["O*NET_standardized_work_style_impact_0_to_100"].isnull().sum()
        assert missing == 753  # Known from inspection


# ---------------------------------------------------------------------------
# Accessor tests
# ---------------------------------------------------------------------------
class TestAccessors:
    def test_get_roles_returns_20(self, loader: DatasetLoader):
        roles = loader.get_roles()
        assert len(roles) == 20
        assert roles == sorted(roles)  # alphabetical

    def test_get_roles_contains_expected(self, loader: DatasetLoader):
        roles = loader.get_roles()
        expected = ["Data Analyst", "Data Scientist", "Software Engineer",
                     "Machine Learning Engineer", "Product Manager",
                     "Cybersecurity Analyst", "Financial Analyst", "Research Scientist"]
        for r in expected:
            assert r in roles, f"Missing expected role: {r}"

    def test_get_domains_returns_50(self, loader: DatasetLoader):
        domains = loader.get_domains()
        assert len(domains) == 50

    def test_get_role_scope(self, loader: DatasetLoader):
        scope = loader.get_role_scope("Data Analyst")
        assert scope is not None
        assert len(scope) > 10

    def test_get_role_scope_unknown(self, loader: DatasetLoader):
        assert loader.get_role_scope("Nonexistent Role") is None

    def test_get_role_domains_sorted_descending(self, loader: DatasetLoader):
        df = loader.get_role_domains("Data Scientist")
        assert len(df) == 50
        scores = df["project_relevance_score_1_to_5"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_get_domain_roles(self, loader: DatasetLoader):
        df = loader.get_domain_roles("D01")
        assert len(df) == 20  # All roles should have D01

    def test_get_construct_families(self, loader: DatasetLoader):
        families = loader.get_construct_families()
        assert len(families) >= 10
        assert "Cognitive ability" in families

    def test_get_dataset_info(self, loader: DatasetLoader):
        info = loader.get_dataset_info()
        assert info.num_roles == 20
        assert info.num_domains == 50
        assert info.num_mappings == 1000
        assert info.score_range == (1, 5)
        assert info.duplicate_role_domain == 0
        assert info.csv_xlsx_scores_agree is True
        assert "Healthcare Professional" in info.weak_proxy_roles


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------
class TestErrorHandling:
    def test_missing_file_raises(self, tmp_path: Path):
        config = AppConfig()
        # Override csv_path to nonexistent
        import dataclasses
        config = dataclasses.replace(config, csv_path=tmp_path / "nonexistent.csv")
        ldr = DatasetLoader(config=config)
        with pytest.raises(DataLoadError, match="not found"):
            ldr.load()

    def test_missing_columns_raises(self, tmp_path: Path):
        df = pd.DataFrame({"role": ["A"], "domain_id": ["D01"]})
        path = tmp_path / "bad.csv"
        df.to_csv(path, index=False)
        config = AppConfig()
        import dataclasses
        config = dataclasses.replace(config, csv_path=path)
        ldr = DatasetLoader(config=config)
        with pytest.raises(DataLoadError, match="Missing columns"):
            ldr.load()

    def test_reload_refreshes_data(self, sample_csv: Path):
        config = AppConfig()
        import dataclasses
        config = dataclasses.replace(config, csv_path=sample_csv)
        ldr = DatasetLoader(config=config)
        df1 = ldr.load()
        df2 = ldr.reload()
        assert df1.shape == df2.shape


# ---------------------------------------------------------------------------
# Consistency tests
# ---------------------------------------------------------------------------
class TestConsistency:
    def test_all_roles_have_evidence_status(self, loader: DatasetLoader):
        assert loader.df["evidence_status"].isnull().sum() == 0

    def test_all_roles_have_review_priority(self, loader: DatasetLoader):
        assert loader.df["review_priority"].isnull().sum() == 0

    def test_score_interpretation_matches_score(self, loader: DatasetLoader):
        """Score interpretation labels should be consistent with scores."""
        df = loader.df
        score_5 = df[df["project_relevance_score_1_to_5"] == 5]["score_interpretation"].unique()
        score_1 = df[df["project_relevance_score_1_to_5"] == 1]["score_interpretation"].unique()
        # Score 5 should be "Core / highly salient"
        assert "Core / highly salient" in score_5
        # Score 1 should be "Low / indirect"
        assert "Low / indirect" in score_1

    def test_onet_impact_within_range(self, loader: DatasetLoader):
        """O*NET impact scores should be 0-100 where present."""
        impacts = loader.df["O*NET_standardized_work_style_impact_0_to_100"].dropna()
        if len(impacts) > 0:
            assert impacts.min() >= 0
            assert impacts.max() <= 100
