"""Tests for the ranking engine module."""

from __future__ import annotations

import pytest
import pandas as pd

from src.config import AppConfig
from src.data_loader import DatasetLoader
from src.ranking_engine import (
    rank_domains,
    get_full_profile,
    filter_domains,
    domains_to_dataframe,
    RankedDomain,
    RankingResult,
)


@pytest.fixture
def loader() -> DatasetLoader:
    config = AppConfig()
    ldr = DatasetLoader(config=config)
    ldr.load()
    return ldr


ALL_ROLES = [
    "Customer Support Specialist", "Cybersecurity Analyst", "Data Analyst",
    "Data Scientist", "Entrepreneur", "Financial Analyst", "HR Manager",
    "Healthcare Professional", "Lawyer", "Machine Learning Engineer",
    "Marketing Manager", "Mechanical Engineer", "Operations Manager",
    "Product Manager", "Project Manager", "Research Scientist",
    "Sales Representative", "Software Engineer", "Teacher", "UX Designer",
]


# ---------------------------------------------------------------------------
# Core ranking tests
# ---------------------------------------------------------------------------
class TestRanking:
    def test_top_10_returns_at_least_10(self, loader: DatasetLoader):
        result = rank_domains("Data Scientist", loader, top_n=10)
        assert result is not None
        assert result.returned_count >= 10

    def test_scores_are_descending(self, loader: DatasetLoader):
        result = rank_domains("Data Analyst", loader, top_n=50)
        assert result is not None
        scores = [d.relevance_score for d in result.domains]
        assert scores == sorted(scores, reverse=True)

    def test_all_scores_in_range(self, loader: DatasetLoader):
        result = get_full_profile("Software Engineer", loader)
        assert result is not None
        for d in result.domains:
            assert 1 <= d.relevance_score <= 5

    def test_full_profile_has_50_domains(self, loader: DatasetLoader):
        result = get_full_profile("Data Scientist", loader)
        assert result is not None
        assert result.total_domains == 50
        assert len(result.domains) == 50

    @pytest.mark.parametrize("role", ALL_ROLES)
    def test_all_roles_retrievable(self, role: str, loader: DatasetLoader):
        result = get_full_profile(role, loader)
        assert result is not None
        assert result.role == role
        assert len(result.domains) == 50

    def test_unknown_role_returns_none(self, loader: DatasetLoader):
        result = rank_domains("Nonexistent Role", loader)
        assert result is None


class TestTieHandling:
    def test_ties_included_at_cutoff(self, loader: DatasetLoader):
        """When tie inclusion is on, domains tied at the boundary appear."""
        result_with_ties = rank_domains("Data Analyst", loader, top_n=5, include_ties=True)
        result_no_ties = rank_domains("Data Analyst", loader, top_n=5, include_ties=False)
        assert result_with_ties is not None
        assert result_no_ties is not None
        # With ties should include >= without ties
        assert result_with_ties.returned_count >= result_no_ties.returned_count

    def test_rank_matches_stored_column(self, loader: DatasetLoader):
        raw = loader.get_role_domains("Data Analyst")
        stored = dict(zip(raw["domain_id"], raw["within_role_rank_ties_respected"].astype(int)))
        result = get_full_profile("Data Analyst", loader)
        assert result is not None
        for domain in result.domains:
            assert domain.rank == stored[domain.domain_id]

    def test_equal_scores_treated_as_ties(self, loader: DatasetLoader):
        result = get_full_profile("Data Analyst", loader)
        assert result is not None
        # Check that domains with equal scores have equal ranks
        score_to_rank = {}
        for d in result.domains:
            if d.relevance_score in score_to_rank:
                assert d.rank == score_to_rank[d.relevance_score], (
                    f"Score {d.relevance_score} has inconsistent ranks"
                )
            else:
                score_to_rank[d.relevance_score] = d.rank


class TestRankedDomainFields:
    def test_all_fields_populated(self, loader: DatasetLoader):
        result = rank_domains("Data Scientist", loader, top_n=5)
        assert result is not None
        for d in result.domains:
            assert d.domain_id
            assert d.domain_name
            assert d.construct_family
            assert d.operational_definition
            assert d.rationale
            assert d.confidence
            assert d.score_status
            assert d.evidence_status
            assert d.onet_reference_occupation
            assert d.onet_source_url

    def test_onet_impact_is_float_or_none(self, loader: DatasetLoader):
        result = get_full_profile("Data Analyst", loader)
        assert result is not None
        for d in result.domains:
            if d.onet_impact_score is not None:
                assert isinstance(d.onet_impact_score, float)
                assert 0 <= d.onet_impact_score <= 100


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
class TestFiltering:
    def test_filter_by_family(self, loader: DatasetLoader):
        result = get_full_profile("Data Analyst", loader)
        assert result is not None
        filtered = filter_domains(result, construct_family="Cognitive ability")
        assert all(d.construct_family == "Cognitive ability" for d in filtered)
        assert len(filtered) > 0

    def test_filter_by_min_score(self, loader: DatasetLoader):
        result = get_full_profile("Data Analyst", loader)
        assert result is not None
        filtered = filter_domains(result, min_score=4)
        assert all(d.relevance_score >= 4 for d in filtered)

    def test_filter_by_max_score(self, loader: DatasetLoader):
        result = get_full_profile("Data Analyst", loader)
        assert result is not None
        filtered = filter_domains(result, max_score=2)
        assert all(d.relevance_score <= 2 for d in filtered)

    def test_filter_combined(self, loader: DatasetLoader):
        result = get_full_profile("Software Engineer", loader)
        assert result is not None
        filtered = filter_domains(result, min_score=3, max_score=5)
        for d in filtered:
            assert 3 <= d.relevance_score <= 5


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
class TestExport:
    def test_domains_to_dataframe(self, loader: DatasetLoader):
        result = rank_domains("Data Scientist", loader, top_n=10)
        assert result is not None
        df = domains_to_dataframe(result.domains)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == result.returned_count
        assert "Relevance Score (1-5)" in df.columns
        assert "Domain" in df.columns

    def test_empty_domains_to_dataframe(self):
        df = domains_to_dataframe([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_csv_export_roundtrip(self, loader: DatasetLoader):
        result = rank_domains("Data Analyst", loader, top_n=5)
        assert result is not None
        df = domains_to_dataframe(result.domains)
        csv = df.to_csv(index=False)
        df_read = pd.read_csv(pd.io.common.StringIO(csv))
        assert len(df_read) == len(df)


# ---------------------------------------------------------------------------
# Score consistency
# ---------------------------------------------------------------------------
class TestScoreConsistency:
    @pytest.mark.parametrize("role", ALL_ROLES)
    def test_stored_scores_not_modified(self, role: str, loader: DatasetLoader):
        """Verify that ranking doesn't modify the stored scores."""
        # Get raw data
        raw = loader.get_role_domains(role)
        raw_scores = dict(zip(raw["domain_id"], raw["project_relevance_score_1_to_5"]))

        # Get ranked data
        result = get_full_profile(role, loader)
        assert result is not None
        for d in result.domains:
            assert d.relevance_score == raw_scores[d.domain_id], (
                f"Score mismatch for {role}/{d.domain_id}: "
                f"raw={raw_scores[d.domain_id]}, ranked={d.relevance_score}"
            )
