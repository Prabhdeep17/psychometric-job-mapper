"""Tests for the role matcher module."""

from __future__ import annotations

import pytest

from src.role_matcher import match_role, MatchType, ROLE_ALIASES


KNOWN_ROLES = [
    "Customer Support Specialist", "Cybersecurity Analyst", "Data Analyst",
    "Data Scientist", "Entrepreneur", "Financial Analyst", "HR Manager",
    "Healthcare Professional", "Lawyer", "Machine Learning Engineer",
    "Marketing Manager", "Mechanical Engineer", "Operations Manager",
    "Product Manager", "Project Manager", "Research Scientist",
    "Sales Representative", "Software Engineer", "Teacher", "UX Designer",
]


class TestExactMatch:
    def test_exact_case_match(self):
        result = match_role("Data Scientist", KNOWN_ROLES)
        assert result.match_type == MatchType.EXACT
        assert result.matched_role == "Data Scientist"
        assert result.confidence == 1.0
        assert result.proxy_warning is None

    def test_case_insensitive_match(self):
        result = match_role("data scientist", KNOWN_ROLES)
        assert result.match_type == MatchType.EXACT
        assert result.matched_role == "Data Scientist"

    def test_extra_whitespace(self):
        result = match_role("  Data   Scientist  ", KNOWN_ROLES)
        assert result.match_type == MatchType.EXACT
        assert result.matched_role == "Data Scientist"

    @pytest.mark.parametrize("role", KNOWN_ROLES)
    def test_all_stored_roles_match_exactly(self, role):
        result = match_role(role, KNOWN_ROLES)
        assert result.match_type == MatchType.EXACT
        assert result.matched_role == role


class TestAliasMatch:
    def test_alias_ml_engineer(self):
        result = match_role("ML Engineer", KNOWN_ROLES)
        assert result.match_type == MatchType.ALIAS
        assert result.matched_role == "Machine Learning Engineer"
        assert result.proxy_warning is not None

    def test_alias_developer(self):
        result = match_role("developer", KNOWN_ROLES)
        assert result.match_type == MatchType.ALIAS
        assert result.matched_role == "Software Engineer"

    def test_alias_security_analyst(self):
        result = match_role("Security Analyst", KNOWN_ROLES)
        assert result.match_type == MatchType.ALIAS
        assert result.matched_role == "Cybersecurity Analyst"

    def test_alias_nurse(self):
        result = match_role("nurse", KNOWN_ROLES)
        assert result.match_type == MatchType.ALIAS
        assert result.matched_role == "Healthcare Professional"

    def test_alias_professor(self):
        result = match_role("Professor", KNOWN_ROLES)
        assert result.match_type == MatchType.ALIAS
        assert result.matched_role == "Teacher"

    def test_alias_confidence_below_exact(self):
        result = match_role("AI Engineer", KNOWN_ROLES)
        assert result.match_type == MatchType.ALIAS
        assert result.confidence < 1.0

    def test_all_aliases_resolve_to_known_roles(self):
        """Every alias must map to a role that exists in KNOWN_ROLES."""
        known_lower = {r.lower() for r in KNOWN_ROLES}
        for alias, target in ROLE_ALIASES.items():
            assert target.lower() in known_lower, (
                f"Alias '{alias}' maps to '{target}' which is not a known role"
            )


class TestFuzzyMatch:
    def test_fuzzy_close_title(self):
        result = match_role("Data Analysts", KNOWN_ROLES)
        assert result.match_type in (MatchType.FUZZY, MatchType.EXACT)
        # Should match Data Analyst (singular)

    def test_fuzzy_provides_alternatives(self):
        result = match_role("Senior Data Engineer", KNOWN_ROLES)
        assert len(result.alternatives) > 0

    def test_fuzzy_has_proxy_warning(self):
        result = match_role("Chief Data Officer", KNOWN_ROLES)
        if result.match_type == MatchType.FUZZY:
            assert result.proxy_warning is not None


class TestNoMatch:
    def test_completely_unrelated(self):
        result = match_role("Astronaut Space Commander", KNOWN_ROLES)
        assert result.match_type in (MatchType.NONE, MatchType.FUZZY)
        if result.match_type == MatchType.NONE:
            assert result.matched_role is None
            assert result.proxy_warning is not None

    def test_no_match_has_alternatives(self):
        result = match_role("xyz123_random_title", KNOWN_ROLES)
        assert len(result.alternatives) > 0

    def test_empty_string(self):
        result = match_role("", KNOWN_ROLES)
        # Should gracefully handle empty input
        assert result is not None


class TestEdgeCases:
    def test_special_characters(self):
        result = match_role("Data Analyst!", KNOWN_ROLES)
        # Should still work
        assert result is not None

    def test_very_long_title(self):
        result = match_role("Senior Principal Staff Data Scientist III" * 3, KNOWN_ROLES)
        assert result is not None
