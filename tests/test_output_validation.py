"""Tests for the output validation module."""

from __future__ import annotations

import json

import pytest

from src.output_validation import (
    validate_domain_references,
    validate_jd_analysis_response,
    sanitize_llm_output,
    separate_known_domain_items,
    invented_score_warnings,
)


KNOWN_IDS = {f"D{i:02d}" for i in range(1, 51)}


class TestDomainReferenceValidation:
    def test_valid_references(self):
        text = "The domain D01 (Analytical Thinking) and D05 (Conscientiousness) are relevant."
        result = validate_domain_references(text, KNOWN_IDS)
        assert result.is_valid
        assert len(result.warnings) == 0

    def test_unknown_reference(self):
        text = "The domain D99 is very important."
        result = validate_domain_references(text, KNOWN_IDS)
        assert not result.is_valid
        assert len(result.warnings) > 0
        assert "D99" in result.warnings[0]

    def test_no_references(self):
        text = "This text has no domain references."
        result = validate_domain_references(text, KNOWN_IDS)
        assert result.is_valid

    def test_mixed_valid_invalid(self):
        text = "D01 is good, D55 is unknown, D50 is fine."
        result = validate_domain_references(text, KNOWN_IDS)
        assert not result.is_valid
        assert "D55" in result.warnings[0]


class TestJdAnalysisValidation:
    def test_valid_json(self):
        data = {
            "primary_domains": [
                {"domain_id": "D01", "domain_name": "Analytical Thinking", "reasoning": "Core skill."}
            ],
            "secondary_domains": [],
            "ambiguity_notes": "None",
            "suggested_reference_roles": ["Data Analyst"],
            "review_notes": "Looks good.",
            "unverified_suggestions": [],
        }
        result = validate_jd_analysis_response(json.dumps(data), KNOWN_IDS)
        assert result.is_valid
        assert result.parsed_data is not None

    def test_json_in_code_block(self):
        raw = '```json\n{"primary_domains": [], "secondary_domains": []}\n```'
        result = validate_jd_analysis_response(raw, KNOWN_IDS)
        assert result.is_valid
        assert result.parsed_data is not None

    def test_invalid_json(self):
        result = validate_jd_analysis_response("This is not JSON at all.", KNOWN_IDS)
        assert not result.is_valid
        assert len(result.errors) > 0

    def test_unknown_domain_in_response(self):
        data = {
            "primary_domains": [
                {"domain_id": "D99", "domain_name": "Fake Domain", "reasoning": "..."}
            ],
            "secondary_domains": [],
        }
        result = validate_jd_analysis_response(json.dumps(data), KNOWN_IDS)
        assert result.is_valid  # Still parses, but has warnings
        assert len(result.warnings) > 0
        assert "D99" in result.warnings[0]

    def test_missing_keys_warning(self):
        data = {"other_field": "value"}
        result = validate_jd_analysis_response(json.dumps(data), KNOWN_IDS)
        assert result.is_valid
        assert len(result.warnings) > 0  # Missing expected keys


class TestSanitization:
    def test_normal_text_unchanged(self):
        text = "This domain is highly relevant because..."
        result = sanitize_llm_output(text)
        assert result == text

    def test_invented_score_pattern(self):
        text = "I would rate this domain 4/5 for relevance."
        result = sanitize_llm_output(text)
        assert "4/5" in result
        assert invented_score_warnings(text)


class TestScoreSeparation:
    def test_unknown_domain_is_unverified_and_score_is_dropped(self):
        known, unverified = separate_known_domain_items(
            [
                {"domain_id": "D01", "domain_name": "Analytical Thinking", "score": 5, "reasoning": "Stored id"},
                {"domain_id": "D99", "domain_name": "Invented", "reasoning": "Not in the list"},
            ],
            KNOWN_IDS,
        )
        assert known[0]["domain_id"] == "D01"
        assert "score" not in known[0]
        assert any("Invented" in item for item in unverified)
