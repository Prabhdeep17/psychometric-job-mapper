"""Tests for optional LLM output handling. No network calls."""

from __future__ import annotations

from src.groq_router import GroqRouter, InferenceResult
from src.llm_service import LlmService
from src.config import GroqConfig


def test_job_description_drops_unknown_ids_and_does_not_require_network():
    router = GroqRouter(config=GroqConfig(api_key="test-key-not-real", max_retries=1))

    def fake_chat(messages, temperature=0.2, max_tokens=1800, response_format=None):
        return InferenceResult(
            model_id="model-b",
            content=(
                '{"primary_domains": [{"domain_id": "D01", "domain_name": "Analytical Thinking", '
                '"score": 9, "reasoning": "The posting asks for analysis."}, '
                '{"domain_id": "D99", "domain_name": "Made Up", "reasoning": "Not stored."}], '
                '"secondary_domains": [], "ambiguity_notes": "Title is broad.", '
                '"suggested_reference_roles": ["Data Analyst"], "review_notes": "Check tasks.", '
                '"unverified_suggestions": []}'
            ),
            fallback_used=True,
            attempts=2,
        )

    router.chat_completion = fake_chat  # type: ignore[method-assign]
    service = LlmService(router)
    result = service.analyze_job_description(
        "Analyze sales data and build dashboards.",
        [{"domain_id": "D01", "domain_name": "Analytical Thinking", "construct_family": "Cognitive ability", "operational_definition": "Break information down."}],
        ["Data Analyst"],
        {"D01"},
    )
    assert result.ok
    assert result.primary_domains[0]["domain_id"] == "D01"
    assert "score" not in result.primary_domains[0]
    assert result.unverified_suggestions
    assert result.fallback_used is True
    assert "Data Analyst" in result.suggested_reference_roles


def test_missing_api_key_does_not_raise():
    router = GroqRouter(config=GroqConfig(api_key=None))
    service = LlmService(router)
    result = service.analyze_job_description("text", [], [], set())
    assert result.ok is False
    assert "GROQ_API_KEY" in (result.error or "")
