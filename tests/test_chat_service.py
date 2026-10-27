"""Chat grounding tests. No network calls."""

from __future__ import annotations

from src.chat_service import build_chat_messages, build_context, reply_to_chat
from src.config import AppConfig, GroqConfig
from src.data_loader import DatasetLoader
from src.groq_router import GroqRouter, InferenceResult


def _loader() -> DatasetLoader:
    loader = DatasetLoader(config=AppConfig())
    loader.load()
    return loader


def test_question_uses_named_role_and_stored_score():
    context = build_context(
        "What domains matter most for a Data Scientist?",
        _loader(),
    )
    assert context.matched_role == "Data Scientist"
    assert context.match_type == "exact"
    assert "D01 Analytical Thinking: stored score 5/5" in context.data_block
    assert "**Analytical Thinking**" in context.offline_reply
    assert "Score: 5/5, Core / highly salient" in context.offline_reply
    assert "It is core to the Data Scientist role." not in context.offline_reply
    assert "Why it fits:" not in context.offline_reply
    assert "%" not in context.offline_reply
    assert "Breaking complex information into components and identifying patterns." in context.offline_reply


def test_follow_up_keeps_previous_role():
    context = build_context(
        "Why is that score so high?",
        _loader(),
        previous_role="Data Analyst",
    )
    assert context.matched_role == "Data Analyst"
    assert "Numerical Reasoning" in context.data_block or "D14" in context.data_block


def test_alias_is_labeled_proxy():
    context = build_context("ML Engineer", _loader())
    assert context.matched_role == "Machine Learning Engineer"
    assert context.match_type == "alias"
    assert context.proxy_warning


def test_unknown_title_does_not_invent_a_role():
    context = build_context("Astronaut Space Commander", _loader())
    assert context.matched_role is None or context.match_type in {"fuzzy_proxy", "no_match"}
    if context.matched_role is None:
        assert "could not match" in context.offline_reply.lower() or "Closest" in context.offline_reply


def test_messages_include_rules_and_do_not_drop_the_question():
    loader = _loader()
    context = build_context("Data Analyst", loader)
    messages = build_chat_messages(
        context,
        [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "ok"}],
    )
    assert messages[0]["role"] == "system"
    assert "Never invent" in messages[0]["content"]
    assert messages[-1]["content"].endswith("Data Analyst") or "Data Analyst" in messages[-1]["content"]
    assert "stored score" in messages[-1]["content"]


def test_role_reply_uses_model_reason_for_each_domain():
    router = GroqRouter(config=GroqConfig(api_key="test-key-not-real", max_retries=1))

    def fake_chat(messages, temperature=0.2, max_tokens=1200, response_format=None):
        return InferenceResult(
            model_id="llama-test",
            content=(
                '{"reasons": [{"domain_id": "D43", "why": '
                '"A data scientist investigates patterns and evaluates evidence, '
                'and research orientation is the work of formulating the questions '
                'and gathering that evidence."}]}'
            ),
            usage={},
        )

    router.chat_completion = fake_chat  # type: ignore[method-assign]
    text, context, warnings = reply_to_chat("Data Scientist", _loader(), router, [])
    assert context.matched_role == "Data Scientist"
    assert "**Analytical Thinking**" in text
    assert "Score: 5/5, Core / highly salient" in text
    assert "It is core to the Data Scientist role." not in text
    assert "formulating the questions and gathering that evidence" in text
    assert "What it means:" in text
    assert "%" not in text
    assert any("Analytical Thinking" in warning for warning in warnings)
