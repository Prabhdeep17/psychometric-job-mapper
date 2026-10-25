"""Groq writes explanation text only. Failures use the preview sentence."""
import json

import pytest
from fastapi.testclient import TestClient

import explain_domains as L
from api import app
from rank_domains import analyze

FAKE_KEY = "gsk_TESTFAKEKEY_do_not_leak"


def _valid(payload):
    return json.dumps({"explanations": [{
        "domain_id": d["domain_id"],
        "relevance_explanation": "This role uses this strength in its daily work.",
        "related_job_responsibilities": ["Invented task"],
        "confidence_explanation": "Invented confidence",
        "limitations": ["Invented limitation"],
        "evidence_reference_ids": d["evidence_reference_ids"][:1],
    } for d in payload["domains"]]})


class RateLimitError(Exception):
    pass


class AuthenticationError(Exception):
    pass


def test_groq_config_from_env(monkeypatch):
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("GROQ_API_KEY", FAKE_KEY)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    cfg = L.config()
    assert cfg["provider"] == "groq" and cfg["model"] == "openai/gpt-oss-20b"
    assert L.is_configured(cfg)


def test_fallback_explanations_are_specific_to_each_domain(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "preview")
    r = analyze(onet_soc_code="15-2051.00", with_explanations=True)
    assert r.is_llm_generated is False and r.llm_provider == "preview"
    texts = [d.relevance_explanation for d in r.psychometric_domains if d.relevance_explanation]
    assert len(texts) >= 2 and len(set(texts)) == len(texts)
    weak = [d for d in r.psychometric_domains if d.evidence_strength in {"partial", "proxy", "insufficient"}]
    assert weak and all("provisional" in d.relevance_explanation.lower() for d in weak)
    assert all("does not by itself establish" in d.relevance_explanation or d.evidence_strength == "insufficient"
               or "insufficient" in d.relevance_explanation.lower() for d in weak)


def test_missing_key_stays_preview(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "preview")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert not L.is_configured()


def test_mock_success_does_not_change_score_tasks_or_review_flag(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", FAKE_KEY)
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-oss-20b")
    before = analyze(onet_soc_code="15-2051.00", top_per_family=1, with_explanations=False)
    saved = {d.domain_id: (d.preliminary_score, list(d.related_job_responsibilities), d.human_review_required)
             for d in before.psychometric_domains}
    r = analyze(onet_soc_code="15-2051.00", top_per_family=1, llm_callable=_valid)
    assert r.is_llm_generated
    for d in r.psychometric_domains:
        score, tasks, flag = saved[d.domain_id]
        assert d.preliminary_score == score
        assert d.related_job_responsibilities == tasks
        assert d.human_review_required is flag is True
        assert "Invented" not in " ".join(d.limitations)


def test_malformed_response_falls_back():
    r = analyze(onet_soc_code="15-2051.00", top_per_family=1, llm_callable=lambda p: "not json")
    assert not r.is_llm_generated and r.llm_provider == "preview"


def test_mixed_groq_and_fallback_are_labelled_per_domain(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", FAKE_KEY)
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-oss-20b")

    def only_first(payload):
        d = payload["domains"][0]
        return json.dumps({"explanations": [{
            "domain_id": d["domain_id"],
            "relevance_explanation": "This occupation is rated on the supplied requirement. The connection to this domain is direct.",
            "evidence_reference_ids": d.get("evidence_reference_ids", [])[:1],
        }]})

    r = analyze(onet_soc_code="15-2051.00", top_per_family=1, llm_callable=only_first)
    sources = {d.explanation_source for d in r.psychometric_domains}
    assert "groq" in sources and "fallback" in sources
    assert r.is_llm_generated is True


def test_tasks_are_not_copied_onto_every_domain():
    r = analyze(onet_soc_code="15-2051.00", with_explanations=False)
    lists = [tuple(d.related_job_responsibilities) for d in r.psychometric_domains]
    assert not (len(set(lists)) == 1 and lists and len(lists[0]) >= 2)


def test_invented_evidence_reference_falls_back():
    bad = lambda p: json.dumps({"explanations": [{"domain_id": "P001", "relevance_explanation": "x",
                                                   "evidence_reference_ids": ["EV-INVENTED-999"]}]})
    r = analyze(onet_soc_code="15-2051.00", top_per_family=1, llm_callable=bad)
    assert not r.is_llm_generated


@pytest.mark.parametrize("exc", [RateLimitError("429"), AuthenticationError("401"), TimeoutError("timed out")])
def test_provider_errors_fall_back_without_leaking_key(monkeypatch, exc):
    monkeypatch.setenv("GROQ_API_KEY", FAKE_KEY)
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-oss-20b")
    r = analyze(onet_soc_code="15-2051.00", top_per_family=1, llm_callable=lambda p: (_ for _ in ()).throw(exc))
    assert not r.is_llm_generated
    assert FAKE_KEY not in r.model_dump_json()


def test_key_redacted_from_error_message(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", FAKE_KEY)
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("LLM_MODEL", "m")
    cfg = L.config()
    assert FAKE_KEY not in L._safe_error(Exception(f"failed with {FAKE_KEY}"), cfg)


def test_retry_transient_only():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RateLimitError("429")
        return "ok"

    assert L.call_with_retry(flaky, {"max_retries": 2, "retry_base_s": 0.01}, sleep=lambda s: None) == "ok"
    with pytest.raises(AuthenticationError):
        L.call_with_retry(lambda: (_ for _ in ()).throw(AuthenticationError("401")),
                          {"max_retries": 3, "retry_base_s": 0.01}, sleep=lambda s: None)


def test_health_does_not_return_the_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", FAKE_KEY)
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-oss-20b")
    body = TestClient(app).get("/health").json()
    assert body["llm"]["configured"] is True
    assert FAKE_KEY not in json.dumps(body)
