import json
import pytest
from src import llm_explainer as L
from src.analysis import analyze

FAKE_KEY = "gsk_TESTFAKEKEY_do_not_leak"

def _valid_payload_json(payload):
    return json.dumps({"explanations": [{"domain_id": d["domain_id"],
        "relevance_explanation": "The source rates elements linked to this domain for this occupation; this suggests relevance to its listed tasks.",
        "related_job_responsibilities": d["tasks"][:1], "confidence_explanation": "Pending review — preliminary.",
        "limitations": [], "evidence_reference_ids": d["evidence_reference_ids"][:1]} for d in payload["domains"]]})

class RateLimitError(Exception): pass
class AuthenticationError(Exception): pass

def test_groq_config_from_env(monkeypatch):
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: None)  # isolate from the developer's real .env
    monkeypatch.setenv("GROQ_API_KEY", FAKE_KEY); monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False); monkeypatch.delenv("LLM_BASE_URL", raising=False)
    cfg = L.config()
    assert cfg["provider"] == "groq" and cfg["model"] == "openai/gpt-oss-20b"
    assert cfg["base_url"] == "https://api.groq.com/openai/v1" and L.is_configured(cfg)

def test_missing_key_stays_preview(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False); monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "preview")
    assert not L.is_configured()
    r = analyze(onet_soc_code="15-2051.00")
    assert r.llm_provider == "preview" and not r.is_llm_generated

def test_groq_mock_success_marks_live(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", FAKE_KEY); monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-oss-20b")
    r = analyze(onet_soc_code="15-2051.00", llm_callable=_valid_payload_json)
    assert r.is_llm_generated and r.llm_provider == "groq" and r.llm_model == "openai/gpt-oss-20b"
    assert all(d.human_review_required for d in r.psychometric_domains)  # pending flags unchanged

@pytest.mark.parametrize("exc", [RateLimitError("429 rate limited"), AuthenticationError("401 invalid key"), TimeoutError("timed out")])
def test_provider_errors_fall_back_without_leaking_key(monkeypatch, exc):
    monkeypatch.setenv("GROQ_API_KEY", FAKE_KEY); monkeypatch.setenv("LLM_PROVIDER", "groq")
    def boom(p): raise exc
    r = analyze(onet_soc_code="15-2051.00", llm_callable=boom)
    assert not r.is_llm_generated and r.llm_provider == "preview"
    assert FAKE_KEY not in " ".join(r.limitations) and FAKE_KEY not in r.model_dump_json()

def test_key_redacted_from_error_message(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", FAKE_KEY)
    cfg = L.config()
    assert FAKE_KEY not in L._safe_error(Exception(f"failed with {FAKE_KEY}"), cfg)

def test_malformed_groq_response_falls_back():
    r = analyze(onet_soc_code="15-2051.00", llm_callable=lambda p: "Here is my answer: not json")
    assert not r.is_llm_generated and r.llm_provider == "preview"

def test_invented_evidence_reference_rejected():
    bad = lambda p: json.dumps({"explanations": [{"domain_id": "P001", "relevance_explanation": "x",
                        "evidence_reference_ids": ["EV-INVENTED-999"]}]})
    r = analyze(onet_soc_code="15-2051.00", llm_callable=bad)
    assert not r.is_llm_generated  # validator rejects, preview used — LLM cannot invent evidence

def test_health_reports_groq_mode_when_configured(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", FAKE_KEY); monkeypatch.setenv("LLM_PROVIDER", "groq"); monkeypatch.delenv("LLM_MODEL", raising=False)
    from fastapi.testclient import TestClient
    from src.api import app
    b = TestClient(app).get("/health").json()
    assert b["llm"]["provider"] == "groq" and b["llm"]["configured"] is True and b["llm"]["mode"] == "live"
    assert FAKE_KEY not in json.dumps(b)
