import json
import pytest
from fastapi.testclient import TestClient
from src import llm_explainer as L
from src.analysis import analyze
from src.api import app

client = TestClient(app)

def _mock_ok(payload):
    items = [{"domain_id": d["domain_id"], "relevance_explanation": f"The source rates elements linked to {d['domain_name']} for this occupation; this suggests relevance to its tasks.",
              "related_job_responsibilities": d["tasks"][:1], "confidence_explanation": "Pending review.",
              "limitations": [], "evidence_reference_ids": d["evidence_reference_ids"][:1]} for d in payload["domains"]]
    return json.dumps({"explanations": items})

def test_mock_llm_valid():
    r = analyze(onet_soc_code="15-2051.00", llm_callable=_mock_ok)
    assert r.is_llm_generated and r.psychometric_domains[0].relevance_explanation

@pytest.mark.parametrize("bad", ["not json", json.dumps({"explanations":[{"domain_id":"P001","relevance_explanation":"x","evidence_reference_ids":["FAKE-1"]}]})])
def test_invalid_llm_falls_back_to_preview(bad):
    r = analyze(onet_soc_code="15-2051.00", llm_callable=lambda p: bad)
    assert not r.is_llm_generated and r.llm_provider == "preview"

def test_unsupported_claim_rejected(data_scientists):
    with pytest.raises(ValueError):
        L.validate_llm_output(json.dumps({"explanations":[{"domain_id":"P001","relevance_explanation":"This trait guarantees job performance","evidence_reference_ids":[data_scientists.evidence_references[0].evidence_id]}]}), data_scientists.psychometric_domains)

def test_provider_timeout_falls_back():
    def boom(p): raise TimeoutError("timed out")
    r = analyze(onet_soc_code="15-2051.00", llm_callable=boom)
    assert not r.is_llm_generated

def test_missing_api_key_is_preview(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False); monkeypatch.setenv("LLM_PROVIDER", "preview")
    r = analyze(onet_soc_code="15-2051.00")
    assert r.llm_provider == "preview" and not r.is_llm_generated

def test_health():
    b = client.get("/health").json(); assert b["status"] in ("ok","degraded") and "llm" in b

def test_search_and_ambiguous_analyze():
    s = client.get("/roles/search", params={"q": "Data Analyst"})
    assert s.status_code == 200 and len(s.json()["candidates"]) >= 2
    a = client.post("/analyze", json={"job_title": "Data Analyst"})
    assert a.json()["resolution_status"] == "ambiguous_clarification_required"

def test_resolved_analyze_and_invalid_input():
    a = client.post("/analyze", json={"onet_soc_code": "15-2051.00"})
    body = a.json(); assert body["resolution_status"] == "resolved" and body["construct_families"]
    assert client.post("/analyze", json={}).status_code == 422
    bad = client.post("/analyze", json={"onet_soc_code": "99-9999.99"})
    assert bad.json()["resolution_status"] == "invalid_code"

def test_response_has_limitations_and_evidence(data_scientists):
    assert data_scientists.limitations and data_scientists.evidence_references and data_scientists.data_gaps
