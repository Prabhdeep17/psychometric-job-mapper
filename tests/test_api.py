"""API checks. Groq is mocked here so the test does not spend API credits."""
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def _no_groq(recs, occupation, job_description="", llm_callable=None, cfg=None):
    return recs, {"provider": "preview", "model": "deterministic-preview", "is_llm": False}


def test_health_loads_the_runtime_tables():
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["data"]["occupations"] == 1016
    assert body["data"]["domains"] == 50
    assert body["llm"]["mode"] in {"live", "preview"}
    assert "api_key" not in json_text(body)


def test_ambiguous_title_is_not_chosen_for_you():
    search = client.get("/roles/search", params={"q": "Data Analyst"}).json()
    assert search["resolution_status"] != "resolved"
    assert len(search["candidates"]) > 1
    result = client.post("/analyze", json={"job_title": "Data Analyst"}).json()
    assert result["resolution_status"] == "ambiguous_clarification_required"
    assert result["psychometric_domains"] == []


def test_known_job_returns_evidence_and_keeps_the_score(monkeypatch):
    import rank_domains
    monkeypatch.setattr(rank_domains, "explain", _no_groq)
    result = client.post("/analyze", json={"onet_soc_code": "15-2051.00", "top_per_family": 5}).json()
    assert result["resolved_occupation"]["official_occupation_title"] == "Data Scientists"
    row = next(d for d in result["psychometric_domains"] if d["domain_id"] == "P002")
    assert abs(row["preliminary_score"] - 96.91) < 0.02
    assert row["evidence_references"]
    assert result["is_llm_generated"] in {True, False}


def json_text(body) -> str:
    import json
    return json.dumps(body)
