"""API behaviour. The analysis route is patched to skip live LLM calls,
so these tests are fast and offline."""
import pytest
from fastapi.testclient import TestClient

import api
from analysis import analyze as real_analyze

client = TestClient(api.app)


@pytest.fixture(autouse=True)
def offline_analysis(monkeypatch):
    monkeypatch.setattr(api, "analyze",
                        lambda **kw: real_analyze(**{**kw, "with_explanations": False}))


def test_health():
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["data"]["occupations"] == 1016
    assert "configured" in body["llm"]


def test_role_search():
    body = client.get("/roles/search", params={"q": "Data Scientist"}).json()
    assert body["candidates"] and body["candidates"][0]["onet_soc_code"] == "15-2051.00"


def test_analyze_requires_input():
    assert client.post("/analyze", json={}).status_code == 422


def test_analyze_ambiguous_asks():
    body = client.post("/analyze", json={"job_title": "Data Analyst"}).json()
    assert body["resolution_status"] == "ambiguous_clarification_required"
    assert body["psychometric_domains"] == [] and body["occupation_candidates"]


def test_analyze_by_code():
    body = client.post("/analyze", json={"onet_soc_code": "15-2051.00"}).json()
    assert body["resolved_occupation"]["official_occupation_title"] == "Data Scientists"
    assert body["psychometric_domains"] and body["evidence_references"]
    assert body["human_review_required"] is True
