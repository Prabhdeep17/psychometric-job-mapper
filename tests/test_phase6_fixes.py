import time
import pytest
from src import analysis, llm_explainer as L

class RateLimitError(Exception): pass
class AuthenticationError(Exception): pass

def test_retry_transient_then_success():
    calls = {"n": 0}
    def flaky():
        calls["n"] += 1
        if calls["n"] < 3: raise RateLimitError("429")
        return "ok"
    assert L.call_with_retry(flaky, {"max_retries": 2, "retry_base_s": 0.01}, sleep=lambda s: None) == "ok"
    assert calls["n"] == 3

def test_no_retry_on_auth():
    calls = {"n": 0}
    def bad():
        calls["n"] += 1; raise AuthenticationError("401")
    with pytest.raises(AuthenticationError): L.call_with_retry(bad, {"max_retries": 3, "retry_base_s": 0.01}, sleep=lambda s: None)
    assert calls["n"] == 1

def test_retry_exhaustion_propagates():
    def always(): raise RateLimitError("429")
    with pytest.raises(RateLimitError): L.call_with_retry(always, {"max_retries": 1, "retry_base_s": 0.01}, sleep=lambda s: None)

def test_analysis_cache(monkeypatch):
    analysis.clear_cache()
    calls = {"n": 0}
    import json
    def mock(p):
        calls["n"] += 1
        return json.dumps({"explanations": [{"domain_id": d["domain_id"], "relevance_explanation": "Source-based relevance.",
            "related_job_responsibilities": [], "confidence_explanation": "pending", "limitations": [],
            "evidence_reference_ids": d["evidence_reference_ids"][:1]} for d in p["domains"]]})
    # cache applies only to the real-provider path (llm_callable=None), so verify key/invalidation logic directly:
    r1 = analysis.analyze(onet_soc_code="15-2051.00", top_per_family=1)
    assert len(analysis._CACHE) == 1
    r2 = analysis.analyze(onet_soc_code="15-2051.00", top_per_family=1)
    assert r1.model_dump() == r2.model_dump() and r1 is not r2  # deep copy, same content
    analysis.clear_cache(); assert len(analysis._CACHE) == 0
