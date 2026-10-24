"""Core behaviour: datasets, title resolution, evidence, ranking, LLM safety.
All tests run offline (LLM is mocked or explanations are disabled)."""
import json

import pandas as pd
import pytest

import llm
from analysis import (analyze, clean_plain, data_status, get_bundle,
                      has_banned_jargon, ranked_domains, resolve_title)

DS = "15-2051.00"  # Data Scientists


def _valid_mock(payload):
    return json.dumps({"explanations": [
        {"domain_id": d["domain_id"], "relevance_explanation": "This domain fits the daily work of this role.",
         "related_job_responsibilities": [], "confidence_explanation": "ok", "limitations": [],
         "evidence_reference_ids": d["evidence_reference_ids"][:1]} for d in payload["domains"]]})


@pytest.fixture(scope="module")
def ds():
    return analyze(onet_soc_code=DS, llm_callable=_valid_mock)


def test_datasets_load():
    st = data_status()
    assert st == {"grid_rows": 50800, "evidence_rows": 120770, "domains": 50,
                  "occupations": 1016, "files_loaded": True}


def test_title_resolution():
    r = resolve_title("Data Scientist")
    assert r.status == "resolved" and r.candidates[0].onet_soc_code == DS
    assert resolve_title("datascientist").candidates[0].onet_soc_code == DS
    assert resolve_title("HR Manager").candidates[0].onet_soc_code == "11-3121.00"


def test_ambiguous_title_never_auto_selected():
    r = resolve_title("Data Analyst")
    assert r.status == "ambiguous_clarification_required" and len(r.candidates) > 1
    resp = analyze(job_title="Data Analyst")
    assert resp.resolution_status == "ambiguous_clarification_required"
    assert resp.psychometric_domains == []


def test_invalid_code_is_clean_error():
    resp = analyze(onet_soc_code="99-9999.00")
    assert resp.resolution_status == "invalid_code" and resp.psychometric_domains == []


def test_evidence_references_resolve(ds):
    ev = pd.read_csv("data/processed/occupation_domain_evidence.csv", dtype=str,
                     usecols=["onet_soc_code", "domain_id", "onet_element_id", "scale_id"])
    valid = set(ev.itertuples(index=False, name=None))
    checked = 0
    for ref in ds.evidence_references:
        if ref.onet_element_id:
            checked += 1
            domain_id = ref.source_record.split("[")[1].split("|")[1]
            assert (ref.onet_soc_code, domain_id, ref.onet_element_id, ref.scale_id) in valid
    assert checked > 0


def test_scores_and_review_flags(ds):
    assert ds.human_review_required and all(d.human_review_required for d in ds.psychometric_domains)
    assert any("pending_review" in lim for lim in ds.limitations)
    grid = pd.read_csv("data/processed/occupation_domain_mapping.csv", dtype=str, keep_default_na=False)
    assert (grid.mapping_status != "validated").all()
    # missing scores stay missing in the source data (never zero-filled)
    assert (grid.preliminary_priority_score == "").any()


def test_ranked_output_shape(ds):
    items = ranked_domains(ds)
    assert 1 <= len(items) <= 8
    scores = [i["score"] for i in items]
    assert scores == sorted(scores, reverse=True) and all(s >= 60 for s in scores)
    for i in items:
        assert i["name"] and i["what_it_means"].endswith(".") and i["why_it_fits"]
        assert not has_banned_jargon(i["what_it_means"] + i["why_it_fits"])


def test_clean_plain_strips_codes():
    out = clean_plain("Analyzes complex data (IM=92). Second sentence follows.")
    assert "IM" not in out and "Second" not in out


def test_llm_mock_explanations_accepted(ds):
    recs, info = llm.explain(ds.psychometric_domains, {"onet_soc_code": DS}, llm_callable=_valid_mock)
    assert info["is_llm"] and all(d.relevance_explanation for d in recs)


def test_llm_invented_reference_falls_back(ds):
    def bad(payload):
        data = json.loads(_valid_mock(payload))
        data["explanations"][0]["evidence_reference_ids"] = ["EV-INVENTED"]
        return json.dumps(data)
    _, info = llm.explain(ds.psychometric_domains, {"onet_soc_code": DS}, llm_callable=bad)
    assert not info["is_llm"] and "fallback_reason" in info


def test_preview_without_credentials():
    cfg = {"provider": "preview", "model": "", "api_key": "", "base_url": "", "timeout": 5}
    assert not llm.is_configured(cfg)
    bundle = get_bundle(DS)
    assert bundle["mapping"].shape[0] == 50
