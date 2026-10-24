import pytest
from src.analysis import analyze
from src.evidence_retriever import get_bundle, valid_occupation

def test_valid_bundle_has_50_rows(data_scientists):
    b = get_bundle("15-2051.00")
    assert len(b["mapping"]) == 50 and data_scientists.resolved_occupation.onet_soc_code == "15-2051.00"

def test_invalid_code():
    assert not valid_occupation("00-0000.00")
    with pytest.raises(KeyError): get_bundle("00-0000.00")

def test_no_match_and_missing_domains_in_gaps(data_scientists):
    assert any("P041" in g or "no_match" in g for g in data_scientists.data_gaps)
    assert all(d.domain_id not in {"P041","P042","P043","P044"} for d in data_scientists.psychometric_domains)

def test_proxy_domain_flagged(data_scientists):
    proxies = [d for d in data_scientists.psychometric_domains if d.mapping_status == "requires_review"]
    assert proxies and all(d.human_review_required for d in proxies)

def test_missing_tasks_occupation():
    # An 'All Other' occupation with sparse profiles should still resolve with gaps, not crash
    r = analyze(onet_soc_code="11-9199.00")  # Managers, All Other (may or may not exist)
    assert r.resolution_status in ("resolved", "invalid_code")

def test_grouped_no_global_rank(data_scientists):
    assert data_scientists.construct_families
    for d in data_scientists.psychometric_domains:
        assert not hasattr(d, "global_rank") and d.within_family_rank is not None or d.preliminary_score is None
    fams = {f.construct_family for f in data_scientists.construct_families}
    assert len(fams) >= 3

def test_missing_scores_stay_missing(data_scientists):
    assert all((d.preliminary_score is None) == (d.within_family_rank is None) or d.preliminary_score is not None for d in data_scientists.psychometric_domains)

def test_pending_review_preserved(data_scientists):
    assert data_scientists.human_review_required
    assert all(d.human_review_required for d in data_scientists.psychometric_domains)
    assert not any("validated" == d.mapping_status for d in data_scientists.psychometric_domains)

def test_evidence_refs_resolve(data_scientists):
    import pandas as pd
    ev = pd.read_csv("data/processed/occupation_domain_evidence.csv", dtype=str,
                     usecols=["onet_soc_code", "domain_id", "onet_element_id", "scale_id"])
    keys = set(map(tuple, ev.values.tolist()))
    refs = [r for r in data_scientists.evidence_references if r.onet_element_id]
    assert refs, "expected rating evidence references"
    for r in refs:  # domain for this ref must be recovered from its deterministic ID
        domain = r.evidence_id.split("-")[3]
        assert (r.onet_soc_code, domain, r.onet_element_id, r.scale_id) in keys, r.evidence_id
