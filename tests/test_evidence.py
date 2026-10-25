"""Evidence, missing data, family ranking, and the rebuilt mapping."""
import pandas as pd
import pytest

from rank_domains import analyze, get_bundle, ranked_domains, valid_occupation

DS = "15-2051.00"


@pytest.fixture(scope="module")
def data_scientists():
    return analyze(onet_soc_code=DS, with_explanations=False)


def test_mapping_is_one_row_per_occupation_domain():
    grid = pd.read_csv("data/processed/occupation_domain_mapping.csv", dtype=str, keep_default_na=False)
    assert len(grid) == 50800
    assert grid.groupby(["onet_soc_code", "domain_id"]).ngroups == 50800
    assert grid.onet_soc_code.nunique() == 1016 and grid.domain_id.nunique() == 50


def test_removed_links_do_not_remain_in_active_evidence():
    ev = pd.read_csv("data/processed/occupation_domain_evidence.csv", dtype=str, keep_default_na=False)
    gone = ev[((ev.domain_id == "P013") & (ev.onet_element_id == "1.A.1.b.5"))
              | ((ev.domain_id == "P005") & (ev.onet_element_id == "1.A.1.g.2"))]
    assert gone.empty


def test_p002_score_matches_rebuilt_evidence():
    grid = pd.read_csv("data/processed/occupation_domain_mapping.csv", dtype=str, keep_default_na=False)
    row = grid[(grid.onet_soc_code == DS) & (grid.domain_id == "P002")].iloc[0]
    assert float(row.preliminary_priority_score) == 96.91
    assert "1.A.1.b.5" not in row.mapped_onet_element_ids


def test_estimated_relevance_is_reproducible_and_excludes_removed_links():
    from rank_domains import estimate_match
    p002 = estimate_match("15-2051.00", "P002")
    assert p002["estimated_match_pct"] is not None
    assert "1.A.1.b.5" not in p002["components"]
    assert "1.A.1.g.2" not in estimate_match("15-2051.00", "P005")["components"]
    for domain in ("P041", "P042", "P043", "P044"):
        assert estimate_match("15-2051.00", domain)["evidence_strength"] == "insufficient"
        assert estimate_match("15-2051.00", domain)["estimated_match_pct"] is None


def test_software_engineer_returns_scored_domains():
    resolved = __import__("rank_domains", fromlist=["resolve_title"]).resolve_title("Software Engineer")
    assert resolved.candidates
    resp = analyze(onet_soc_code=resolved.candidates[0].onet_soc_code, with_explanations=False)
    assert any(d.estimated_match_pct is not None for d in resp.psychometric_domains)


def test_work_values_stay_blank():
    grid = pd.read_csv("data/processed/occupation_domain_mapping.csv", dtype=str, keep_default_na=False)
    vals = grid[grid.domain_id.isin(["P041", "P042", "P043", "P044"])]
    assert (vals.mapping_status == "no_match").all()
    assert (vals.preliminary_priority_score == "").all()


def test_bundle_and_invalid_code():
    assert len(get_bundle(DS)["mapping"]) == 50
    assert not valid_occupation("00-0000.00")
    with pytest.raises(KeyError):
        get_bundle("00-0000.00")


def test_gaps_flags_and_no_validated_status(data_scientists):
    assert any("P041" in g or "no_match" in g for g in data_scientists.data_gaps)
    assert data_scientists.human_review_required
    assert all(d.human_review_required for d in data_scientists.psychometric_domains)
    assert not any(d.mapping_status == "validated" for d in data_scientists.psychometric_domains)


def test_evidence_references_resolve(data_scientists):
    ev = pd.read_csv("data/processed/occupation_domain_evidence.csv", dtype=str,
                     usecols=["onet_soc_code", "domain_id", "onet_element_id", "scale_id"])
    keys = set(map(tuple, ev.values.tolist()))
    refs = [r for r in data_scientists.evidence_references if r.onet_element_id]
    assert refs
    for r in refs:
        domain = r.evidence_id.split("-")[3]
        assert (r.onet_soc_code, domain, r.onet_element_id, r.scale_id) in keys, r.evidence_id


def test_display_uses_estimated_relevance_not_the_old_percentile(data_scientists):
    items = ranked_domains(data_scientists)
    assert items
    assert [it["estimated_match_pct"] for it in items] == sorted((it["estimated_match_pct"] for it in items), reverse=True)
    p002 = next(d for d in data_scientists.psychometric_domains if d.domain_id == "P002")
    assert p002.preliminary_score == 96.91
    assert p002.estimated_match_pct is not None and p002.estimated_match_pct != p002.preliminary_score
