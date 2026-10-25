"""Checks the data, the within-family display, and that Groq cannot change scores."""
import json

from explain_domains import explain
from rank_domains import analyze, ranked_domains, resolve_title

DS = "15-2051.00"


def test_title_and_score_come_from_the_files():
    assert resolve_title("Data Scientist").candidates[0].onet_soc_code == DS
    assert resolve_title("Data Analyst").status != "resolved"
    resp = analyze(onet_soc_code=DS, with_explanations=False)
    row = next(d for d in resp.psychometric_domains if d.domain_id == "P002")
    assert row.domain_name == "Quantitative Reasoning"
    assert abs(row.preliminary_score - 96.91) < 0.02
    assert row.preliminary_score is not None
    assert resp.data_gaps
    assert all(d.preliminary_score is not None for d in resp.psychometric_domains)
    import pandas as pd
    grid = pd.read_csv("data/processed/occupation_domain_mapping.csv", dtype=str, keep_default_na=False)
    missing = grid[(grid.onet_soc_code == DS) & (grid.mapping_status == "no_match")]
    assert len(missing) >= 1 and (missing.preliminary_priority_score == "").all()


def test_clean_plain_keeps_the_job_connection_sentence():
    from rank_domains import clean_plain
    raw = ("This occupation is rated on Mathematical Reasoning. "
           "That requirement matches quantitative reasoning used when analysing data with statistical models.")
    out = clean_plain(raw)
    assert "Mathematical Reasoning" in out
    assert "statistical models" in out
    assert out.count(".") >= 2


def test_display_ranks_by_estimated_relevance():
    items = ranked_domains(analyze(onet_soc_code=DS, with_explanations=False))
    assert 1 <= len(items) <= 10
    scores = [it["estimated_match_pct"] for it in items]
    assert scores == sorted(scores, reverse=True)
    assert all(it["preliminary_score"] != it["estimated_match_pct"] or it["estimated_match_pct"] is not None for it in items)
    assert "job_level" not in items[0]


def test_groq_text_cannot_change_score_or_tasks():
    resp = analyze(onet_soc_code=DS, top_per_family=1, with_explanations=False)
    before = {d.domain_id: (d.preliminary_score, list(d.related_job_responsibilities), d.within_family_rank)
              for d in resp.psychometric_domains}

    def fake(_payload):
        return json.dumps({"explanations": [{
            "domain_id": d.domain_id,
            "relevance_explanation": "This job uses this strength in its daily work.",
            "related_job_responsibilities": ["Invented task that is not in the file."],
            "confidence_explanation": "Invented confidence.",
            "limitations": ["Invented limitation."],
            "evidence_reference_ids": [d.evidence_references[0].evidence_id],
        } for d in resp.psychometric_domains if d.evidence_references]})

    updated, info = explain(resp.psychometric_domains, {"onet_soc_code": DS, "official_title": "Data Scientists"},
                            llm_callable=fake)
    assert info["is_llm"] is True
    for d in updated:
        score, tasks, rank = before[d.domain_id]
        assert d.preliminary_score == score
        assert d.related_job_responsibilities == tasks
        assert d.within_family_rank == rank
        assert "Invented" not in " ".join(d.limitations)
        if d.evidence_references:
            assert d.relevance_explanation == "This job uses this strength in its daily work."
