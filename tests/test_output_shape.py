"""Output-shape tests: the chat answer must be one ranked list of at most 8
domains (60%+), each with a score, a what-it-means line and one short plain
why-it-fits sentence, with no jargon anywhere."""
from src.presentation import ranked_domains, clean_plain, has_banned_jargon
from src.analysis import analyze
from src import job_title_resolver as resolver

DS = ("Data Scientists", "15-2051.00")

def _items(title, soc):
    return ranked_domains(analyze(job_title=title, onet_soc_code=soc, top_per_family=5))

def test_ranked_single_list_rules():
    items = _items(*DS)
    assert 1 <= len(items) <= 8
    scores = [it["score"] for it in items]
    assert scores == sorted(scores, reverse=True) and all(s >= 60 for s in scores)
    assert [it["rank"] for it in items] == list(range(1, len(items) + 1))
    for it in items:
        assert it["name"] and it["what_it_means"].endswith(".")
        assert it["why_it_fits"] and it["why_it_fits"].endswith((".", "…"))
        assert len(it["why_it_fits"].split()) <= 26

def test_no_jargon_anywhere():
    items = _items(*DS)
    blob = " ".join(f"{it['name']} {it['what_it_means']} {it['why_it_fits']}" for it in items)
    assert not has_banned_jargon(blob)
    for bad in (" IM=", " LV=", " WI=", "crosswalk", "percentile"):
        assert bad not in blob

def test_clean_plain_strips_codes_to_one_sentence():
    raw = "Analyzes complex data sets (IM=92). Uses LV=81 scores. Second sentence here."
    out = clean_plain(raw)
    assert "IM" not in out and "=" not in out and "Second" not in out

def test_weaker_roles_still_clean_and_capped():
    items = _items("Human Resources Managers", "11-3121.00")
    assert len(items) <= 8 and all(it["score"] >= 60 for it in items)

def test_ambiguous_title_still_asks():
    assert resolver.resolve_title("Data Analyst").status != "resolved"
