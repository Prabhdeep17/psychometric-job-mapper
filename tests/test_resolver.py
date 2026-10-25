"""Job-title resolution, including titles that must not be auto-selected."""
import rank_domains as R


def test_exact_official_title():
    r = R.resolve_title("Software Developers")
    assert r.status in ("resolved", "ambiguous_clarification_required")
    assert any(c.onet_soc_code == "15-1252.00" for c in r.candidates)


def test_alternate_and_ambiguous_data_analyst():
    r = R.resolve_title("Data Analyst")
    assert r.status == "ambiguous_clarification_required" and len(r.candidates) >= 2


def test_case_insensitive():
    assert R.resolve_title("software developers").status == R.resolve_title("Software Developers").status


def test_unknown():
    assert R.resolve_title("Zqxwv Unknown Role 123").status == "not_found"


def test_explicit_code_and_invalid():
    assert R.lookup_by_code("15-2051.00") is not None
    assert R.lookup_by_code("99-9999.99") is None


def test_description_suggestion_remains_suggestion():
    r = R.resolve_title("Data Analyst")
    out = R.suggest_from_description(r.candidates, "build statistical models and analyse data with machine learning")
    assert any(c.is_suggested for c in out)


def test_spacing_and_plural_insensitive():
    for q in ["datascientist", "Data Scientist", "data scientists"]:
        r = R.resolve_title(q)
        assert r.status == "resolved" and r.candidates[0].onet_soc_code == "15-2051.00", q


def test_no_absurd_fuzzy_options():
    r = R.resolve_title("Data Scientist")
    assert not any("Chemist" in c.official_occupation_title or "Animal" in c.official_occupation_title for c in r.candidates)


def test_hr_abbreviation():
    r = R.resolve_title("HR Manager")
    assert any(c.onet_soc_code == "11-3121.00" for c in r.candidates)
