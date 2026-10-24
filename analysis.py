"""Core logic: data loading, title resolution, evidence, ranking, analysis.

Everything except the UI (app.py), the API (api.py) and the LLM client
(llm.py) lives here. Scores and rankings are computed locally from the
CSV data; the LLM only writes explanation text (see llm.py).
"""
from __future__ import annotations

import difflib
import hashlib
import re
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from pydantic import BaseModel, Field

from llm import explain, is_configured

DATA = Path("data/processed")

# ---------------------------------------------------------------- schemas

class OccupationCandidate(BaseModel):
    onet_soc_code: str
    official_occupation_title: str
    matched_title: str = ""
    title_match_type: str = ""  # official | alternate | reported | fuzzy | explicit_code
    title_source: str = ""
    mapping_status: str = ""
    match_score: Optional[float] = None  # title similarity only, NOT occupational relevance
    match_rationale: str = ""
    sample_tasks: list[str] = Field(default_factory=list)
    is_suggested: bool = False  # description-based suggestion; never auto-resolved


class ResolvedOccupation(BaseModel):
    onet_soc_code: str
    official_occupation_title: str
    resolution_method: str  # explicit_code | exact_title | fuzzy_confirmed
    matched_title: str = ""


class EvidenceReference(BaseModel):
    evidence_id: str  # deterministic, derived (raw data untouched)
    onet_soc_code: str
    onet_table: str = ""
    source_file: str = ""
    onet_element_id: str = ""
    onet_element_name: str = ""
    scale_id: str = ""
    scale_name: str = ""
    original_rating: str = ""  # original value as stored; "" = missing, never 0
    task_id: str = ""
    onet_release: str = "31.0"
    source_record: str = ""  # where this record lives in the processed data


class DomainRecommendation(BaseModel):
    domain_id: str
    domain_name: str
    construct_family: str
    domain_definition: str = ""
    relevance_explanation: str = ""  # filled by LLM or deterministic preview
    related_job_responsibilities: list[str] = Field(default_factory=list)
    mapping_status: str  # provisional_evidence | requires_review (recommendations only)
    evidence_type: str = ""
    crosswalk_link_types: list[str] = Field(default_factory=list)
    source_rating_summary: str = ""
    preliminary_score: Optional[float] = None  # percentile mean; NOT a psychometric score
    within_family_rank: Optional[int] = None  # rank ONLY within this family
    contextual_relevance: str = ""  # job-description-based, kept separate from score
    evidence_references: list[EvidenceReference] = Field(default_factory=list)
    confidence_explanation: str = ""
    human_review_required: bool = True
    limitations: list[str] = Field(default_factory=list)


class ConstructFamilyGroup(BaseModel):
    construct_family: str
    score_scope_note: str = ("Ranked only within this family using preliminary percentiles; "
                             "not comparable across families and not a validated psychometric score.")
    domains: list[DomainRecommendation] = Field(default_factory=list)


class AnalysisResponse(BaseModel):
    input_job_title: str = ""
    input_job_description: str = ""
    resolution_status: str  # resolved | ambiguous_clarification_required | not_found | invalid_code
    resolved_occupation: Optional[ResolvedOccupation] = None
    occupation_candidates: list[OccupationCandidate] = Field(default_factory=list)
    analysis_status: str  # completed | clarification_required | not_analyzed | completed_with_limitations
    construct_families: list[ConstructFamilyGroup] = Field(default_factory=list)
    psychometric_domains: list[DomainRecommendation] = Field(default_factory=list)
    evidence_references: list[EvidenceReference] = Field(default_factory=list)
    data_gaps: list[str] = Field(default_factory=list)
    human_review_required: bool = True
    limitations: list[str] = Field(default_factory=list)
    llm_provider: str = "preview"
    llm_model: str = "deterministic-preview"
    is_llm_generated: bool = False


class AnalyzeRequest(BaseModel):
    job_title: str = Field(default="", max_length=300)
    job_description: str = Field(default="", max_length=20000)
    onet_soc_code: Optional[str] = Field(default=None, max_length=20)
    top_per_family: int = Field(default=5, ge=1, le=14)


class RoleSearchResult(BaseModel):
    query: str
    resolution_status: str
    candidates: list[OccupationCandidate] = Field(default_factory=list)


# ---------------------------------------------------------------- data

@lru_cache(maxsize=1)
def _grid() -> pd.DataFrame:
    return pd.read_csv(DATA / "occupation_domain_mapping.csv", dtype=str, keep_default_na=False)


@lru_cache(maxsize=1)
def _evidence() -> pd.DataFrame:
    return pd.read_csv(DATA / "occupation_domain_evidence.csv", dtype=str, keep_default_na=False)


@lru_cache(maxsize=1)
def _domains() -> pd.DataFrame:
    return pd.read_csv(DATA / "psychometric_domains.csv", dtype=str, keep_default_na=False)


@lru_cache(maxsize=1)
def _lookup() -> pd.DataFrame:
    path = DATA / "job_title_lookup.csv"
    if not path.exists():
        raise FileNotFoundError(f"Title lookup missing: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


@lru_cache(maxsize=1)
def _sample_tasks() -> dict[str, list[str]]:
    """First listed tasks per occupation, from the mapping file itself."""
    out: dict[str, list[str]] = {}
    for soc, tasks in zip(_grid().onet_soc_code, _grid().supporting_tasks):
        if soc not in out and tasks:
            out[soc] = tasks.split("; ")[:3]
    return out


def data_status() -> dict:
    return {"grid_rows": len(_grid()), "evidence_rows": len(_evidence()),
            "domains": len(_domains()), "occupations": int(_grid().onet_soc_code.nunique()),
            "files_loaded": True}


# ---------------------------------------------------------------- evidence

def valid_occupation(code: str) -> bool:
    return bool(code) and code in set(_grid().onet_soc_code)


def _evidence_id(soc, domain, element, scale) -> str:
    return f"EV-{soc}-{domain}-{element}-{scale}".replace(" ", "")


def get_bundle(onet_soc_code: str) -> dict:
    """All 50 mapping rows + grouped evidence + definitions + gaps for one occupation."""
    if not valid_occupation(onet_soc_code):
        raise KeyError(f"Unknown O*NET-SOC code: {onet_soc_code}")
    grid = _grid()
    mapping = grid[grid.onet_soc_code == onet_soc_code].copy()
    evid = _evidence()
    evid = evid[evid.onet_soc_code == onet_soc_code].copy()
    doms = _domains().set_index("domain_id")

    refs_by_domain: dict[str, list[EvidenceReference]] = {}
    for r in evid.itertuples():
        refs_by_domain.setdefault(r.domain_id, []).append(EvidenceReference(
            evidence_id=_evidence_id(r.onet_soc_code, r.domain_id, r.onet_element_id, r.scale_id),
            onet_soc_code=r.onet_soc_code, onet_table=r.onet_table, source_file=r.source_file,
            onet_element_id=r.onet_element_id, onet_element_name=r.onet_element_name,
            scale_id=r.scale_id, scale_name=r.scale_name, original_rating=r.data_value,
            source_record=f"occupation_domain_evidence[{r.onet_soc_code}|{r.domain_id}|{r.onet_element_id}|{r.scale_id}]"))

    task_refs: dict[str, list[EvidenceReference]] = {}
    for r in mapping.itertuples():
        task_refs[r.domain_id] = [EvidenceReference(
            evidence_id=f"TASK-{r.onet_soc_code}-{tid}", onet_soc_code=r.onet_soc_code,
            onet_table="task_statements.csv", source_file="task_statements.csv", task_id=tid,
            source_record=f"task_statements[{r.onet_soc_code}|{tid}] (occupation context only)")
            for tid in [t for t in str(r.supporting_task_ids).split("; ") if t]]

    gaps = [f"{r.domain_id} {r.domain_name}: {r.mapping_status} — {r.evidence_summary}"
            for r in mapping.itertuples()
            if r.mapping_status in {"no_evidence", "no_match", "missing_source_data"}]
    return {"onet_soc_code": onet_soc_code,
            "occupation_title": mapping.occupation_title.iloc[0] if len(mapping) else "",
            "mapping": mapping, "evidence": evid, "domains": doms,
            "refs_by_domain": refs_by_domain, "task_refs_by_domain": task_refs,
            "data_gaps": gaps,
            "supporting_tasks": (mapping.supporting_tasks.iloc[0].split("; ")
                                 if len(mapping) and mapping.supporting_tasks.iloc[0] else [])}


# ---------------------------------------------------------------- title resolution

STOP = set("the and or of for a an to in with on at by from as is are job work".split())


def _normalize(s: str) -> str:
    return " ".join(str(s).lower().split())


def _canon(s: str) -> str:
    """Spacing/plural-insensitive key: 'datascientist', 'Data Scientist' and
    'Data Scientists' all map to 'datascientist'."""
    toks = re.findall(r"[a-z]+", str(s).lower())
    toks = [x for t in toks for x in (["human", "resource"] if t == "hr" else [t])]
    return "".join(t[:-1] if len(t) > 3 and t.endswith("s") else t for t in toks)


def _tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z]+", str(s).lower()) if t not in STOP and len(t) > 2}


def _candidate(row, match_type=None, score=None, rationale="") -> OccupationCandidate:
    return OccupationCandidate(
        onet_soc_code=row["onet_soc_code"], official_occupation_title=row["official_occupation_title"],
        matched_title=row["original_title"], title_match_type=match_type or row["title_match_type"],
        title_source=row["title_source"], mapping_status=row["mapping_status"],
        match_score=score, match_rationale=rationale,
        sample_tasks=_sample_tasks().get(row["onet_soc_code"], []))


class TitleResolution:
    def __init__(self, status, candidates):
        self.status, self.candidates = status, candidates


def resolve_title(title: str, limit: int = 25) -> TitleResolution:
    """Exact title -> its occupations; otherwise strict fuzzy. Ambiguous
    titles are never silently resolved to one occupation."""
    q = _normalize(title or "")
    if not q:
        return TitleResolution("not_found", [])
    df = _lookup()
    exact = df[df.normalized_title == q]
    if len(exact):
        by_soc = {}
        for _, r in exact.iterrows():
            if r.onet_soc_code not in by_soc or r.title_match_type == "official":
                by_soc[r.onet_soc_code] = r
        cands = [_candidate(r, score=1.0, rationale="Exact normalized title match")
                 for r in by_soc.values()]
        return TitleResolution("resolved" if len(cands) == 1 else "ambiguous_clarification_required",
                               cands[:limit])
    # Spacing/plural-insensitive match (official titles first)
    df = df.assign(_canon=df.normalized_title.map(_canon))
    qk = _canon(q)
    for pool, label in [(df[df.title_match_type == "official"], "official"), (df, "alternate")]:
        hits = pool[pool._canon == qk]
        if len(hits):
            by_soc = {}
            for _, r in hits.iterrows():
                if r.onet_soc_code not in by_soc or r.title_match_type == "official":
                    by_soc[r.onet_soc_code] = r
            cands = [_candidate(r, score=1.0, rationale=f"Title match ({label}, spacing/plural-insensitive)")
                     for r in by_soc.values()]
            return TitleResolution("resolved" if len(cands) == 1 else "ambiguous_clarification_required",
                                   cands[:limit])
    # Fuzzy: strict cutoff, only candidates near the best score, so generic
    # one-word overlaps (e.g. every '... Scientist') never appear.
    uniques = df.drop_duplicates("normalized_title")
    scored = sorted(((difflib.SequenceMatcher(None, qk, _canon(t)).ratio(), t)
                     for t in uniques.normalized_title), reverse=True)
    if not scored or scored[0][0] < 0.85:
        return TitleResolution("not_found", [])
    best = scored[0][0]
    close = [t for s, t in scored if s >= best - 0.03 and s >= 0.85][:5]
    cands, seen = [], set()
    for c in close:
        score = round(difflib.SequenceMatcher(None, qk, _canon(c)).ratio(), 3)
        for _, r in df[df.normalized_title == c].iterrows():
            if r.onet_soc_code in seen:
                continue
            seen.add(r.onet_soc_code)
            cands.append(_candidate(r, match_type="fuzzy", score=score,
                                    rationale=f"Fuzzy title similarity {score} to '{r.original_title}' — requires user confirmation"))
    return TitleResolution("ambiguous_clarification_required", cands[:limit])


def lookup_by_code(code: str) -> OccupationCandidate | None:
    df = _lookup()
    rows = df[df.onet_soc_code == (code or "").strip()]
    if rows.empty:
        return None
    official = rows[rows.title_match_type == "official"]
    row = official.iloc[0] if len(official) else rows.iloc[0]
    return _candidate(row, match_type="explicit_code", score=1.0,
                      rationale="Occupation explicitly selected by O*NET-SOC code")


def suggest_from_description(candidates, job_description: str):
    """Order existing candidates by overlap with their tasks + title.
    Marks the best as suggested; the status stays ambiguous."""
    jd = _tokens(job_description or "")
    if not jd or not candidates:
        return candidates
    scored = []
    for c in candidates:
        overlap = jd & _tokens(c.official_occupation_title + " " + " ".join(c.sample_tasks))
        scored.append((len(overlap), c, sorted(overlap)))
    scored.sort(key=lambda x: -x[0])
    out = []
    for i, (n, c, words) in enumerate(scored):
        cc = c.model_copy()
        cc.is_suggested = (i == 0 and n > 0)
        cc.match_rationale = (c.match_rationale
                              + f" | Description overlap on {n} task/title term(s): {', '.join(words[:8]) or 'none'}"
                              + (" — SUGGESTED only, please confirm" if cc.is_suggested else ""))
        out.append(cc)
    return out


def search_roles(query: str, limit: int = 25) -> TitleResolution:
    """Substring search for the API/UI search box."""
    q = _normalize(query or "")
    if not q:
        return TitleResolution("not_found", [])
    df = _lookup()
    sub = df[df.normalized_title.str.contains(re.escape(q), na=False)]
    if sub.empty:
        return resolve_title(query, limit)
    by_soc = {}
    for _, r in sub.iterrows():
        if r.onet_soc_code not in by_soc or r.title_match_type == "official":
            by_soc[r.onet_soc_code] = r
    cands = [_candidate(r, score=1.0 if r.normalized_title == q else None, rationale="Title search match")
             for r in list(by_soc.values())[:limit]]
    return TitleResolution("resolved" if len(cands) == 1 else "ambiguous_clarification_required", cands)


# ---------------------------------------------------------------- ranking

RECOMMENDABLE = {"provisional_evidence", "requires_review"}


def rank(bundle: dict, job_description: str = "", top_per_family: int = 5):
    """Group an occupation's domains by family and rank within each family.
    Scores come from the mapping file; nothing is rescored here."""
    mapping, doms = bundle["mapping"], bundle["domains"]
    jd = _tokens(job_description or "")
    recs: list[DomainRecommendation] = []
    for r in mapping.itertuples():
        if r.mapping_status not in RECOMMENDABLE:
            continue
        refs = bundle["refs_by_domain"].get(r.domain_id, []) + bundle["task_refs_by_domain"].get(r.domain_id, [])
        # Job-description overlap is context only; it never changes the score.
        ctx_text = _tokens(r.domain_name + " " + r.mapped_onet_element_names + " " + bundle["supporting_tasks"][0]
                           if bundle["supporting_tasks"] else r.domain_name)
        overlap = jd & ctx_text if jd else set()
        ctx = (f"Job-description overlap on: {', '.join(sorted(overlap)[:6])} (contextual only; does not change the score)"
               if overlap else ("No job description supplied; no contextual adjustment applied" if not jd
                                else "No direct term overlap with the supplied description; ranking rests on occupational ratings only"))
        d = doms.loc[r.domain_id] if r.domain_id in doms.index else None
        lims = [r.evidence_summary] if r.evidence_summary else []
        if d is not None and str(d["limitations"]):
            lims.append(str(d["limitations"]))
        if r.mapping_status == "requires_review":
            lims.append("Only proxy evidence supports this domain — treat as a hypothesis for expert review, not an established requirement.")
        evid = bundle["evidence"]
        recs.append(DomainRecommendation(
            domain_id=r.domain_id, domain_name=r.domain_name, construct_family=r.construct_family,
            domain_definition=(str(d["operational_definition"]) if d is not None else ""),
            related_job_responsibilities=bundle["supporting_tasks"][:3],
            mapping_status=r.mapping_status, evidence_type=r.evidence_type,
            crosswalk_link_types=sorted(set(evid.loc[evid.domain_id == r.domain_id, "crosswalk_link_type"])) if len(evid) else [],
            source_rating_summary=r.source_signal_summary,
            preliminary_score=(float(r.preliminary_priority_score) if str(r.preliminary_priority_score) not in ("", "nan") else None),
            within_family_rank=(int(float(r.within_family_rank)) if str(r.within_family_rank) not in ("", "nan") else None),
            contextual_relevance=ctx, evidence_references=refs,
            confidence_explanation=f"Mapping confidence: {r.mapping_confidence} (crosswalk pending review; confidence is about the construct link, not job performance).",
            human_review_required=True, limitations=lims))

    families: list[ConstructFamilyGroup] = []
    if recs:
        frame = pd.DataFrame([x.model_dump() for x in recs])
        for fam, group in frame.groupby("construct_family"):
            ordered = sorted(group.to_dict("records"),
                             key=lambda x: (x["within_family_rank"] is None, x["within_family_rank"] or 999, x["domain_id"]))[:top_per_family]
            families.append(ConstructFamilyGroup(construct_family=fam,
                                                 domains=[DomainRecommendation(**o) for o in ordered]))
    families.sort(key=lambda f: f.construct_family)
    flat = [d for fam in families for d in fam.domains]
    return families, flat


# ---------------------------------------------------------------- display (Streamlit cards)

MIN_SCORE, TOP_N = 60, 8
_BANNED_WORDS = ["crosswalk", "percentile", "o*net", "proxy", "partial link",
                 "scale id", "evidence reference", "distinctiveness"]


def clean_plain(text: str, max_words: int = 25) -> str:
    """One short, plain sentence: strip rating codes and jargon, keep the first sentence."""
    t = text or ""
    t = re.sub(r"\([^)]*\b(IM|LV|WI|DR|OI|IH)\b[^)]*\)", "", t)
    t = re.sub(r"\b\d+\.[A-Z]\.[\d.a-z.]+\b", "", t)
    t = re.sub(r"\b(IM|LV|WI|DR|OI|IH)\s*[=]?\s*\d+(\.\d+)?", "", t)
    t = re.sub(r"[^.]*\b(crosswalk|percentile|proxy|link is (direct|partial))\b[^.]*\.", "", t, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip()
    first = (t.split(". ")[0].rstrip(".") + ".") if t else ""
    words = first.split()
    if not first:
        return ""
    return " ".join(words[:max_words]).rstrip(",;:") + ("…" if len(words) > max_words
                                                        else ("" if first.endswith(".") else "."))


def _first_sentence(text: str) -> str:
    t = (text or "").strip()
    return (t.split(". ")[0].rstrip(".") + ".") if t else ""


def ranked_domains(resp) -> list[dict]:
    """Top domains across all families, strongest first, 60%+ only, max 8.
    Each item: rank, name, score (int %), what_it_means, why_it_fits."""
    strong = [d for d in resp.psychometric_domains if (d.preliminary_score or 0) >= MIN_SCORE]
    strong.sort(key=lambda d: -(d.preliminary_score or 0))
    return [{"rank": i, "domain_id": d.domain_id, "name": d.domain_name,
             "score": round(d.preliminary_score),
             "what_it_means": _first_sentence(d.domain_definition),
             "why_it_fits": clean_plain(d.relevance_explanation)}
            for i, d in enumerate(strong[:TOP_N], 1)]


def has_banned_jargon(text: str) -> bool:
    low = (text or "").lower()
    return any(b in low for b in _BANNED_WORDS) or bool(re.search(r"\b(IM|LV|WI|DR|OI|IH)\s*=", text or ""))


# ---------------------------------------------------------------- analysis

_CACHE: dict = {}


def clear_cache():
    _CACHE.clear()


BASE_LIMITATIONS = [
    "Scores are preliminary occupation-relative percentiles within an element/scale — not validated psychometric scores and not comparable across construct families.",
    "All domain crosswalk links remain pending_review; no mapping is scientifically validated.",
    "Occupational ratings describe job requirements, not individual candidates; this output must not be used for hiring or screening decisions.",
]


def analyze(job_title: str = "", job_description: str = "", onet_soc_code: Optional[str] = None,
            top_per_family: int = 5, llm_callable: Optional[Callable] = None,
            with_explanations: bool = True) -> AnalysisResponse:
    base = dict(input_job_title=job_title or "", input_job_description=job_description or "",
                limitations=list(BASE_LIMITATIONS), human_review_required=True)
    # An explicit code wins; an invalid code is a clean error, never a guess.
    if onet_soc_code:
        cand = lookup_by_code(onet_soc_code)
        if cand is None:
            return AnalysisResponse(
                **base, resolution_status="invalid_code", analysis_status="not_analyzed",
                data_gaps=[f"O*NET-SOC code '{onet_soc_code}' not found in the occupation data. No occupation was substituted."])
        # Repeat requests with a real provider are cached; preview fallbacks are not.
        key = (cand.onet_soc_code, top_per_family,
               hashlib.sha256((job_description or "").encode()).hexdigest())
        if llm_callable is None and key in _CACHE:
            return _CACHE[key].model_copy(deep=True)
        resp = _run(cand.onet_soc_code, cand.official_occupation_title, "explicit_code",
                    cand.matched_title, job_description, top_per_family, llm_callable, base, with_explanations)
        if llm_callable is None and (resp.is_llm_generated or not is_configured()):
            _CACHE[key] = resp.model_copy(deep=True)
        return resp

    res = resolve_title(job_title or "")
    if res.status == "resolved":
        c = res.candidates[0]
        return _run(c.onet_soc_code, c.official_occupation_title, "exact_title", c.matched_title,
                    job_description, top_per_family, llm_callable, base, with_explanations)
    if res.status == "ambiguous_clarification_required":
        cands = suggest_from_description(res.candidates, job_description) if job_description else res.candidates
        return AnalysisResponse(
            **base, resolution_status="ambiguous_clarification_required",
            analysis_status="clarification_required", occupation_candidates=cands,
            data_gaps=["Title is ambiguous: select an occupation explicitly (onet_soc_code) to run the analysis. No profile was merged or auto-selected."])
    # Not found: description-based suggestions are offered, still not resolved.
    cands = []
    if job_description:
        seed = search_roles(job_title, limit=10) if job_title else res
        cands = suggest_from_description(seed.candidates, job_description) if seed.candidates else []
    return AnalysisResponse(
        **base, resolution_status="not_found", analysis_status="clarification_required",
        occupation_candidates=cands,
        data_gaps=["No reliable O*NET title match. Provide a more specific title, a job description, or select an occupation from search. No O*NET code was invented."])


def _run(code, title, method, matched, jd, top_per_family, llm_callable, base, with_explanations=True) -> AnalysisResponse:
    bundle = get_bundle(code)
    families, flat = rank(bundle, jd, top_per_family)
    if with_explanations:
        flat, info = explain(flat, {"onet_soc_code": code, "official_title": title}, jd, llm_callable=llm_callable)
    else:  # definition-only mode: no LLM call, instant, no token cost
        info = {"provider": "none", "model": "domain-definitions", "is_llm": False}
    by_id = {d.domain_id: d for d in flat}
    for fam in families:
        fam.domains = [by_id[d.domain_id] for d in fam.domains]
    all_refs, seen = [], set()
    for d in flat:
        for r in d.evidence_references:
            if r.evidence_id not in seen:
                seen.add(r.evidence_id)
                all_refs.append(r)
    gaps = list(bundle["data_gaps"])
    if not bundle["supporting_tasks"]:
        gaps.append("No Core task statements available for this occupation in O*NET 31.0; responsibility context is limited.")
    base = {**base, "limitations": list(base["limitations"]) + ([info["fallback_reason"]] if "fallback_reason" in info else [])}
    return AnalysisResponse(
        **base, resolution_status="resolved",
        resolved_occupation=ResolvedOccupation(onet_soc_code=code, official_occupation_title=title,
                                               resolution_method=method, matched_title=matched),
        analysis_status="completed" if not gaps else "completed_with_limitations",
        construct_families=families, psychometric_domains=flat, evidence_references=all_refs,
        data_gaps=gaps, llm_provider=info["provider"], llm_model=info["model"],
        is_llm_generated=info["is_llm"])
