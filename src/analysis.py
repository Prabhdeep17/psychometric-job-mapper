"""Orchestration shared by API and Streamlit UI (single code path)."""
from __future__ import annotations
from typing import Callable, Optional
from src import job_title_resolver as resolver
from src.domain_ranker import rank
from src.evidence_retriever import get_bundle, valid_occupation
from src.llm_explainer import explain
from src.schemas import AnalysisResponse, ResolvedOccupation

_CACHE: dict = {}
def clear_cache(): _CACHE.clear()

BASE_LIMITATIONS = [
    "Phase 3 preliminary scores are occupation-relative percentiles within an element/scale — not validated psychometric scores and not comparable across construct families.",
    "All Phase 2 crosswalk links remain pending_review; no mapping is scientifically validated.",
    "Occupational ratings describe job requirements, not individual candidates; this output must not be used for hiring or screening decisions.",
]

def analyze(job_title: str = "", job_description: str = "", onet_soc_code: Optional[str] = None,
            top_per_family: int = 5, llm_callable: Optional[Callable] = None,
            with_explanations: bool = True) -> AnalysisResponse:
    base = dict(input_job_title=job_title or "", input_job_description=job_description or "",
                limitations=list(BASE_LIMITATIONS), human_review_required=True)
    # explicit code wins; invalid code is a clean error response, never a guess
    if onet_soc_code:
        cand = resolver.lookup_by_code(onet_soc_code)
        if cand is None:
            return AnalysisResponse(**base, resolution_status="invalid_code", analysis_status="not_analyzed",
                data_gaps=[f"O*NET-SOC code '{onet_soc_code}' not found in the audited occupation data. No occupation was substituted."])
        # In-memory explanation cache (real-provider path only): identical repeat
        # requests return instantly and cost no provider tokens (Phase 6.2 fix).
        import hashlib
        key = (cand.onet_soc_code, top_per_family, hashlib.sha256((job_description or "").encode()).hexdigest())
        if llm_callable is None and key in _CACHE: return _CACHE[key].model_copy(deep=True)
        resp = _run(cand.onet_soc_code, cand.official_occupation_title, "explicit_code", cand.matched_title, job_description, top_per_family, llm_callable, base, with_explanations)
        # Never cache a preview FALLBACK when a live provider is configured — a later retry may succeed live.
        from src.llm_explainer import is_configured
        if llm_callable is None and (resp.is_llm_generated or not is_configured()):
            _CACHE[key] = resp.model_copy(deep=True)
        return resp
    res = resolver.resolve_title(job_title or "")
    if res.status == "resolved":
        c = res.candidates[0]
        return _run(c.onet_soc_code, c.official_occupation_title, "exact_title", c.matched_title, job_description, top_per_family, llm_callable, base, with_explanations)
    if res.status == "ambiguous_clarification_required":
        cands = resolver.suggest_from_description(res.candidates, job_description) if job_description else res.candidates
        return AnalysisResponse(**base, resolution_status="ambiguous_clarification_required", analysis_status="clarification_required",
            occupation_candidates=cands, data_gaps=["Title is ambiguous: select an occupation explicitly (onet_soc_code) to run the analysis. No profile was merged or auto-selected."])
    # not found: offer description-based suggestions if possible, still not resolved
    cands = []
    if job_description:
        from src.job_title_resolver import search_roles
        seed = search_roles(job_title, limit=10) if job_title else res
        cands = resolver.suggest_from_description(seed.candidates, job_description) if seed.candidates else []
    return AnalysisResponse(**base, resolution_status="not_found", analysis_status="clarification_required",
        occupation_candidates=cands, data_gaps=["No reliable O*NET title match. Provide a more specific title, a job description, or select an occupation from search. No O*NET code was invented."])

def _run(code, title, method, matched, jd, top_per_family, llm_callable, base, with_explanations=True) -> AnalysisResponse:
    bundle = get_bundle(code)
    families, flat = rank(bundle, jd, top_per_family)
    if with_explanations:
        flat, info = explain(flat, {"onet_soc_code": code, "official_title": title}, jd, llm_callable=llm_callable)
    else:  # definition-only mode (simple UI): no LLM call, instant, no token cost
        info = {"provider": "none", "model": "domain-definitions", "is_llm": False}
    by_id = {d.domain_id: d for d in flat}
    for fam in families: fam.domains = [by_id[d.domain_id] for d in fam.domains]
    all_refs, seen = [], set()
    for d in flat:
        for r in d.evidence_references:
            if r.evidence_id not in seen: seen.add(r.evidence_id); all_refs.append(r)
    gaps = list(bundle["data_gaps"])
    if not bundle["supporting_tasks"]: gaps.append("No Core task statements available for this occupation in O*NET 31.0; responsibility context is limited.")
    base = {**base, "limitations": list(base["limitations"]) + ([info["fallback_reason"]] if "fallback_reason" in info else [])}
    return AnalysisResponse(**base, resolution_status="resolved",
        resolved_occupation=ResolvedOccupation(onet_soc_code=code, official_occupation_title=title, resolution_method=method, matched_title=matched),
        analysis_status="completed" if not gaps else "completed_with_limitations",
        construct_families=families, psychometric_domains=flat, evidence_references=all_refs,
        data_gaps=gaps, llm_provider=info["provider"], llm_model=info["model"], is_llm_generated=info["is_llm"])
