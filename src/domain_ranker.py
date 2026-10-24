"""Within-family ranking only. Uses Phase 3 within_family_rank /
preliminary_priority_score strictly as documented (occupation-relative
percentiles). NEVER creates a cross-family/global score, never scores
unscored domains, never counts mapped elements as importance."""
from __future__ import annotations
import re
import pandas as pd
from src.evidence_retriever import get_bundle
from src.schemas import ConstructFamilyGroup, DomainRecommendation

RECOMMENDABLE = {"provisional_evidence", "requires_review"}
STOP = set("the and or of for a an to in with on at by from as is are job work".split())
def _tokens(s): return {t for t in re.findall(r"[a-z]+", str(s).lower()) if t not in STOP and len(t) > 2}

def rank(bundle: dict, job_description: str = "", top_per_family: int = 5) -> tuple[list[ConstructFamilyGroup], list[DomainRecommendation]]:
    mapping: pd.DataFrame = bundle["mapping"]; doms = bundle["domains"]
    jd = _tokens(job_description or "")
    recs: list[DomainRecommendation] = []
    for r in mapping.itertuples():
        if r.mapping_status not in RECOMMENDABLE: continue
        refs = bundle["refs_by_domain"].get(r.domain_id, []) + bundle["task_refs_by_domain"].get(r.domain_id, [])
        # JD contextual relevance — qualitative, SEPARATE from preliminary score
        ctx_text = _tokens(r.domain_name + " " + r.mapped_onet_element_names + " " + bundle["supporting_tasks"][0] if bundle["supporting_tasks"] else r.domain_name)
        overlap = jd & ctx_text if jd else set()
        ctx = (f"Job-description overlap on: {', '.join(sorted(overlap)[:6])} (contextual only; does not change the Phase 3 score)"
               if overlap else ("No job description supplied; no contextual adjustment applied" if not jd else "No direct term overlap with the supplied description; ranking rests on occupational ratings only"))
        d = doms.loc[r.domain_id] if r.domain_id in doms.index else None
        lims = [r.evidence_summary] if r.evidence_summary else []
        if d is not None and str(d["limitations"]): lims.append(str(d["limitations"]))
        if r.mapping_status == "requires_review": lims.append("Only proxy evidence supports this domain — treat as a hypothesis for expert review, not an established requirement.")
        recs.append(DomainRecommendation(
            domain_id=r.domain_id, domain_name=r.domain_name, construct_family=r.construct_family,
            domain_definition=(str(d["operational_definition"]) if d is not None else ""),
            related_job_responsibilities=bundle["supporting_tasks"][:3],
            mapping_status=r.mapping_status, evidence_type=r.evidence_type,
            crosswalk_link_types=sorted(set(bundle["evidence"].loc[bundle["evidence"].domain_id == r.domain_id, "crosswalk_link_type"])) if len(bundle["evidence"]) else [],
            source_rating_summary=r.source_signal_summary,
            preliminary_score=(float(r.preliminary_priority_score) if str(r.preliminary_priority_score) not in ("", "nan") else None),
            within_family_rank=(int(float(r.within_family_rank)) if str(r.within_family_rank) not in ("", "nan") else None),
            contextual_relevance=ctx, evidence_references=refs,
            confidence_explanation=f"Mapping confidence: {r.mapping_confidence} (Phase 2 crosswalk pending review; confidence is about the construct link, not job performance).",
            human_review_required=True, limitations=lims))
    families: list[ConstructFamilyGroup] = []
    for fam, group in pd.DataFrame([x.model_dump() for x in recs]).groupby("construct_family") if recs else []:
        ordered = sorted(group.to_dict("records"), key=lambda x: (x["within_family_rank"] is None, x["within_family_rank"] or 999, x["domain_id"]))[:top_per_family]
        families.append(ConstructFamilyGroup(construct_family=fam, domains=[DomainRecommendation(**o) for o in ordered]))
    families.sort(key=lambda f: f.construct_family)
    flat = [d for f in families for d in f.domains]
    return families, flat
