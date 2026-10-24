"""Occupational evidence retrieval from Phase 3 outputs (cached, read-only).
Missing values stay missing (''), never 0. Evidence IDs are deterministic
derived identifiers: EV-<soc>-<domain>-<element>-<scale>."""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import pandas as pd
from src.schemas import EvidenceReference

PROC, RAW = Path("data/processed"), Path("data/raw/onet_31_0")

@lru_cache(maxsize=1)
def _grid() -> pd.DataFrame:
    return pd.read_csv(PROC/"occupation_domain_mapping.csv", dtype=str, keep_default_na=False)

@lru_cache(maxsize=1)
def _evidence() -> pd.DataFrame:
    return pd.read_csv(PROC/"occupation_domain_evidence.csv", dtype=str, keep_default_na=False)

@lru_cache(maxsize=1)
def _domains() -> pd.DataFrame:
    return pd.read_csv(PROC/"psychometric_domains.csv", dtype=str, keep_default_na=False)

@lru_cache(maxsize=1)
def _element_desc() -> dict[str, str]:
    cmr = pd.read_csv(RAW/"content_model_reference.csv", dtype=str, keep_default_na=False)
    return dict(zip(cmr["Element ID"], cmr["Description"]))

@lru_cache(maxsize=1)
def _scales() -> dict[str, dict]:
    s = pd.read_csv(RAW/"scales_reference.csv", dtype=str, keep_default_na=False)
    return {r["Scale ID"]: {"name": r["Scale Name"], "min": r["Minimum"], "max": r["Maximum"]} for _, r in s.iterrows()}

def data_status() -> dict:
    return {"grid_rows": len(_grid()), "evidence_rows": len(_evidence()), "domains": len(_domains()),
            "occupations": int(_grid().onet_soc_code.nunique()), "files_loaded": True}

def valid_occupation(code: str) -> bool:
    return bool(code) and code in set(_grid().onet_soc_code)

def evidence_id(soc, domain, element, scale) -> str:
    return f"EV-{soc}-{domain}-{element}-{scale}".replace(" ", "")

def get_bundle(onet_soc_code: str) -> dict:
    """All 50 mapping rows + grouped evidence + definitions + gaps for one occupation."""
    if not valid_occupation(onet_soc_code): raise KeyError(f"Unknown O*NET-SOC code: {onet_soc_code}")
    grid = _grid(); mapping = grid[grid.onet_soc_code == onet_soc_code].copy()
    ev = _evidence(); evid = ev[ev.onet_soc_code == onet_soc_code].copy()
    doms = _domains().set_index("domain_id")
    refs_by_domain: dict[str, list[EvidenceReference]] = {}
    for r in evid.itertuples():
        refs_by_domain.setdefault(r.domain_id, []).append(EvidenceReference(
            evidence_id=evidence_id(r.onet_soc_code, r.domain_id, r.onet_element_id, r.scale_id),
            onet_soc_code=r.onet_soc_code, onet_table=r.onet_table, source_file=r.source_file,
            onet_element_id=r.onet_element_id, onet_element_name=r.onet_element_name, scale_id=r.scale_id,
            scale_name=r.scale_name, original_rating=r.data_value,
            phase3_record=f"occupation_domain_evidence[{r.onet_soc_code}|{r.domain_id}|{r.onet_element_id}|{r.scale_id}]"))
    task_refs: dict[str, list[EvidenceReference]] = {}
    for r in mapping.itertuples():
        refs = []
        for tid in [t for t in str(r.supporting_task_ids).split("; ") if t]:
            refs.append(EvidenceReference(evidence_id=f"TASK-{r.onet_soc_code}-{tid}", onet_soc_code=r.onet_soc_code,
                onet_table="task_statements.csv", source_file="task_statements.csv", task_id=tid,
                phase3_record=f"task_statements[{r.onet_soc_code}|{tid}] (occupation context only)"))
        task_refs[r.domain_id] = refs
    gaps = [f"{r.domain_id} {r.domain_name}: {r.mapping_status} — {r.evidence_summary}" for r in mapping.itertuples()
            if r.mapping_status in {"no_evidence", "no_match", "missing_source_data"}]
    return {"onet_soc_code": onet_soc_code,
            "occupation_title": mapping.occupation_title.iloc[0] if len(mapping) else "",
            "mapping": mapping, "evidence": evid, "domains": doms,
            "refs_by_domain": refs_by_domain, "task_refs_by_domain": task_refs,
            "element_descriptions": _element_desc(), "scales": _scales(), "data_gaps": gaps,
            "supporting_tasks": (mapping.supporting_tasks.iloc[0].split("; ") if len(mapping) and mapping.supporting_tasks.iloc[0] else [])}
