"""Job-title resolution against data/processed/job_title_lookup.csv.
Ambiguous titles are NEVER silently resolved. Fuzzy matching uses stdlib
difflib (transparent, no extra dependency). Scores here are title-string
similarity only."""
from __future__ import annotations
import difflib, re
from functools import lru_cache
from pathlib import Path
import pandas as pd
from src.schemas import OccupationCandidate

LOOKUP = Path("data/processed/job_title_lookup.csv")
TASKS = Path("data/raw/onet_31_0/task_statements.csv")
STOP = set("the and or of for a an to in with on at by from as is are job work".split())

def normalize(s: str) -> str:
    return " ".join(str(s).lower().split())

def _canon(s: str) -> str:
    """Spacing/plural-insensitive key: 'datascientist', 'Data Scientist' and
    'Data Scientists' all map to 'datascientist'."""
    toks = re.findall(r"[a-z]+", str(s).lower())
    toks = [x for t in toks for x in (["human", "resource"] if t == "hr" else [t])]  # HR = Human Resources
    toks = [t[:-1] if len(t) > 3 and t.endswith("s") else t for t in toks]
    return "".join(toks)

def _tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z]+", str(s).lower()) if t not in STOP and len(t) > 2}

@lru_cache(maxsize=1)
def _lookup() -> pd.DataFrame:
    if not LOOKUP.exists(): raise FileNotFoundError(f"Title lookup missing: {LOOKUP}")
    return pd.read_csv(LOOKUP, dtype=str, keep_default_na=False)

@lru_cache(maxsize=1)
def _sample_tasks() -> dict[str, list[str]]:
    if not TASKS.exists(): return {}
    ts = pd.read_csv(TASKS, dtype=str, usecols=["O*NET-SOC Code", "Task", "Task Type"])
    core = ts[ts["Task Type"] == "Core"].groupby("O*NET-SOC Code").head(3)
    return {soc: list(g["Task"]) for soc, g in core.groupby("O*NET-SOC Code")}

def _candidate(row, match_type=None, score=None, rationale="") -> OccupationCandidate:
    return OccupationCandidate(onet_soc_code=row["onet_soc_code"], official_occupation_title=row["official_occupation_title"],
        matched_title=row["original_title"], title_match_type=match_type or row["title_match_type"], title_source=row["title_source"],
        mapping_status=row["mapping_status"], match_score=score, match_rationale=rationale,
        sample_tasks=_sample_tasks().get(row["onet_soc_code"], []))

class TitleResolution:
    def __init__(self, status, candidates): self.status, self.candidates = status, candidates

def resolve_title(title: str, limit: int = 25) -> TitleResolution:
    """exact (normalized) -> all occupations for that title; else transparent fuzzy."""
    q = normalize(title or "")
    if not q: return TitleResolution("not_found", [])
    df = _lookup()
    exact = df[df.normalized_title == q]
    if len(exact):
        by_soc = {}
        for _, r in exact.iterrows():  # keep best (official preferred) row per occupation
            if r.onet_soc_code not in by_soc or r.title_match_type == "official": by_soc[r.onet_soc_code] = r
        cands = [_candidate(r, score=1.0, rationale="Exact normalized title match") for r in by_soc.values()]
        return TitleResolution("resolved" if len(cands) == 1 else "ambiguous_clarification_required", cands[:limit])
    # Tier B/C: spacing/plural-insensitive canonical match (official titles first)
    df = df.assign(_canon=df.normalized_title.map(_canon))
    qk = _canon(q)
    for pool, label in [(df[df.title_match_type == "official"], "official"), (df, "alternate")]:
        hits = pool[pool._canon == qk]
        if len(hits):
            by_soc = {}
            for _, r in hits.iterrows():
                if r.onet_soc_code not in by_soc or r.title_match_type == "official": by_soc[r.onet_soc_code] = r
            cands = [_candidate(r, score=1.0, rationale=f"Title match ({label}, spacing/plural-insensitive)") for r in by_soc.values()]
            return TitleResolution("resolved" if len(cands) == 1 else "ambiguous_clarification_required", cands[:limit])
    # Tier D: fuzzy — strict cutoff and only candidates near the BEST score,
    # so generic one-word overlaps (e.g. every '... Scientist') never appear.
    uniques = df.drop_duplicates("normalized_title")
    scored = sorted(((difflib.SequenceMatcher(None, qk, _canon(t)).ratio(), t) for t in uniques.normalized_title), reverse=True)
    if not scored or scored[0][0] < 0.85: return TitleResolution("not_found", [])
    best = scored[0][0]
    close = [t for s, t in scored if s >= best - 0.03 and s >= 0.85][:5]
    if not close: return TitleResolution("not_found", [])
    cands, seen = [], set()
    for c in close:
        score = round(difflib.SequenceMatcher(None, qk, _canon(c)).ratio(), 3)
        for _, r in df[df.normalized_title == c].iterrows():
            if r.onet_soc_code in seen: continue
            seen.add(r.onet_soc_code)
            cands.append(_candidate(r, match_type="fuzzy", score=score, rationale=f"Fuzzy title similarity {score} to '{r.original_title}' — requires user confirmation"))
    return TitleResolution("ambiguous_clarification_required", cands[:limit])

def lookup_by_code(code: str) -> OccupationCandidate | None:
    df = _lookup(); rows = df[df.onet_soc_code == (code or "").strip()]
    if rows.empty: return None
    r = rows[rows.title_match_type == "official"]
    row = (r.iloc[0] if len(r) else rows.iloc[0])
    return _candidate(row, match_type="explicit_code", score=1.0, rationale="Occupation explicitly selected by O*NET-SOC code")

def valid_code(code: str) -> bool:
    return lookup_by_code(code) is not None

def suggest_from_description(candidates: list[OccupationCandidate], job_description: str) -> list[OccupationCandidate]:
    """Rank EXISTING candidates by token overlap with their Core tasks + title.
    Marks the best as is_suggested with a rationale; status stays ambiguous."""
    jd = _tokens(job_description or "")
    if not jd or not candidates: return candidates
    scored = []
    for c in candidates:
        text = _tokens(c.official_occupation_title + " " + " ".join(c.sample_tasks))
        overlap = jd & text
        scored.append((len(overlap), c, sorted(overlap)))
    scored.sort(key=lambda x: -x[0])
    out = []
    for i, (n, c, words) in enumerate(scored):
        cc = c.model_copy()
        cc.is_suggested = (i == 0 and n > 0)
        cc.match_rationale = (c.match_rationale + f" | Description overlap on {n} task/title term(s): {', '.join(words[:8]) or 'none'}"
                              + (" — SUGGESTED only, please confirm" if cc.is_suggested else ""))
        out.append(cc)
    return out

def search_roles(query: str, limit: int = 25) -> TitleResolution:
    """Substring + exact search for the API/UI search box."""
    q = normalize(query or "")
    if not q: return TitleResolution("not_found", [])
    df = _lookup()
    sub = df[df.normalized_title.str.contains(re.escape(q), na=False)]
    if sub.empty: return resolve_title(query, limit)
    by_soc = {}
    for _, r in sub.iterrows():
        if r.onet_soc_code not in by_soc or r.title_match_type == "official": by_soc[r.onet_soc_code] = r
    cands = [_candidate(r, score=1.0 if r.normalized_title == q else None, rationale="Title search match") for r in list(by_soc.values())[:limit]]
    return TitleResolution("resolved" if len(cands) == 1 else "ambiguous_clarification_required", cands)
