"""Presentation layer for the chat UI: turns an AnalysisResponse into ONE
simple ranked list. No families, no codes, no jargon. Pure/testable."""
from __future__ import annotations
import re

MIN_SCORE = 60
TOP_N = 8
BANNED = ["crosswalk", "percentile", "o*net", "proxy", "partial link", "scale id", "evidence reference", "distinctiveness"]

def clean_plain(text: str, max_words: int = 25) -> str:
    """One short, plain sentence: strip rating codes/jargon, keep first sentence."""
    t = text or ""
    t = re.sub(r"\([^)]*\b(IM|LV|WI|DR|OI|IH)\b[^)]*\)", "", t)
    t = re.sub(r"\b\d+\.[A-Z]\.[\d.a-z.]+\b", "", t)
    t = re.sub(r"\b(IM|LV|WI|DR|OI|IH)\s*[=]?\s*\d+(\.\d+)?", "", t)
    t = re.sub(r"[^.]*\b(crosswalk|percentile|proxy|link is (direct|partial))\b[^.]*\.", "", t, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip()
    first = (t.split(". ")[0].rstrip(".") + ".") if t else ""
    words = first.split()
    return (" ".join(words[:max_words]).rstrip(",;:") + ("…" if len(words) > max_words else ("" if first.endswith(".") else "."))) if first else ""

def first_sentence(text: str) -> str:
    t = (text or "").strip()
    return (t.split(". ")[0].rstrip(".") + ".") if t else ""

def ranked_domains(resp) -> list[dict]:
    """Top domains across all families, strongest first, 60%+ only, max 8.
    Each item: rank, name, score (int %), what_it_means, why_it_fits."""
    strong = [d for d in resp.psychometric_domains if (d.preliminary_score or 0) >= MIN_SCORE]
    strong.sort(key=lambda d: -(d.preliminary_score or 0))
    out = []
    for i, d in enumerate(strong[:TOP_N], 1):
        out.append({"rank": i, "domain_id": d.domain_id, "name": d.domain_name,
                    "score": round(d.preliminary_score),
                    "what_it_means": first_sentence(d.domain_definition),
                    "why_it_fits": clean_plain(d.relevance_explanation)})
    return out

def has_banned_jargon(text: str) -> bool:
    low = (text or "").lower()
    return any(b in low for b in BANNED) or bool(re.search(r"\b(IM|LV|WI|DR|OI|IH)\s*=", text or ""))
