"""Writes the one-line "why this domain fits this job" sentence."""
from __future__ import annotations
import json, os, time
from typing import Callable, Optional
from pydantic import BaseModel, Field

class LLMExplanationItem(BaseModel):
    domain_id: str
    relevance_explanation: str
    related_job_responsibilities: list[str] = Field(default_factory=list)
    confidence_explanation: str = ""
    limitations: list[str] = Field(default_factory=list)
    evidence_reference_ids: list[str] = Field(default_factory=list)

class LLMOutput(BaseModel):
    explanations: list[LLMExplanationItem] = Field(default_factory=list)

SYSTEM_PROMPT = """You explain psychometric-domain relevance for ONE resolved O*NET occupation using ONLY the supplied retrieved evidence and domain definitions.
Rules (mandatory):
1. Use only supplied evidence; never fabricate citations, O*NET values, occupation codes, ratings or task statements.
2. Keep abilities, personality traits, work styles, vocational interests and work values conceptually distinct.
3. Relate each domain to the occupation's actual responsibilities (supplied tasks).
4. State whether each link is direct, partial or proxy.
5. Distinguish source facts from your interpretation (say 'the source rates...' vs 'this suggests...').
6. Report missing evidence and pending_review status; never promote pending_review to validated.
7. Every explanation must cite supplied evidence_reference_ids that actually support it.
8. Never invent a rating where one is absent; never convert missing to zero.
9. An occupational rating describes the JOB, never an individual candidate; never present it as a candidate score.
10. Never claim a trait guarantees performance, is mandatory, or should screen/hire anyone.
11. Do not create new scores or cross-family rankings.
12. AUDIENCE STYLE (strict): relevance_explanation must be EXACTLY ONE short plain-English sentence (max 20 words) saying why this domain fits this job's actual work, written for a general audience. NEVER mention rating codes or scale names (IM, LV, WI, DR, OI, IH), numeric rating values, percentiles, the words 'crosswalk', 'proxy', 'partial link', 'the source rates', 'evidence', or 'percentile' in relevance_explanation.
Return ONLY JSON: {"explanations":[{"domain_id":..., "relevance_explanation":..., "related_job_responsibilities":[...], "confidence_explanation":..., "limitations":[...], "evidence_reference_ids":[...]}]}"""

BANNED = ["guarantees job performance", "guaranteed to perform", "must possess", "candidate's score", "candidate score is", "scientifically validated requirement", "validated psychometric score"]

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_DEFAULT_MODEL = "openai/gpt-oss-20b"  # initial model, subject to account availability

def config() -> dict:
    try:
        from dotenv import load_dotenv; load_dotenv()  # local .env only; never committed
    except Exception: pass
    groq_key = os.getenv("GROQ_API_KEY", "")
    provider = os.getenv("LLM_PROVIDER", "").lower() or ("groq" if groq_key else "preview")
    if provider == "groq":
        return {"provider": "groq", "model": os.getenv("LLM_MODEL", "") or os.getenv("GROQ_MODEL", "") or GROQ_DEFAULT_MODEL,
                "api_key": groq_key, "base_url": os.getenv("LLM_BASE_URL", "") or GROQ_BASE_URL,
                "timeout": float(os.getenv("LLM_TIMEOUT", "30"))}
    return {"provider": provider,
            "model": os.getenv("LLM_MODEL", ""), "api_key": os.getenv("OPENAI_API_KEY", ""),
            "base_url": os.getenv("LLM_BASE_URL", ""), "timeout": float(os.getenv("LLM_TIMEOUT", "30"))}

def is_configured(cfg=None) -> bool:
    cfg = cfg or config()
    return cfg["provider"] in ("openai", "groq") and bool(cfg["api_key"]) and bool(cfg["model"])

def _safe_error(e: Exception, cfg: dict) -> str:
    """Sanitised provider error: class + message with any key redacted, truncated."""
    msg = str(e)
    if cfg.get("api_key"): msg = msg.replace(cfg["api_key"], "***")
    return f"{type(e).__name__}: {msg[:300]}"

def _preview_text(d: object) -> str:
    tasks = [t.split(";")[0].strip().rstrip(".") for t in d.related_job_responsibilities if t.strip()]
    if tasks:
        task = tasks[sum(map(ord, d.domain_id)) % len(tasks)]  # vary task per domain, deterministically
        task = task[0].lower() + task[1:]
        return f"This job involves work such as {task}, so strengths in {d.domain_name.lower()} matter here."
    return f"The day-to-day work in this job calls for strengths in {d.domain_name.lower()}."

def preview(recs: list[object]) -> tuple[list[object], dict]:
    for d in recs:
        if not d.relevance_explanation: d.relevance_explanation = _preview_text(d)
        if not d.confidence_explanation: d.confidence_explanation = "Preview mode: no LLM call was made."
    return recs, {"provider": "preview", "model": "deterministic-preview", "is_llm": False}

def validate_llm_output(raw_text: str, recs: list[object]) -> LLMOutput:
    """Raises ValueError on malformed JSON, unknown domains, unresolved refs, banned claims."""
    text = (raw_text or "").strip()
    if text.startswith("```"): text = text.strip("`").removeprefix("json").strip()
    try: data = json.loads(text)
    except Exception:
        s, e2 = text.find("{"), text.rfind("}")  # tolerate prose/fence wrapping, still require real JSON object
        if s < 0 or e2 <= s: raise ValueError("LLM returned invalid JSON: no JSON object found")
        try: data = json.loads(text[s:e2+1])
        except Exception as e: raise ValueError(f"LLM returned invalid JSON: {e}") from e
    out = LLMOutput(**data)
    allowed_ids = {d.domain_id for d in recs}
    allowed_refs = {r.evidence_id for d in recs for r in d.evidence_references}
    for item in out.explanations:
        if item.domain_id not in allowed_ids: raise ValueError(f"LLM explained unknown domain {item.domain_id}")
        bad = set(item.evidence_reference_ids) - allowed_refs
        if bad: raise ValueError(f"LLM cited unresolved evidence references: {sorted(bad)}")
        if not item.evidence_reference_ids: raise ValueError(f"LLM explanation for {item.domain_id} has no evidence references")
        text = item.relevance_explanation.lower()
        for b in BANNED:
            if b in text: raise ValueError(f"LLM made unsupported claim ('{b}') for {item.domain_id}")
    return out

TRANSIENT = ("RateLimitError", "APITimeoutError", "APIConnectionError", "InternalServerError")

def _retry_delay(e: Exception, attempt: int, base: float) -> float:
    """Retry-After header if the provider sent one, else bounded exponential backoff."""
    try:
        ra = e.response.headers.get("retry-after")  # openai SDK errors carry .response
        if ra: return min(float(ra), 60.0)
    except Exception: pass
    return min(base * (2 ** attempt), 60.0)

def call_with_retry(fn, cfg: dict, sleep=time.sleep):
    """Retry ONLY transient provider errors (rate limit/timeout/connection/5xx).
    Auth and other errors propagate immediately — never retried."""
    attempts = int(cfg.get("max_retries", 2)); base = float(cfg.get("retry_base_s", 5.0))
    for attempt in range(attempts + 1):
        try: return fn()
        except Exception as e:
            if type(e).__name__ not in TRANSIENT or attempt >= attempts: raise
            sleep(_retry_delay(e, attempt, base))
    raise RuntimeError("unreachable")

def _provider_call(payload: dict, cfg: dict) -> str:
    """OpenAI-compatible chat call (OpenAI, or Groq via api.groq.com/openai/v1).
    No response_format is forced: returned JSON is validated with Pydantic by
    validate_llm_output, so providers/models without JSON mode still work.
    Transient errors are retried with bounded backoff (Retry-After honoured);
    final failures propagate to explain(), which falls back to preview with a
    sanitised reason — keys are never logged."""
    from openai import OpenAI  # installed; imported lazily so preview never needs it
    client = OpenAI(api_key=cfg["api_key"], timeout=cfg["timeout"], max_retries=0,
                    **({"base_url": cfg["base_url"]} if cfg["base_url"] else {}))
    def once():
        resp = client.chat.completions.create(model=cfg["model"], temperature=0, max_tokens=int(cfg.get("max_tokens", 4096)),
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": json.dumps(payload)}])
        return resp.choices[0].message.content or ""
    return call_with_retry(once, cfg)

_openai_call = _provider_call  # backwards-compatible alias

def explain(recs: list[object], occupation: dict, job_description: str = "",
            llm_callable: Optional[Callable[[dict], str]] = None, cfg: Optional[dict] = None) -> tuple[list[object], dict]:
    """Returns (recommendations with explanations, provider_info). Falls back to preview on any provider failure."""
    cfg = cfg or config()
    # Compact, batched payloads: Groq on-demand TPM limits (8k/min observed) and
    # long reference lists make a single all-domain call fail. Cap refs/tasks and
    # call in chunks of 6 domains; every chunk is validated independently.
    def compact(ds):
        return {"occupation": occupation, "job_description": (job_description or "")[:1000],
               "domains": [{"domain_id": d.domain_id, "domain_name": d.domain_name, "family": d.construct_family,
                            "mapping_status": d.mapping_status, "link_types": d.crosswalk_link_types,
                            "rating_summary": (d.source_rating_summary or "")[:300], "tasks": [t[:120] for t in d.related_job_responsibilities[:2]],
                            "evidence_reference_ids": [r.evidence_id for r in d.evidence_references if r.onet_element_id][:4] or [r.evidence_id for r in d.evidence_references][:2],
                            "limitations": [l[:200] for l in d.limitations[:1]]} for d in ds]}
    payload = compact(recs)  # retained for mock callables/tests that inspect a single payload
    call = llm_callable
    if call is None and is_configured(cfg): call = lambda p: _provider_call(p, cfg)
    if call is None:
        return preview(recs)
    try:
        # Injected callables (tests/mocks) get one call; real providers get batched calls.
        if llm_callable is not None:
            outs = [validate_llm_output(call(payload), recs)]
        else:
            outs = [validate_llm_output(call(compact(recs[i:i+6])), recs) for i in range(0, len(recs), 6)]
        class _Merged: pass
        out = _Merged(); out.explanations = [e for o in outs for e in o.explanations]
    except Exception as e:  # auth, rate limit, timeout, malformed, unsupported -> graceful preview fallback
        recs2, info = preview(recs)
        info["fallback_reason"] = f"LLM output/provider failed validation ({_safe_error(e, cfg)}); deterministic preview used. No live LLM result was produced."
        return recs2, info
    by_id = {i.domain_id: i for i in out.explanations}
    for d in recs:
        if d.domain_id in by_id:
            i = by_id[d.domain_id]
            d.relevance_explanation, d.related_job_responsibilities = i.relevance_explanation, i.related_job_responsibilities or d.related_job_responsibilities
            d.confidence_explanation = i.confidence_explanation or d.confidence_explanation
            d.limitations = list(dict.fromkeys(d.limitations + i.limitations))
    return recs, {"provider": cfg["provider"], "model": cfg["model"] or "mock", "is_llm": True}
