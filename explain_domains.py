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
6. Report missing evidence honestly; crosswalk links have been reviewed but are not independently validated by a licensed psychometrician.
7. Every explanation must cite supplied evidence_reference_ids that actually support it.
8. Never invent a rating where one is absent; never convert missing to zero.
9. An occupational rating describes the JOB, never an individual candidate; never present it as a candidate score.
10. Never claim a trait guarantees performance, is mandatory, or should screen/hire anyone.
11. Do not create new scores or cross-family rankings.
12. relevance_explanation must be TWO short plain-English sentences. Do not answer with only a restatement of the definition; What it means already shows that. Sentence 1: name the supplied occupational requirement and what it involves. Sentence 2: say how that requirement relates to this domain, and whether the connection is direct or only provisional. If evidence_strength is direct, you may say the occupation is rated on that requirement. If evidence_strength is partial, proxy, or insufficient, say the relationship is provisional and do not claim a task proves the trait. A survey, sampling, or data task does not establish leadership or an emotional trait. Each domain must mention its own element name. NEVER mention rating codes (IM, LV, WI, DR, OI, IH), numeric values, percentiles, or the words crosswalk or proxy. NEVER state a percentage.
Return ONLY JSON: {"explanations":[{"domain_id":..., "relevance_explanation":..., "related_job_responsibilities":[...], "confidence_explanation":..., "limitations":[...], "evidence_reference_ids":[...]}]}"""

BANNED = ["guarantees job performance", "guaranteed to perform", "must possess", "candidate's score", "candidate score is", "scientifically validated requirement", "validated psychometric score"]

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# Ordered fallback chain — rotate on RateLimitError
GROQ_MODEL_CHAIN = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
]
GROQ_DEFAULT_MODEL = GROQ_MODEL_CHAIN[0]

def config() -> dict:
    try:
        from pathlib import Path
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent / ".env")  # does not override an existing env var
    except Exception: pass
    groq_key = os.getenv("GROQ_API_KEY", "")
    provider = os.getenv("LLM_PROVIDER", "").lower() or ("groq" if groq_key else "preview")
    if provider == "groq":
        # If user pinned a model via env, use it; otherwise use the full auto-switch chain
        pinned = os.getenv("LLM_MODEL", "") or os.getenv("GROQ_MODEL", "")
        chain = [pinned] if pinned else GROQ_MODEL_CHAIN
        return {"provider": "groq", "model": chain[0], "model_chain": chain,
                "api_key": groq_key, "base_url": os.getenv("LLM_BASE_URL", "") or GROQ_BASE_URL,
                "timeout": float(os.getenv("LLM_TIMEOUT", "30"))}
    return {"provider": provider,
            "model": os.getenv("LLM_MODEL", ""), "model_chain": [],
            "api_key": os.getenv("OPENAI_API_KEY", ""),
            "base_url": os.getenv("LLM_BASE_URL", ""), "timeout": float(os.getenv("LLM_TIMEOUT", "30"))}

def is_configured(cfg=None) -> bool:
    cfg = cfg or config()
    return cfg["provider"] in ("openai", "groq") and bool(cfg["api_key"]) and bool(cfg["model"])

def _safe_error(e: Exception, cfg: dict) -> str:
    """Sanitised provider error: class + message with any key redacted, truncated."""
    msg = str(e)
    if cfg.get("api_key"): msg = msg.replace(cfg["api_key"], "***")
    return f"{type(e).__name__}: {msg[:300]}"

def _element_names(d: object) -> list[str]:
    names = []
    for ref in getattr(d, "evidence_references", []) or []:
        name = getattr(ref, "onet_element_name", "") or ""
        if name and name not in names:
            names.append(name)
    return names[:3]


def _preview_text(d: object) -> str:
    """Deterministic explanation tied to this domain's own job requirement, not a copied definition line."""
    meaning = ((getattr(d, "domain_definition", "") or "").split(". ")[0].strip().rstrip(".") or d.domain_name)
    elements = _element_names(d)
    strength = getattr(d, "evidence_strength", "") or "insufficient"
    named = ", ".join(elements)
    if strength == "insufficient" or not elements:
        return (f"No occupational rating was available to connect {d.domain_name} ({meaning}) to this job. "
                "Supporting evidence is insufficient, so the relationship remains provisional.")
    if strength == "direct":
        return (f"This occupation is rated on {named}. That job requirement matches {d.domain_name} ({meaning}), "
                "so the connection is supported by the occupational rating rather than by assuming a task proves a personal trait.")
    return (f"This occupation is linked to {named}, which covers only part of {d.domain_name} ({meaning}). "
            "The relationship is provisional, and a job task does not by itself establish this trait.")

def preview(recs: list[object]) -> tuple[list[object], dict]:
    for d in recs:
        if not d.relevance_explanation:
            d.relevance_explanation = _preview_text(d)
        d.explanation_source = "fallback"
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
    """OpenAI-compatible chat call with automatic Groq model rotation.
    When multiple models are in the chain, each model is tried ONCE — on
    RateLimitError the next model is tried immediately (no backoff sleep).
    Single-model mode still uses call_with_retry for backoff.
    Keys are never logged."""
    from openai import OpenAI  # installed; imported lazily so preview never needs it
    client = OpenAI(api_key=cfg["api_key"], timeout=cfg["timeout"], max_retries=0,
                    **({"base_url": cfg["base_url"]} if cfg["base_url"] else {}))
    chain = list(cfg.get("model_chain") or [cfg["model"]])

    # Single model: use normal retry logic
    if len(chain) == 1:
        def once():
            resp = client.chat.completions.create(
                model=chain[0], temperature=0, max_tokens=int(cfg.get("max_tokens", 4096)),
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                           {"role": "user",   "content": json.dumps(payload)}])
            return resp.choices[0].message.content or ""
        return call_with_retry(once, cfg)

    # Multiple models: try each once, switch immediately on rate limit
    last_exc: Exception = RuntimeError("no models in chain")
    for model in chain:
        try:
            resp = client.chat.completions.create(
                model=model, temperature=0, max_tokens=int(cfg.get("max_tokens", 4096)),
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                           {"role": "user",   "content": json.dumps(payload)}])
            result = resp.choices[0].message.content or ""
            if model != chain[0]:
                import sys
                print(f"Groq auto-switched to model: {model}", file=sys.stderr)
            return result
        except Exception as e:
            last_exc = e
            if type(e).__name__ == "RateLimitError" and model != chain[-1]:
                import sys
                print(f"Groq RateLimitError on {model}, trying next…", file=sys.stderr)
                continue
            raise
    raise last_exc

_openai_call = _provider_call  # backwards-compatible alias

def fallback_kind(exc: Exception, cfg: dict) -> str:
    """Safe reason for the terminal. Never includes the API key."""
    if not cfg.get("api_key"):
        kind = "missing_key"
    else:
        name = type(exc).__name__
        if "Timeout" in name:
            kind = "timeout"
        elif name in {"RateLimitError", "APIConnectionError", "APIStatusError", "AuthenticationError",
                      "InternalServerError", "PermissionDeniedError", "NotFoundError", "BadRequestError"}:
            kind = "api_error"
        else:
            kind = "invalid_response"
    import sys
    print(f"Explanation fallback: {kind} ({type(exc).__name__})", file=sys.stderr)
    return kind


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
                            "definition": (d.domain_definition or "")[:240],
                            "evidence_strength": d.evidence_strength,
                            "elements": _element_names(d),
                            "mapping_status": d.mapping_status, "link_types": d.crosswalk_link_types,
                            "tasks": [t[:120] for t in d.related_job_responsibilities[:2]],
                            "evidence_reference_ids": [r.evidence_id for r in d.evidence_references if r.onet_element_id][:4] or [r.evidence_id for r in d.evidence_references][:2],
                            "limitations": [l[:200] for l in d.limitations[:1]]} for d in ds]}
    payload = compact(recs)  # retained for mock callables/tests that inspect a single payload
    call = llm_callable
    if call is None and is_configured(cfg): call = lambda p: _provider_call(p, cfg)
    if call is None:
        info = {"provider": "preview", "model": "deterministic-preview", "is_llm": False,
                "explanation_note": "Deterministic fallback: missing key"}
        print("Explanation fallback: missing_key", file=__import__("sys").stderr)
        recs2, _ = preview(recs)
        return recs2, info
    chunks = [recs] if llm_callable is not None else [recs[i:i+6] for i in range(0, len(recs), 6)]
    applied, failed = 0, []
    for chunk in chunks:
        try:
            out = validate_llm_output(call(compact(chunk) if llm_callable is None else payload), recs)
        except Exception as e:
            failed.append(fallback_kind(e, cfg))
            for d in chunk:
                d.relevance_explanation = _preview_text(d)
                d.explanation_source = "fallback"
            continue
        by_id = {i.domain_id: i for i in out.explanations}
        for d in chunk:
            item = by_id.get(d.domain_id)
            if item and item.relevance_explanation:
                d.relevance_explanation = item.relevance_explanation
                d.explanation_source = "groq"
                applied += 1
            else:
                d.relevance_explanation = _preview_text(d)
                d.explanation_source = "fallback"
    if applied == 0:
        return recs, {"provider": "preview", "model": "deterministic-preview", "is_llm": False,
                      "explanation_note": "Deterministic fallback: " + (failed[0] if failed else "invalid_response").replace("_", " ")}
    note = "Live Groq inference" if not failed else "Live Groq inference; some domains used deterministic fallback"
    return recs, {"provider": cfg["provider"], "model": cfg["model"] or "mock", "is_llm": True, "explanation_note": note}
