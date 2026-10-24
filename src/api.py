"""FastAPI app. Run: uvicorn src.api:app --port 8000
Secrets are never returned; errors are sanitised (no stack traces)."""
from __future__ import annotations
from fastapi import FastAPI, HTTPException, Query
from src import job_title_resolver as resolver
from src.analysis import analyze
from src.evidence_retriever import data_status
from src.llm_explainer import config, is_configured
from src.schemas import AnalyzeRequest, RoleSearchResult

app = FastAPI(title="Psychometric Job-Role Analysis (O*NET 31.0, Phase 4)", version="0.4.0")

@app.get("/health")
def health():
    try:
        st = data_status(); data_ok = True
    except Exception as e:  # sanitised
        st, data_ok = {"error": type(e).__name__}, False
    cfg = config()
    return {"status": "ok" if data_ok else "degraded", "data": st,
            "llm": {"provider": cfg["provider"], "model": cfg["model"] or None, "configured": is_configured(cfg),
                    "mode": "live" if is_configured(cfg) else "deterministic_preview"}}

@app.get("/roles/search", response_model=RoleSearchResult)
def roles_search(q: str = Query(..., min_length=1, max_length=300)):
    res = resolver.search_roles(q)
    return RoleSearchResult(query=q, resolution_status=res.status, candidates=res.candidates)

@app.post("/analyze")
def analyze_endpoint(req: AnalyzeRequest):
    if not req.onet_soc_code and not req.job_title.strip():
        raise HTTPException(status_code=422, detail="Provide job_title or onet_soc_code.")
    try:
        resp = analyze(job_title=req.job_title, job_description=req.job_description,
                       onet_soc_code=req.onet_soc_code, top_per_family=req.top_per_family)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e).strip("'"))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Required dataset unavailable: {e.filename}")
    except Exception:
        raise HTTPException(status_code=500, detail="Internal error during analysis (details withheld).")
    return resp
