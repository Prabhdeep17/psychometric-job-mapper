"""FastAPI backend. Run: uvicorn api:app --port 8000
Secrets are never returned; errors are simplified (no stack traces)."""
from fastapi import FastAPI, HTTPException, Query

from analysis import AnalyzeRequest, RoleSearchResult, analyze, data_status, search_roles
from llm import config, is_configured

app = FastAPI(title="Psychometric Job Mapper", version="1.0.0")


@app.get("/health")
def health():
    try:
        data, data_ok = data_status(), True
    except Exception as e:  # simplified error
        data, data_ok = {"error": type(e).__name__}, False
    cfg = config()
    return {"status": "ok" if data_ok else "degraded", "data": data,
            "llm": {"provider": cfg["provider"], "model": cfg["model"] or None,
                    "configured": is_configured(cfg),
                    "mode": "live" if is_configured(cfg) else "preview"}}


@app.get("/roles/search", response_model=RoleSearchResult)
def roles_search(q: str = Query(..., min_length=1, max_length=300)):
    res = search_roles(q)
    return RoleSearchResult(query=q, resolution_status=res.status, candidates=res.candidates)


@app.post("/analyze")
def analyze_endpoint(req: AnalyzeRequest):
    if not req.onet_soc_code and not req.job_title.strip():
        raise HTTPException(status_code=422, detail="Provide job_title or onet_soc_code.")
    try:
        return analyze(job_title=req.job_title, job_description=req.job_description,
                       onet_soc_code=req.onet_soc_code, top_per_family=req.top_per_family)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e).strip("'"))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Required dataset unavailable: {e.filename}")
    except Exception:
        raise HTTPException(status_code=500, detail="Internal error during analysis (details withheld).")
