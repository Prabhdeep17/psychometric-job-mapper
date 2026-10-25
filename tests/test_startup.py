"""The website module loads, and the API process can serve /health."""
import ast
from pathlib import Path

from fastapi.testclient import TestClient

from api import app


def test_streamlit_app_parses_and_is_clean():
    source = Path("app.py").read_text()
    ast.parse(source)
    assert "Relevance" in source
    assert "% match" not in source
    assert "% probability of job fit" not in source
    assert "Why it fits" in source
    assert "ranked_domains" in source


def test_api_process_health():
    body = TestClient(app).get("/health").json()
    assert body["status"] == "ok"
    assert body["data"]["grid_rows"] == 50800
