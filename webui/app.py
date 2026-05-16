"""Telltale web UI — a real, thin surface over the real drift core.

Not a mock: /api/run executes the genuine Gemini-backed drift pipeline
(telltale.webrun.live_run); /api/evidence serves the committed real
headline runs. Fail-closed: no GEMINI_API_KEY -> 503 with an honest error,
never a fabricated verdict.
"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from telltale import webrun  # noqa: E402
from telltale.config import MissingCredential  # noqa: E402

app = FastAPI(title="Telltale")
INDEX = (Path(__file__).parent / "index.html").read_text()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX


@app.get("/api/evidence")
def evidence() -> JSONResponse:
    return JSONResponse(webrun.load_evidence())


@app.post("/api/run")
def run() -> JSONResponse:
    """Real small live drift run (model downgrade pro->flash). Slow & honest."""
    try:
        cand = ROOT / "candidate_config.json"
        if not cand.exists():
            from telltale.agent import AgentConfig
            import dataclasses
            dataclasses.replace(AgentConfig.baseline(), version="cand-model",
                                model="gemini-2.5-flash").save(str(cand))
        return JSONResponse(webrun.live_run(str(cand), n_probes=4, k=2))
    except MissingCredential as e:
        return JSONResponse({"error": "fail-closed", "detail": str(e)}, status_code=503)
