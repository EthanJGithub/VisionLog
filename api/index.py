"""Vercel serverless API — lightweight detection LOGGING only.

All inference runs client-side (WebGPU), so this function never imports onnxruntime/
OpenCV/ultralytics — it only persists the detections the browser produces, to Postgres.
Reuses the project's SQLAlchemy store. Exposed as an ASGI `app` for the Vercel Python runtime.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable (so `src` resolves on Vercel).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from src import store  # noqa: E402  (light: SQLAlchemy only)
from src.agent import chat_graph  # noqa: E402  (torch-free: langgraph/langchain-groq only)
from src.api import schemas  # noqa: E402  (light: pydantic only)

app = FastAPI(title="VisionLog logging API", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_initialized = False


def _ensure_db() -> None:
    global _initialized
    if not _initialized:
        store.init_db()
        _initialized = True


@app.get("/api/v1/health", response_model=schemas.HealthOut)
def health() -> schemas.HealthOut:
    # This deployment runs all models client-side (WebGPU); no server inference.
    return schemas.HealthOut(
        status="ok",
        model_version="client-webgpu",
        model_loaded=True,
        sample_fps=0,
        max_upload_mb=0,
        max_duration_seconds=0,
        server_models={},
        open_vocab_available=False,
        # Chatbot is torch-free (LangGraph + Groq), so it runs on this Vercel function when a
        # GROQ_API_KEY is set; otherwise chat_graph.available() is False and /chat returns 503.
        chat_available=chat_graph.available(),
    )


@app.post("/api/v1/chat", response_model=schemas.ChatOut)
def chat(body: schemas.ChatIn) -> schemas.ChatOut:
    # LangGraph multi-agent text-to-SQL over the detections DB (mirrors src/api/routes.py).
    # No silent fallback: without GROQ_API_KEY (+ deps) we surface a clear 503.
    if not chat_graph.available():
        raise HTTPException(
            status_code=503,
            detail="Chatbot needs langgraph/langchain-groq and a GROQ_API_KEY. "
            "Set GROQ_API_KEY in the Vercel env to enable it.",
        )
    if not body.question.strip():
        raise HTTPException(status_code=422, detail="Ask a question about the detections.")
    _ensure_db()
    try:
        return schemas.ChatOut(**chat_graph.ask(body.question.strip()))
    except Exception as exc:  # pragma: no cover - LLM/runtime errors
        raise HTTPException(status_code=500, detail=f"chat failed: {str(exc)[:200]}") from exc


@app.post("/api/v1/client-sessions", response_model=schemas.SourceOut)
def create_client_session(body: schemas.ClientSessionIn) -> schemas.SourceOut:
    _ensure_db()
    source_id = store.create_source(
        body.kind,
        filename=None,
        fps=None,
        model_version=body.model_version,
        conf_threshold=body.conf_threshold,
    )
    row = next((s for s in store.list_sources() if s["id"] == source_id), None)
    return schemas.SourceOut(**row)


@app.post(
    "/api/v1/client-sessions/{source_id}/detections",
    response_model=schemas.ClientLogResult,
)
def log_client_detections(source_id: int, body: schemas.ClientLogIn) -> schemas.ClientLogResult:
    _ensure_db()
    logged = 0
    for frame in body.items:
        dets = [d.model_dump() for d in frame.detections]
        logged += store.add_detections(source_id, frame.frame_number, frame.ts_seconds, dets)
    return schemas.ClientLogResult(logged=logged)


@app.get("/api/v1/sources", response_model=list[schemas.SourceOut])
def get_sources() -> list[schemas.SourceOut]:
    _ensure_db()
    return [schemas.SourceOut(**s) for s in store.list_sources()]


@app.get("/api/v1/sources/{source_id}/detections", response_model=list[schemas.DetectionOut])
def source_detections(source_id: int) -> list[schemas.DetectionOut]:
    _ensure_db()
    return [schemas.DetectionOut(**d) for d in store.get_detections(source_id)]


@app.get("/api/v1/stats", response_model=schemas.StatsOut)
def stats() -> schemas.StatsOut:
    _ensure_db()
    return schemas.StatsOut(
        totals=store.totals(),
        class_counts=[schemas.ClassCount(**c) for c in store.class_counts()],
    )
