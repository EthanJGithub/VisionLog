"""VisionLog FastAPI app: API + serves the built Vite/React frontend."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src import config, store
from src.api.routes import router

_FRONTEND_DIST = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.init_db()
    yield


app = FastAPI(title="VisionLog", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)


# Serve the built SPA (if present). API routes take precedence (registered above).
if _FRONTEND_DIST.exists():
    app.mount(
        "/assets", StaticFiles(directory=_FRONTEND_DIST / "assets"), name="assets"
    )
    # Client-side (WebGPU) models are fetched by the browser from /models/*.onnx.
    _models_dir = _FRONTEND_DIST / "models"
    if _models_dir.exists():
        app.mount("/models", StaticFiles(directory=_models_dir), name="models")
    # Benchmark report + plots for the Benchmarks tab.
    _bench_dir = _FRONTEND_DIST / "benchmarks"
    if _bench_dir.exists():
        app.mount("/benchmarks", StaticFiles(directory=_bench_dir), name="benchmarks")

    @app.get("/")
    def spa_root() -> FileResponse:
        return FileResponse(_FRONTEND_DIST / "index.html")
