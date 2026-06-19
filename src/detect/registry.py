"""Server-side model registry.

Closed-vocab YOLO26 variants (n/s/m/x) run via the lean onnxruntime DetectionEngine.
Open-vocabulary (YOLOE-26) is handled separately in `openvocab.py` (needs ultralytics).
Engines are cached per model id. No silent fallback: a requested ONNX that hasn't been
exported raises FileNotFoundError with the exact export command.
"""
from __future__ import annotations

import threading
from pathlib import Path

from src import config
from src.detect.engine import DetectionEngine

ROOT = Path(config.ROOT)

# model id -> onnx path (relative to repo root). Only the small variants are served via the
# lean onnxruntime path; m/x are handled by src/detect/ultra.py (ultralytics, full image).
SERVER_ONNX: dict[str, Path] = {
    "yolo26n": ROOT / "models" / "yolo26n.onnx",
    "yolo26s": ROOT / "models" / "yolo26s.onnx",
}

_cache: dict[str, DetectionEngine] = {}
_lock = threading.Lock()


def available_models() -> dict[str, bool]:
    """Map model id -> whether its ONNX file is present."""
    return {mid: path.exists() for mid, path in SERVER_ONNX.items()}


def get_onnx_engine(model_id: str) -> DetectionEngine:
    if model_id not in SERVER_ONNX:
        raise KeyError(
            f"Unknown server model '{model_id}'. Known: {sorted(SERVER_ONNX)}."
        )
    if model_id not in _cache:
        with _lock:
            if model_id not in _cache:
                path = SERVER_ONNX[model_id]
                if not path.exists():
                    raise FileNotFoundError(
                        f"Model '{model_id}' is not exported ({path}). Export it with: "
                        f"python -m src.detect.export --model {model_id}"
                    )
                _cache[model_id] = DetectionEngine(
                    model_path=str(path), model_version=f"{model_id}-onnx"
                )
    return _cache[model_id]
