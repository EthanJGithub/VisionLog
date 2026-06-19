"""Server-side closed-vocab detection for the heavy YOLO26 variants (m / x).

n/s run client-side (WebGPU) or via the lean onnxruntime path; m/x are too large to ship to
the browser (x ONNX ≈ 220 MB) so they run here via `ultralytics` (weights auto-download on
first use). Like open-vocab, this needs the "full" image — gated by available(); otherwise
the API returns 503 (no silent fallback). On the free tier the server is CPU-only, so this
is slow, which the UI states plainly.
"""
from __future__ import annotations

import threading
from typing import Any

import numpy as np

PT = {"yolo26m": "yolo26m.pt", "yolo26x": "yolo26x.pt"}

_lock = threading.Lock()
_models: dict[str, Any] = {}


def available() -> bool:
    try:
        import ultralytics  # noqa: F401
        return True
    except Exception:
        return False


def supports(model_id: str) -> bool:
    return model_id in PT


def detect_frame(model_id: str, frame_bgr: np.ndarray, conf: float) -> list[dict[str, Any]]:
    from ultralytics import YOLO

    with _lock:
        if model_id not in _models:
            _models[model_id] = YOLO(PT[model_id])
        model = _models[model_id]
        results = model.predict(frame_bgr, conf=conf, verbose=False)

    dets: list[dict[str, Any]] = []
    for r in results:
        boxes = getattr(r, "boxes", None)
        names = r.names
        if boxes is None:
            continue
        for b in boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            cls = int(b.cls[0])
            dets.append(
                {
                    "class_id": cls,
                    "class_label": names.get(cls, str(cls)) if isinstance(names, dict) else str(cls),
                    "confidence": round(float(b.conf[0]), 4),
                    "bbox_x": round(x1, 2),
                    "bbox_y": round(y1, 2),
                    "bbox_w": round(x2 - x1, 2),
                    "bbox_h": round(y2 - y1, 2),
                }
            )
    return dets
