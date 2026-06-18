"""Open-vocabulary detection via YOLOE-26 (text prompts).

The user supplies class names (e.g. "forklift, hard hat"); YOLOE detects them even if
never explicitly trained on them. After set_classes() the prompts are baked in, so
inference runs at normal YOLO speed. This needs `ultralytics` at runtime (heavier than
the lean ONNX path), so it is guarded by `available()` — if unavailable the API returns
a clear 503 rather than silently degrading.

The checkpoint is env-configurable (VISIONLOG_YOLOE_MODEL). First use downloads weights.
"""
from __future__ import annotations

import os
import threading
from typing import Any

import numpy as np

YOLOE_MODEL = os.getenv("VISIONLOG_YOLOE_MODEL", "yoloe-11s-seg.pt")

_lock = threading.Lock()
_model = None
_current_classes: list[str] | None = None


def available() -> bool:
    try:
        import ultralytics  # noqa: F401
        return True
    except Exception:
        return False


def model_version() -> str:
    return f"yoloe:{YOLOE_MODEL}"


def _ensure_model(classes: list[str]):
    global _model, _current_classes
    from ultralytics import YOLOE

    if _model is None:
        _model = YOLOE(YOLOE_MODEL)
    if classes != _current_classes:
        _model.set_classes(classes, _model.get_text_pe(classes))
        _current_classes = classes
    return _model


def detect_frame(frame_bgr: np.ndarray, prompts: list[str], conf: float) -> list[dict[str, Any]]:
    """Run open-vocab detection on a single BGR frame for the given text prompts."""
    classes = [p.strip() for p in prompts if p.strip()]
    if not classes:
        return []
    with _lock:
        model = _ensure_model(classes)
        results = model.predict(frame_bgr, conf=conf, verbose=False)

    dets: list[dict[str, Any]] = []
    for r in results:
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue
        for b in boxes:
            xyxy = b.xyxy[0].tolist()
            cls_idx = int(b.cls[0])
            label = classes[cls_idx] if 0 <= cls_idx < len(classes) else str(cls_idx)
            x1, y1, x2, y2 = xyxy
            dets.append(
                {
                    "class_id": cls_idx,
                    "class_label": label,
                    "confidence": round(float(b.conf[0]), 4),
                    "bbox_x": round(x1, 2),
                    "bbox_y": round(y1, 2),
                    "bbox_w": round(x2 - x1, 2),
                    "bbox_h": round(y2 - y1, 2),
                }
            )
    return dets
