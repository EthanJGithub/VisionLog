"""YOLO26n object-detection engine (lean CPU inference via onnxruntime).

YOLO26's default head is end-to-end / NMS-free, exporting to ONNX as (1, 300, 6) where
each row is [x1, y1, x2, y2, score, class_id] in letterboxed input-space pixels. We
undo the letterbox to map boxes back to original-frame coordinates. No NMS step needed.

No silent fallback (CLAUDE.md): if the model file is missing, __init__ raises with
actionable guidance pointing at the export script.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from src import config
from src.detect.labels import COCO_CLASSES


@dataclass
class Detection:
    class_id: int
    class_label: str
    confidence: float
    # bbox in original-frame pixels, top-left origin
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class DetectionEngine:
    """Thread-safe wrapper around an onnxruntime session for YOLO26n."""

    def __init__(
        self,
        model_path: str | None = None,
        input_size: int | None = None,
        conf_threshold: float | None = None,
        model_version: str | None = None,
    ) -> None:
        self.model_path = Path(model_path or config.MODEL_PATH)
        self.input_size = input_size or config.INPUT_SIZE
        self.conf_threshold = (
            conf_threshold if conf_threshold is not None else config.CONF_THRESHOLD
        )
        self._model_version = model_version
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"YOLO26 ONNX model not found at '{self.model_path}'. "
                "Export it first with:  python -m src.detect.export  "
                "(requires the dev dependency `ultralytics`)."
            )
        self._lock = threading.Lock()
        self.session = ort.InferenceSession(
            str(self.model_path), providers=["CPUExecutionProvider"]
        )
        self._input_name = self.session.get_inputs()[0].name
        self.model_version = self._model_version or config.MODEL_VERSION

    # -- preprocessing ----------------------------------------------------------------
    def _letterbox(self, frame: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        """Resize keeping aspect ratio, pad to a square `input_size`."""
        h, w = frame.shape[:2]
        scale = min(self.input_size / w, self.input_size / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        pad_x, pad_y = (self.input_size - nw) // 2, (self.input_size - nh) // 2
        canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized
        return canvas, scale, pad_x, pad_y

    def _preprocess(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        canvas, scale, pad_x, pad_y = self._letterbox(frame_bgr)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]  # NCHW
        return np.ascontiguousarray(tensor), scale, pad_x, pad_y

    # -- inference --------------------------------------------------------------------
    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Run detection on a single BGR frame; return boxes in original coords."""
        h, w = frame_bgr.shape[:2]
        tensor, scale, pad_x, pad_y = self._preprocess(frame_bgr)
        with self._lock:
            outputs = self.session.run(None, {self._input_name: tensor})
        preds = np.asarray(outputs[0])
        if preds.ndim == 3:  # (1, 300, 6)
            preds = preds[0]

        results: list[Detection] = []
        for row in preds:
            x1, y1, x2, y2, score, cls = (
                float(row[0]), float(row[1]), float(row[2]),
                float(row[3]), float(row[4]), int(round(float(row[5]))),
            )
            if score < self.conf_threshold:
                continue
            # undo letterbox: subtract pad, divide by scale, clamp to frame
            ox1 = max(0.0, (x1 - pad_x) / scale)
            oy1 = max(0.0, (y1 - pad_y) / scale)
            ox2 = min(float(w), (x2 - pad_x) / scale)
            oy2 = min(float(h), (y2 - pad_y) / scale)
            if ox2 <= ox1 or oy2 <= oy1:
                continue
            label = COCO_CLASSES[cls] if 0 <= cls < len(COCO_CLASSES) else str(cls)
            results.append(
                Detection(
                    class_id=cls,
                    class_label=label,
                    confidence=round(score, 4),
                    bbox_x=round(ox1, 2),
                    bbox_y=round(oy1, 2),
                    bbox_w=round(ox2 - ox1, 2),
                    bbox_h=round(oy2 - oy1, 2),
                )
            )
        return results


_engine: DetectionEngine | None = None
_engine_lock = threading.Lock()


def get_engine() -> DetectionEngine:
    """Process-wide singleton (lazy) so the ONNX session is created once."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = DetectionEngine()
    return _engine
