"""Shared test fixtures. Hermetic: in-memory SQLite, a stub engine, and a synthetic
video generated with OpenCV — no network, no ONNX model, no GPU required.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src import store


@pytest.fixture
def engine():
    """A throwaway in-memory SQLite engine with the schema created."""
    eng = store.get_engine("sqlite:///:memory:")
    store.Base.metadata.create_all(eng)
    return eng


class StubDetection:
    def __init__(self, d: dict):
        self._d = d

    def as_dict(self) -> dict:
        return self._d


class StubEngine:
    """Deterministic engine: reports one 'person' per frame. Avoids the ONNX model."""

    model_version = "stub-engine"
    conf_threshold = 0.35

    def detect(self, frame):
        return [
            StubDetection(
                {
                    "class_id": 0,
                    "class_label": "person",
                    "confidence": 0.9,
                    "bbox_x": 10.0,
                    "bbox_y": 10.0,
                    "bbox_w": 50.0,
                    "bbox_h": 80.0,
                }
            )
        ]


@pytest.fixture
def stub_engine() -> StubEngine:
    return StubEngine()


@pytest.fixture
def sample_video() -> Path:
    """A short synthetic mp4 (10 frames @ 10 fps) with a moving rectangle."""
    path = Path(tempfile.mkstemp(suffix=".mp4")[1])
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (160, 120))
    for i in range(10):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        cv2.rectangle(frame, (i * 5, 30), (i * 5 + 40, 90), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    yield path
    try:
        path.unlink(missing_ok=True)
    except PermissionError:
        pass  # best-effort: Windows can briefly hold the temp handle
