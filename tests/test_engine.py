"""Engine test against the real YOLO26n ONNX model.

Skips automatically if the model hasn't been exported yet (keeps CI hermetic when the
weights aren't bundled). Uses a synthetic image to assert output shape/contract.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src import config
from src.detect.engine import Detection, DetectionEngine

pytestmark = pytest.mark.skipif(
    not Path(config.MODEL_PATH).exists(),
    reason="yolo26n.onnx not exported; run `python -m src.detect.export`",
)


def test_detect_returns_well_formed_detections():
    engine = DetectionEngine()
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)  # neutral gray frame
    dets = engine.detect(frame)
    assert isinstance(dets, list)
    for d in dets:
        assert isinstance(d, Detection)
        assert 0.0 <= d.confidence <= 1.0
        assert d.confidence >= engine.conf_threshold
        # bbox within frame bounds
        assert 0 <= d.bbox_x <= 640 and 0 <= d.bbox_y <= 480
        assert d.bbox_w > 0 and d.bbox_h > 0
        assert d.class_label
