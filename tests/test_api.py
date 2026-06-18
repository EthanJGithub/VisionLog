"""End-to-end API contract test: upload -> detections logged -> /stats aggregates.

Hermetic: the ONNX engine is replaced with a stub, the DB is in-memory SQLite, and the
video is the synthetic fixture. No network / model / GPU.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src import store
from src.api import routes
import src.api.main as main_module


@pytest.fixture
def client(engine, stub_engine, monkeypatch):
    # Point the store at the in-memory engine for every call.
    monkeypatch.setattr(store, "get_engine", lambda *a, **k: engine)
    # Replace the real ONNX engine registry with the deterministic stub.
    monkeypatch.setattr(routes.registry, "get_onnx_engine", lambda model_id: stub_engine)
    monkeypatch.setattr(
        routes.registry, "available_models", lambda: {"yolo26n": True}
    )
    return TestClient(main_module.app)


def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["model_version"]
    assert body["sample_fps"] > 0


def test_upload_logs_detections_and_stats(client, sample_video):
    with open(sample_video, "rb") as fh:
        r = client.post("/api/v1/sources", files={"file": ("clip.mp4", fh, "video/mp4")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["frames_analysed"] >= 1
    assert body["detections_logged"] >= 1
    assert body["source"]["kind"] == "upload"
    assert body["detections"][0]["class_label"] == "person"

    sid = body["source"]["id"]
    dets = client.get(f"/api/v1/sources/{sid}/detections").json()
    assert len(dets) == body["detections_logged"]

    stats = client.get("/api/v1/stats").json()
    assert stats["totals"]["detections"] == body["detections_logged"]
    assert any(c["class_label"] == "person" for c in stats["class_counts"])


def test_client_session_logging(client):
    """In-browser (WebGPU) path: open a session, post sampled detections, see them logged."""
    sess = client.post(
        "/api/v1/client-sessions",
        json={"kind": "webcam", "model_version": "yolo26n-webgpu", "conf_threshold": 0.35},
    ).json()
    assert sess["kind"] == "webcam"
    sid = sess["id"]

    det = {
        "class_id": 0, "class_label": "person", "confidence": 0.88,
        "bbox_x": 1.0, "bbox_y": 2.0, "bbox_w": 3.0, "bbox_h": 4.0,
    }
    r = client.post(
        f"/api/v1/client-sessions/{sid}/detections",
        json={"items": [
            {"frame_number": 0, "ts_seconds": 0.0, "detections": [det, det]},
            {"frame_number": 1, "ts_seconds": 0.2, "detections": [det]},
        ]},
    )
    assert r.status_code == 200, r.text
    assert r.json()["logged"] == 3
    assert client.get("/api/v1/stats").json()["totals"]["detections"] == 3


def test_upload_too_large_rejected(client, monkeypatch):
    from src import config

    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 0.0001)
    r = client.post(
        "/api/v1/sources",
        files={"file": ("big.mp4", b"x" * 5000, "video/mp4")},
    )
    assert r.status_code == 413
