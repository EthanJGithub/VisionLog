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


def _log_session(client, kind, dets):
    sid = client.post(
        "/api/v1/client-sessions",
        json={"kind": kind, "model_version": "yolo26n-webgpu", "conf_threshold": 0.4},
    ).json()["id"]
    client.post(
        f"/api/v1/client-sessions/{sid}/detections",
        json={"items": [{"frame_number": 0, "ts_seconds": 0.0, "detections": dets}]},
    )
    return sid


def _det(label, cid, track):
    return {
        "class_id": cid, "class_label": label, "confidence": 0.9, "track_id": track,
        "bbox_x": 1.0, "bbox_y": 2.0, "bbox_w": 3.0, "bbox_h": 4.0,
    }


def test_stats_scoped_by_source(client):
    s1 = _log_session(client, "upload", [_det("car", 2, 1), _det("car", 2, 2), _det("person", 0, 3)])
    s2 = _log_session(client, "webcam", [_det("dog", 16, 1)])

    all_stats = client.get("/api/v1/stats").json()
    assert all_stats["totals"]["sources"] == 2
    assert all_stats["totals"]["detections"] == 4
    # unique objects = distinct (source_id, track_id): s1 has 3, s2 has 1 (track_id 1 reused) = 4
    assert all_stats["totals"]["objects"] == 4

    s1_stats = client.get(f"/api/v1/stats?source_id={s1}").json()
    assert s1_stats["totals"]["sources"] == 1
    assert s1_stats["totals"]["detections"] == 3
    assert s1_stats["totals"]["objects"] == 3
    assert {c["class_label"] for c in s1_stats["class_counts"]} == {"car", "person"}

    s2_stats = client.get(f"/api/v1/stats?source_id={s2}").json()
    assert s2_stats["totals"]["detections"] == 1
    assert [c["class_label"] for c in s2_stats["class_counts"]] == ["dog"]


def test_delete_one_source_and_clear_all(client):
    s1 = _log_session(client, "upload", [_det("car", 2, 1)])
    s2 = _log_session(client, "webcam", [_det("dog", 16, 1)])
    assert client.get("/api/v1/stats").json()["totals"]["sources"] == 2

    # delete one run -> its detections go too (cascade), other run untouched
    r = client.delete(f"/api/v1/sources/{s1}")
    assert r.status_code == 200 and r.json()["deleted"] == s1
    after = client.get("/api/v1/stats").json()
    assert after["totals"]["sources"] == 1
    assert after["totals"]["detections"] == 1
    assert client.get(f"/api/v1/sources/{s1}/detections").json() == []

    # deleting a missing run -> 404
    assert client.delete(f"/api/v1/sources/{s1}").status_code == 404

    # reset clears everything
    r = client.delete("/api/v1/sources")
    assert r.status_code == 200
    cleared = client.get("/api/v1/stats").json()
    assert cleared["totals"] == {"sources": 0, "detections": 0, "objects": 0}
    _ = s2  # second run was removed by the reset


def test_upload_too_large_rejected(client, monkeypatch):
    from src import config

    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 0.0001)
    r = client.post(
        "/api/v1/sources",
        files={"file": ("big.mp4", b"x" * 5000, "video/mp4")},
    )
    assert r.status_code == 413
