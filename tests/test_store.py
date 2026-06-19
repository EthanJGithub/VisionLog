"""Store CRUD + aggregation against in-memory SQLite (hermetic)."""
from __future__ import annotations

from src import store


def _det(label: str, conf: float = 0.9) -> dict:
    return {
        "class_id": 0, "class_label": label, "confidence": conf,
        "bbox_x": 1.0, "bbox_y": 2.0, "bbox_w": 3.0, "bbox_h": 4.0,
    }


def test_create_source_records_audit_fields(engine):
    sid = store.create_source(
        "upload", filename="clip.mp4", fps=5.0,
        model_version="yolo26n-onnx", conf_threshold=0.35, engine=engine,
    )
    rows = store.list_sources(engine=engine)
    assert len(rows) == 1
    assert rows[0]["id"] == sid
    assert rows[0]["model_version"] == "yolo26n-onnx"
    assert rows[0]["conf_threshold"] == 0.35


def test_add_detections_and_counts(engine):
    sid = store.create_source(
        "upload", model_version="m", conf_threshold=0.3, engine=engine
    )
    n1 = store.add_detections(sid, 0, 0.0, [_det("person"), _det("car")], engine=engine)
    n2 = store.add_detections(sid, 1, 0.2, [_det("person")], engine=engine)
    assert (n1, n2) == (2, 1)

    dets = store.get_detections(sid, engine=engine)
    assert len(dets) == 3
    assert dets[0]["ts_seconds"] == 0.0

    counts = {c["class_label"]: c["count"] for c in store.class_counts(engine=engine)}
    assert counts == {"person": 2, "car": 1}

    t = store.totals(engine=engine)
    assert t["sources"] == 1 and t["detections"] == 3
    assert "objects" in t  # distinct track_id count (0 here — no track_id supplied)


def test_track_id_persisted_and_counted(engine):
    sid = store.create_source("webcam", model_version="m", conf_threshold=0.4, engine=engine)
    d1 = {**_det("person"), "track_id": 7}
    d2 = {**_det("person"), "track_id": 7}  # same object, later frame
    d3 = {**_det("car"), "track_id": 8}
    store.add_detections(sid, 0, 0.0, [d1, d3], engine=engine)
    store.add_detections(sid, 1, 0.2, [d2], engine=engine)

    dets = store.get_detections(sid, engine=engine)
    assert all("track_id" in d for d in dets)
    assert {d["track_id"] for d in dets} == {7, 8}
    # two distinct objects (track_id 7 and 8) though 3 detection rows
    assert store.totals(engine=engine) == {"sources": 1, "detections": 3, "objects": 2}


def test_empty_detections_noop(engine):
    sid = store.create_source("webcam", model_version="m", conf_threshold=0.3, engine=engine)
    assert store.add_detections(sid, 0, 0.0, [], engine=engine) == 0
    assert store.totals(engine=engine)["detections"] == 0
