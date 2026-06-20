"""Store CRUD + aggregation against in-memory SQLite (hermetic)."""
from __future__ import annotations

import pytest

from src import store
from src.config import _normalize_db_url


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("postgres://u:p@h/db?sslmode=require", "postgresql+psycopg://u:p@h/db?sslmode=require"),
        ("postgresql://u:p@h/db", "postgresql+psycopg://u:p@h/db"),
        ("postgresql+psycopg://u:p@h/db", "postgresql+psycopg://u:p@h/db"),
        ("sqlite:///x.db", "sqlite:///x.db"),
    ],
)
def test_normalize_db_url_pins_psycopg_driver(raw, expected):
    # Managed Postgres (Neon) hands out postgres[ql]:// URLs; we must pin the psycopg v3 driver
    # so they connect without a manual scheme edit. SQLite/qualified URLs pass through unchanged.
    assert _normalize_db_url(raw) == expected


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


def test_object_crops_upsert_keeps_best(engine):
    sid = store.create_source("upload", model_version="m", conf_threshold=0.4, engine=engine)
    store.upsert_object_crop(sid, 1, "car", 0.70, "data:image/jpeg;base64,AAA", engine=engine)
    store.upsert_object_crop(sid, 1, "car", 0.90, "data:image/jpeg;base64,BBB", engine=engine)  # higher
    store.upsert_object_crop(sid, 1, "car", 0.50, "data:image/jpeg;base64,CCC", engine=engine)  # lower, ignored
    store.upsert_object_crop(sid, 2, "person", 0.80, "data:image/jpeg;base64,DDD", engine=engine)

    crops = store.get_object_crops(engine=engine)
    assert len(crops) == 2  # one per (source, track)
    car = next(c for c in crops if c["track_id"] == 1)
    assert car["confidence"] == 0.90 and car["thumb"].endswith("BBB")  # kept the best
    assert {c["class_label"] for c in crops} == {"car", "person"}
    assert [c["track_id"] for c in store.get_object_crops(engine=engine, class_label="PERSON")] == [2]


def test_crop_embeddings_and_semantic_search(engine):
    sid = store.create_source("upload", model_version="m", conf_threshold=0.4, engine=engine)
    for tid, lbl in [(1, "car"), (2, "person"), (3, "cat")]:
        store.upsert_object_crop(sid, tid, lbl, 0.9, f"data:image/png;base64,{lbl}", engine=engine)
    # orthogonal-ish unit embeddings so cosine ranking is unambiguous
    store.set_crop_embeddings(sid, [
        {"track_id": 1, "embedding": [1.0, 0.0, 0.0]},
        {"track_id": 2, "embedding": [0.0, 1.0, 0.0]},
        {"track_id": 3, "embedding": [0.9, 0.1, 0.0]},  # closest to the [1,0,0] query after car
    ], engine=engine)

    hits = store.search_object_crops([1.0, 0.0, 0.0], engine=engine)
    assert [h["track_id"] for h in hits[:2]] == [1, 3]      # car, then the near-car cat
    assert hits[0]["similarity"] >= hits[1]["similarity"]
    # class-scoped search
    cats = store.search_object_crops([1.0, 0.0, 0.0], engine=engine, class_label="cat")
    assert [h["track_id"] for h in cats] == [3]
    # crops without embeddings are skipped
    store.upsert_object_crop(sid, 9, "bus", 0.9, "data:image/png;base64,bus", engine=engine)
    assert 9 not in {h["track_id"] for h in store.search_object_crops([1.0, 0.0, 0.0], engine=engine)}


def test_crop_caption_roundtrip(engine):
    sid = store.create_source("upload", model_version="m", conf_threshold=0.4, engine=engine)
    store.upsert_object_crop(sid, 1, "dog", 0.9, "data:image/png;base64,dog", engine=engine)
    assert store.get_object_crops(engine=engine)[0]["caption"] is None
    store.set_crop_caption(sid, 1, "a small brown dog sitting", engine=engine)
    assert store.get_object_crops(engine=engine)[0]["caption"] == "a small brown dog sitting"


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
