"""Hermetic tests for the chatbot's safety guard + SQL execution (no LLM / no network)."""
from __future__ import annotations

import pytest
from sqlalchemy import text

from src import store
from src.agent.schema import UnsafeSQLError, sanitize_sql


def test_guard_allows_select_and_adds_limit():
    out = sanitize_sql("SELECT class_label, COUNT(*) FROM detections GROUP BY class_label")
    assert out.lower().startswith("select")
    assert "limit" in out.lower()  # auto-appended


def test_guard_strips_fences_and_prose():
    raw = "Here is the query:\n```sql\nSELECT * FROM sources\n```"
    out = sanitize_sql(raw)
    assert out.lower().startswith("select * from sources")


def test_guard_keeps_existing_limit():
    out = sanitize_sql("SELECT * FROM detections LIMIT 5")
    assert out.lower().count("limit") == 1


@pytest.mark.parametrize("bad", [
    "DROP TABLE detections",
    "DELETE FROM detections",
    "UPDATE sources SET kind='x'",
    "INSERT INTO sources VALUES (1)",
    "SELECT 1; DROP TABLE sources",          # multi-statement
    "ALTER TABLE detections ADD COLUMN x INT",
    "PRAGMA table_info(detections)",
])
def test_guard_rejects_writes_and_multistatement(bad):
    with pytest.raises(UnsafeSQLError):
        sanitize_sql(bad)


def test_guarded_sql_executes_readonly(engine):
    # seed a couple of detections, then run a guarded aggregate query
    sid = store.create_source("webcam", model_version="m", conf_threshold=0.4, engine=engine)
    det = {"class_id": 0, "class_label": "person", "confidence": 0.9,
           "bbox_x": 1, "bbox_y": 2, "bbox_w": 3, "bbox_h": 4, "track_id": 1}
    store.add_detections(sid, 0, 0.0, [det, {**det, "class_label": "car", "track_id": 2}], engine=engine)

    sql = sanitize_sql("SELECT class_label, COUNT(*) AS n FROM detections GROUP BY class_label")
    with engine.connect() as conn:
        rows = {r[0]: r[1] for r in conn.execute(text(sql)).fetchall()}
    assert rows == {"person": 1, "car": 1}
