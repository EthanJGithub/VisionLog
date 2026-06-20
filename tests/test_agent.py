"""Hermetic tests for the chatbot's safety guard + SQL execution (no LLM / no network)."""
from __future__ import annotations

import pytest
from sqlalchemy import text

from src import store
from src.agent import chat_graph
from src.agent.schema import UnsafeSQLError, sanitize_sql


@pytest.mark.parametrize("q,intent", [
    ("How many cars were detected?", "analytics"),
    ("What is the average confidence per class?", "analytics"),
    ("Which run has the most detections?", "analytics"),
    ("What's been detected?", "overview"),
    ("Give me a summary of the data", "overview"),
    ("tell me about the footage", "overview"),
    ("What can I ask about this data?", "schema"),
    ("what classes do you track?", "schema"),
    ("show me the trucks", "gallery"),
    ("what did the cats look like", "gallery"),
    ("show me a picture of the cats", "gallery"),
    ("random vague question", "overview"),  # safe default
])
def test_keyword_intent_router(q, intent):
    assert chat_graph._keyword_intent(q) == intent


@pytest.mark.parametrize("raw,expected", [
    ("analytics", "analytics"),
    ("Intent: overview", "overview"),
    ("schema", "schema"),
    ("out_of_scope", "out_of_scope"),
    ("gibberish", None),
])
def test_normalize_intent(raw, expected):
    assert chat_graph._normalize_intent(raw) == expected


@pytest.mark.parametrize("q,expected", [
    ("show me the people", "person"),          # synonym
    ("show me the cars", "car"),               # simple plural
    ("show me the buses", "bus"),              # -es plural
    ("show me a picture of the cat", "cat"),   # singular substring
    ("show me the hard hats", "hard hat"),     # multi-word + plural
    ("show me everything", None),              # no class named
])
def test_gallery_wanted_class(q, expected):
    labels = {"person", "car", "bus", "cat", "hard hat"}
    assert chat_graph._wanted_class(q, labels) == expected


def test_overview_queries_pass_the_guard(engine):
    # The overview agent runs PRE-DEFINED queries (not LLM-authored); they must all be guard-safe
    # SELECTs and execute on the schema.
    store.create_source("upload", model_version="m", conf_threshold=0.4, engine=engine)
    for sql in chat_graph.OVERVIEW_QUERIES.values():
        safe = sanitize_sql(sql)
        assert safe.lower().lstrip().startswith("select")
        with engine.connect() as conn:
            conn.execute(text(safe)).fetchall()  # must not raise


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
