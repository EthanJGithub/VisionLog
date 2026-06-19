"""Schema grounding + SQL safety guard for the detections chatbot.

These are LLM-free and hermetically testable. The guard enforces read-only, single-statement
SELECT queries (defense-in-depth on top of the prompt) so the agent can never mutate the DB.
"""
from __future__ import annotations

import re

# Plain-language schema handed to the SQL-author agent. Kept in sync with src/store.py.
SCHEMA_DESCRIPTION = """\
You are querying a read-only detections database with two tables (SQLite or PostgreSQL).

TABLE sources  -- one row per video/webcam session
  id             INTEGER PRIMARY KEY
  kind           TEXT     -- 'upload' or 'webcam'
  filename       TEXT     -- original file name (NULL for webcam)
  fps            REAL     -- analysed/sample fps
  frame_count    INTEGER
  model_version  TEXT     -- e.g. 'yolo26n-webgpu', 'yolo26-ppe-webgpu'
  conf_threshold REAL
  created_at     TIMESTAMP

TABLE detections  -- one row per detected object per frame
  id            INTEGER PRIMARY KEY
  source_id     INTEGER  -- FK -> sources.id
  frame_number  INTEGER
  ts_seconds    REAL     -- time offset within the session
  track_id      INTEGER  -- Object ID: stable across frames for the same physical object
  class_label   TEXT     -- e.g. 'person', 'car', 'helmet', 'vest'
  class_id      INTEGER
  confidence    REAL     -- 0..1
  bbox_x, bbox_y, bbox_w, bbox_h  REAL  -- pixels
  created_at    TIMESTAMP

Notes for correct answers:
- A unique physical object = a distinct (source_id, track_id). To COUNT objects use
  COUNT(DISTINCT track_id) within a source (track_id repeats across frames).
- detections has many rows per object (one per frame); don't confuse rows with objects.
- Class names are lowercase except PPE 'Person'. Use case-insensitive matching (LOWER()).
"""

_WRITE = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|replace|attach|detach|"
    r"pragma|grant|revoke|vacuum|merge|call|exec|execute)\b",
    re.IGNORECASE,
)


class UnsafeSQLError(ValueError):
    pass


def sanitize_sql(raw: str, default_limit: int = 200) -> str:
    """Return a safe single-statement SELECT, or raise UnsafeSQLError.

    - strips markdown fences / prose, keeps the SQL
    - must be a single statement starting with SELECT or WITH
    - rejects any write/DDL keyword
    - appends a LIMIT if absent
    """
    if not raw or not raw.strip():
        raise UnsafeSQLError("empty query")

    sql = raw.strip()
    # strip ```sql ... ``` fences
    fence = re.search(r"```(?:sql)?\s*(.+?)```", sql, re.DOTALL | re.IGNORECASE)
    if fence:
        sql = fence.group(1).strip()
    # if the model added prose, grab from the first SELECT/WITH
    m = re.search(r"\b(select|with)\b", sql, re.IGNORECASE)
    if m:
        sql = sql[m.start():]
    sql = sql.rstrip(";").strip()

    if ";" in sql:
        raise UnsafeSQLError("multiple statements are not allowed")
    if not re.match(r"^(select|with)\b", sql, re.IGNORECASE):
        raise UnsafeSQLError("only SELECT/WITH queries are allowed")
    if _WRITE.search(sql):
        raise UnsafeSQLError("write/DDL statements are not allowed")

    if not re.search(r"\blimit\b", sql, re.IGNORECASE):
        sql = f"{sql}\nLIMIT {default_limit}"
    return sql
