"""Persistence layer for VisionLog (SQLAlchemy 2.0).

Postgres (Neon) in production via DATABASE_URL; SQLite locally and in tests. The same
schema/code path serves both — an explicit, configured choice (CLAUDE.md: no silent
fallback). Every source row records the model_version + conf_threshold used, so each
detection run is reproducible/auditable.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    Column, DateTime, Float, ForeignKey, Integer, String, create_engine, func, select,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship
from sqlalchemy.pool import StaticPool

from src import config


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Source(Base):
    __tablename__ = "sources"

    id = Column(Integer, primary_key=True, autoincrement=True)
    kind = Column(String(16), nullable=False)            # "upload" | "webcam"
    filename = Column(String(512))
    fps = Column(Float)                                   # analysed fps (sample rate)
    frame_count = Column(Integer, default=0)
    model_version = Column(String(64), nullable=False)
    conf_threshold = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    detections = relationship(
        "DetectionRow", back_populates="source", cascade="all, delete-orphan"
    )


class DetectionRow(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_id = Column(
        Integer, ForeignKey("sources.id", ondelete="CASCADE"), nullable=False, index=True
    )
    frame_number = Column(Integer, nullable=False)
    ts_seconds = Column(Float, nullable=False)            # offset within the video/stream
    track_id = Column(Integer, index=True)                # Object ID — stable across frames
    class_label = Column(String(64), nullable=False, index=True)
    class_id = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=False)
    bbox_x = Column(Float, nullable=False)
    bbox_y = Column(Float, nullable=False)
    bbox_w = Column(Float, nullable=False)
    bbox_h = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    source = relationship("Source", back_populates="detections")


_engine = None


def get_engine(database_url: str | None = None):
    global _engine
    if _engine is None or database_url is not None:
        url = database_url or config.DATABASE_URL
        kwargs: dict = {"future": True}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
            if ":memory:" in url:
                # Share one in-memory DB across all connections/threads (tests).
                kwargs["poolclass"] = StaticPool
        eng = create_engine(url, **kwargs)
        if database_url is not None:
            return eng
        _engine = eng
    return _engine


def init_db(database_url: str | None = None) -> None:
    Base.metadata.create_all(get_engine(database_url))


# --- write paths ---------------------------------------------------------------------
def create_source(
    kind: str,
    *,
    filename: str | None = None,
    fps: float | None = None,
    model_version: str,
    conf_threshold: float,
    engine=None,
) -> int:
    with Session(engine or get_engine()) as session:
        src = Source(
            kind=kind,
            filename=filename,
            fps=fps,
            frame_count=0,
            model_version=model_version,
            conf_threshold=conf_threshold,
        )
        session.add(src)
        session.commit()
        return src.id


def add_detections(
    source_id: int,
    frame_number: int,
    ts_seconds: float,
    detections: list[dict[str, Any]],
    *,
    engine=None,
) -> int:
    """Append detection rows for one frame; bumps the source frame_count. Returns count."""
    if not detections:
        return 0
    with Session(engine or get_engine()) as session:
        rows = [
            DetectionRow(
                source_id=source_id,
                frame_number=frame_number,
                ts_seconds=ts_seconds,
                track_id=d.get("track_id"),
                class_label=d["class_label"],
                class_id=d["class_id"],
                confidence=d["confidence"],
                bbox_x=d["bbox_x"],
                bbox_y=d["bbox_y"],
                bbox_w=d["bbox_w"],
                bbox_h=d["bbox_h"],
            )
            for d in detections
        ]
        session.add_all(rows)
        src = session.get(Source, source_id)
        if src is not None:
            src.frame_count = (src.frame_count or 0) + 1
        session.commit()
        return len(rows)


# --- read paths ----------------------------------------------------------------------
def list_sources(engine=None, limit: int = 50) -> list[dict[str, Any]]:
    with Session(engine or get_engine()) as session:
        rows = session.scalars(
            select(Source).order_by(Source.created_at.desc()).limit(limit)
        ).all()
        return [
            {
                "id": s.id,
                "kind": s.kind,
                "filename": s.filename,
                "fps": s.fps,
                "frame_count": s.frame_count,
                "model_version": s.model_version,
                "conf_threshold": s.conf_threshold,
                "created_at": s.created_at.isoformat() if s.created_at else None,
            }
            for s in rows
        ]


def get_detections(source_id: int, engine=None) -> list[dict[str, Any]]:
    with Session(engine or get_engine()) as session:
        rows = session.scalars(
            select(DetectionRow)
            .where(DetectionRow.source_id == source_id)
            .order_by(DetectionRow.ts_seconds, DetectionRow.id)
        ).all()
        return [
            {
                "id": r.id,
                "frame_number": r.frame_number,
                "ts_seconds": r.ts_seconds,
                "track_id": r.track_id,
                "class_label": r.class_label,
                "class_id": r.class_id,
                "confidence": r.confidence,
                "bbox_x": r.bbox_x,
                "bbox_y": r.bbox_y,
                "bbox_w": r.bbox_w,
                "bbox_h": r.bbox_h,
            }
            for r in rows
        ]


def class_counts(engine=None, source_id: int | None = None) -> list[dict[str, Any]]:
    """Counts per class (for the Nivo bar chart)."""
    with Session(engine or get_engine()) as session:
        stmt = select(
            DetectionRow.class_label, func.count(DetectionRow.id)
        ).group_by(DetectionRow.class_label).order_by(func.count(DetectionRow.id).desc())
        if source_id is not None:
            stmt = stmt.where(DetectionRow.source_id == source_id)
        return [{"class_label": label, "count": n} for label, n in session.execute(stmt)]


def totals(engine=None, source_id: int | None = None) -> dict[str, int]:
    """Totals, optionally scoped to one source/run. A unique object is a distinct
    (source_id, track_id) PAIR (track_id is only unique within a source), so we count
    distinct pairs via a portable subquery rather than COUNT(DISTINCT track_id)."""
    with Session(engine or get_engine()) as session:
        det_q = select(func.count(DetectionRow.id))
        pair_q = (
            select(DetectionRow.source_id, DetectionRow.track_id)
            .where(DetectionRow.track_id.isnot(None))
            .distinct()
        )
        if source_id is not None:
            det_q = det_q.where(DetectionRow.source_id == source_id)
            pair_q = pair_q.where(DetectionRow.source_id == source_id)
            sources = 1 if session.get(Source, source_id) is not None else 0
        else:
            sources = session.scalar(select(func.count(Source.id))) or 0
        return {
            "sources": sources,
            "detections": session.scalar(det_q) or 0,
            "objects": session.scalar(select(func.count()).select_from(pair_q.subquery())) or 0,
        }


def delete_source(source_id: int, engine=None) -> bool:
    """Delete one run and its detections (cascade). Returns False if it didn't exist."""
    with Session(engine or get_engine()) as session:
        src = session.get(Source, source_id)
        if src is None:
            return False
        session.delete(src)  # cascade removes its detection rows
        session.commit()
        return True


def clear_all(engine=None) -> dict[str, int]:
    """Delete ALL runs + detections (user-initiated reset). Returns counts removed."""
    with Session(engine or get_engine()) as session:
        dets = session.query(DetectionRow).delete()  # children first (FK)
        srcs = session.query(Source).delete()
        session.commit()
        return {"sources": srcs, "detections": dets}
