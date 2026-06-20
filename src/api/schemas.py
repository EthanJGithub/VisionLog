"""Pydantic response models for the VisionLog API."""
from __future__ import annotations

from pydantic import BaseModel


class DetectionCore(BaseModel):
    track_id: int | None = None  # Object ID — stable across frames (from the tracker)
    class_label: str
    class_id: int
    confidence: float
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float


class DetectionOut(DetectionCore):
    id: int | None = None
    frame_number: int
    ts_seconds: float


class SourceOut(BaseModel):
    id: int
    kind: str
    filename: str | None = None
    fps: float | None = None
    frame_count: int
    model_version: str
    conf_threshold: float
    created_at: str | None = None


class UploadResult(BaseModel):
    source: SourceOut
    frames_analysed: int
    detections_logged: int
    detections: list[DetectionOut]


class ClassCount(BaseModel):
    class_label: str
    count: int


class StatsOut(BaseModel):
    totals: dict[str, int]
    class_counts: list[ClassCount]


class HealthOut(BaseModel):
    status: str
    model_version: str
    model_loaded: bool
    sample_fps: float
    max_upload_mb: float
    max_duration_seconds: float
    server_models: dict[str, bool] = {}
    open_vocab_available: bool = False
    chat_available: bool = False


# --- Client-side (WebGPU) logging ---------------------------------------------------
class ClientSessionIn(BaseModel):
    kind: str = "webcam"
    model_version: str
    conf_threshold: float = 0.35


class ClientFrameIn(BaseModel):
    frame_number: int
    ts_seconds: float
    detections: list[DetectionCore]


class ClientLogIn(BaseModel):
    items: list[ClientFrameIn]


class ClientLogResult(BaseModel):
    logged: int


# --- Chatbot (LangGraph + Groq text-to-SQL over the detections DB) -------------------
class ChatIn(BaseModel):
    question: str


class ChatOut(BaseModel):
    answer: str
    intent: str | None = None  # which agent handled it (analytics/overview/schema/out_of_scope)
    sql: str | None = None
    rows: list[dict] = []
    attempts: int = 0
    error: str | None = None
