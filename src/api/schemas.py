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
    thumb: str | None = None  # optional small JPEG data URL, sent once per new confirmed track


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
    # optional CLIP text embedding of the question (computed client-side) — enables semantic
    # visual search in the gallery agent without running CLIP on the server.
    query_embedding: list[float] | None = None


class ObjectCropOut(BaseModel):
    source_id: int
    track_id: int
    class_label: str
    confidence: float
    thumb: str
    similarity: float | None = None  # set by semantic search (cosine), else None
    caption: str | None = None       # set by the Groq-vision caption agent, else None


class CropEmbeddingItem(BaseModel):
    track_id: int
    embedding: list[float]


class CropEmbeddingsIn(BaseModel):
    items: list[CropEmbeddingItem]


class CropEmbeddingsResult(BaseModel):
    updated: int


class ChatOut(BaseModel):
    answer: str
    intent: str | None = None  # which agent handled it (analytics/overview/schema/gallery/...)
    sql: str | None = None
    rows: list[dict] = []
    crops: list[ObjectCropOut] = []  # visual recall: object thumbnails for the gallery agent
    attempts: int = 0
    error: str | None = None
