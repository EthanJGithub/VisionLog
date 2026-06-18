"""Video decoding + frame sampling for uploaded files (OpenCV, headless).

Enforces the free-tier guardrails explicitly (size + duration caps) and samples frames
at a fixed rate so CPU inference stays within budget. Raises ValueError with a clear
message on violations — surfaced to the user, never silently truncated.
"""
from __future__ import annotations

import tempfile
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from src import config


class VideoTooLargeError(ValueError):
    pass


def save_upload(data: bytes, suffix: str = ".mp4") -> Path:
    max_bytes = int(config.MAX_UPLOAD_MB * 1024 * 1024)
    if len(data) > max_bytes:
        raise VideoTooLargeError(
            f"Upload is {len(data) / 1e6:.1f} MB; the free-tier limit is "
            f"{config.MAX_UPLOAD_MB:.0f} MB. Please upload a shorter/smaller clip."
        )
    tmp = Path(tempfile.mkstemp(suffix=suffix)[1])
    tmp.write_bytes(data)
    return tmp


def probe(path: Path) -> dict[str, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise ValueError("Could not decode the uploaded video. Is it a valid video file?")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    duration = (frames / fps) if fps > 0 else 0.0
    return {"fps": fps, "frame_count": frames, "duration": duration}


def iter_sampled_frames(
    path: Path, sample_fps: float | None = None
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_number, ts_seconds, BGR frame) sampled at ~sample_fps.

    Enforces MAX_DURATION_SECONDS up front (no silent truncation).
    """
    sample_fps = sample_fps or config.SAMPLE_FPS
    meta = probe(path)
    if meta["duration"] > config.MAX_DURATION_SECONDS + 0.5:
        raise VideoTooLargeError(
            f"Video is {meta['duration']:.0f}s; the free-tier limit is "
            f"{config.MAX_DURATION_SECONDS:.0f}s. Please upload a shorter clip."
        )

    cap = cv2.VideoCapture(str(path))
    src_fps = meta["fps"] or 30.0
    step = max(1, int(round(src_fps / sample_fps)))
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                ts = idx / src_fps
                yield idx, round(ts, 3), frame
            idx += 1
    finally:
        cap.release()


def decode_jpeg(data: bytes) -> np.ndarray:
    """Decode a JPEG/PNG byte buffer (webcam frame) to a BGR ndarray."""
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode webcam frame.")
    return frame
