"""Central configuration for VisionLog.

All tunables live here with pinned, deterministic defaults (CLAUDE.md: "deterministic
where it counts"). Override via environment / .env. No silent fallbacks: if the model
file is missing at startup the engine raises with actionable guidance rather than
quietly degrading.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = os.getenv("VISIONLOG_MODEL", str(ROOT / "models" / "yolo26n.onnx"))

# --- Model / inference ---------------------------------------------------------------
MODEL_VERSION = os.getenv("VISIONLOG_MODEL_VERSION", "yolo26n-onnx")
INPUT_SIZE = int(os.getenv("VISIONLOG_INPUT_SIZE", "640"))  # square letterbox
CONF_THRESHOLD = float(os.getenv("VISIONLOG_CONF_THRESHOLD", "0.35"))

# --- Video / free-tier guardrails (explicit, surfaced in the UI) ---------------------
SAMPLE_FPS = float(os.getenv("VISIONLOG_SAMPLE_FPS", "5"))      # frames analysed/sec
MAX_UPLOAD_MB = float(os.getenv("VISIONLOG_MAX_UPLOAD_MB", "50"))
MAX_DURATION_SECONDS = float(os.getenv("VISIONLOG_MAX_DURATION_SECONDS", "60"))

# --- Storage -------------------------------------------------------------------------
# Postgres (Neon) in prod; SQLite locally / in tests. Explicit, configured choice — not
# a silent fallback. e.g. postgresql+psycopg://user:pw@host/db
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{ROOT / 'data' / 'visionlog.db'}")

# --- App -----------------------------------------------------------------------------
CORS_ORIGINS = os.getenv("VISIONLOG_CORS_ORIGINS", "*").split(",")
