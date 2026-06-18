# VisionLog — single-image deploy for Hugging Face Spaces (Docker, free CPU tier).
# Builds the Vite frontend, then serves API + static SPA from one FastAPI app on :7860.

# --- Stage 1: build the React/Vite frontend -----------------------------------------
FROM node:18-slim AS frontend
WORKDIR /fe
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# --- Stage 2: runtime ----------------------------------------------------------------
FROM python:3.13-slim
WORKDIR /app

# OpenCV (headless) needs libGL + glib at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/
COPY --from=frontend /fe/dist ./frontend/dist

# HF Spaces routes traffic to 7860.
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
