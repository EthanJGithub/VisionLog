# Always-on stream bridge: needs ffmpeg, so it ships as a container (unlike the serverless API).
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY server.py .

ENV PORT=8000 STREAM_FPS=12 STREAM_WIDTH=960
EXPOSE 8000
# Single worker: one shared ffmpeg reader fans out to all WebSocket clients.
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT}"]
