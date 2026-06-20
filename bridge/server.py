"""VisionLog UDP→browser bridge (always-on worker; NOT serverless).

Browsers can't read raw UDP and Vercel functions can't hold a socket, so this small service
is the missing piece for live streams: it runs ONE ffmpeg process that ingests a stream
(UDP/RTP/RTSP/…), samples it to JPEG frames, and broadcasts them to connected browsers over a
WebSocket. Detection still runs CLIENT-SIDE on the visitor's GPU — this only moves pixels, it
never runs a model. Deploy it on any always-on host (Fly.io/Render/a VM); see README.md.

Config via env:
  STREAM_SOURCE  ffmpeg input args. Default = a moving test pattern so the bridge is demoable
                 with no real feed. Real examples:
                   "-i udp://@:1234"                         (raw UDP / MPEG-TS)
                   "-rtsp_transport tcp -i rtsp://host/path"  (RTSP camera)
  STREAM_FPS     frames/sec to forward (default 12 — plenty for detection, easy on bandwidth)
  STREAM_WIDTH   downscale width in px, keeps aspect (default 960)
  JPEG_QUALITY   ffmpeg mjpeg -q:v, 2(best)..31(worst) (default 6)
  ALLOW_ORIGINS  comma-separated CORS origins (default *)
"""
from __future__ import annotations

import asyncio
import os
import shlex
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

FPS = float(os.getenv("STREAM_FPS", "12"))
WIDTH = int(os.getenv("STREAM_WIDTH", "960"))
QUALITY = os.getenv("JPEG_QUALITY", "6")
SOURCE = os.getenv("STREAM_SOURCE", "-f lavfi -i testsrc=size=1280x720:rate=30")
ORIGINS = [o.strip() for o in os.getenv("ALLOW_ORIGINS", "*").split(",") if o.strip()]

_clients: set[WebSocket] = set()
_SOI, _EOI = b"\xff\xd8", b"\xff\xd9"  # JPEG start/end-of-image markers


def _ffmpeg_cmd() -> list[str]:
    # -fflags nobuffer + -flags low_delay keep latency down on live sources.
    cmd = (
        f"ffmpeg -hide_banner -loglevel error -fflags nobuffer -flags low_delay {SOURCE} "
        f"-vf fps={FPS},scale={WIDTH}:-2 -f mjpeg -q:v {QUALITY} pipe:1"
    )
    return shlex.split(cmd)


async def _broadcast(frame: bytes) -> None:
    for ws in list(_clients):
        try:
            await ws.send_bytes(frame)
        except Exception:
            _clients.discard(ws)


async def _reader() -> None:
    """Run ffmpeg, split its MJPEG stdout into JPEGs, broadcast. Restart if the source drops."""
    while True:
        try:
            proc = await asyncio.create_subprocess_exec(
                *_ffmpeg_cmd(),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except FileNotFoundError:
            raise RuntimeError("ffmpeg not found — install it (the Docker image does).")
        buf = bytearray()
        assert proc.stdout is not None
        while True:
            chunk = await proc.stdout.read(65536)
            if not chunk:
                break  # ffmpeg exited (source ended/dropped) — restart below
            buf += chunk
            while True:
                start = buf.find(_SOI)
                end = buf.find(_EOI, start + 2) if start >= 0 else -1
                if start < 0 or end < 0:
                    break
                if _clients:
                    await _broadcast(bytes(buf[start:end + 2]))
                del buf[: end + 2]
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        await asyncio.sleep(1)  # backoff before reconnecting to the source


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_reader())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(title="VisionLog stream bridge", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=ORIGINS, allow_methods=["*"], allow_headers=["*"]
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "source": SOURCE, "fps": FPS, "width": WIDTH, "clients": len(_clients)}


@app.websocket("/ws")
async def ws(ws: WebSocket) -> None:
    await ws.accept()
    _clients.add(ws)
    try:
        while True:
            await ws.receive_text()  # we only push frames; reads just detect disconnect
    except WebSocketDisconnect:
        pass
    finally:
        _clients.discard(ws)
