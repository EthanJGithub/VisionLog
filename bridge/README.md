# VisionLog stream bridge (UDP/RTP/RTSP → browser)

Browsers can't read raw UDP and Vercel serverless functions can't hold a socket, so live
network streams need **one always-on worker**. This service runs a single `ffmpeg` process that
ingests a stream, samples it to JPEG frames, and broadcasts them to connected browsers over a
WebSocket. **Detection still runs client-side on the visitor's GPU** — the bridge only moves
pixels, it never loads a model.

```
 camera / drone ──UDP/RTP/RTSP──▶ [ ffmpeg → MJPEG → WebSocket ]  (this worker, always-on)
                                          │  binary JPEG frames
                                          ▼
                          browser: decode → WebGPU YOLO26 → tracker → Postgres
```

## Configure (env vars)

| var | default | meaning |
|-----|---------|---------|
| `STREAM_SOURCE` | `-f lavfi -i testsrc=...` | ffmpeg **input** args. Default is a moving test pattern so the bridge is demoable with no real feed. |
| `STREAM_FPS` | `12` | frames/sec forwarded |
| `STREAM_WIDTH` | `960` | downscale width (keeps aspect) |
| `JPEG_QUALITY` | `6` | ffmpeg mjpeg `-q:v` (2 best … 31 worst) |
| `ALLOW_ORIGINS` | `*` | CORS origins (set to your site for production) |

Real-source examples for `STREAM_SOURCE`:
- Raw UDP / MPEG-TS:  `-i udp://@:1234`
- RTSP camera:        `-rtsp_transport tcp -i rtsp://user:pass@host:554/stream`
- RTP (SDP file):     `-protocol_whitelist file,udp,rtp -i stream.sdp`

## Run locally (Docker)

```bash
cd bridge
docker build -t visionlog-bridge .
# demo test-pattern source:
docker run --rm -p 8000:8000 visionlog-bridge
# or a real UDP source (host networking so UDP reaches the container):
docker run --rm --network host -e STREAM_SOURCE="-i udp://@:1234" visionlog-bridge
```
Check `http://localhost:8000/health`. In the app's **Live stream (UDP)** tab, set the bridge URL
to `ws://localhost:8000/ws` and click **Connect**.

## Deploy (Fly.io — always-on, ~free)

```bash
cd bridge
fly launch --no-deploy            # creates the app from fly.toml + Dockerfile
fly secrets set STREAM_SOURCE="-i udp://@:1234"   # your real source (optional; omit for demo)
fly secrets set ALLOW_ORIGINS="https://vision-log-lilac.vercel.app"
fly deploy
```
Your bridge is then at `https://<app>.fly.dev`; use `wss://<app>.fly.dev/ws` in the app.
For a **raw UDP** source you must also expose a UDP port (`fly ips allocate-v4` + a `[[services]]`
block with `protocol = "udp"`); RTSP/RTP-over-TCP sources need only the default HTTPS port.

Render works too: New → Web Service → this `bridge/` dir (Docker), set the same env vars, and keep
one instance always on (free instances sleep — fine for a demo, not for 24/7 monitoring).

## Notes
- One shared ffmpeg reader fans out to all WebSocket clients (cheap; scales to many viewers).
- ffmpeg auto-restarts if the source drops.
- The browser drops frames while an inference is in flight, so it always processes the freshest
  frame (low latency) rather than building a backlog.
