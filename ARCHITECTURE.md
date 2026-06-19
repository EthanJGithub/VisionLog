# ARCHITECTURE — VisionLog

VisionLog runs **YOLO26 object detection on user-supplied video** and writes **structured
detection logs to PostgreSQL**, with a live React/Nivo dashboard. It uses a **hybrid**
compute model so it is both *free to run* and *fast*:

- **Webcam *and* n/s file uploads → client-side, on the visitor's own GPU** (onnxruntime-web
  + **WebGPU**). Real-time on a capable GPU; falls back to CPU (WASM) with the backend shown.
  If the browser can run inference for the webcam, it can run it for an uploaded file too —
  so n/s uploads never touch the server's compute.
- **Heavy models → server-side** (onnxruntime): **YOLO26 m / x** and **YOLOE-26
  open-vocabulary**. On the free tier the server is **CPU-only (no GPU)**, so these are much
  slower than the in-browser path — surfaced as an explicit UI disclaimer, not hidden.

This document explains *why* each piece is built this way, the data flow, observability,
and known limitations. Claims are grounded in the code and in measured numbers.

---

## 1. Component choices — the "why"

| Decision | Why |
|---|---|
| **Webcam runs in the browser (WebGPU)** | The only way to use the *visitor's* GPU — and it makes the live demo free-scaling (each visitor brings their own compute) and private (raw video never leaves the device). **Measured: YOLO26n at 640px ≈ 47.8 fps on an NVIDIA GPU via WebGPU** (warm, inference only). Verified end-to-end in-browser; output tensor `(1,300,6)`. |
| **opset=12 ONNX export** | Required for onnxruntime-web WebGPU compatibility. Static (fixed 640) export is used because dynamic shapes measurably *reduced* WebGPU throughput (47.8→34.5 fps) without speeding up smaller inputs. |
| **No silent fallback for backend** | The client picks WebGPU if `navigator.gpu` yields an adapter, else WASM/CPU — and the UI badge says which, plus live fps. CPU mode is labelled "not real-time". |
| **Uploads run server-side** | Bigger models (m/x) and open-vocab are too heavy to ship to every browser, and the upload path must work even without WebGPU. Lean **onnxruntime** runtime; `ultralytics`/torch are dev-only for the closed-vocab path. |
| **YOLO26 is end-to-end / NMS-free** | The one-to-one head emits final detections `(1,300,6)` = `[x1,y1,x2,y2,score,class]`, so neither the JS nor the Python path needs NMS post-processing. |
| **Open-vocab = YOLOE-26** | After `set_classes()` the text prompts are re-parameterized into a standard model, so inference runs at normal YOLO speed. Needs `ultralytics` + a ~570MB CLIP text encoder at runtime → server-side, "full image" only (see below). |
| **SQLAlchemy over DATABASE_URL** | Postgres (Neon) in prod, SQLite locally/in tests — one code path, explicit config, no silent fallback ([src/store.py](src/store.py)). |
| **Uploads don't re-encode video** | Detections are keyed by `ts_seconds`; the frontend overlays boxes on the native `<video>` synced to `currentTime` ([frontend/src/components/VideoUpload.jsx](frontend/src/components/VideoUpload.jsx)). |

---

## 2. Data flow

```
                         ┌──────────────────────── Browser (Vite/React/Nivo) ───────────────────────┐
                         │                                                                            │
  Webcam (client-side):  │  getUserMedia → ClientDetector (onnxruntime-web, WebGPU/WASM)             │
                         │      frontend/src/webgpu/detector.js  → boxes drawn live (real-time)      │
                         │      sampled detections (~1/sec) ──POST /client-sessions/{id}/detections──┐
                         │                                                                           │ │
  Upload (server-side):  │  POST /api/v1/sources (file + model[, prompts]) ──────────────┐          │ │
                         └───────────────────────────────────────────────────────────────┼──────────┘ │
                                                                                          ▼            ▼
                                                              FastAPI (src/api/routes.py)  ───────────────
                                                                  │                                      │
                                          ┌───────────────────────┼─────────────────────────┐          │
                                   src/video.py            src/detect/registry.py      src/detect/      │
                                   decode+sample           onnxruntime n/s/m/x          openvocab.py     │
                                   size/duration guard     (lean)                       YOLOE-26 (full)  │
                                                                  │                                      │
                                                          src/store.py → Postgres / SQLite ◀────────────┘
                                                          sources + detections (model_version + threshold)
                                                                  │
                                          Dashboard ← GET /stats, /sources, /sources/{id}/detections
```

---

## 3. Models

| Model | Where it runs | Notes |
|---|---|---|
| YOLO26n (9.5 MB) | client **and** server | Default. Real-time webcam on most GPUs. |
| YOLO26s (37 MB) | client **and** server | More accurate; needs a stronger GPU for real-time webcam. |
| YOLO26m / x | server only | Larger/slower; upload path. Export with `python -m src.detect.export --all`. |
| YOLOE-26 (open-vocab) | server only ("full" image) | Text-prompt detection of arbitrary classes via [src/detect/openvocab.py](src/detect/openvocab.py). Needs `requirements-full.txt` + first-run CLIP download (~570 MB). |

Client model registry: [frontend/src/models.js](frontend/src/models.js). Server registry +
availability: [src/detect/registry.py](src/detect/registry.py). `/api/v1/health` reports
which server models are present and whether open-vocab is available — no silent gaps.

---

## 4. Determinism, guardrails, observability

- Pinned thresholds/sizes in [src/config.py](src/config.py); every `source` row stores the
  `model_version` + `conf_threshold` used → each run is reproducible/auditable.
- Free-tier guardrails are explicit: uploads over `MAX_UPLOAD_MB`/`MAX_DURATION_SECONDS`
  get `413` ([src/video.py](src/video.py)); webcam logging is **sampled** (~1/sec) to keep
  the DB lean even at 45+ fps.
- `GET /api/v1/health` (backends/models/limits), `/stats`, `/sources`,
  `/sources/{id}/detections`.

---

## 5. Deployment

- **Full app (free):** single Docker image ([Dockerfile](Dockerfile)) builds the Vite
  bundle and serves API + SPA + the client `/models/*.onnx` on `:7860` for
  **Hugging Face Spaces (free CPU)**. The webcam path needs no server compute (runs on the
  visitor's GPU); the server only handles uploads + logging. DB = **Neon free Postgres**
  via `DATABASE_URL`.
- **Client-only demo (free):** because the webcam path is pure static assets, the frontend
  can also be deployed to any static host (Vercel/Pages); logging just needs the API reachable.
- **Open-vocab:** requires the "full" image (`requirements-full.txt`) — not part of the free
  lean deploy.

---

## 6. Known limitations & roadmap

- **Server upload path on free CPU** is still CPU-bound (sampled at `SAMPLE_FPS`); an
  optional CUDA onnxruntime backend can be enabled where GPU hardware exists.
- **Open-vocab cost:** the YOLOE CLIP text encoder is ~570 MB; impractical on the smallest
  free tiers, hence "full image" only.
- **Tracking/segmentation** (object IDs across frames) not implemented — detection only.
- **Client-side open-vocab** (fixed-prompt YOLOE in the browser) is a natural next step.
