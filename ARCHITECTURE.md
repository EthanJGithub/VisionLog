# ARCHITECTURE — VisionLog

VisionLog runs **YOLO26 object detection on user-supplied video** and writes **structured
detection logs to PostgreSQL**, with a live React/Nivo dashboard. It uses a **hybrid**
compute model so it is both *free to run* and *fast*:

- **Everything → client-side, on the visitor's own GPU** (onnxruntime-web + **WebGPU**),
  webcam and uploads alike. Real-time on a capable GPU; falls back to CPU (WASM) with the
  backend shown. Measured: n≈48, m≈20, x≈11 fps (NVIDIA).
- **All five models run in-browser:** YOLO26 **n/s/m/x** (NMS-free end-to-end head) and
  **YOLOE-26 open-vocabulary** (a curated vocabulary baked into a 41 MB ONNX; one-to-many
  head decoded with JS-side NMS — `frontend/src/webgpu/detector.js`). No CLIP text encoder
  is shipped to the browser; instead the vocabulary's text embeddings are baked in at export.
- **Optional server path** (`src/detect/{ultra,openvocab}.py`) remains for *arbitrary* typed
  open-vocab prompts (needs the CLIP encoder) and as a no-WebGPU fallback. Off by default;
  CPU-only/slow on the free tier — surfaced as an explicit disclaimer, not hidden.

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
  Webcam (client-side):  │  getUserMedia ─┐                                                          │
  Live stream (UDP):     │  WebSocket ◀ bridge worker (ffmpeg) ─┼─▶ ClientDetector (onnxruntime-web, WebGPU/WASM) │
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

| Model | Size | Where it runs | Notes |
|---|---|---|---|
| YOLO26n | 9.5 MB | client (GPU) | Default. ~48 fps. Committed. |
| YOLO26s | 37 MB | client (GPU) | ~real-time on a decent GPU. Committed. |
| YOLO26m | 82 MB | client (GPU) | ~20 fps. Committed. |
| YOLO26x | 223 MB | client (GPU) | ~10 fps; needs a strong GPU. Exceeds GitHub's 100 MB limit → served on deploy, not committed (UI shows unavailable if absent). |
| YOLOE-26 open-vocab | 41 MB | client (GPU) | Curated vocabulary baked in; client-side, no CLIP encoder. Committed (`yoloe26-vocab.onnx` + `.json`). |
| YOLOE-26 (arbitrary prompts) | — | server (optional) | True free-text prompts; needs the CLIP encoder (~570 MB, `requirements-full.txt`). Off by default. |

Client model registry: [frontend/src/models.js](frontend/src/models.js). Server registry +
availability: [src/detect/registry.py](src/detect/registry.py). `/api/v1/health` reports
which server models are present and whether open-vocab is available — no silent gaps.

---

## 3a. Robust logging — tracking + confirmation

To keep the SQL data "mostly correct", logged detections are not raw per-frame boxes:

- **Object IDs (tracking).** A lightweight client-side **SORT-style** tracker
  ([frontend/src/webgpu/tracker.js](frontend/src/webgpu/tracker.js)) — class-aware, with a
  constant-velocity motion model: each track is predicted forward by its velocity and matched on
  the *predicted* box (IoU), with a center-distance/size fallback for fast or briefly-occluded
  objects. This assigns a **stable `track_id`** to each physical object across frames (pure
  frame-to-frame IoU re-IDed anything moving faster than ~its own width/frame). Persisted as the
  `detections.track_id` column (the "Object ID").
- **Confirmation gating.** A track must be seen in ≥ `minHits` (default 3) frames before any
  of its detections are logged — this drops single-frame false-positive flicker.
- **Adjustable confidence.** A UI slider (default **0.40**) sets the detection threshold live
  (applied in `detector.js`, no model reload). Higher = fewer false positives in the log.

Each logged row therefore carries **Object ID (`track_id`), class, confidence** (+ bbox,
frame, timestamp). `totals.objects` = distinct tracked objects.

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

**Live: https://vision-log-lilac.vercel.app** (Vercel, GitHub-connected, auto-deploys on push).

- **App = Vercel** ([vercel.json](vercel.json)): `@vercel/static-build` serves the Vite SPA +
  client `/models/*.onnx` at the root; `@vercel/python` runs the FastAPI logging + chatbot
  function ([api/index.py](api/index.py)). All detection runs on the visitor's GPU, so the
  function is torch-free and tiny (`excludeFiles: frontend/**`). DB = **Neon free Postgres**
  via `DATABASE_URL` (auto-pinned to the psycopg v3 driver in [src/config.py](src/config.py)).
- **Chatbot** runs on the same function, gated on `GROQ_API_KEY` (LangGraph + Groq, no torch).
- **`x` (223 MB)** exceeds GitHub's 100 MB limit → not bundled; missing `/models/*` returns a
  real 404 and the UI says "not deployed". To enable it, host it on a CDN (e.g. HF Hub) and set
  `VITE_MODELS_BASE`.
- **Live-stream bridge (optional, always-on):** [bridge/](bridge/) — an ffmpeg→WebSocket worker
  that relays a UDP/RTP/RTSP stream to the browser; **not** serverless (Vercel can't hold a
  socket), so it deploys separately (Fly.io/Render). Detection still runs client-side; set the
  bridge URL in the **Live stream (UDP)** tab. See [bridge/README.md](bridge/README.md).
- **Open-vocab (arbitrary prompts):** requires the "full" image (`requirements-full.txt`) — not
  part of the Vercel deploy.

---

## 6. Known limitations & roadmap

- **Open-vocab is a baked 69-class vocabulary, not arbitrary free text.** Fully client-side
  (no CLIP encoder shipped). To detect new classes you re-export `export_vocab.py` with an
  updated `VOCAB`.
- **Arbitrary typed prompts fully client-side — investigated, deferred (real R&D).** Spike
  findings: YOLOE's MobileCLIP text encoder is loaded *transiently* from a ~570 MB TorchScript
  inside `get_text_pe` and released (`clip_model` is `None` afterward), so it isn't a clean
  module to export. Truly-arbitrary client-side text would need (a) extracting that text tower
  to a browser ONNX + replicating its tokenizer in JS, and (b) a *text-conditioned* YOLOE ONNX
  that takes embeddings as a runtime input (standard export bakes classes in). Both are
  multi-step with real uncertainty. An optional server path for arbitrary prompts exists
  (`src/detect/openvocab.py`), off by default.
- **Tracking** is a SORT-style motion tracker (constant-velocity + IoU/center association), not
  a full appearance-model tracker (e.g. ByteTrack/Re-ID) — IDs can still switch under heavy
  mutual occlusion of same-class objects. Tracking-based analytics (counting/zones) are roadmap.
- **FP16 client models** (smaller download, faster) — pending a proper `half=True` export
  (a naive post-hoc float16 convert breaks on the model's `Resize` ops).
