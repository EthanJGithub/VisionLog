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

- **Object IDs (tracking).** A lightweight client-side tracker
  ([frontend/src/webgpu/tracker.js](frontend/src/webgpu/tracker.js)) — greedy IoU,
  class-aware association — assigns a **stable `track_id`** to each physical object across
  frames. Persisted as the `detections.track_id` column (the "Object ID").
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
- **Tracking** (object IDs across frames, e.g. ByteTrack → counting) not implemented.
- **FP16 client models** (smaller download, faster) — pending a proper `half=True` export
  (a naive post-hoc float16 convert breaks on the model's `Resize` ops).
