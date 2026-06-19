<!-- Hugging Face Spaces reads this frontmatter when this repo is deployed as a Docker Space. -->
---
title: VisionLog
emoji: 🎥
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: agpl-3.0
---

# VisionLog

**YOLO26 object detection on your video → structured detection logs in PostgreSQL →
live, browsable dashboard.** A **hybrid** computer-vision app:

- **Everything runs in your browser, on your own GPU** (onnxruntime-web + **WebGPU**),
  webcam and uploaded files alike. Measured on an NVIDIA card: **n≈48, m≈20, x≈11 fps**;
  automatic CPU/WASM fallback. Your video never leaves your device — only sampled detections
  are logged.
- **Model picker:** YOLO26 **n / s / m / x** (speed↔accuracy) plus **YOLOE-26
  open-vocabulary** — a curated vocabulary (incl. classes COCO lacks: hard hat, forklift,
  pallet, license plate…) baked into a 41 MB ONNX, with a click-to-pick class filter. Fully
  client-side; **no server inference and no 570 MB CLIP text encoder.**
- An optional server path (open-vocab with *arbitrary* typed prompts via ultralytics) exists
  but is off by default; on a free tier it's CPU-only/slow (flagged in the UI).

Built to run **free**: the webcam path uses the visitor's GPU (no server compute), and the
server (uploads + logging) fits a free Hugging Face Space + Neon free Postgres.

> Part of a production-focused ML/AI portfolio alongside CredAgent (multi-agent loan
> underwriting) and FraudPulse (real-time fraud detection). See
> [ARCHITECTURE.md](ARCHITECTURE.md), [COMPLIANCE.md](COMPLIANCE.md), and
> [MODEL_CARD.md](MODEL_CARD.md) (measured mAP + latency; reproduce with
> `python -m src.eval.benchmark`). Live in-app **Benchmarks** tab.

---

## Stack

- **Detector:** YOLO26 (Ultralytics, COCO) — end-to-end / NMS-free, opset-12 ONNX; YOLOE-26
  for open-vocabulary.
- **Client inference:** onnxruntime-web (WebGPU / WASM) — [frontend/src/webgpu/detector.js](frontend/src/webgpu/detector.js).
- **Server inference:** onnxruntime (lean); `ultralytics`/torch are dev-only except for the
  optional open-vocab "full" image.
- **API:** FastAPI (server upload + client-detection logging).
- **Storage:** SQLAlchemy → PostgreSQL (prod) / SQLite (local & tests).
- **Frontend:** Vite + React + Nivo.

## Quickstart (local)

```bash
# 1. Backend
python -m venv .venv && .venv/Scripts/activate      # (Windows) or source .venv/bin/activate
pip install -r requirements-dev.txt
python -m src.detect.export                          # one-time: export YOLO26n -> models/yolo26n.onnx
uvicorn src.api.main:app --reload --port 8000

# 2. Frontend (separate terminal)
cd frontend && npm install && npm run dev            # http://127.0.0.1:5175 (proxies /api to :8000)
```

Open the dev URL, upload a clip or start the webcam, and watch detections appear and log.

## Tests

```bash
pytest        # hermetic: no network, no GPU; engine test runs if the ONNX model is present
```

## Configuration

See [.env.example](.env.example). Key vars: `DATABASE_URL` (Neon Postgres in prod),
`VISIONLOG_CONF_THRESHOLD`, `VISIONLOG_SAMPLE_FPS`, `VISIONLOG_MAX_UPLOAD_MB`,
`VISIONLOG_MAX_DURATION_SECONDS`.

## Deploy (free)

- **App:** Hugging Face Spaces (Docker). The [Dockerfile](Dockerfile) builds the frontend
  and serves API + SPA on `:7860`.
- **Database:** create a free Neon Postgres DB and set `DATABASE_URL` as a Space secret
  (`postgresql+psycopg://...?sslmode=require`).

## License

AGPL-3.0 ([LICENSE](LICENSE)) — required because YOLO26 is AGPL-3.0. See
[COMPLIANCE.md](COMPLIANCE.md) §5.
