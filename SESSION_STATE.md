# SESSION_STATE — VisionLog

_Last updated: 2026-06-18_

## What this is
Third portfolio project (after CredAgent, FraudPulse): a CV web app where **all inference
runs client-side on the visitor's GPU** (onnxruntime-web + WebGPU; CPU/WASM fallback).
- **Webcam AND uploads → in-browser on the user's GPU.** Verified NVIDIA: n≈48, m≈20, x≈11 fps.
- **All 5 models client-side:** YOLO26 n/s/m/x (NMS-free) + **YOLOE-26 open-vocab** (curated
  vocab baked into 41MB ONNX, YOLOv8 head + JS NMS — no CLIP encoder, no server). Open-vocab
  "prompting" = class-filter chips. x (223MB) served on deploy only (GitHub 100MB limit).
- Optional server path (ultra.py/openvocab.py) for arbitrary typed prompts / no-WebGPU, off by default.
- Detections logged to PostgreSQL (SQLite locally). Vite/React/Nivo dashboard.

Repo: **https://github.com/EthanJGithub/VisionLog** (pushed, branch `main`).

## State: built + verified locally + pushed to GitHub. Live cloud deploy = one auth step away.

### Verified this session (commands + numbers)
- `pytest` → **8 passed** (engine, store, api incl. client-logging + model-select upload).
- **In-browser WebGPU inference proven** (Playwright + real NVIDIA GPU): YOLO26n loads via
  onnxruntime-web, output `(1,300,6)`, **≈47.8 fps warm at 640px** (20.9 ms/frame).
  Confirmed dynamic-shape export is slower → kept static 640 (fastest).
- **Server upload** (live API): `model=yolo26s` over sample video → 90 detections logged;
  `/models/*.onnx` served (200); `/stats` aggregates.
- **Open-vocab (YOLOE-26)** proven in Python: prompts `bus,person,backpack` → correct dets.
  NOTE: first use downloads a **~570 MB CLIP text encoder** → "full image" only, not free-tier.
- UI verified via screenshots: dashboard, model pickers, webcam tab, Nivo charts.

### Architecture notes
- Client detector: [frontend/src/webgpu/detector.js](frontend/src/webgpu/detector.js)
  (WebGPU→WASM fallback, backend shown in UI, no silent fallback).
- Webcam logs **sampled** detections (~1/sec) via `/client-sessions` to keep DB lean at 45+fps.
- Server models: [src/detect/registry.py](src/detect/registry.py) (n/s exported; m/x export on
  demand: `python -m src.detect.export --all`). Open-vocab: [src/detect/openvocab.py](src/detect/openvocab.py).
- Client models committed in BOTH `models/` (server) and `frontend/public/models/` (client).

### Deploy — next steps (needs YOUR cloud login; `gh`/HF token not available in this env)
**Recommended: Hugging Face Spaces.** Repo README has the HF Docker frontmatter
(`sdk: docker`, `app_port: 7860`).
- **Lean free build (default):** webcam + n/s uploads run on the visitor's GPU; server does
  logging only. m/x/open-vocab return a clear 503.
- **Full build (m/x + open-vocab):** build the image with `--build-arg INSTALL_FULL=1`
  (installs ultralytics). Heavier + downloads weights on first use; server is CPU-only on free
  tier so m/x/open-vocab are slow (by design, flagged in UI). Verified locally:
  `yolo26x-ultralytics` upload logged 75 dets; YOLOE-26 prompts `bus,person,backpack` correct.
1. Create a Neon free Postgres DB → copy its `postgresql+psycopg://...?sslmode=require` URL.
2. Create a new HF Space (SDK: Docker). Add a secret `DATABASE_URL` = that URL.
3. Push this repo to the Space's git remote (or use the HF GitHub sync). Build serves on :7860.
   - Webcam works on the visitor's GPU with zero server compute; uploads run on the Space CPU.
   - Open-vocab on the Space needs the "full image" (swap to `requirements-full.txt`) + ~570MB
     CLIP download — skip for the lean free Space.
**Alt: Vercel static** for an instant live *webcam-only* demo (client-side WebGPU). Logging/
upload need the API, so this is a partial demo unless the API is also hosted.

### Env gotchas
- ultralytics downloads stray `*.pt` + `mobileclip_blt.ts` (599 MB) into CWD on export/open-vocab
  → gitignored. Don't commit them.
- onnxruntime-web wasm/jsep loaded from jsDelivr CDN (pinned 1.26.0) — set in detector.js.
- Models exported at fixed 640, opset=12 (WebGPU requirement). Dynamic shapes were slower.
- In-memory SQLite needs `StaticPool`; Windows temp unlink is best-effort.
- `.venv` on D:, caches under `visionlog/.cache`.
