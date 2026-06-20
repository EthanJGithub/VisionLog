# SESSION_STATE — VisionLog (handoff)

_Last updated: 2026-06-20_

## What this is
Ethan's 3rd portfolio project (CV): a browser app where **all detection runs client-side on
the visitor's GPU** (onnxruntime-web + WebGPU; CPU/WASM fallback). Repo (pushed):
**https://github.com/EthanJGithub/VisionLog**, branch `main`. Latest commit ~`83de750`.

## ✅ Complete + committed (verified)
- **Client-side detection, all on the user's GPU** (measured NVIDIA: n≈48, m≈20, x≈11 fps):
  webcam AND file uploads. Models: YOLO26 **n/s/m/x** (NMS-free), **YOLOE-26 open-vocab**
  (69-class bundled vocab, JS NMS), and **YOLO26n-PPE** (fine-tuned). x (223MB) gitignored →
  served on deploy only; UI shows "not deployed" if absent (HEAD check).
- **Robust logging:** client-side tracker (`frontend/src/webgpu/tracker.js`) → stable
  **Object IDs** (`detections.track_id`); confirmation gating (≥3 frames before logging) kills
  false-positive flicker; **confidence slider** (default 0.40). SQL stores track_id + class +
  confidence (+ bbox/frame/ts). Postgres (Neon) prod / SQLite local.
- **Domain fine-tune** (`src/train/finetune.py`): YOLO26n on Construction-PPE,
  **before mAP@50 0.0003 → after 0.458**. Exported ONNX selectable as "YOLO26n-PPE".
- **Evaluation** (`src/eval/benchmark.py`): measured COCO128 mAP + CPU/WebGPU latency →
  `MODEL_CARD.md` + in-app **Benchmarks** tab (per-class AP, confusion matrix, PR curve).
- **Chatbot** (`src/agent/`, "Ask the data" tab): **LangGraph multi-agent text-to-SQL**
  (author→guard→execute→answer + self-correction), **Groq** LLM. Read-only guard (SELECT-only).
  Gated on `GROQ_API_KEY` (+ deps) → 503 otherwise. Mirrors CredAgent pattern.
- **Lighthouse:** Accessibility/Best-Practices/SEO = **100**, Performance **98** local (lazy-load
  ort+nivo, inline CSS, GZip+immutable cache, deferred charts, favicon, aria labels). Expect
  ~100 on a CDN deploy.
- **Tests: 20 pass** (`pytest`), hermetic. Frontend builds clean.

## 🎯 DEPLOY DECISION (made this session — implement next)
**Vercel for the app + Hugging Face Hub CDN for the model files. Chatbot + logging as light
Vercel serverless functions (no torch). Training stays OFFLINE (reproducible script).**
Rationale: fps is client-side (host-agnostic); this is always-on, zero-ops, auto-deploy on git
push, no Docker/LFS. The only thing not hosted is *live* training (needs heavy/GPU/always-on —
incompatible with free+low-ops); the PPE fine-tune + benchmarks already ship.
(Alternative if live-training MUST be hosted: HF Spaces Docker full image — worse on cold-start
/ fps / setup. HF frontmatter already in README; `requirements-full.txt` + `Dockerfile`
INSTALL_FULL exist for that path.)

### NEXT STEPS — implementation (NOT yet done; do these first)
1. **Configurable model CDN base.** In `frontend/src/models.js`, prefix every `url`/`vocabUrl`
   with `const MODELS_BASE = import.meta.env.VITE_MODELS_BASE || "/models"` (so prod points at
   the HF Hub CDN, local stays `/models`). `yolo26-ppe.json`/`yoloe26-vocab.json` too.
2. **Run the chatbot on the Vercel function** (it's torch-free): in `api/index.py` replace the
   `/api/v1/chat` 503 stub with the real route (import `src.agent.chat_graph`; gate on
   `chat_graph.available()`); set `chat_available` in its `/health`. Add to
   `api/requirements.txt`: `langgraph`, `langchain-core`, `langchain-groq`. Add
   `{"functions": {"api/index.py": {"maxDuration": 60}}}` to `vercel.json` (LLM latency).
   Note: api/index.py already reuses `src/store.py` (light, SQLAlchemy only) — no torch pulled.
3. Build + commit + push.

### NEXT STEPS — one-time user actions (then it's auto forever)
- Complete **Vercel sign-in** (OAuth was started; tool `mcp__plugin_vercel_vercel__authenticate`
  → user pastes the `localhost:3118/callback?...` URL into `complete_authentication`). STILL PENDING.
- Create **Neon Postgres** (Vercel Marketplace = 1-click) → `DATABASE_URL`
  (`postgresql+psycopg://...?sslmode=require`).
- Set Vercel env vars: `DATABASE_URL`, `GROQ_API_KEY` (Groq free key), `VITE_MODELS_BASE`
  (the HF Hub resolve URL).
- Upload the ONNX models + their `.json` to a **HF Hub model repo** (one-time; n/s/m/x/vocab/ppe
  from `frontend/public/models/` — present locally, gitignored for the big ones). Then they
  serve from HF's CDN.
- After sign-in, the assistant can drive `vercel deploy` via the Vercel MCP/skills.

## 🔜 Queued (not started)
- **Website training UI** (user chose "BYO-compute"): server-triggered training job (pick/upload
  dataset → train → live progress → export → client inference). Needs the full container backend
  (torch/ultralytics) → only on the HF Spaces path, NOT Vercel. Conflicts with the Vercel
  decision; revisit if they still want it hosted.
- **CI** (GitHub Actions: pytest + frontend build + optional Lighthouse check) — quick, expected.
- Tracking-based analytics (counting/zones); README demo GIF.
- Arbitrary-text open-vocab fully client-side (deferred R&D — see ARCHITECTURE.md §6).

## Verification commands
- `pytest` (20 pass, hermetic). `cd frontend && npm run build`.
- Local full run: `uvicorn src.api.main:app --port 8000` + serves built `dist/`. Detection,
  logging, benchmarks verified in-browser via Playwright. Chatbot returns 503 w/o GROQ key
  (verified); live LLM run needs a key.
- Lighthouse (local): `npx lighthouse@12 http://127.0.0.1:8000/ --chrome-flags="--headless=new"`.

## Gotchas
- **torch is CPU-only** here; training/eval run on CPU (slow but fine). NVIDIA GPU exists but
  torch isn't CUDA — fine since inference is client-side WebGPU.
- ONNX models exported **static 640, opset 12** (WebGPU needs opset 12; dynamic shapes were
  slower). FP16 deferred (post-hoc convert breaks on Resize; needs `half=True` export).
- ultralytics writes scratch to repo root (`*.pt`, `mobileclip_blt.ts` ~599MB, `runs/`,
  `.cache/datasets`) — all gitignored. Don't commit. Open-vocab server path needs the ~570MB
  CLIP encoder (full image only).
- onnxruntime-web wasm loaded from jsDelivr CDN (pinned 1.26.0), set in
  `frontend/src/webgpu/detector.js`.
- Windows: kill stale uvicorn before restart (multiple bg servers caused a stale-code 404 once);
  in-memory SQLite needs StaticPool; temp-file unlink is best-effort.
- Big files: GitHub rejects >100MB (x stays out of git); HF rejects >10MB non-LFS (hence the
  HF-Hub-CDN-for-models decision avoids LFS entirely).
- `.venv` + caches on **D:** (`HF_HOME`/`PIP_CACHE_DIR` → `visionlog/.cache`).
