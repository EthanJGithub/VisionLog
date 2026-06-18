# SESSION_STATE — VisionLog

_Last updated: 2026-06-18_

## What this is
Third portfolio project (after CredAgent, FraudPulse): a **computer-vision** web app —
YOLO26 detection on user video (upload + webcam) → detection logs in PostgreSQL → live
Vite/React/Nivo dashboard. Built to deploy **free** (HF Spaces CPU + Neon Postgres).
Full intent in `../handoff docs/CV-PROJECT-HANDOFF.md`; design in
`C:\Users\ethan\.claude\plans\moonlit-crafting-yao.md`.

## State: core pipeline COMPLETE and verified end-to-end (local). Not yet deployed.

### Accomplished this session
- Scaffolded `visionlog/` mirroring FraudPulse conventions (`src/{api,detect}`, `frontend`).
- **Exported YOLO26n → `models/yolo26n.onnx`** (9.5 MB, committed). Confirmed output is
  end-to-end `(1,300,6)` = `[x1,y1,x2,y2,score,class]` (NMS-free).
- Backend: `config.py`, `detect/engine.py` (onnxruntime CPU), `detect/export.py`,
  `video.py` (decode/sample + size/duration guards), `store.py` (SQLAlchemy,
  Postgres/SQLite), `api/{main,routes,schemas}.py` (upload + WebSocket webcam).
- Frontend: Vite+React+Nivo — upload w/ box overlay synced to video time, live webcam
  (throttled ~4 fps over WS), Nivo bar (class counts) + line (timeline), source feed,
  health bar. **Builds clean.**
- Governance: `ARCHITECTURE.md`, `COMPLIANCE.md`, `README.md`, AGPL-3.0 `LICENSE`,
  `Dockerfile` (HF Spaces, :7860), `.env.example`, `.gitignore`.

### Verification (commands used)
- `pytest` → **7 passed, 1 warning** (engine, store, api; hermetic). Engine test runs
  against the real ONNX model.
- Engine on real `tests/fixtures/bus.jpg` → bus 0.925 + 4 person (0.52–0.92), correct bboxes.
- **Live server** (`uvicorn src.api.main:app --port 8000`): `POST /api/v1/sources` with a
  3 s sample video → 15 frames @5fps, **75 detections logged** (60 person + 15 bus);
  `GET /api/v1/stats` and direct SQLite query both confirm 75 rows persisted.
- Frontend `npm run build` → OK (442 kB JS). Served SPA at `/` returns 200.

### Known gaps / next steps (priority order)
1. **Browser visual check of the box overlay** is the one unverified piece — Playwright
   MCP was locked by another session (`Browser is already in use`), so the live overlay
   wasn't screenshotted. Logic is sound (API fully verified); confirm visually:
   `uvicorn ... :8000` + `cd frontend && npm run dev`, upload `tests/fixtures/sample_bus.mp4`.
2. **Verify the Postgres path** end-to-end (only SQLite exercised so far). `psycopg 3.3.4`
   is installed and the code is driver-agnostic; point `DATABASE_URL` at a Neon DB and
   re-run an upload.
3. **Deploy:** push to a public GitHub repo, create an HF Space (Docker). HF needs a
   README with frontmatter on the Space: `---\nsdk: docker\napp_port: 7860\n---`. Set
   `DATABASE_URL` (Neon, `postgresql+psycopg://...?sslmode=require`) as a Space secret.
4. **Roadmap differentiator (deferred):** open-vocabulary / NL "detect only X" control to
   bridge CV + LLM-agent skills (noted in ARCHITECTURE.md §6).

### Env gotchas
- Windows: temp-file unlink after `cv2.VideoCapture` can `PermissionError`; cleanup is
  best-effort (`except PermissionError: pass`) in `routes.py` + test fixture.
- In-memory SQLite needs `StaticPool` (in `store.get_engine`) so all threads share one DB.
- `.venv` on D:, caches under `visionlog/.cache` (HF_HOME/PIP_CACHE_DIR) per CLAUDE.md.
- Vite dev proxy targets `http://127.0.0.1:8000` with `ws: true` for the webcam stream.
