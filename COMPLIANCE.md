# COMPLIANCE — VisionLog

VisionLog processes **user-supplied video**, which can contain people and faces. This
document states, honestly, how the system handles privacy, explainability, auditability,
licensing, and access control. Each control is marked **Implemented**, **Designed-for
(not enforced)**, or **Roadmap** — no control is described as done when it is not.

---

## 1. PII / biometric data & data minimization

Video of people is potentially **biometric/PII-sensitive**. The design minimizes what is
retained.

| Control | Status | Detail |
|---|---|---|
| Webcam video **never leaves the device** | **Implemented** | Live detection runs entirely in the browser (onnxruntime-web/WebGPU, [frontend/src/webgpu/detector.js](frontend/src/webgpu/detector.js)). Only *sampled detection rows* (class/confidence/bbox) are POSTed to the server — never frames or pixels. Strongest possible data-minimization for the live path. |
| Raw uploaded video is **not persisted** | **Implemented** | The upload is written to a temp file, processed, then deleted in a `finally` block ([src/api/routes.py](src/api/routes.py)). Only structured detections are stored. |
| No face recognition / identity inference | **Implemented** | The model is COCO object detection (class `person`, not *which* person). No biometric template is computed or stored. |
| No raw pixels in the database | **Implemented** | `detections` stores class, confidence, bbox, and timestamps only ([src/store.py](src/store.py)) — no image crops. |
| Retention / TTL purge of old detections | **Roadmap** | Detections persist indefinitely today; a scheduled purge is planned. |
| Upload consent / usage notice in UI | **Designed-for (not enforced)** | The UI states what is logged; an explicit consent gate is not yet enforced. |

---

## 2. Model explainability & model card

| Control | Status | Detail |
|---|---|---|
| Per-detection provenance | **Implemented** | Every `source` row records `model_version` and `conf_threshold`; every `detection` records frame/timestamp/class/confidence/bbox ([src/store.py](src/store.py)), so any result is explainable and reproducible. |
| Confidence surfaced to the user | **Implemented** | Boxes are labelled with class + confidence ([frontend/src/components/BoxOverlay.jsx](frontend/src/components/BoxOverlay.jsx)). |
| Documented model + metrics | **Implemented** | YOLO26n / COCO origin and the NMS-free `(1,300,6)` output contract are documented in [ARCHITECTURE.md](ARCHITECTURE.md). |

---

## 3. Auditability

| Control | Status | Detail |
|---|---|---|
| Append-only detection log | **Implemented** | Detections are inserted, never mutated ([src/store.py](src/store.py)). |
| Reproducible runs | **Implemented** | Pinned thresholds/seeds in [src/config.py](src/config.py) + stored model version per source. |
| Queryable history | **Implemented** | `GET /api/v1/sources` and `/sources/{id}/detections` expose the full record. |

---

## 4. Prompt / open-vocab guardrails

Open-vocabulary (YOLOE-26) takes **user-supplied text prompts** (class names). It is not a
generative LLM — the prompts are encoded into class embeddings, so there is no free-text
generation to jailbreak. Controls:

| Control | Status | Detail |
|---|---|---|
| Prompts validated / bounded | **Implemented** | Empty/whitespace prompts rejected (`422`); prompts are split to a class list, not executed ([src/api/routes.py](src/api/routes.py), [src/detect/openvocab.py](src/detect/openvocab.py)). |
| Open-vocab gated behind capability check | **Implemented** | If the full image (`ultralytics`) isn't present the API returns `503` with guidance — never a silent closed-vocab substitution. |
| Prompt-content policy (e.g. blocking biometric-targeting prompts) | **Roadmap** | No content policy on prompt terms yet. |

---

## 5. Licensing compliance

| Control | Status | Detail |
|---|---|---|
| YOLO26 **AGPL-3.0** obligation | **Implemented** | YOLO26 weights/code are AGPL-3.0. As a network-deployed service this requires offering corresponding source; satisfied because this repo is **public** under AGPL-3.0 ([LICENSE](LICENSE)). For any closed/commercial deployment an Ultralytics Enterprise license would be required instead. |

---

## 6. Access control & secrets

| Control | Status | Detail |
|---|---|---|
| Secrets via env, never committed | **Implemented** | `DATABASE_URL` and config come from env / platform secrets; `.env` is gitignored ([.gitignore](.gitignore), [.env.example](.env.example)). |
| Upload size/duration limits (abuse/DoS guard) | **Implemented** | Enforced in [src/video.py](src/video.py) (`413` on violation). |
| AuthN/AuthZ on endpoints | **Designed-for (not enforced)** | The demo is intentionally open. CORS is configurable ([src/config.py](src/config.py)); API keys / auth would be added for non-demo use. |
| Rate limiting | **Roadmap** | Not implemented; relevant for a public, unauthenticated demo. |
