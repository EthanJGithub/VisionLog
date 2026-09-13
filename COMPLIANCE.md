# VisionLog ? privacy and deployment boundaries

Browser inference processes video locally. Logging sends sampled detection rows; representative object crops and embeddings are also sent when object recall is enabled. The optional server-upload path processes a temporary video and attempts to remove it afterward. These paths must not be described as storing no pixels.

## Workspace isolation

The browser generates a 256-bit capability retained in local storage, sent in X-Workspace-Key. API-only callers can use an HTTP-only session cookie. Only its SHA-256 hash is stored on sources. Possession grants workspace access; this is not identity verification or an account-recovery system. Clearing browser storage loses access.

API context scopes source, detection and crop queries. Writes check ownership; reset removes only owned runs. Legacy unowned records remain preserved and are not public. Local maintenance outside HTTP remains privileged.

Generated SQL runs against a read-only ephemeral SQLite snapshot containing only the caller's sources/detections, with no production credentials, other workspaces, crops or capability hashes. The demo limits snapshots to 50,000 rows per table. Cross-workspace tests cover both FastAPI entrypoints.

Full account authentication, durable abuse quotas and automated retention are future deployment work. Rows/crops currently persist until deleted by the workspace; no automatic TTL is claimed.

## Model boundaries

Results record model/threshold provenance. IDs associate observations within a run and do not establish real-world identity. Tracking is not guaranteed under arbitrary occlusion. See MODEL_CARD.md and research/tracking/HEAVY_MODE.md. The code/model distribution uses AGPL-3.0; review upstream terms against the intended commercial distribution. No legal certification or safety-critical suitability is claimed.
