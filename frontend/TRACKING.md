# Browser object tracking

The optional Advanced object IDs mode uses a browser-oriented, ByteTrack-style two-stage association pipeline:

- High-confidence detections receive global Hungarian assignment against Kalman-predicted tracks.
- A second pass associates low-confidence observations with established tracks; weak observations cannot spawn identities.
- Body-centered spatial color/luminance histograms add appearance cues. Crowded detections do not update the appearance template.
- Unmatched tracks remain recoverable for a bounded buffer. IDs are monotonic within a run and not global identities across sessions.
- Rendered boxes are detector observations. The tracker does not draw extrapolated boxes for fully missing detections.

Reference: https://github.com/FoundationVision/ByteTrack . This implementation adds lightweight appearance cues; it is not the official ByteTrack implementation, BoT-SORT, or a learned person-ReID model. No algorithm guarantees perfect identity under every occlusion or camera condition.

Tracking runs on the CPU. Detection uses WebGPU when available. Turning tracking off also skips object crops and CLIP semantic embeddings, reducing device workload; the detector still runs. Change mode between runs so existing source IDs cannot silently change meaning. No external compute is charged by this browser tracking code.

Each webcam/upload frame is captured before inference; appearance and crops use the same captured pixels. The network stream already uses a single decoded bitmap.

Validation: `node --test tests/advancedTracker.test.mjs`. Tests cover global assignment, detector-order changes, low-confidence recovery, short occlusions, expiration, and class gating. Browser checks exercise both modes with real ONNX detection and inspect logging payloads and model requests.
