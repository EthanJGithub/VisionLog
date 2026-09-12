# Tracking quality research

The portfolio keeps its verified seven-second browser result until a **continuous 15-second or longer** replacement passes review. These offline experiments are not the deployed browser tracker and are not approved showcase footage.

## Current comparison

The isolated research environment uses Ultralytics 8.4.148, PyTorch 2.7.0+cu118 and an RTX 3060. `candidate-report.json` records exact configurations and provenance. Three profiles processed all 150 frames starting at OpenCV vtest frame 404:

- YOLO26m / 640 + ByteTrack: baseline.
- YOLO26x / 960 + TrackTrack with native detector appearance features and sparse optical-flow camera-motion compensation.
- YOLO26x / 960 + BoT-SORT with native detector appearance features and sparse optical-flow camera-motion compensation.

The larger detector and appearance-aware profiles generated fewer distinct IDs, but that **does not establish greater accuracy**. Detector size and resolution differ as well as the tracker. Review still found missed people during overlap and identity instability; none qualifies for publication. No claim of perfect tracking, benchmark leadership or general performance is supported by this experiment.

A second source, Roboflow Supervision's `people-walking.mp4`, is only 13.64 seconds and fails the duration requirement before quality evaluation.

## Reproduce

Use an isolated environment with a suitable PyTorch installation, `ultralytics==8.4.148`, OpenCV, PyYAML and lap. This is research tooling; do not upgrade the production API environment just to run it.

```powershell
python research/tracking/benchmark.py --source /path/to/vtest.avi --model-dir . --start-frame 404 --frames 150 --output /path/to/review-output
```

The script emits per-frame tracked boxes, IDs, configurations, timings and review stills. It does not modify website assets. No source footage or model weights are committed here.

## Promotion gate

1. Use at least 15 seconds of continuous source footage at its native playback speed. Preserve source provenance and permission to publish.
2. Record actual inference and tracking output for every frame. Do not manually rewrite IDs, delete inconvenient people, or fabricate boxes.
3. Review every frame, including partial entrances and overlaps, against person-level ground truth. Require zero identity switches, missed visible people and extra boxes for the selected showcase.
4. Compare candidates with the same detector/resolution and annotated inputs before attributing gains to a tracking algorithm. Report ID switches, false positives, misses, HOTA/IDF1 and device cost when a properly annotated evaluation set is available.
5. Render counts with the same frame as boxes, verify complete preloading and looping, and test mobile playback before publication.

Next candidates: stronger person-specific ReID embeddings (native detector features are not a dedicated person-ReID model), observation-centric motion handling, and high-resolution detection around crowded objects. Evaluate these changes; do not describe them as deployed until integrated and tested in VisionLog.

References: [Ultralytics tracking](https://docs.ultralytics.com/modes/track), [BoT-SORT](https://github.com/NirAharon/BoT-SORT), [ByteTrack](https://github.com/FoundationVision/ByteTrack), [Supervision sample assets](https://github.com/roboflow/supervision/blob/develop/src/supervision/assets/list.py).
