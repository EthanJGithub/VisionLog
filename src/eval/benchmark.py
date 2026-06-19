"""Dev-only: evaluate the YOLO26 variants and emit a benchmarks report.

Produces, for each variant:
  • Accuracy (REAL, measured): mAP@50 and mAP@50-95 via `model.val()` on the COCO128
    benchmark subset (128 COCO images; auto-downloads ~7 MB). Per-class AP + the confusion
    matrix / PR curve plots are saved by Ultralytics.
  • Latency (REAL, measured): onnxruntime CPU ms/frame at 640px (this machine).

Writes frontend/public/benchmarks/benchmarks.json (+ copies key plots) so the live app can
render a Benchmarks view, and feeds MODEL_CARD.md.

    python -m src.eval.benchmark
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np

from src import config

VARIANTS = ["yolo26n", "yolo26s", "yolo26m", "yolo26x"]
OUT_DIR = Path(config.ROOT) / "frontend" / "public" / "benchmarks"

# In-browser WebGPU fps measured earlier on an NVIDIA GPU (640px, warm) — recorded here so
# the report shows the client-side story alongside server CPU latency. Source: this project.
WEBGPU_FPS = {"yolo26n": 47.8, "yolo26s": 38.9, "yolo26m": 19.8, "yolo26x": 10.6}


def measure_cpu_latency(model_id: str, runs: int = 20) -> float | None:
    """onnxruntime CPU ms/frame at 640px, if the ONNX is present."""
    from src.detect.engine import DetectionEngine

    onnx = Path(config.ROOT) / "frontend" / "public" / "models" / f"{model_id}.onnx"
    if not onnx.exists():
        return None
    eng = DetectionEngine(model_path=str(onnx), model_version=model_id)
    frame = np.full((640, 640, 3), 128, dtype=np.uint8)
    eng.detect(frame)  # warmup
    t0 = time.perf_counter()
    for _ in range(runs):
        eng.detect(frame)
    return round((time.perf_counter() - t0) / runs * 1000, 1)


def main() -> None:
    from ultralytics import YOLO, settings

    settings.update({"datasets_dir": str(Path(config.ROOT) / ".cache" / "datasets")})
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    report: dict = {"dataset": "COCO128 (128-image COCO subset)", "input_size": 640, "models": []}
    per_class = None

    for v in VARIANTS:
        entry: dict = {"id": v}
        try:
            m = YOLO(f"{v}.pt")
            metrics = m.val(data="coco128.yaml", imgsz=640, plots=True, verbose=False)
            entry["map50"] = round(float(metrics.box.map50), 4)
            entry["map50_95"] = round(float(metrics.box.map), 4)
            entry["precision"] = round(float(metrics.box.mp), 4)
            entry["recall"] = round(float(metrics.box.mr), 4)
            entry["params_m"] = round(sum(p.numel() for p in m.model.parameters()) / 1e6, 1)
            if per_class is None:  # capture per-class AP from the smallest model
                names = metrics.names
                maps = metrics.box.maps  # per-class mAP50-95
                per_class = sorted(
                    ({"class": names[i], "ap": round(float(a), 4)} for i, a in enumerate(maps)),
                    key=lambda x: x["ap"], reverse=True,
                )[:15]
                # copy the diagnostic plots for this model
                sd = Path(metrics.save_dir)
                for src_name, dst_name in [
                    ("confusion_matrix_normalized.png", "confusion_matrix_normalized.png"),
                    ("BoxPR_curve.png", "PR_curve.png"),
                ]:
                    src = sd / src_name
                    if src.exists():
                        shutil.copy(src, OUT_DIR / dst_name)
        except Exception as exc:  # keep going; report what we have
            entry["error"] = str(exc)[:160]
        entry["cpu_ms"] = measure_cpu_latency(v)
        entry["webgpu_fps"] = WEBGPU_FPS.get(v)
        report["models"].append(entry)
        print(f"{v}: {entry}")

    report["per_class_ap"] = per_class or []
    (OUT_DIR / "benchmarks.json").write_text(json.dumps(report, indent=2))
    print(f"\nWrote {OUT_DIR / 'benchmarks.json'}")


if __name__ == "__main__":
    main()
