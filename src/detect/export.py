"""Dev-only: export YOLO26 variants to ONNX for the onnxruntime runtime.

Needs the `ultralytics` dev dep + network (fetches weights). YOLO26's default head is
end-to-end (NMS-free), so the runtime needs no NMS post-processing. opset=12 is required
for onnxruntime-web WebGPU compatibility (client-side path).

    python -m src.detect.export                 # exports yolo26n (default)
    python -m src.detect.export --model yolo26s
    python -m src.detect.export --all           # n, s, m, x
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src import config

ALL = ["yolo26n", "yolo26s", "yolo26m", "yolo26x"]


def export_one(model_id: str) -> Path:
    from ultralytics import YOLO  # dev-only dependency

    out_dir = Path(config.ROOT) / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(f"{model_id}.pt")
    exported = model.export(
        format="onnx", imgsz=config.INPUT_SIZE, opset=12, simplify=True, dynamic=False
    )
    target = out_dir / f"{model_id}.onnx"
    src_path = Path(exported)
    if src_path.resolve() != target.resolve():
        src_path.replace(target)
    print(f"Exported -> {target}")
    return target


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26n", choices=ALL)
    ap.add_argument("--all", action="store_true", help="export n, s, m, x")
    args = ap.parse_args()
    for mid in (ALL if args.all else [args.model]):
        export_one(mid)


if __name__ == "__main__":
    main()
