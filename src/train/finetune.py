"""Dev-only: domain fine-tune YOLO26n on the Construction-PPE dataset.

Demonstrates transfer learning: the COCO-pretrained base has NONE of the PPE classes
(helmet, vest, goggles, …), so it scores ~0 on this data; fine-tuning teaches them.
Reports before/after mAP, exports the fine-tuned model to ONNX for the in-browser app,
and writes a report the Benchmarks tab reads.

    python -m src.train.finetune --epochs 16 --imgsz 416

Real data (auto-downloads ~178 MB), no API key. CPU-friendly defaults.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src import config

DATA = "construction-ppe.yaml"
PUBLIC_MODELS = Path(config.ROOT) / "frontend" / "public" / "models"
BENCH = Path(config.ROOT) / "frontend" / "public" / "benchmarks"


def _map(metrics) -> dict:
    return {"map50": round(float(metrics.box.map50), 4), "map50_95": round(float(metrics.box.map), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    from ultralytics import YOLO, settings

    settings.update({"datasets_dir": str(Path(config.ROOT) / ".cache" / "datasets")})

    # 1) BEFORE — the COCO-pretrained base on the PPE val set (expected ~0; classes differ).
    print("== evaluating base (before) ==")
    before = _map(YOLO("yolo26n.pt").val(data=DATA, imgsz=args.imgsz, verbose=False))
    print("before:", before)

    # 2) Fine-tune.
    print("== fine-tuning ==")
    model = YOLO("yolo26n.pt")
    results = model.train(
        data=DATA, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device="cpu", workers=4, project="runs/ppe", name="finetune", exist_ok=True,
        plots=True, verbose=True,
    )
    best = Path(results.save_dir) / "weights" / "best.pt"

    # 3) AFTER — the fine-tuned model on the same val set.
    print("== evaluating fine-tuned (after) ==")
    ft = YOLO(str(best))
    after = _map(ft.val(data=DATA, imgsz=args.imgsz, verbose=False))
    print("after:", after)

    # 4) Export the fine-tuned model for the in-browser app (end-to-end ONNX) + class names.
    PUBLIC_MODELS.mkdir(parents=True, exist_ok=True)
    exported = ft.export(format="onnx", imgsz=640, opset=12, simplify=True, dynamic=False)
    Path(exported).replace(PUBLIC_MODELS / "yolo26-ppe.onnx")
    names = [ft.names[i] for i in sorted(ft.names)]
    (PUBLIC_MODELS / "yolo26-ppe.json").write_text(json.dumps({"classes": names}))

    # 5) Report for the Benchmarks tab.
    BENCH.mkdir(parents=True, exist_ok=True)
    (BENCH / "finetune.json").write_text(json.dumps({
        "dataset": "Construction-PPE (1132 train / 143 val)",
        "base_model": "yolo26n (COCO-pretrained)",
        "classes": names,
        "epochs": args.epochs, "imgsz": args.imgsz,
        "before": before, "after": after,
    }, indent=2))
    print("DONE. before:", before, "after:", after, "classes:", names)


if __name__ == "__main__":
    main()
