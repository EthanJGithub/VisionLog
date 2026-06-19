"""Dev-only: export the client-side open-vocabulary YOLOE-26 model.

Bakes a curated vocabulary's text embeddings into a standard ONNX (one-to-many head),
so the browser runs open-vocab detection on the GPU with no CLIP text encoder. Edit
VOCAB and re-run to change the available classes.

    python -m src.detect.export_vocab
"""
from __future__ import annotations

import json
from pathlib import Path

from src import config

# Curated open-vocabulary. Deliberately includes many classes COCO does NOT have
# (safety/industrial/retail/logistics) to showcase open-vocabulary detection.
VOCAB = [
    # people & vehicles
    "person", "child", "car", "truck", "bus", "van", "bicycle", "motorcycle",
    "scooter", "forklift", "train", "boat", "airplane", "wheelchair",
    # safety / PPE / industrial
    "hard hat", "safety vest", "gloves", "goggles", "face mask", "fire extinguisher",
    "traffic cone", "barrier", "pallet", "cardboard box", "ladder", "toolbox",
    "gas cylinder", "crane", "drone",
    # retail / logistics
    "shopping cart", "shopping basket", "price tag", "barcode", "package",
    "license plate",
    # animals
    "dog", "cat", "bird", "horse", "cow", "sheep",
    # personal items
    "backpack", "handbag", "suitcase", "umbrella", "cell phone", "laptop", "tablet",
    "headphones", "watch", "glasses", "wallet", "keys",
    # household / scene
    "chair", "table", "couch", "bed", "tv", "bottle", "cup", "plate", "knife",
    "scissors", "book", "potted plant", "door", "window", "stop sign",
    "traffic light",
]


def main() -> None:
    from ultralytics import YOLOE  # dev-only

    out_dir = Path(config.ROOT) / "frontend" / "public" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLOE("yoloe-11s-seg.pt")
    model.set_classes(VOCAB, model.get_text_pe(VOCAB))
    exported = model.export(format="onnx", imgsz=config.INPUT_SIZE, opset=12, simplify=True, dynamic=False)

    onnx_path = out_dir / "yoloe26-vocab.onnx"
    Path(exported).replace(onnx_path)
    (out_dir / "yoloe26-vocab.json").write_text(
        json.dumps({"classes": VOCAB, "reg": 16, "mask": 32})
    )
    print(f"Exported {onnx_path} ({onnx_path.stat().st_size/1e6:.1f} MB), {len(VOCAB)} classes")


if __name__ == "__main__":
    main()
