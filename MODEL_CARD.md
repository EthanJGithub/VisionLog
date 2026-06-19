# Model Card — VisionLog detectors

VisionLog runs the **Ultralytics YOLO26** object-detection family plus **YOLOE-26**
(open-vocabulary), all in the browser on the visitor's GPU (WebGPU). This card documents
what the models are, how they were evaluated, measured accuracy/latency, and limitations.

Reproduce everything here with: `python -m src.eval.benchmark`.

---

## Models

| Model | Params | Head | Where it runs |
|---|---|---|---|
| YOLO26n / s / m / x | 2.4 / 9.5 / 20.4 / ~60 M | end-to-end (NMS-free) | client GPU (WebGPU) |
| YOLOE-26 (open-vocab) | ~12 M | one-to-many (+JS NMS) | client GPU; 69-class baked vocabulary |

- **Training data (upstream):** COCO (80 classes) for YOLO26; YOLOE adds open-vocabulary
  pretraining (Objects365 / grounding data). Weights are Ultralytics-pretrained — VisionLog
  does **not** retrain them (a domain fine-tune is on the roadmap).
- **Task:** object detection (boxes + class + confidence). Tracking adds stable Object IDs.
- **License:** AGPL-3.0 (see [LICENSE](LICENSE)).

---

## Accuracy — measured

Evaluated with `model.val()` at 640px on the **COCO128 benchmark subset** (128 COCO images).
These are *real, measured* numbers from this repo. ⚠️ COCO128 is a small quick-benchmark
subset (not a large held-out split), so absolute mAP is optimistic vs full COCO val2017 —
treat it as a **relative** comparison across model sizes. Full-COCO numbers are published in
the upstream [Ultralytics YOLO26 docs](https://docs.ultralytics.com/models/yolo26).

| Model | mAP@50 | mAP@50-95 | Precision | Recall |
|---|---|---|---|---|
| YOLO26n | 63.7% | 47.9% | 68.4% | 54.4% |
| YOLO26s | 74.5% | 57.5% | 73.3% | 65.7% |
| YOLO26m | 78.5% | 61.8% | 70.3% | 71.0% |
| YOLO26x | **81.7%** | **65.6%** | 75.6% | 72.6% |

Accuracy rises monotonically with model size — the basis for the in-app speed↔accuracy
picker. Per-class AP, the confusion matrix, and the PR curve are in the app's **Benchmarks**
tab (`frontend/public/benchmarks/`).

---

## Latency / throughput — measured (this machine)

| Model | CPU (onnxruntime, ms/frame) | In-browser WebGPU (fps, NVIDIA) |
|---|---|---|
| YOLO26n | 23.5 ms | 47.8 |
| YOLO26s | 49.5 ms | 38.9 |
| YOLO26m | 174.4 ms | 19.8 |
| YOLO26x | 492.1 ms | 10.6 |

The WebGPU column is why detection runs client-side: even YOLO26x is interactive on a GPU,
while on CPU it is ~0.5 s/frame. n/s are comfortably real-time.

---

## Intended use & limitations

- **Intended:** demos / analytics of common objects in user-provided video; open-vocab for
  classes in the baked 69-word vocabulary.
- **Not for:** safety-critical decisions; identity/biometric inference (none is performed —
  see [COMPLIANCE.md](COMPLIANCE.md)).
- **Known limits:** COCO/-vocab classes only (no domain fine-tune yet); accuracy degrades on
  small/occluded/low-light objects; COCO128 eval is a subset; weak/integrated GPUs may not
  hit real-time on m/x. Roadmap (see [ARCHITECTURE.md](ARCHITECTURE.md) §6): domain fine-tune
  with before/after mAP, FP16, tracking-based analytics, arbitrary-text open-vocab.
