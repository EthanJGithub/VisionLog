// Client-side YOLO26 detector — runs in the browser via onnxruntime-web.
// Uses the WebGPU execution provider when available (real-time on a capable GPU),
// otherwise falls back to WASM (CPU). The active backend is reported, never hidden.
import * as ort from "onnxruntime-web/webgpu";
import { COCO_CLASSES } from "./labels";

// Serve the ORT wasm/jsep binaries from the matching CDN version (avoids bundler config).
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/";

/** Returns "webgpu" if the browser exposes a usable GPU adapter, else "wasm". */
export async function pickBackend() {
  if (typeof navigator !== "undefined" && navigator.gpu) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) return "webgpu";
    } catch {
      /* fall through */
    }
  }
  return "wasm";
}

export class ClientDetector {
  constructor(modelUrl, { inputSize = 640, confThreshold = 0.35 } = {}) {
    this.modelUrl = modelUrl;
    this.inputSize = inputSize;
    this.confThreshold = confThreshold;
    this.session = null;
    this.backend = null;
    this.inputName = null;
    // reusable preprocessing canvas + buffer
    this._canvas = document.createElement("canvas");
    this._ctx = this._canvas.getContext("2d", { willReadFrequently: true });
    this._chw = new Float32Array(inputSize * inputSize * 3);
  }

  async load() {
    this.backend = await pickBackend();
    this.session = await ort.InferenceSession.create(this.modelUrl, {
      executionProviders: [this.backend],
      graphOptimizationLevel: "all",
    });
    this.inputName = this.session.inputNames[0];
    return this.backend;
  }

  // Letterbox the source frame into a square `inputSize` canvas, return CHW float32.
  _preprocess(source, srcW, srcH) {
    const s = this.inputSize;
    const scale = Math.min(s / srcW, s / srcH);
    const nw = Math.round(srcW * scale);
    const nh = Math.round(srcH * scale);
    const padX = Math.floor((s - nw) / 2);
    const padY = Math.floor((s - nh) / 2);

    this._canvas.width = s;
    this._canvas.height = s;
    const ctx = this._ctx;
    ctx.fillStyle = "rgb(114,114,114)";
    ctx.fillRect(0, 0, s, s);
    ctx.drawImage(source, 0, 0, srcW, srcH, padX, padY, nw, nh);

    const { data } = ctx.getImageData(0, 0, s, s); // RGBA
    const chw = this._chw;
    const area = s * s;
    for (let i = 0; i < area; i++) {
      const j = i * 4;
      chw[i] = data[j] / 255; // R
      chw[i + area] = data[j + 1] / 255; // G
      chw[i + 2 * area] = data[j + 2] / 255; // B
    }
    return { scale, padX, padY };
  }

  /** Run detection on a video/canvas/image element. Returns detections in source px. */
  async detect(source, srcW, srcH) {
    const { scale, padX, padY } = this._preprocess(source, srcW, srcH);
    const s = this.inputSize;
    const tensor = new ort.Tensor("float32", this._chw, [1, 3, s, s]);
    const outputs = await this.session.run({ [this.inputName]: tensor });
    const out = outputs[this.session.outputNames[0]];
    const data = out.data; // (1,300,6) flattened => [x1,y1,x2,y2,score,cls] * 300

    const dets = [];
    for (let i = 0; i < data.length; i += 6) {
      const score = data[i + 4];
      if (score < this.confThreshold) continue;
      const cls = Math.round(data[i + 5]);
      const ox1 = Math.max(0, (data[i] - padX) / scale);
      const oy1 = Math.max(0, (data[i + 1] - padY) / scale);
      const ox2 = Math.min(srcW, (data[i + 2] - padX) / scale);
      const oy2 = Math.min(srcH, (data[i + 3] - padY) / scale);
      if (ox2 <= ox1 || oy2 <= oy1) continue;
      dets.push({
        class_id: cls,
        class_label: COCO_CLASSES[cls] ?? String(cls),
        confidence: +score.toFixed(4),
        bbox_x: +ox1.toFixed(2),
        bbox_y: +oy1.toFixed(2),
        bbox_w: +(ox2 - ox1).toFixed(2),
        bbox_h: +(oy2 - oy1).toFixed(2),
      });
    }
    return dets;
  }

  dispose() {
    this.session?.release?.();
    this.session = null;
  }
}
