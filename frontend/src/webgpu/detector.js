// Client-side YOLO26 detector — runs in the browser via onnxruntime-web.
// Uses the WebGPU execution provider when available (real-time on a capable GPU),
// otherwise falls back to WASM (CPU). The active backend is reported, never hidden.
import { COCO_CLASSES } from "./labels";

// onnxruntime-web is large (~1MB JS + wasm); load it ON DEMAND (first detection) so it
// stays out of the initial bundle — keeps the page's first load fast (Lighthouse).
let _ortPromise = null;
function getOrt() {
  if (!_ortPromise) {
    _ortPromise = import("onnxruntime-web/webgpu").then((ort) => {
      ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/";
      return ort;
    });
  }
  return _ortPromise;
}

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

// Class-aware non-max suppression for the YOLOE (one-to-many) head.
function iou(a, b) {
  const x1 = Math.max(a.x1, b.x1), y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2), y2 = Math.min(a.y2, b.y2);
  const w = Math.max(0, x2 - x1), h = Math.max(0, y2 - y1);
  const inter = w * h;
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (areaA + areaB - inter + 1e-9);
}
function nms(boxes, iouThresh = 0.45) {
  boxes.sort((a, b) => b.score - a.score);
  const keep = [];
  for (const b of boxes) {
    if (keep.every((k) => k.cls !== b.cls || iou(b, k) <= iouThresh)) keep.push(b);
  }
  return keep;
}

export class ClientDetector {
  constructor(modelUrl, { inputSize = 640, confThreshold = 0.35, head = "e2e", classNames = null } = {}) {
    this.modelUrl = modelUrl;
    this.inputSize = inputSize;
    this.confThreshold = confThreshold;
    this.head = head; // "e2e" (n/s/m/x, NMS-free) | "v8seg" (YOLOE open-vocab)
    this.classNames = classNames; // for v8seg: the baked vocabulary
    this.session = null;
    this.backend = null;
    this.inputName = null;
    this._ort = null;
    // reusable preprocessing canvas + buffer
    this._canvas = document.createElement("canvas");
    this._ctx = this._canvas.getContext("2d", { willReadFrequently: true });
    this._chw = new Float32Array(inputSize * inputSize * 3);
  }

  async load() {
    this._ort = await getOrt();
    this.backend = await pickBackend();
    this.session = await this._ort.InferenceSession.create(this.modelUrl, {
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
    const tensor = new this._ort.Tensor("float32", this._chw, [1, 3, s, s]);
    const outputs = await this.session.run({ [this.inputName]: tensor });
    const out = outputs[this.session.outputNames[0]];
    const map = (x1, y1, x2, y2) => ({
      ox1: Math.max(0, (x1 - padX) / scale),
      oy1: Math.max(0, (y1 - padY) / scale),
      ox2: Math.min(srcW, (x2 - padX) / scale),
      oy2: Math.min(srcH, (y2 - padY) / scale),
    });
    const names = this.classNames || COCO_CLASSES; // custom (vocab/fine-tune) else COCO
    const push = (arr, cls, score, x1, y1, x2, y2) => {
      const { ox1, oy1, ox2, oy2 } = map(x1, y1, x2, y2);
      if (ox2 <= ox1 || oy2 <= oy1) return;
      arr.push({
        class_id: cls,
        class_label: names?.[cls] ?? String(cls),
        confidence: +score.toFixed(4),
        bbox_x: +ox1.toFixed(2), bbox_y: +oy1.toFixed(2),
        bbox_w: +(ox2 - ox1).toFixed(2), bbox_h: +(oy2 - oy1).toFixed(2),
      });
    };

    // --- YOLOE one-to-many head: (1, 4+K+32, 8400) → threshold → class-aware NMS ---
    // Also stashes the 32 mask coefficients per kept detection + this frame's prototype masks,
    // so a segmentation cutout can be assembled later (only when a crop is captured — cheap).
    if (this.head === "v8seg") {
      const K = this.classNames.length;
      const A = out.dims[2]; // anchors (8400)
      const d = out.data;
      const protoOut = outputs[this.session.outputNames[1]]; // (1,32,160,160)
      this._proto = protoOut
        ? { data: protoOut.data, mc: protoOut.dims[1], mh: protoOut.dims[2], mw: protoOut.dims[3] }
        : null;
      this._segGeom = { scale, padX, padY, inputSize: s };
      const cand = [];
      for (let a = 0; a < A; a++) {
        let best = -1, bestC = -1;
        for (let c = 0; c < K; c++) {
          const v = d[(4 + c) * A + a];
          if (v > best) { best = v; bestC = c; }
        }
        if (best < this.confThreshold) continue;
        const cx = d[a], cy = d[A + a], w = d[2 * A + a], h = d[3 * A + a];
        cand.push({ x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2, score: best, cls: bestC, a });
      }
      const dets = [];
      for (const b of nms(cand, 0.45)) {
        const before = dets.length;
        push(dets, b.cls, b.score, b.x1, b.y1, b.x2, b.y2);
        if (this._proto && dets.length > before) {
          const mc = this._proto.mc;
          const coeffs = new Float32Array(mc);
          for (let k = 0; k < mc; k++) coeffs[k] = d[(4 + K + k) * A + b.a];
          dets[dets.length - 1]._coeffs = coeffs;
        }
      }
      return dets;
    }

    // --- End-to-end head (n/s/m/x): (1,300,6) = [x1,y1,x2,y2,score,cls], NMS-free ---
    const data = out.data;
    const dets = [];
    for (let i = 0; i < data.length; i += 6) {
      const score = data[i + 4];
      if (score < this.confThreshold) continue;
      push(dets, Math.round(data[i + 5]), score, data[i], data[i + 1], data[i + 2], data[i + 3]);
    }
    return dets;
  }

  /**
   * Assemble the segmentation mask for ONE detection from the last frame's prototypes:
   * mask = sigmoid(Σ coeff·proto), cropped to the detection's box. Returns a small canvas whose
   * ALPHA channel is the soft mask (white RGB), ready to composite over a crop. v8seg only.
   * Called at most once per confirmed track (capture time), so the per-instance cost is cheap.
   */
  instanceMaskCanvas(det) {
    if (!this._proto || !det?._coeffs || !this._segGeom) return null;
    const { data, mh, mw } = this._proto;
    const { scale, padX, padY, inputSize } = this._segGeom;
    const f = mw / inputSize; // input(640) → proto(160) factor
    // det bbox is in SOURCE px → input(640) px → proto(160) px.
    let px1 = Math.floor((det.bbox_x * scale + padX) * f);
    let px2 = Math.ceil(((det.bbox_x + det.bbox_w) * scale + padX) * f);
    let py1 = Math.floor((det.bbox_y * scale + padY) * f);
    let py2 = Math.ceil(((det.bbox_y + det.bbox_h) * scale + padY) * f);
    px1 = Math.max(0, Math.min(mw, px1)); px2 = Math.max(0, Math.min(mw, px2));
    py1 = Math.max(0, Math.min(mh, py1)); py2 = Math.max(0, Math.min(mh, py2));
    const rw = px2 - px1, rh = py2 - py1;
    if (rw < 1 || rh < 1) return null;

    if (!this._maskCanvas) {
      this._maskCanvas = document.createElement("canvas");
      this._maskCtx = this._maskCanvas.getContext("2d");
    }
    this._maskCanvas.width = rw;
    this._maskCanvas.height = rh;
    const img = this._maskCtx.createImageData(rw, rh);
    const coeffs = det._coeffs, mc = coeffs.length;
    for (let yy = 0; yy < rh; yy++) {
      for (let xx = 0; xx < rw; xx++) {
        const py = py1 + yy, px = px1 + xx;
        let acc = 0;
        for (let k = 0; k < mc; k++) acc += coeffs[k] * data[(k * mh + py) * mw + px];
        const alpha = 1 / (1 + Math.exp(-acc)); // sigmoid
        const o = (yy * rw + xx) * 4;
        img.data[o] = 255; img.data[o + 1] = 255; img.data[o + 2] = 255;
        img.data[o + 3] = alpha > 0.5 ? 255 : Math.round(alpha * 255); // crisp interior, soft edge
      }
    }
    this._maskCtx.putImageData(img, 0, 0);
    return this._maskCanvas;
  }

  dispose() {
    this.session?.release?.();
    this.session = null;
  }
}

/**
 * Build + load a ClientDetector for a model-registry entry. For the open-vocab model
 * (head "v8seg") it fetches the bundled vocabulary first. Returns { detector, backend,
 * classNames }.
 */
export async function createDetector(model, { inputSize = 640, conf = 0.35 } = {}) {
  let classNames = null;
  const head = model.head === "v8seg" ? "v8seg" : "e2e";
  // Any model with a vocabUrl carries custom class names (open-vocab OR a fine-tune).
  if (model.vocabUrl) {
    const meta = await (await fetch(model.vocabUrl)).json();
    classNames = meta.classes;
  }
  const detector = new ClientDetector(model.url, {
    inputSize,
    confThreshold: conf,
    head,
    classNames,
  });
  const backend = await detector.load();
  return { detector, backend, classNames };
}
