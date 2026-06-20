// Lightweight online multi-object tracker (SORT-style, class-aware, no deps).
// Assigns a stable Object ID (track_id) to each detection across frames, and gates
// logging behind a "min hits" confirmation so single-frame false positives aren't
// persisted. Runs in the same loop as detection.
//
// Why a motion model: pure frame-to-frame IoU breaks for objects that move fast (a car
// crossing the frame moves more than its own width between sampled frames, so its box no
// longer overlaps the previous one and it gets a NEW id every few frames). We instead:
//   1. predict each track forward with a constant-velocity model, match IoU on the PREDICTION;
//   2. fall back to a center-distance + size gate when boxes don't overlap (fast / briefly
//      occluded objects), so the same object keeps its id;
//   3. remember unmatched tracks for maxAge frames (survives a car passing behind another);
//   4. (optional) a lightweight COLOR-HISTOGRAM appearance cue (when a frame source is passed):
//      refines association ties and widens the gate when appearance strongly matches, so an
//      object re-identifies after occlusion instead of getting a new id. No model — cheap.

let _hc = null, _hctx = null;
// 4x4x4 RGB histogram (64 bins) from a 16x16 downsample of the box region; normalized.
function colorHist(source, box) {
  if (typeof document === "undefined") return null;
  const w = box.x2 - box.x1, h = box.y2 - box.y1;
  if (w < 2 || h < 2) return null;
  if (!_hc) {
    _hc = document.createElement("canvas");
    _hc.width = _hc.height = 16;
    _hctx = _hc.getContext("2d", { willReadFrequently: true });
  }
  let data;
  try {
    _hctx.clearRect(0, 0, 16, 16);
    _hctx.drawImage(source, box.x1, box.y1, w, h, 0, 0, 16, 16);
    data = _hctx.getImageData(0, 0, 16, 16).data;
  } catch {
    return null;
  }
  const hist = new Float32Array(64);
  for (let p = 0; p < data.length; p += 4) {
    hist[((data[p] >> 6) * 4 + (data[p + 1] >> 6)) * 4 + (data[p + 2] >> 6)]++;
  }
  let s = 0;
  for (const v of hist) s += v;
  if (s > 0) for (let i = 0; i < 64; i++) hist[i] /= s;
  return hist;
}
// Histogram intersection in [0,1]; null if either side is missing.
function histSim(a, b) {
  if (!a || !b) return null;
  let s = 0;
  for (let i = 0; i < 64; i++) s += Math.min(a[i], b[i]);
  return s;
}
// EMA blend of two normalized histograms (stays normalized).
function emaHist(prev, next, a) {
  const out = new Float32Array(64);
  for (let i = 0; i < 64; i++) out[i] = prev[i] * (1 - a) + next[i] * a;
  return out;
}

function iou(a, b) {
  const x1 = Math.max(a.x1, b.x1), y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2), y2 = Math.min(a.y2, b.y2);
  const w = Math.max(0, x2 - x1), h = Math.max(0, y2 - y1);
  const inter = w * h;
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (areaA + areaB - inter + 1e-9);
}
const toBox = (d) => ({ x1: d.bbox_x, y1: d.bbox_y, x2: d.bbox_x + d.bbox_w, y2: d.bbox_y + d.bbox_h });
const cx = (b) => (b.x1 + b.x2) / 2;
const cy = (b) => (b.y1 + b.y2) / 2;
const bw = (b) => b.x2 - b.x1;
const bh = (b) => b.y2 - b.y1;
const shift = (b, dx, dy) => ({ x1: b.x1 + dx, y1: b.y1 + dy, x2: b.x2 + dx, y2: b.y2 + dy });

export class SimpleTracker {
  // iouThresh: forgiving (prediction makes overlap easier). maxAge: ~1s at ~30fps.
  // centerGate: max center jump to still count as the same object, in units of the box's
  // mean dimension (so a car can move ~2 box-widths/frame and keep its id). minSizeRatio:
  // reject a match whose box is a very different size (a different object nearby).
  constructor({ iouThresh = 0.2, maxAge = 30, minHits = 3, centerGate = 2.0, minSizeRatio = 0.35,
                appWeight = 0.3 } = {}) {
    this.iouThresh = iouThresh;
    this.maxAge = maxAge;
    this.minHits = minHits;
    this.centerGate = centerGate;
    this.minSizeRatio = minSizeRatio;
    this.appWeight = appWeight; // how much color appearance refines association
    this.tracks = [];
    this.nextId = 1;
    this.frame = 0;
  }

  /**
   * Annotate detections with track_id + confirmed flag.
   * @param dets         detections for this frame
   * @param frameSource  optional drawable frame (video/canvas/bitmap) → enables the color
   *                     appearance cue for better re-identification.
   * @returns dets each extended with { track_id, _confirmed }.
   */
  update(dets, frameSource = null) {
    this.frame += 1;
    const boxes = dets.map(toBox);
    const hists = frameSource ? boxes.map((b) => colorHist(frameSource, b)) : null;
    const detTrack = new Array(dets.length).fill(null);
    const used = new Set();

    // Constant-velocity prediction: project each track forward by however many frames it has
    // gone unmatched, so a moving object is matched where it WILL be, not where it was.
    for (const t of this.tracks) {
      const steps = this.frame - t.lastFrame; // >= 1
      t.pred = shift(t.box, t.vx * steps, t.vy * steps);
      t.steps = steps;
    }

    // Greedy association, established tracks (more hits) first so they win contested detections.
    const order = [...this.tracks].sort((a, b) => b.hits - a.hits);
    for (const t of order) {
      let bestScore = 0, bi = -1;
      const pcx = cx(t.pred), pcy = cy(t.pred);
      // Gate grows with the prediction horizon (more uncertainty after a long miss).
      const gate = this.centerGate * Math.max(8, (bw(t.pred) + bh(t.pred)) / 2) * Math.sqrt(t.steps);
      for (let i = 0; i < dets.length; i++) {
        if (used.has(i) || dets[i].class_id !== t.classId) continue;
        const ov = iou(boxes[i], t.pred);
        const app = hists && t.hist ? histSim(hists[i], t.hist) : null; // 0..1 or null
        // IoU match dominates (score >= 1); else fall back to center-distance + size similarity.
        // Appearance refines ties (IoU path) and reweights/widens the fallback path.
        let score = 0;
        if (ov >= this.iouThresh) {
          score = 1 + ov + (app != null ? this.appWeight * app : 0);
        } else {
          // Strong appearance match widens the gate → recover an object that moved during occlusion.
          const effGate = gate * (app != null && app > 0.8 ? 1.5 : 1);
          const dist = Math.hypot(cx(boxes[i]) - pcx, cy(boxes[i]) - pcy);
          const sr = (Math.min(bw(boxes[i]), bw(t.pred)) / Math.max(bw(boxes[i]), bw(t.pred)))
                   * (Math.min(bh(boxes[i]), bh(t.pred)) / Math.max(bh(boxes[i]), bh(t.pred)));
          if (dist <= effGate && sr >= this.minSizeRatio) {
            score = (1 - dist / effGate) * sr * (app != null ? 0.6 + 0.4 * app : 1);
          }
        }
        if (score > bestScore) { bestScore = score; bi = i; }
      }
      if (bi >= 0) {
        const nb = boxes[bi];
        const gap = t.steps; // frames since this track last matched (>= 1)
        // EMA velocity from the per-frame center delta (gap-normalized so multi-frame misses
        // don't inflate velocity).
        t.vx = 0.5 * t.vx + 0.5 * (cx(nb) - cx(t.box)) / gap;
        t.vy = 0.5 * t.vy + 0.5 * (cy(nb) - cy(t.box)) / gap;
        if (hists && hists[bi]) t.hist = t.hist ? emaHist(t.hist, hists[bi], 0.3) : hists[bi];
        used.add(bi);
        t.box = nb;
        t.lastFrame = this.frame;
        t.hits += 1;
        detTrack[bi] = t;
      }
    }

    // Unmatched detections start new tracks.
    for (let i = 0; i < dets.length; i++) {
      if (detTrack[i]) continue;
      const t = {
        id: this.nextId++, classId: dets[i].class_id, box: boxes[i],
        vx: 0, vy: 0, lastFrame: this.frame, hits: 1,
        hist: hists ? hists[i] : null,
      };
      this.tracks.push(t);
      detTrack[i] = t;
    }

    // Drop tracks unseen for longer than maxAge.
    this.tracks = this.tracks.filter((t) => this.frame - t.lastFrame <= this.maxAge);

    return dets.map((d, i) => ({
      ...d,
      track_id: detTrack[i].id,
      _confirmed: detTrack[i].hits >= this.minHits,
    }));
  }
}
