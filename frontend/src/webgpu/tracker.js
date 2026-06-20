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
//   3. remember unmatched tracks for maxAge frames (survives a car passing behind another).

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
  constructor({ iouThresh = 0.2, maxAge = 30, minHits = 3, centerGate = 2.0, minSizeRatio = 0.35 } = {}) {
    this.iouThresh = iouThresh;
    this.maxAge = maxAge;
    this.minHits = minHits;
    this.centerGate = centerGate;
    this.minSizeRatio = minSizeRatio;
    this.tracks = [];
    this.nextId = 1;
    this.frame = 0;
  }

  /**
   * Annotate detections with track_id + confirmed flag.
   * @returns dets each extended with { track_id, _confirmed }.
   */
  update(dets) {
    this.frame += 1;
    const boxes = dets.map(toBox);
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
        // IoU match dominates (score >= 1); else fall back to center-distance + size similarity.
        let score = ov >= this.iouThresh ? 1 + ov : 0;
        if (score === 0) {
          const dist = Math.hypot(cx(boxes[i]) - pcx, cy(boxes[i]) - pcy);
          const sr = (Math.min(bw(boxes[i]), bw(t.pred)) / Math.max(bw(boxes[i]), bw(t.pred)))
                   * (Math.min(bh(boxes[i]), bh(t.pred)) / Math.max(bh(boxes[i]), bh(t.pred)));
          if (dist <= gate && sr >= this.minSizeRatio) score = (1 - dist / gate) * sr; // < 1
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
