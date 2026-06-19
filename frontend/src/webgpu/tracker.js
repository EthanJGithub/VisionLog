// Lightweight online multi-object tracker (greedy IoU association, class-aware).
// Assigns a stable Object ID (track_id) to each detection across frames, and gates
// logging behind a "min hits" confirmation so single-frame false positives aren't
// persisted. No deps — runs in the same loop as detection.

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

export class SimpleTracker {
  constructor({ iouThresh = 0.3, maxAge = 15, minHits = 3 } = {}) {
    this.iouThresh = iouThresh;
    this.maxAge = maxAge;     // frames a track survives unmatched before removal
    this.minHits = minHits;   // frames before a track is "confirmed" (logged)
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

    // Prefer established tracks (more hits) when competing for a detection.
    const order = [...this.tracks].sort((a, b) => b.hits - a.hits);
    for (const t of order) {
      let best = this.iouThresh, bi = -1;
      for (let i = 0; i < dets.length; i++) {
        if (used.has(i) || dets[i].class_id !== t.classId) continue;
        const v = iou(boxes[i], t.box);
        if (v >= best) { best = v; bi = i; }
      }
      if (bi >= 0) {
        used.add(bi);
        t.box = boxes[bi];
        t.lastFrame = this.frame;
        t.hits += 1;
        detTrack[bi] = t;
      }
    }

    // Unmatched detections start new tracks.
    for (let i = 0; i < dets.length; i++) {
      if (detTrack[i]) continue;
      const t = { id: this.nextId++, classId: dets[i].class_id, box: boxes[i], lastFrame: this.frame, hits: 1 };
      this.tracks.push(t);
      detTrack[i] = t;
    }

    // Drop stale tracks.
    this.tracks = this.tracks.filter((t) => this.frame - t.lastFrame <= this.maxAge);

    return dets.map((d, i) => ({
      ...d,
      track_id: detTrack[i].id,
      _confirmed: detTrack[i].hits >= this.minHits,
    }));
  }
}
