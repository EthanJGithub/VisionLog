// Browser-oriented ByteTrack-style association, with color appearance and a
// constant-velocity Kalman filter. This is not a port of BoT-SORT or learned ReID.
// Reference: https://github.com/FoundationVision/ByteTrack
let appearanceCanvas;
// Body-centered spatial histograms retain clothing/brightness differences while
// reducing background contamination. This is lightweight appearance, not neural ReID.
function descriptor(source, d) {
  if (!source || typeof document === 'undefined') return null;
  const c=appearanceCanvas || (appearanceCanvas=document.createElement('canvas'));
  c.width=18;c.height=36;const ctx=c.getContext('2d',{willReadFrequently:true});
  ctx.drawImage(source,d.bbox_x+d.bbox_w*.25,d.bbox_y+d.bbox_h*.08,d.bbox_w*.5,d.bbox_h*.84,0,0,18,36);
  const px=ctx.getImageData(0,0,18,36).data, hist=new Float32Array(240);
  for(let y=0;y<36;y++) for(let x=0;x<18;x++) {const i=(y*18+x)*4, band=Math.floor(y/12)*80,r=px[i],g=px[i+1],b=px[i+2];hist[band+(r>>6)*16+(g>>6)*4+(b>>6)]+=.5/648;hist[band+64+Math.min(15,Math.floor((r*.299+g*.587+b*.114)/16))]+=.5/648;}
  return hist;
}

// Rectangular Hungarian assignment. Dummy columns allow tracks to stay unmatched.
export function assign(costs, unmatched = .85) {
  const n = costs.length, real = costs[0]?.length || 0, m = real + n;
  if (!n || !real) return [];
  const u = Array(n + 1).fill(0), v = Array(m + 1).fill(0), p = Array(m + 1).fill(0), way = Array(m + 1).fill(0);
  for (let i = 1; i <= n; i++) {
    p[0] = i; let j0 = 0;
    const min = Array(m + 1).fill(Infinity), used = Array(m + 1).fill(false);
    do {
      used[j0] = true; const i0 = p[j0]; let delta = Infinity, j1 = 0;
      for (let j = 1; j <= m; j++) if (!used[j]) {
        const c = (j <= real ? costs[i0 - 1][j - 1] : unmatched) - u[i0] - v[j];
        if (c < min[j]) { min[j] = c; way[j] = j0; }
        if (min[j] < delta) { delta = min[j]; j1 = j; }
      }
      for (let j = 0; j <= m; j++) { if (used[j]) { u[p[j]] += delta; v[j] -= delta; } else min[j] -= delta; }
      j0 = j1;
    } while (p[j0]);
    do { const j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
  }
  return p.flatMap((row, col) => row && col && col <= real && costs[row - 1][col - 1] < unmatched ? [[row - 1, col - 1]] : []);
}

class Axis {
  constructor(x) { this.x = x; this.v = 0; this.a = 16; this.b = 0; this.c = 100; }
  predict() { this.x += this.v; this.a += 2 * this.b + this.c + 1; this.b += this.c; this.c += 1; }
  correct(z) {
    const variance = this.a + 4, k = this.a / variance, kv = this.b / variance, error = z - this.x;
    this.x += k * error; this.v += kv * error;
    this.c -= kv * this.b; this.b *= 1 - k; this.a *= 1 - k;
  }
}
const measure = d => [d.bbox_x + d.bbox_w / 2, d.bbox_y + d.bbox_h / 2, d.bbox_w, d.bbox_h];
const rect = d => ({ x1: d.bbox_x, y1: d.bbox_y, x2: d.bbox_x + d.bbox_w, y2: d.bbox_y + d.bbox_h });
function overlap(a, b) {
  const intersection = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1)) * Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  return intersection / Math.max(1, (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - intersection);
}
function similarity(a, b) { return a && b ? a.reduce((s, x, i) => s + Math.min(x, b[i]), 0) : .5; }

export class AdvancedTracker {
  constructor({ highThreshold = .4, lowThreshold = .1, maxAge = 30, minHits = 2 } = {}) {
    Object.assign(this, { highThreshold, lowThreshold, maxAge, minHits }); this.tracks = []; this.nextId = 1;
  }
  update(detections, source = null) {
    const dets = detections.filter(d => d.confidence >= this.lowThreshold);
    const appearance = dets.map(d => source ? descriptor(source, d) : null);
    for (const t of this.tracks) { t.age++; t.axes.forEach(a => a.predict()); }
    const matched = new Map(), used = new Set();
    const associate = (tracks, indices, low) => {
      const costs = tracks.map(t => indices.map(i => {
        const d = dets[i]; if (d.class_id !== t.classId) return 1e6;
        const [x,y,w,h] = t.axes.map(a => a.x), z = measure(d);
        const distance = Math.hypot((z[0]-x)/Math.max(10,w), (z[1]-y)/Math.max(10,h));
        const size = Math.min(w,z[2])/Math.max(w,z[2]) * Math.min(h,z[3])/Math.max(h,z[3]);
        const app = similarity(t.hist, appearance[i]);
        if (distance > (app > .7 ? 3 : Math.min(2, 1 + t.age * .1)) || size < .18) return 1e6;
        if (t.hits > 3 && t.hist && appearance[i] && app < .46) return 1e6;
        const iou = overlap({x1:x-w/2,y1:y-h/2,x2:x+w/2,y2:y+h/2}, rect(d));
        if (low && iou < .1) return 1e6;
        return .35*(1-iou) + .15*Math.min(1,distance) + .5*(1-app);
      }));
      for (const [row, col] of assign(costs, low ? .65 : .8)) {
        const t = tracks[row], i = indices[col]; matched.set(i,t); used.add(t);
      }
    };
    associate(this.tracks, dets.flatMap((d,i)=>d.confidence>=this.highThreshold?[i]:[]), false);
    associate(this.tracks.filter(t=>!used.has(t) && t.hits>=this.minHits), dets.flatMap((d,i)=>d.confidence<this.highThreshold?[i]:[]), true);
    for (const [i,t] of matched) {
      measure(dets[i]).forEach((z,j)=>t.axes[j].correct(z)); t.age=0; t.hits++;
      const crowded = dets.some((d,j)=>j!==i && overlap(rect(d),rect(dets[i]))>.12);
      if (appearance[i] && !crowded && dets[i].confidence >= this.highThreshold) t.hist = t.hist ? t.hist.map((x,j)=>.98*x+.02*appearance[i][j]) : appearance[i];
    }
    // New IDs are monotonic and scoped to one run; low confidence cannot spawn IDs.
    for (let i=0;i<dets.length;i++) if (!matched.has(i) && dets[i].confidence>=this.highThreshold) {
      const t={id:this.nextId++,classId:dets[i].class_id,axes:measure(dets[i]).map(x=>new Axis(x)),hist:appearance[i],age:0,hits:1};
      this.tracks.push(t); matched.set(i,t);
    }
    this.tracks=this.tracks.filter(t=>t.age<=this.maxAge);
    return [...matched].map(([i,t])=>({...dets[i],track_id:t.id,_confirmed:t.hits>=this.minHits}));
  }
}
