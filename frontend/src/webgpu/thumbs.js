// Crop a detected object's bounding box from the current frame into a tiny JPEG data URL,
// for the chatbot's visual object gallery. Reuses one offscreen canvas; cheap because we only
// call it ONCE per confirmed track (not every frame).
const _c = document.createElement("canvas");
const _ctx = _c.getContext("2d", { willReadFrequently: false });

/**
 * @param source  a drawable frame: HTMLVideoElement | HTMLCanvasElement | ImageBitmap
 * @param bbox    {x, y, w, h} in source pixels
 * @param max     longest thumbnail edge in px (default 256 — crisp in the gallery + when enlarged
 *                in the lightbox; only ever downscales, so small/distant objects keep native res)
 * @returns       "data:image/jpeg;base64,..." or null if the crop is degenerate
 */
export function cropThumb(source, bbox, max = 256) {
  const sw = Math.round(bbox.w);
  const sh = Math.round(bbox.h);
  if (sw < 4 || sh < 4) return null;
  const scale = Math.min(1, max / Math.max(sw, sh));
  const w = Math.max(1, Math.round(sw * scale));
  const h = Math.max(1, Math.round(sh * scale));
  _c.width = w;
  _c.height = h;
  _ctx.imageSmoothingQuality = "high";
  try {
    _ctx.drawImage(source, Math.round(bbox.x), Math.round(bbox.y), sw, sh, 0, 0, w, h);
    return _c.toDataURL("image/jpeg", 0.85);
  } catch {
    return null; // e.g. tainted canvas / source not yet drawable
  }
}

/**
 * Attach a thumbnail to each tracked detection the FIRST time its track is confirmed (mutates the
 * detections in place + records the track id in `thumbedSet`). The thumb then rides along in the
 * logged payload and is stored once per object for the gallery.
 */
export function attachThumbs(tracked, source, thumbedSet) {
  for (const t of tracked) {
    if (t._confirmed && t.track_id != null && !thumbedSet.has(t.track_id)) {
      const thumb = cropThumb(source, { x: t.bbox_x, y: t.bbox_y, w: t.bbox_w, h: t.bbox_h });
      if (thumb) {
        t.thumb = thumb;
        thumbedSet.add(t.track_id);
      }
    }
  }
}
