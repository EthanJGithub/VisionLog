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
export function cropThumb(source, bbox, { max = 256, maskCanvas = null } = {}) {
  const sw = Math.round(bbox.w);
  const sh = Math.round(bbox.h);
  if (sw < 4 || sh < 4) return null;
  const scale = Math.min(1, max / Math.max(sw, sh));
  const w = Math.max(1, Math.round(sw * scale));
  const h = Math.max(1, Math.round(sh * scale));
  _c.width = w;
  _c.height = h;
  _ctx.clearRect(0, 0, w, h);
  _ctx.globalCompositeOperation = "source-over";
  _ctx.imageSmoothingQuality = "high";
  try {
    _ctx.drawImage(source, Math.round(bbox.x), Math.round(bbox.y), sw, sh, 0, 0, w, h);
    if (maskCanvas) {
      // Keep only the masked (object) pixels → transparent background. PNG preserves alpha.
      _ctx.globalCompositeOperation = "destination-in";
      _ctx.drawImage(maskCanvas, 0, 0, w, h);
      _ctx.globalCompositeOperation = "source-over";
      return _c.toDataURL("image/png");
    }
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
export function attachThumbs(tracked, source, thumbedSet, detector = null) {
  for (const t of tracked) {
    if (t._confirmed && t.track_id != null && !thumbedSet.has(t.track_id)) {
      // Segmentation cutout (transparent bg) for seg models (YOLOE); plain crop otherwise.
      const maskCanvas =
        detector && detector.head === "v8seg" && t._coeffs
          ? detector.instanceMaskCanvas(t)
          : null;
      const thumb = cropThumb(
        source, { x: t.bbox_x, y: t.bbox_y, w: t.bbox_w, h: t.bbox_h }, { maskCanvas }
      );
      if (thumb) {
        t.thumb = thumb;
        thumbedSet.add(t.track_id);
      }
    }
  }
}
