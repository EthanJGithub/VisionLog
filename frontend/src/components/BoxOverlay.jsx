import { useEffect, useRef } from "react";

// Palette keyed by class id (stable colors per class).
const COLORS = [
  "#22d3ee", "#f472b6", "#a3e635", "#fbbf24", "#818cf8",
  "#fb7185", "#34d399", "#e879f9", "#60a5fa", "#facc15",
];

function colorFor(classId) {
  return COLORS[((classId % COLORS.length) + COLORS.length) % COLORS.length];
}

/**
 * Draws bounding boxes on a transparent canvas overlaid on a media element.
 * Detections are in *original-frame* pixels; we scale to the displayed size.
 */
export default function BoxOverlay({ detections, nativeWidth, nativeHeight, displayWidth, displayHeight }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !nativeWidth || !nativeHeight) return;
    canvas.width = displayWidth;
    canvas.height = displayHeight;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const sx = displayWidth / nativeWidth;
    const sy = displayHeight / nativeHeight;
    ctx.lineWidth = 2;
    ctx.font = "14px ui-monospace, monospace";
    ctx.textBaseline = "top";

    for (const d of detections) {
      const x = d.bbox_x * sx;
      const y = d.bbox_y * sy;
      const w = d.bbox_w * sx;
      const h = d.bbox_h * sy;
      const color = colorFor(d.class_id);
      ctx.strokeStyle = color;
      ctx.strokeRect(x, y, w, h);

      const label = `${d.class_label} ${(d.confidence * 100).toFixed(0)}%`;
      const tw = ctx.measureText(label).width + 8;
      ctx.fillStyle = color;
      ctx.fillRect(x, Math.max(0, y - 18), tw, 18);
      ctx.fillStyle = "#0b1020";
      ctx.fillText(label, x + 4, Math.max(0, y - 17));
    }
  }, [detections, nativeWidth, nativeHeight, displayWidth, displayHeight]);

  return (
    <canvas
      ref={canvasRef}
      style={{ position: "absolute", top: 0, left: 0, pointerEvents: "none" }}
    />
  );
}
