/** Renderer shared by live detections and exported evidence. Every box and label
 * comes from actual model/tracker output. */
const COLORS = ["#c5eca9", "#a6c9dc", "#e2c28c", "#bdb5dd", "#8dd7c6", "#d8ac9c"];
export function drawDetections(ctx, detections, nativeWidth, nativeHeight, width, height) {
  if (!nativeWidth || !nativeHeight || !width || !height) return;
  const sx = width / nativeWidth, sy = height / nativeHeight;
  const fontSize = Math.max(10, Math.min(14, width / 55));
  ctx.font = `500 ${fontSize}px ui-monospace, SFMono-Regular, Consolas, monospace`;
  ctx.textBaseline = "middle";
  for (const d of detections) {
    const color = COLORS[((d.class_id % COLORS.length) + COLORS.length) % COLORS.length];
    const x = Math.max(1, d.bbox_x * sx), y = Math.max(1, d.bbox_y * sy);
    const w = Math.min(d.bbox_w * sx, width - x - 1), h = Math.min(d.bbox_h * sy, height - y - 1);
    if (w <= 0 || h <= 0) continue;
    ctx.lineWidth = 1; ctx.strokeStyle = color + '99'; ctx.strokeRect(x, y, w, h);
    ctx.lineWidth = 2.5; ctx.strokeStyle = color;
    const corner = Math.min(13, w / 4, h / 4);
    for (const [cx, cy, dx, dy] of [[x,y,1,1],[x+w,y,-1,1],[x,y+h,1,-1],[x+w,y+h,-1,-1]]) {
      ctx.beginPath(); ctx.moveTo(cx + dx * corner, cy); ctx.lineTo(cx,cy); ctx.lineTo(cx,cy + dy * corner); ctx.stroke();
    }
    const label = `${d.class_label.toUpperCase()}${d.track_id != null ? ` / ${String(d.track_id).padStart(2,'0')}` : ''}  ${(d.confidence * 100).toFixed(0)}%`;
    const labelWidth = Math.min(width - 4, ctx.measureText(label).width + 18), labelHeight = fontSize + 12;
    const labelX = Math.max(2, Math.min(x, width - labelWidth - 2));
    const labelY = y >= labelHeight + 3 ? y - labelHeight - 2 : y + 3;
    ctx.fillStyle = '#102017ed'; ctx.fillRect(labelX, labelY, labelWidth, labelHeight);
    ctx.fillStyle = color; ctx.fillRect(labelX,labelY,2,labelHeight); ctx.fillText(label,labelX+9,labelY+labelHeight/2);
  }
}
