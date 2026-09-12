/** Renderer shared by live detections and exported evidence. Every box and label
 * comes from actual model/tracker output. */
const COLORS = ["#c5eca9", "#a6c9dc", "#e2c28c", "#bdb5dd", "#8dd7c6", "#d8ac9c"];
export function drawDetections(ctx, detections, nativeWidth, nativeHeight, width, height) {
  if (!nativeWidth || !nativeHeight || !width || !height) return;
  const sx = width / nativeWidth, sy = height / nativeHeight;
  const fontSize = Math.max(10, Math.min(14, width / 55));
  ctx.font = `500 ${fontSize}px ui-monospace, SFMono-Regular, Consolas, monospace`;
  ctx.textBaseline = "middle";
  const occupied = [];
  const people = detections.map(d => ({ x: d.bbox_x * sx, y: d.bbox_y * sy, w: d.bbox_w * sx, h: d.bbox_h * sy }));
  const overlaps = (a, b) => a.x < b.x + b.w + 3 && a.x + a.w + 3 > b.x && a.y < b.y + b.h + 3 && a.y + a.h + 3 > b.y;
  // Spatial order keeps label placement stable when confidence ranking changes.
  for (const d of [...detections].sort((a, b) => a.bbox_x - b.bbox_x)) {
    const colorIndex = d.track_id != null ? d.track_id - 1 : d.class_id;
    const color = COLORS[((colorIndex % COLORS.length) + COLORS.length) % COLORS.length];
    const x = Math.max(1, d.bbox_x * sx), y = Math.max(1, d.bbox_y * sy);
    const w = Math.min(d.bbox_w * sx, width - x - 1), h = Math.min(d.bbox_h * sy, height - y - 1);
    if (w <= 0 || h <= 0) continue;
    ctx.lineWidth = 1; ctx.strokeStyle = color + '99'; ctx.strokeRect(x, y, w, h);
    ctx.lineWidth = 2.5; ctx.strokeStyle = color;
    const corner = Math.min(13, w / 4, h / 4);
    for (const [cx, cy, dx, dy] of [[x,y,1,1],[x+w,y,-1,1],[x,y+h,1,-1],[x+w,y+h,-1,-1]]) {
      ctx.beginPath(); ctx.moveTo(cx + dx * corner, cy); ctx.lineTo(cx,cy); ctx.lineTo(cx,cy + dy * corner); ctx.stroke();
    }
    const label = d.track_id != null ? `ID ${String(d.track_id).padStart(2,'0')}` : `${d.class_label.toUpperCase()}  ${(d.confidence * 100).toFixed(0)}%`;
    const labelWidth = Math.min(width - 4, ctx.measureText(label).width + 18), labelHeight = fontSize + 12;
    const labelX = Math.max(2, Math.min(x, width - labelWidth - 2));
    let labelY = Math.max(2, y - labelHeight - 5);
    // Move crowded labels above the scene instead of covering other labels or people.
    for (let attempt = 0; attempt < 20; attempt++) {
      const candidate = { x: labelX, y: labelY, w: labelWidth, h: labelHeight };
      if (![...occupied, ...people].some(box => overlaps(candidate, box))) break;
      labelY -= labelHeight + 5;
      if (labelY < 2) { labelY = Math.min(height - labelHeight - 2, y + h + 5); break; }
    }
    occupied.push({ x: labelX, y: labelY, w: labelWidth, h: labelHeight });
    if (Math.abs(labelY + labelHeight - y) > 8) {
      ctx.strokeStyle = color + '99'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(labelX + 4, labelY + labelHeight); ctx.stroke();
    }
    ctx.fillStyle = '#102017ed'; ctx.fillRect(labelX, labelY, labelWidth, labelHeight);
    ctx.fillStyle = color; ctx.fillRect(labelX,labelY,2,labelHeight); ctx.fillText(label,labelX+9,labelY+labelHeight/2);
  }
}
