import { useEffect, useRef } from "react";
import { drawDetections } from "../webgpu/overlay";

export default function BoxOverlay({ detections, nativeWidth, nativeHeight, displayWidth, displayHeight }) {
  const canvasRef = useRef(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !nativeWidth || !nativeHeight) return;
    const render = () => {
      const media = canvas.parentElement?.querySelector('video, img');
      const width = media?.clientWidth || displayWidth, height = media?.clientHeight || displayHeight;
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
      canvas.style.width = `${width}px`; canvas.style.height = `${height}px`;
      const ctx = canvas.getContext("2d");
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      ctx.clearRect(0, 0, width, height);
      drawDetections(ctx, detections, nativeWidth, nativeHeight, width, height);
    };
    render();
    const observer = new ResizeObserver(render);
    if (canvas.parentElement) observer.observe(canvas.parentElement);
    return () => observer.disconnect();
  }, [detections, nativeWidth, nativeHeight, displayWidth, displayHeight]);
  return <canvas ref={canvasRef} aria-hidden="true" style={{position:"absolute",top:0,left:0,pointerEvents:"none",width:displayWidth,height:displayHeight}} />;
}
