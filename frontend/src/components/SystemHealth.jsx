/** Backend health + free-tier limits + totals header. */
export default function SystemHealth({ health, totals }) {
  if (!health) return null;
  const ok = health.model_loaded;
  return (
    <div className="health">
      <span className={`dot ${ok ? "dot-ok" : "dot-bad"}`} />
      <span>{ok ? "Model loaded" : "Model NOT loaded"}</span>
      <span className="sep">·</span>
      <span>{health.model_version}</span>
      <span className="sep">·</span>
      <span>{totals?.objects ?? 0} objects · {totals?.detections ?? 0} detections / {totals?.sources ?? 0} sources</span>
      <span className="sep">·</span>
      <span className="muted">
        limits: {health.max_upload_mb}MB / {health.max_duration_seconds}s / {health.sample_fps}fps
      </span>
    </div>
  );
}
