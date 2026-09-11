/** API connectivity; browser models load only after a detection is started. */
export default function SystemHealth({ health }) {
  if (!health) return <div className="health"><span className="muted">Connecting to the logging service…</span></div>;
  const ok = health.status === "ok";
  return <div className="health"><span className={`dot ${ok ? "dot-ok" : "dot-bad"}`} /><span>{ok ? "Logging service connected" : "Logging service degraded"}</span>{health.max_upload_mb > 0 && <span className="muted"> · uploads up to {health.max_upload_mb} MB</span>}</div>;
}
