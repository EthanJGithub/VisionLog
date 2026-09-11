export default function TrackingToggle({ enabled, onChange, disabled }) {
  return <div className="tracking-option">
    <label><input type="checkbox" checked={enabled} disabled={disabled} onChange={e => onChange(e.target.checked)} />
      <span>Advanced object IDs <small>{enabled ? 'ON' : 'OFF'}</small></span>
    </label>
    <p>{enabled ? 'Kalman motion + global matching + appearance. Associates objects across frames within each run.' : 'Detection only. Skips tracking, object crops and semantic embeddings to reduce device workload.'}
      {disabled && ' Stop the current run to change tracking mode.'}</p>
  </div>;
}
