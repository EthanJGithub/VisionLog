/** Open-vocab "prompting": toggle which vocabulary classes to detect/show. */
export default function ClassFilter({ classes, enabled, onToggle, onAll, onNone }) {
  if (!classes?.length) return null;
  return (
    <div className="classfilter">
      <div className="classfilter-head">
        <span className="muted">Detect which classes:</span>
        <button className="link" onClick={onAll}>all</button>
        <button className="link" onClick={onNone}>none</button>
      </div>
      <div className="chipwrap">
        {classes.map((c) => (
          <button
            key={c}
            className={`fchip ${enabled.has(c) ? "fchip-on" : ""}`}
            onClick={() => onToggle(c)}
          >
            {c}
          </button>
        ))}
      </div>
    </div>
  );
}
