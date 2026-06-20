/** Recent logged sources (upload/webcam sessions) from Postgres.
 *  Click a row to scope the charts to that run; the × deletes the run. */
export default function SourceFeed({ sources, onSelect, onDelete, selectedId }) {
  return (
    <div className="panel">
      <h2>Logged sources</h2>
      {(!sources || sources.length === 0) ? (
        <p className="muted">Nothing logged yet.</p>
      ) : (
        <table className="feed">
          <thead>
            <tr>
              <th>id</th><th>kind</th><th>name</th><th>frames</th><th>when</th><th />
            </tr>
          </thead>
          <tbody>
            {sources.map((s) => (
              <tr
                key={s.id}
                className={s.id === selectedId ? "row-sel" : "row-click"}
                onClick={() => onSelect?.(s.id)}
              >
                <td>{s.id}</td>
                <td><span className={`tag tag-${s.kind}`}>{s.kind}</span></td>
                <td className="ellipsis">{s.filename || "—"}</td>
                <td>{s.frame_count}</td>
                <td className="muted">
                  {s.created_at ? new Date(s.created_at).toLocaleTimeString() : "—"}
                </td>
                <td>
                  <button
                    className="row-del"
                    title={`Delete run #${s.id}`}
                    aria-label={`Delete run #${s.id}`}
                    onClick={(e) => { e.stopPropagation(); onDelete?.(s.id); }}
                  >
                    ×
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
