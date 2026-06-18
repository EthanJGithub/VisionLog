/** Recent logged sources (upload/webcam sessions) from Postgres. */
export default function SourceFeed({ sources, onSelect, selectedId }) {
  return (
    <div className="panel">
      <h2>Logged sources</h2>
      {(!sources || sources.length === 0) ? (
        <p className="muted">Nothing logged yet.</p>
      ) : (
        <table className="feed">
          <thead>
            <tr>
              <th>id</th><th>kind</th><th>name</th><th>frames</th><th>when</th>
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
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
