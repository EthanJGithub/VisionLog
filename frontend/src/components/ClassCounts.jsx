import { ResponsiveBar } from "@nivo/bar";

const theme = {
  text: { fill: "#cbd5e1" },
  axis: { ticks: { text: { fill: "#94a3b8" } } },
  grid: { line: { stroke: "#1e293b" } },
  tooltip: { container: { background: "#0b1020", color: "#e2e8f0" } },
};

export default function ClassCounts({ data, scopeLabel }) {
  const top = (data || []).slice(0, 12);
  return (
    <div className="panel">
      <h2>
        Detections by class{scopeLabel ? <span className="muted"> · {scopeLabel}</span> : null}
      </h2>
      {top.length === 0 ? (
        <p className="muted">No detections logged yet.</p>
      ) : (
        <div style={{ height: 320 }}>
          <ResponsiveBar
            data={top}
            keys={["count"]}
            indexBy="class_label"
            margin={{ top: 10, right: 20, bottom: 70, left: 50 }}
            padding={0.3}
            colors={["#22d3ee"]}
            theme={theme}
            axisBottom={{ tickRotation: -40 }}
            enableLabel={false}
            animate
          />
        </div>
      )}
    </div>
  );
}
