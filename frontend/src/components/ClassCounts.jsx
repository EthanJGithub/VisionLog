import { ResponsiveBar } from "@nivo/bar";

const theme = {
  text: { fill: "#ecf1e9" },
  axis: { ticks: { text: { fill: "#a2b0a5" } } },
  grid: { line: { stroke: "#2d3b34" } },
  tooltip: { container: { background: "#17201d", color: "#ecf1e9" } },
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
            colors={["#bce8a4"]}
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
