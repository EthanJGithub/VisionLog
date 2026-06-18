import { ResponsiveLine } from "@nivo/line";

const theme = {
  text: { fill: "#cbd5e1" },
  axis: { ticks: { text: { fill: "#94a3b8" } } },
  grid: { line: { stroke: "#1e293b" } },
  tooltip: { container: { background: "#0b1020", color: "#e2e8f0" } },
};

/**
 * Detections-over-time for the most recent source. `detections` is the raw list
 * for one source; we bucket counts per second.
 */
export default function DetectionTimeline({ detections }) {
  const buckets = new Map();
  for (const d of detections || []) {
    const sec = Math.floor(d.ts_seconds);
    buckets.set(sec, (buckets.get(sec) || 0) + 1);
  }
  const points = [...buckets.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([x, y]) => ({ x, y }));

  return (
    <div className="panel">
      <h2>Detections over time (latest source)</h2>
      {points.length === 0 ? (
        <p className="muted">No detections logged yet.</p>
      ) : (
        <div style={{ height: 280 }}>
          <ResponsiveLine
            data={[{ id: "detections", data: points }]}
            margin={{ top: 10, right: 20, bottom: 50, left: 50 }}
            xScale={{ type: "linear" }}
            yScale={{ type: "linear", min: 0 }}
            colors={["#a3e635"]}
            theme={theme}
            axisBottom={{ legend: "time (s)", legendOffset: 36, legendPosition: "middle" }}
            axisLeft={{ legend: "count", legendOffset: -40, legendPosition: "middle" }}
            enablePoints
            pointSize={6}
            useMesh
            animate
          />
        </div>
      )}
    </div>
  );
}
