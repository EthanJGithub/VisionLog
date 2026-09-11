import { ResponsiveLine } from "@nivo/line";

const theme = {
  text: { fill: "#ecf1e9" },
  axis: { ticks: { text: { fill: "#a2b0a5" } } },
  grid: { line: { stroke: "#2d3b34" } },
  tooltip: { container: { background: "#17201d", color: "#ecf1e9" } },
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
            colors={["#bce8a4"]}
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
