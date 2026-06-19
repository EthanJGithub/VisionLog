import { useEffect, useState } from "react";
import { ResponsiveBar } from "@nivo/bar";

const theme = {
  text: { fill: "#cbd5e1" },
  axis: { ticks: { text: { fill: "#94a3b8" } } },
  grid: { line: { stroke: "#1e293b" } },
  tooltip: { container: { background: "#0b1020", color: "#e2e8f0" } },
};

const pct = (x) => (x == null ? "—" : (x * 100).toFixed(1) + "%");

/** Model evaluation + latency report (loaded from /benchmarks/benchmarks.json). */
export default function Benchmarks() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    fetch("/benchmarks/benchmarks.json")
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error("no report"))))
      .then(setData)
      .catch(() => setErr("Benchmark report not found — run `python -m src.eval.benchmark`."));
  }, []);

  if (err) return <div className="panel"><h2>Benchmarks</h2><p className="muted">{err}</p></div>;
  if (!data) return <div className="panel"><h2>Benchmarks</h2><p className="muted">Loading…</p></div>;

  const perClass = (data.per_class_ap || []).map((c) => ({ class: c.class, AP: +(c.ap * 100).toFixed(1) }));

  return (
    <div className="grid">
      <div className="col">
        <div className="panel">
          <h2>Accuracy — {data.dataset}</h2>
          <p className="muted">Measured with <code>model.val()</code> at {data.input_size}px. mAP = mean average precision.</p>
          <table className="feed">
            <thead><tr><th>model</th><th>params</th><th>mAP@50</th><th>mAP@50-95</th><th>precision</th><th>recall</th></tr></thead>
            <tbody>
              {data.models.map((m) => (
                <tr key={m.id}>
                  <td>{m.id}</td>
                  <td className="muted">{m.params_m ? `${m.params_m}M` : "—"}</td>
                  <td>{pct(m.map50)}</td>
                  <td>{pct(m.map50_95)}</td>
                  <td>{pct(m.precision)}</td>
                  <td>{pct(m.recall)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="panel">
          <h2>Latency / throughput</h2>
          <p className="muted">CPU = onnxruntime per-frame (server). WebGPU = in-browser fps on an NVIDIA GPU.</p>
          <table className="feed">
            <thead><tr><th>model</th><th>CPU ms/frame</th><th>WebGPU fps (GPU)</th></tr></thead>
            <tbody>
              {data.models.map((m) => (
                <tr key={m.id}>
                  <td>{m.id}</td>
                  <td>{m.cpu_ms != null ? `${m.cpu_ms} ms` : "—"}</td>
                  <td className="chip-ok" style={{ borderRadius: 0 }}>{m.webgpu_fps != null ? `${m.webgpu_fps} fps` : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="col">
        <div className="panel">
          <h2>Per-class AP (top 15)</h2>
          {perClass.length === 0 ? <p className="muted">No per-class data.</p> : (
            <div style={{ height: 360 }}>
              <ResponsiveBar
                data={perClass} keys={["AP"]} indexBy="class" layout="horizontal"
                margin={{ top: 6, right: 20, bottom: 30, left: 110 }} padding={0.3}
                colors={["#a3e635"]} theme={theme} enableLabel={false}
                axisBottom={{ legend: "AP %", legendPosition: "middle", legendOffset: 24 }} animate
              />
            </div>
          )}
        </div>
        <div className="panel">
          <h2>Diagnostics</h2>
          <p className="muted">Confusion matrix &amp; precision–recall curve (Ultralytics <code>val</code>).</p>
          <div className="diag">
            <img src="/benchmarks/confusion_matrix_normalized.png" alt="confusion matrix" onError={(e)=>{e.target.style.display='none';}} />
            <img src="/benchmarks/PR_curve.png" alt="PR curve" onError={(e)=>{e.target.style.display='none';}} />
          </div>
        </div>
      </div>
    </div>
  );
}
