import { lazy, Suspense, useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api";
import VideoUpload from "./components/VideoUpload";
import LiveWebcam from "./components/LiveWebcam";
import LiveStream from "./components/LiveStream";
import SourceFeed from "./components/SourceFeed";
import SystemHealth from "./components/SystemHealth";
// Code-split the heavier / Nivo-using parts out of the initial bundle.
const Benchmarks = lazy(() => import("./components/Benchmarks"));
const Chat = lazy(() => import("./components/Chat"));
const ClassCounts = lazy(() => import("./components/ClassCounts"));
const DetectionTimeline = lazy(() => import("./components/DetectionTimeline"));

export default function App() {
  const [mode, setMode] = useState("upload");
  const [health, setHealth] = useState(null);
  const [stats, setStats] = useState(null);
  const [sources, setSources] = useState([]);
  // selectedId === null → "All runs"; otherwise stats/charts are scoped to that run.
  const [selectedId, setSelectedId] = useState(null);
  const [timeline, setTimeline] = useState([]);
  // Ref so refresh() (a stable callback, also used as onLogged) always reads the live scope.
  const scopeRef = useRef(null);

  const refresh = useCallback(async () => {
    try {
      const scope = scopeRef.current;
      const [s, src] = await Promise.all([api.stats(scope), api.sources()]);
      setStats(s);
      setSources(src);
      // Timeline shows the selected run, or the most recent run when viewing "All".
      const tlId = scope ?? src[0]?.id ?? null;
      setTimeline(tlId != null ? await api.detections(tlId) : []);
    } catch {
      /* surfaced via health dot */
    }
  }, []);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
    refresh();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  async function selectScope(id) {
    setSelectedId(id);
    scopeRef.current = id;
    await refresh();
  }

  async function resetAll() {
    if (!window.confirm("Delete ALL logged runs and detections? This cannot be undone.")) return;
    await api.clearAll();
    await selectScope(null);
  }

  async function deleteRun(id) {
    if (!window.confirm(`Delete run #${id} and its detections?`)) return;
    await api.deleteSource(id);
    await selectScope(scopeRef.current === id ? null : scopeRef.current);
  }

  return (
    <div className="app">
      <header className="topbar">
        <div>
          <h1>VisionLog</h1>
          <p className="tagline">YOLO26 detection on your video → logged to PostgreSQL</p>
        </div>
        <div className="health-slot">
          <SystemHealth health={health} totals={stats?.totals} />
        </div>
      </header>

      <div className="tabs">
        <button className={mode === "upload" ? "tab tab-on" : "tab"} onClick={() => setMode("upload")}>
          Upload video
        </button>
        <button className={mode === "webcam" ? "tab tab-on" : "tab"} onClick={() => setMode("webcam")}>
          Live webcam (your GPU)
        </button>
        <button className={mode === "stream" ? "tab tab-on" : "tab"} onClick={() => setMode("stream")}>
          Live stream (UDP)
        </button>
        <button className={mode === "benchmarks" ? "tab tab-on" : "tab"} onClick={() => setMode("benchmarks")}>
          Benchmarks
        </button>
        <button className={mode === "chat" ? "tab tab-on" : "tab"} onClick={() => setMode("chat")}>
          Ask the data
        </button>
      </div>

      {mode === "benchmarks" ? (
        <Suspense fallback={<div className="panel"><p className="muted">Loading…</p></div>}>
          <Benchmarks />
        </Suspense>
      ) : mode === "chat" ? (
        <Suspense fallback={<div className="panel"><p className="muted">Loading…</p></div>}>
          <Chat />
        </Suspense>
      ) : (
        <div className="grid">
          <div className="col">
            {mode === "upload" ? (
              <VideoUpload onLogged={refresh} />
            ) : mode === "webcam" ? (
              <LiveWebcam onLogged={refresh} />
            ) : (
              <LiveStream onLogged={refresh} />
            )}
            <SourceFeed
              sources={sources}
              onSelect={selectScope}
              onDelete={deleteRun}
              selectedId={selectedId}
            />
          </div>
          <div className="col">
            {sources.length > 0 && (
              <div className="panel">
                <div className="scope-bar">
                  <label>
                    View
                    <select
                      value={selectedId ?? ""}
                      onChange={(e) =>
                        selectScope(e.target.value === "" ? null : Number(e.target.value))
                      }
                    >
                      <option value="">All runs ({sources.length})</option>
                      {sources.map((s) => (
                        <option key={s.id} value={s.id}>
                          #{s.id} · {s.kind}
                          {s.filename ? ` · ${s.filename}` : ""} · {s.frame_count} frames
                        </option>
                      ))}
                    </select>
                  </label>
                  <span className="spacer" />
                  <button className="btn-danger" onClick={resetAll}>
                    Reset all data
                  </button>
                </div>
              </div>
            )}
            {(stats?.totals?.detections ?? 0) > 0 ? (
              <Suspense fallback={<div className="panel charts-ph" />}>
                <ClassCounts
                  data={stats?.class_counts}
                  scopeLabel={selectedId != null ? `run #${selectedId}` : "all runs"}
                />
                <DetectionTimeline detections={timeline} />
              </Suspense>
            ) : (
              <div className="panel charts-ph">
                <h2>Charts</h2>
                <p className="muted">
                  Run a detection (upload or webcam) to populate per-class counts and the
                  detections-over-time chart.
                </p>
              </div>
            )}
          </div>
        </div>
      )}

      <footer className="foot muted">
        YOLO26 (AGPL-3.0) · everything runs in-browser on your GPU (WebGPU) — n/s/m/x +
        open-vocab (YOLOE) · no server inference · detections persisted with model version
        &amp; threshold for auditability
      </footer>
    </div>
  );
}
