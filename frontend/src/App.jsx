import { lazy, Suspense, useCallback, useEffect, useState } from "react";
import { api } from "./api";
import VideoUpload from "./components/VideoUpload";
import LiveWebcam from "./components/LiveWebcam";
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
  const [selectedId, setSelectedId] = useState(null);
  const [timeline, setTimeline] = useState([]);

  const refresh = useCallback(async () => {
    try {
      const [s, src] = await Promise.all([api.stats(), api.sources()]);
      setStats(s);
      setSources(src);
      const sel = selectedId ?? src[0]?.id ?? null;
      setSelectedId(sel);
      if (sel != null) setTimeline(await api.detections(sel));
    } catch {
      /* surfaced via health dot */
    }
  }, [selectedId]);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
    refresh();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  async function selectSource(id) {
    setSelectedId(id);
    setTimeline(await api.detections(id));
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
            ) : (
              <LiveWebcam onLogged={refresh} />
            )}
            <SourceFeed sources={sources} onSelect={selectSource} selectedId={selectedId} />
          </div>
          <div className="col">
            {(stats?.totals?.detections ?? 0) > 0 ? (
              <Suspense fallback={<div className="panel charts-ph" />}>
                <ClassCounts data={stats?.class_counts} />
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
