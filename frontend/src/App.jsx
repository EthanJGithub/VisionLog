import { useCallback, useEffect, useState } from "react";
import { api } from "./api";
import VideoUpload from "./components/VideoUpload";
import LiveWebcam from "./components/LiveWebcam";
import ClassCounts from "./components/ClassCounts";
import DetectionTimeline from "./components/DetectionTimeline";
import SourceFeed from "./components/SourceFeed";
import SystemHealth from "./components/SystemHealth";

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
        <SystemHealth health={health} totals={stats?.totals} />
      </header>

      <div className="tabs">
        <button className={mode === "upload" ? "tab tab-on" : "tab"} onClick={() => setMode("upload")}>
          Upload video
        </button>
        <button className={mode === "webcam" ? "tab tab-on" : "tab"} onClick={() => setMode("webcam")}>
          Live webcam (your GPU)
        </button>
      </div>

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
          <ClassCounts data={stats?.class_counts} />
          <DetectionTimeline detections={timeline} />
        </div>
      </div>

      <footer className="foot muted">
        YOLO26 (AGPL-3.0) · webcam runs in-browser on your GPU (WebGPU) · uploads run
        server-side (n/s/m/x + open-vocab) · detections persisted with model version &
        threshold for auditability
      </footer>
    </div>
  );
}
