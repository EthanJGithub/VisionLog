import { useMemo, useRef, useState } from "react";
import { api } from "../api";
import { SERVER_MODELS } from "../models";
import BoxOverlay from "./BoxOverlay";

/**
 * Upload mode (server-side): POST a video file + chosen model, get back logged
 * detections, then play the native <video> with boxes overlaid — synced to
 * currentTime (no re-encode). Supports YOLO26 n/s/m/x and YOLOE-26 open-vocab.
 */
export default function VideoUpload({ onLogged }) {
  const videoRef = useRef(null);
  const fileRef = useRef(null);
  const [model, setModel] = useState(SERVER_MODELS[0].id);
  const [prompts, setPrompts] = useState("");
  const [videoUrl, setVideoUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [now, setNow] = useState(0);
  const [dims, setDims] = useState({ nw: 0, nh: 0, dw: 0, dh: 0 });

  const selected = SERVER_MODELS.find((m) => m.id === model);
  const isOpenVocab = !!selected?.openVocab;

  async function handleFile(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    if (isOpenVocab && !prompts.trim()) {
      setError("Enter comma-separated class names to detect (e.g. 'forklift, hard hat').");
      e.target.value = "";
      return;
    }
    setError(null);
    setResult(null);
    setVideoUrl(URL.createObjectURL(file));
    setBusy(true);
    try {
      const res = await api.upload(file, { model, prompts });
      setResult(res);
      onLogged?.();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  // Detections within ~1 sample window of the current playback time.
  const window = result ? 1 / (result.source.fps || 5) + 0.05 : 0;
  const active = useMemo(() => {
    if (!result) return [];
    return result.detections.filter((d) => Math.abs(d.ts_seconds - now) <= window);
  }, [result, now, window]);

  function syncDims() {
    const v = videoRef.current;
    if (!v) return;
    setDims({
      nw: v.videoWidth,
      nh: v.videoHeight,
      dw: v.clientWidth,
      dh: v.clientHeight,
    });
  }

  return (
    <div className="panel">
      <h2>Upload a video — server-side</h2>
      <p className="muted">
        Runs the selected model on the server, logs detections to PostgreSQL, then replays
        them as boxes over your video.
      </p>

      <div className="controls">
        <label>
          Model{" "}
          <select value={model} onChange={(e) => setModel(e.target.value)} disabled={busy}>
            {SERVER_MODELS.map((m) => (
              <option key={m.id} value={m.id}>{m.label}</option>
            ))}
          </select>
        </label>
      </div>
      {isOpenVocab && (
        <input
          className="prompts"
          type="text"
          placeholder="classes to detect, comma-separated — e.g. forklift, hard hat, pallet"
          value={prompts}
          onChange={(e) => setPrompts(e.target.value)}
          disabled={busy}
        />
      )}
      <p className="muted" style={{ marginTop: 8 }}>{selected?.note}</p>
      <input ref={fileRef} type="file" accept="video/*" onChange={handleFile} disabled={busy} />
      {busy && <p className="muted">Analysing frames with YOLO26…</p>}
      {error && <p className="error">⚠ {error}</p>}

      {videoUrl && (
        <div className="video-wrap" style={{ position: "relative", marginTop: 12 }}>
          <video
            ref={videoRef}
            src={videoUrl}
            controls
            style={{ width: "100%", borderRadius: 8, display: "block" }}
            onLoadedMetadata={syncDims}
            onTimeUpdate={(e) => setNow(e.target.currentTime)}
          />
          <BoxOverlay
            detections={active}
            nativeWidth={dims.nw}
            nativeHeight={dims.nh}
            displayWidth={dims.dw}
            displayHeight={dims.dh}
          />
        </div>
      )}

      {result && (
        <div className="stat-row" style={{ marginTop: 12 }}>
          <span className="chip">{result.frames_analysed} frames analysed</span>
          <span className="chip">{result.detections_logged} detections logged</span>
          <span className="chip">model {result.source.model_version}</span>
          <span className="chip">conf ≥ {result.source.conf_threshold}</span>
        </div>
      )}
    </div>
  );
}
