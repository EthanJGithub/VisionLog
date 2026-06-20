import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api";
import { createDetector } from "../webgpu/detector";
import { SimpleTracker } from "../webgpu/tracker";
import { attachThumbs } from "../webgpu/thumbs";
import { warmClip } from "../webgpu/clip";
import { createEmbedder } from "../webgpu/embedQueue";
import { CLIENT_MODELS } from "../models";
import BoxOverlay from "./BoxOverlay";
import ClassFilter from "./ClassFilter";

const LOG_FLUSH_MS = 1000;
const DEFAULT_CONF = 0.4;

/**
 * Live network stream (UDP/RTP/RTSP) → detection IN THE BROWSER on the visitor's GPU.
 *
 * Browsers can't read raw UDP, so an always-on bridge worker (bridge/server.py) ingests the
 * stream with ffmpeg and pushes JPEG frames over a WebSocket. Here we decode each frame and run
 * the SAME client-side detector + tracker + Postgres logging as the webcam path — only the pixel
 * source differs. Set the bridge URL below (or VITE_STREAM_BRIDGE at build time).
 */
export default function LiveStream({ onLogged }) {
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const detectorRef = useRef(null);
  const trackerRef = useRef(null);
  const thumbedRef = useRef(new Set()); // track_ids already thumbnailed (one crop per object)
  const embedderRef = useRef(null);     // background CLIP embedder (semantic search)
  const runningRef = useRef(false);
  const busyRef = useRef(false);        // one inference at a time; drop frames to stay live
  const sourceIdRef = useRef(null);
  const pendingRef = useRef([]);
  const frameNoRef = useRef(0);
  const enabledRef = useRef(null);

  const INPUT_SIZE = 640;
  const DEFAULT_BRIDGE = import.meta.env.VITE_STREAM_BRIDGE || "";
  const [bridgeUrl, setBridgeUrl] = useState(DEFAULT_BRIDGE);
  const [modelId, setModelId] = useState(CLIENT_MODELS[0].id);
  const [vocab, setVocab] = useState(null);
  const [enabled, setEnabled] = useState(new Set());
  const [conf, setConf] = useState(DEFAULT_CONF);
  const [backend, setBackend] = useState(null);
  const [loading, setLoading] = useState(false);
  const [running, setRunning] = useState(false);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState(null);
  const [dets, setDets] = useState([]);
  const [fps, setFps] = useState(0);
  const [logged, setLogged] = useState(0);
  const [dims, setDims] = useState({ nw: 0, nh: 0, dw: 0, dh: 0 });
  const selectedModel = CLIENT_MODELS.find((m) => m.id === modelId);

  function changeConf(v) {
    setConf(v);
    if (detectorRef.current) detectorRef.current.confThreshold = v;
  }

  const stop = useCallback(() => {
    runningRef.current = false;
    setRunning(false);
    setConnected(false);
    try { wsRef.current?.close(); } catch { /* ignore */ }
    wsRef.current = null;
    detectorRef.current?.dispose();
    detectorRef.current = null;
    trackerRef.current = null;
    onLogged?.();
  }, [onLogged]);

  useEffect(() => () => stop(), [stop]);

  function toWs(url) {
    const u = url.trim();
    if (/^wss?:\/\//i.test(u)) return u;
    if (/^https?:\/\//i.test(u)) return u.replace(/^http/i, "ws").replace(/\/?$/, "") + "/ws";
    return u; // assume already a ws path
  }

  async function start() {
    setError(null);
    if (!bridgeUrl.trim()) {
      setError("Enter the bridge WebSocket URL (e.g. wss://your-bridge.fly.dev/ws).");
      return;
    }
    setLoading(true);
    try {
      const model = CLIENT_MODELS.find((m) => m.id === modelId);
      const { detector, backend: be, classNames } = await createDetector(model, {
        inputSize: INPUT_SIZE,
        conf,
      });
      detectorRef.current = detector;
      trackerRef.current = new SimpleTracker();
      setBackend(be);
      if (classNames) {
        setVocab(classNames);
        const all = new Set(classNames);
        setEnabled(all);
        enabledRef.current = all;
      } else {
        setVocab(null);
        enabledRef.current = null;
      }

      const session = await api.clientSession({
        kind: "stream",
        model_version: `${model.id}-webgpu`,
        conf_threshold: conf,
      });
      sourceIdRef.current = session.id;
      frameNoRef.current = 0;
      pendingRef.current = [];
      thumbedRef.current = new Set();
      embedderRef.current = createEmbedder(() => sourceIdRef.current, () => runningRef.current);
      warmClip();
      setLogged(0);

      const ws = new WebSocket(toWs(bridgeUrl));
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;
      ws.onopen = () => setConnected(true);
      ws.onerror = () => setError("Could not connect to the bridge. Check the URL / that it's running.");
      ws.onclose = () => { setConnected(false); if (runningRef.current) setError("Bridge connection closed."); };
      ws.onmessage = (ev) => onFrame(ev.data);

      runningRef.current = true;
      setRunning(true);
      setLoading(false);
      startFlush();
    } catch (err) {
      setError(err.message || "Could not start the stream.");
      setLoading(false);
    }
  }

  // Decode one JPEG frame, draw it, run detection. Drop frames while one is in flight so we
  // always process the freshest frame (live, not a growing backlog).
  async function onFrame(data) {
    if (!runningRef.current || busyRef.current || !detectorRef.current) return;
    busyRef.current = true;
    const t0 = performance.now();
    try {
      const bitmap = await createImageBitmap(new Blob([data], { type: "image/jpeg" }));
      const canvas = canvasRef.current;
      if (canvas) {
        if (canvas.width !== bitmap.width) canvas.width = bitmap.width;
        if (canvas.height !== bitmap.height) canvas.height = bitmap.height;
        canvas.getContext("2d").drawImage(bitmap, 0, 0);
      }
      const all = await detectorRef.current.detect(bitmap, bitmap.width, bitmap.height);
      const filtered = enabledRef.current
        ? all.filter((x) => enabledRef.current.has(x.class_label))
        : all;
      const tracked = trackerRef.current ? trackerRef.current.update(filtered, canvas) : filtered;
      if (canvas) attachThumbs(tracked, canvas, thumbedRef.current, detectorRef.current); // crop (+seg cutout)
      for (const t of tracked) {
        if (t.thumb && t.track_id != null) embedderRef.current?.enqueue(t.track_id, t.thumb);
      }
      setDets(tracked);
      const now = performance.now();
      const inst = 1000 / Math.max(1, now - t0);
      setFps((p) => (p ? p * 0.8 + inst * 0.2 : inst));
      setDims({
        nw: bitmap.width, nh: bitmap.height,
        dw: canvasRef.current?.clientWidth || 0, dh: canvasRef.current?.clientHeight || 0,
      });
      const confirmed = tracked.filter((t) => t._confirmed).map(({ _confirmed, _coeffs, ...rest }) => rest);
      pendingRef.current.push({ frame_number: frameNoRef.current++, ts_seconds: now / 1000, detections: confirmed });
      bitmap.close();
    } catch (err) {
      setError(`Inference error: ${err.message}`);
    } finally {
      busyRef.current = false;
    }
  }

  function startFlush() {
    const tick = async () => {
      if (!runningRef.current) return;
      const batch = pendingRef.current;
      pendingRef.current = [];
      if (batch.length && sourceIdRef.current != null) {
        const step = Math.max(1, Math.floor(batch.length / 3));
        // Keep the regular sample PLUS any frame carrying an object thumbnail (captured once per
        // track) so crops are never dropped by sampling.
        const sampled = batch.filter(
          (f, i) => i % step === 0 || f.detections.some((d) => d.thumb)
        );
        try {
          const res = await api.clientDetections(sourceIdRef.current, sampled);
          setLogged((n) => n + (res.logged || 0));
          onLogged?.();
        } catch { /* surfaced via health dot */ }
      }
      if (runningRef.current) setTimeout(tick, LOG_FLUSH_MS);
    };
    setTimeout(tick, LOG_FLUSH_MS);
  }

  const isCpu = backend === "wasm";

  return (
    <div className="panel">
      <h2>Live stream (UDP/RTSP) — runs on your device</h2>
      <p className="muted">
        A network video stream (UDP/RTP/RTSP) is relayed by an always-on{" "}
        <strong>bridge worker</strong> (ffmpeg → WebSocket); detection still runs{" "}
        <strong>in your browser</strong> on your GPU. Only sampled detections are logged.
        See <code>bridge/README.md</code> to deploy the worker.
      </p>

      <div className="controls">
        <label style={{ flex: "1 1 320px" }}>
          Bridge URL{" "}
          <input
            type="text"
            className="prompts"
            style={{ width: "100%" }}
            placeholder="wss://your-bridge.fly.dev/ws"
            value={bridgeUrl}
            onChange={(e) => setBridgeUrl(e.target.value)}
            disabled={running}
          />
        </label>
      </div>
      <div className="controls">
        <label>
          Model{" "}
          <select value={modelId} onChange={(e) => setModelId(e.target.value)} disabled={running}>
            {CLIENT_MODELS.map((m) => (
              <option key={m.id} value={m.id}>{m.label} (~{m.sizeMB}MB)</option>
            ))}
          </select>
        </label>
        <label>
          Confidence ≥ {conf.toFixed(2)}{" "}
          <input type="range" min="0.1" max="0.9" step="0.05" value={conf}
            onChange={(e) => changeConf(+e.target.value)} />
        </label>
        {!running ? (
          <button className="btn" onClick={start} disabled={loading}>
            {loading ? "Loading model…" : "Connect"}
          </button>
        ) : (
          <button className="btn btn-stop" onClick={stop}>Disconnect</button>
        )}
      </div>

      {selectedModel?.openVocab && <p className="muted" style={{ marginTop: 4 }}>{selectedModel.note}</p>}
      {running && vocab && (
        <ClassFilter
          classes={vocab} enabled={enabled}
          onToggle={(c) => { const n = new Set(enabled); n.has(c) ? n.delete(c) : n.add(c); setEnabled(n); enabledRef.current = n; }}
          onAll={() => { const a = new Set(vocab); setEnabled(a); enabledRef.current = a; }}
          onNone={() => { const e = new Set(); setEnabled(e); enabledRef.current = e; }}
          onSet={(s) => { setEnabled(s); enabledRef.current = s; }}
        />
      )}

      {error && <p className="error">⚠ {error}</p>}

      {running && (
        <div className="stat-row" style={{ marginTop: 10 }}>
          <span className={`chip ${connected ? "chip-ok" : "chip-warn"}`}>
            {connected ? "● connected" : "connecting…"}
          </span>
          <span className={`chip ${isCpu ? "chip-warn" : "chip-ok"}`}>
            {backend === "webgpu" ? "⚡ WebGPU (GPU)" : "CPU (WASM)"}
          </span>
          <span className="chip">{fps.toFixed(1)} fps</span>
          <span className="chip">{dets.length} objects</span>
          <span className="chip">{logged} logged</span>
        </div>
      )}

      <div className="video-wrap" style={{ position: "relative", marginTop: 12 }}>
        <canvas ref={canvasRef} aria-label="Live stream with detection overlay"
          style={{ width: "100%", borderRadius: 8, display: "block", background: "#0b1322" }} />
        <BoxOverlay
          detections={dets}
          nativeWidth={dims.nw} nativeHeight={dims.nh}
          displayWidth={canvasRef.current?.clientWidth || 0}
          displayHeight={canvasRef.current?.clientHeight || 0}
        />
      </div>
    </div>
  );
}
