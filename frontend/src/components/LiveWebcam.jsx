import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api";
import { createDetector } from "../webgpu/detector";
import { SimpleTracker } from "../webgpu/tracker";
import { attachThumbs } from "../webgpu/thumbs";
import { CLIENT_MODELS } from "../models";
import BoxOverlay from "./BoxOverlay";
import ClassFilter from "./ClassFilter";

const LOG_FLUSH_MS = 1000; // sample detections to Postgres ~1x/sec (don't spam the DB)
const DEFAULT_CONF = 0.4;  // robust default — fewer false positives in the SQL log

/**
 * Real-time webcam detection that runs IN THE BROWSER on the visitor's GPU
 * (onnxruntime-web + WebGPU; WASM/CPU fallback). Runs as fast as the device allows
 * — real-time on a capable GPU — and samples detections to the server for logging.
 */
export default function LiveWebcam({ onLogged }) {
  const videoRef = useRef(null);
  const detectorRef = useRef(null);
  const runningRef = useRef(false);
  const sourceIdRef = useRef(null);
  const pendingRef = useRef([]); // frames awaiting flush to the server
  const frameNoRef = useRef(0);
  const enabledRef = useRef(null); // Set of enabled class labels (open-vocab), or null = all
  const trackerRef = useRef(null); // assigns stable Object IDs across frames
  const thumbedRef = useRef(new Set()); // track_ids already thumbnailed (one crop per object)

  const INPUT_SIZE = 640; // models are exported at a fixed 640 for peak WebGPU throughput
  const [modelId, setModelId] = useState(CLIENT_MODELS[0].id);
  const [vocab, setVocab] = useState(null);
  const [enabled, setEnabled] = useState(new Set());
  const [conf, setConf] = useState(DEFAULT_CONF);
  const selectedModel = CLIENT_MODELS.find((m) => m.id === modelId);

  function changeConf(v) {
    setConf(v);
    if (detectorRef.current) detectorRef.current.confThreshold = v; // live, no reload
  }
  const [backend, setBackend] = useState(null);
  const [loading, setLoading] = useState(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);
  const [dets, setDets] = useState([]);
  const [fps, setFps] = useState(0);
  const [logged, setLogged] = useState(0);
  const [dims, setDims] = useState({ nw: 0, nh: 0, dw: 0, dh: 0 });

  const stop = useCallback(() => {
    runningRef.current = false;
    setRunning(false);
    const stream = videoRef.current?.srcObject;
    stream?.getTracks().forEach((t) => t.stop());
    if (videoRef.current) videoRef.current.srcObject = null;
    detectorRef.current?.dispose();
    detectorRef.current = null;
    trackerRef.current = null;
    onLogged?.();
  }, [onLogged]);

  useEffect(() => () => stop(), [stop]);

  async function start() {
    setError(null);
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

      let stream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
      } catch (e) {
        // getUserMedia DOMExceptions are cryptic ("Could not start video source") — map them
        // to actionable guidance (no silent failure).
        const hints = {
          NotReadableError:
            "Camera couldn't start — it's likely in use by another app (Zoom, Teams, OBS, " +
            "another browser tab) or blocked by the OS. Close those and try again.",
          NotAllowedError:
            "Camera permission was denied. Allow camera access for this site (the camera icon " +
            "in the address bar) and retry.",
          NotFoundError: "No camera was found on this device.",
          OverconstrainedError: "No camera matched the requested settings.",
          SecurityError: "Camera blocked — the page must be served over HTTPS.",
        };
        throw new Error(hints[e.name] || `Camera error: ${e.message || e.name}`);
      }
      const video = videoRef.current;
      video.srcObject = stream;
      await video.play();

      // Open a logging session on the server (source row with model + threshold).
      const session = await api.clientSession({
        kind: "webcam",
        model_version: `${model.id}-webgpu`,
        conf_threshold: conf,
      });
      sourceIdRef.current = session.id;
      frameNoRef.current = 0;
      pendingRef.current = [];
      thumbedRef.current = new Set();
      setLogged(0);

      runningRef.current = true;
      setRunning(true);
      setLoading(false);
      loop();
      startFlush();
    } catch (err) {
      setError(err.message || "Could not start webcam detection.");
      setLoading(false);
    }
  }

  // Detection loop — runs as fast as the device allows (real-time on a good GPU).
  let lastT = 0;
  function loop() {
    const video = videoRef.current;
    const detector = detectorRef.current;
    if (!runningRef.current || !video || !detector) return;
    if (video.readyState < 2) {
      requestAnimationFrame(loop);
      return;
    }
    const t0 = performance.now();
    detector
      .detect(video, video.videoWidth, video.videoHeight)
      .then((all) => {
        const filtered = enabledRef.current
          ? all.filter((x) => enabledRef.current.has(x.class_label))
          : all;
        const tracked = trackerRef.current ? trackerRef.current.update(filtered) : filtered;
        attachThumbs(tracked, video, thumbedRef.current, detectorRef.current); // crop (+seg cutout) per object
        setDets(tracked);
        const now = performance.now();
        const inst = 1000 / Math.max(1, now - t0);
        setFps((prev) => (prev ? prev * 0.8 + inst * 0.2 : inst));
        setDims({
          nw: video.videoWidth,
          nh: video.videoHeight,
          dw: video.clientWidth,
          dh: video.clientHeight,
        });
        // Log only CONFIRMED objects (seen across several frames) — robust SQL data.
        const confirmed = tracked
          .filter((t) => t._confirmed)
          .map(({ _confirmed, ...rest }) => rest);
        const fn = frameNoRef.current++;
        pendingRef.current.push({ frame_number: fn, ts_seconds: now / 1000, detections: confirmed });
        requestAnimationFrame(loop);
      })
      .catch((err) => {
        setError(`Inference error: ${err.message}`);
        runningRef.current = false;
        setRunning(false);
      });
  }

  function startFlush() {
    const tick = async () => {
      if (!runningRef.current) return;
      const batch = pendingRef.current;
      pendingRef.current = [];
      if (batch.length && sourceIdRef.current != null) {
        // sample: log at most a few frames per flush to keep the DB lean
        const step = Math.max(1, Math.floor(batch.length / 3));
        const sampled = batch.filter((_, i) => i % step === 0);
        try {
          const res = await api.clientDetections(sourceIdRef.current, sampled);
          setLogged((n) => n + (res.logged || 0));
          onLogged?.();
        } catch {
          /* surfaced via health dot */
        }
      }
      if (runningRef.current) setTimeout(tick, LOG_FLUSH_MS);
    };
    setTimeout(tick, LOG_FLUSH_MS);
  }

  const isCpu = backend === "wasm";

  return (
    <div className="panel">
      <h2>Live webcam — runs on your device</h2>
      <p className="muted">
        Detection runs <strong>in your browser</strong> via onnxruntime-web. With a capable
        GPU (WebGPU) it runs in real time; otherwise it falls back to CPU. Only sampled
        detections are sent to the server — your video never leaves your device.
      </p>

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
          <input
            type="range" min="0.1" max="0.9" step="0.05" value={conf}
            onChange={(e) => changeConf(+e.target.value)}
          />
        </label>
        {!running ? (
          <button className="btn" onClick={start} disabled={loading}>
            {loading ? "Loading model…" : "Start"}
          </button>
        ) : (
          <button className="btn btn-stop" onClick={stop}>Stop</button>
        )}
      </div>

      {selectedModel?.openVocab && (
        <p className="muted" style={{ marginTop: 4 }}>{selectedModel.note}</p>
      )}
      {running && vocab && (
        <ClassFilter
          classes={vocab}
          enabled={enabled}
          onToggle={(c) => {
            const next = new Set(enabled);
            next.has(c) ? next.delete(c) : next.add(c);
            setEnabled(next);
            enabledRef.current = next;
          }}
          onAll={() => { const a = new Set(vocab); setEnabled(a); enabledRef.current = a; }}
          onNone={() => { const e = new Set(); setEnabled(e); enabledRef.current = e; }}
          onSet={(s) => { setEnabled(s); enabledRef.current = s; }}
        />
      )}

      {error && <p className="error">⚠ {error}</p>}

      {running && (
        <div className="stat-row" style={{ marginTop: 10 }}>
          <span className={`chip ${isCpu ? "chip-warn" : "chip-ok"}`}>
            {backend === "webgpu" ? "⚡ WebGPU (GPU)" : "CPU (WASM) — not real-time"}
          </span>
          <span className="chip">{fps.toFixed(1)} fps</span>
          <span className="chip">{dets.length} objects</span>
          <span className="chip">{logged} logged</span>
        </div>
      )}

      <div className="video-wrap" style={{ position: "relative", marginTop: 12 }}>
        <video ref={videoRef} muted playsInline aria-label="Live webcam with detection overlay"
          style={{ width: "100%", borderRadius: 8, display: "block" }} />
        <BoxOverlay
          detections={dets}
          nativeWidth={dims.nw}
          nativeHeight={dims.nh}
          displayWidth={videoRef.current?.clientWidth || 0}
          displayHeight={videoRef.current?.clientHeight || 0}
        />
      </div>
    </div>
  );
}
