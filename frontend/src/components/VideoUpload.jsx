import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { MODELS, isClientModel } from "../models";
import { createDetector } from "../webgpu/detector";
import { SimpleTracker } from "../webgpu/tracker";
import { attachThumbs } from "../webgpu/thumbs";
import { warmClip } from "../webgpu/clip";
import { createEmbedder } from "../webgpu/embedQueue";
import BoxOverlay from "./BoxOverlay";
import ClassFilter from "./ClassFilter";

const LOG_FLUSH_MS = 1000;
const INPUT_SIZE = 640;
const DEFAULT_CONF = 0.4;

/**
 * Upload a video file. For YOLO26 n/s the file is decoded and detected **in the browser
 * on your GPU** (same as the webcam path) — free and fast. For the heavy server models
 * (m/x/open-vocab) the file is processed server-side (CPU-only on the free tier — slow,
 * flagged in the UI). Either way, detections are logged to PostgreSQL.
 */
export default function VideoUpload({ onLogged }) {
  const videoRef = useRef(null);
  const detectorRef = useRef(null);
  const runningRef = useRef(false); // detection loop active
  const loggingRef = useRef(false); // log only during the first pass
  const sourceIdRef = useRef(null);
  const pendingRef = useRef([]);
  const frameNoRef = useRef(0);
  const enabledRef = useRef(null); // open-vocab class filter (Set), or null = all
  const trackerRef = useRef(null); // stable Object IDs across frames
  const fileRef = useRef(null);    // the chosen File (kept until the user clicks Start)
  const thumbedRef = useRef(new Set()); // track_ids already thumbnailed (one crop per object)
  const embedderRef = useRef(null);     // background CLIP embedder (semantic search)

  const [vocab, setVocab] = useState(null);
  const [enabled, setEnabled] = useState(new Set());
  const [conf, setConf] = useState(DEFAULT_CONF);
  const [model, setModel] = useState("yolo26n");

  function changeConf(v) {
    setConf(v);
    if (detectorRef.current) detectorRef.current.confThreshold = v;
  }
  const [prompts, setPrompts] = useState("");
  const [videoUrl, setVideoUrl] = useState(null);
  const [busy, setBusy] = useState(false);
  const [running, setRunning] = useState(false); // client detection loop active (for Start/Stop UI)
  const [error, setError] = useState(null);
  const [fileName, setFileName] = useState(null);
  const [dragging, setDragging] = useState(false);

  // client-side (local GPU) state
  const [backend, setBackend] = useState(null);
  const [fps, setFps] = useState(0);
  const [logged, setLogged] = useState(0);
  const [liveDets, setLiveDets] = useState([]);
  const [dims, setDims] = useState({ nw: 0, nh: 0, dw: 0, dh: 0 });

  // server-side state
  const [serverResult, setServerResult] = useState(null);
  const [now, setNow] = useState(0);

  const selected = MODELS.find((m) => m.id === model);
  const clientSide = isClientModel(model);
  const isOpenVocab = !!selected?.openVocab;

  const teardown = useCallback(() => {
    runningRef.current = false;
    loggingRef.current = false;
    detectorRef.current?.dispose();
    detectorRef.current = null;
    trackerRef.current = null;
    setRunning(false);
  }, []);

  useEffect(() => () => teardown(), [teardown]);

  // Preload an open-vocab/fine-tune model's vocabulary as soon as it's SELECTED (not after the
  // model downloads), so the class filter is usable — and you can pre-type "truck" — before Start.
  useEffect(() => {
    const m = MODELS.find((x) => x.id === model);
    if (!m?.vocabUrl) {
      setVocab(null);
      enabledRef.current = null;
      return;
    }
    let cancelled = false;
    fetch(m.vocabUrl)
      .then((r) => r.json())
      .then((meta) => {
        if (cancelled) return;
        setVocab(meta.classes);
        const all = new Set(meta.classes);
        setEnabled(all);
        enabledRef.current = all;
      })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [model]);

  // Switching model resets the loaded clip so the two paths never mix.
  function onModelChange(id) {
    teardown();
    setModel(id);
    setVideoUrl(null);
    setServerResult(null);
    setLiveDets([]);
    setFps(0);
    setLogged(0);
    setError(null);
    setFileName(null);
    fileRef.current = null;
    // vocab/enabled are (re)loaded by the effect on `model`.
  }

  // Selecting a file only STAGES it (shows a paused preview). Detection starts on the Start
  // button — so you can set the model/classes first and it doesn't auto-run.
  function handleFiles(files) {
    const file = files?.[0];
    if (!file) return;
    if (!file.type.startsWith("video/")) {
      setError(`"${file.name}" isn't a video file.`);
      return;
    }
    teardown();
    fileRef.current = file;
    setFileName(file.name);
    setError(null);
    setServerResult(null);
    setLiveDets([]);
    setLogged(0);
    setFps(0);
    setVideoUrl(URL.createObjectURL(file));
  }

  function onDrop(e) {
    e.preventDefault();
    setDragging(false);
    if (!busy && !running) handleFiles(e.dataTransfer.files);
  }

  async function start() {
    if (!fileRef.current || busy || running) return;
    if (clientSide) await runClientSide(videoUrl);
    else await runServerSide(fileRef.current);
  }

  function stop() {
    teardown();
    const v = videoRef.current;
    if (v) v.pause();
  }

  // ---- client-side (local GPU) -----------------------------------------------------
  async function runClientSide(url) {
    setBusy(true);
    try {
      // No silent failure: confirm the model file is actually served here first. A missing
      // big model (e.g. x, gitignored) can come back as the SPA's index.html with a 200, which
      // would otherwise blow up later as "protobuf parsing failed" — so reject HTML too.
      const head = await fetch(selected.url, { method: "HEAD" });
      const ct = head.headers.get("content-type") || "";
      if (!head.ok || ct.includes("text/html")) {
        throw new Error(
          `${selected.id} (~${selected.sizeMB}MB) isn't deployed here. It's too large for ` +
          `GitHub's 100MB limit, so it isn't bundled with the site. Try n/s/m instead, or ` +
          `host it on a CDN${selected.id === "yolo26x" ? " and set VITE_X_MODEL_URL" : " and set VITE_MODELS_BASE"}.`
        );
      }
      const { detector, backend: be, classNames } = await createDetector(selected, {
        inputSize: INPUT_SIZE,
        conf,
      });
      detectorRef.current = detector;
      trackerRef.current = new SimpleTracker();
      setBackend(be);
      if (classNames) {
        setVocab(classNames);
        // Preserve the user's pre-Start class selection; only default to "all" if none set.
        if (!enabledRef.current) {
          const all = new Set(classNames);
          setEnabled(all);
          enabledRef.current = all;
        }
      } else {
        setVocab(null);
        enabledRef.current = null;
      }

      const session = await api.clientSession({
        kind: "upload",
        model_version: `${model}-webgpu`,
        conf_threshold: conf,
      });
      sourceIdRef.current = session.id;
      frameNoRef.current = 0;
      pendingRef.current = [];
      thumbedRef.current = new Set();
      embedderRef.current = createEmbedder(() => sourceIdRef.current, () => runningRef.current);
      warmClip(); // load CLIP in the background so crop embeddings are ready
      runningRef.current = true;
      loggingRef.current = true;
      setRunning(true);
      setBusy(false);

      // src is bound to videoUrl in JSX (set above); by now React has rendered it.
      const v = videoRef.current;
      v.muted = true;
      if (v.readyState < 2) {
        await new Promise((res) => {
          v.onloadeddata = res;
          setTimeout(res, 3000); // safety
        });
      }
      await v.play().catch(() => {});
      loop();
      startFlush();
    } catch (err) {
      setError(err.message || "Could not start in-browser detection.");
      setBusy(false);
    }
  }

  function loop() {
    const v = videoRef.current;
    const detector = detectorRef.current;
    if (!runningRef.current || !v || !detector) return;
    if (v.readyState < 2 || v.paused || v.ended) {
      if (runningRef.current) requestAnimationFrame(loop);
      return;
    }
    const t0 = performance.now();
    detector
      .detect(v, v.videoWidth, v.videoHeight)
      .then((all) => {
        const filtered = enabledRef.current
          ? all.filter((x) => enabledRef.current.has(x.class_label))
          : all;
        const tracked = trackerRef.current ? trackerRef.current.update(filtered) : filtered;
        attachThumbs(tracked, v, thumbedRef.current, detectorRef.current); // crop (+seg cutout) per object
        for (const t of tracked) {
          if (t.thumb && t.track_id != null) embedderRef.current?.enqueue(t.track_id, t.thumb);
        }
        setLiveDets(tracked);
        const inst = 1000 / Math.max(1, performance.now() - t0);
        setFps((p) => (p ? p * 0.8 + inst * 0.2 : inst));
        setDims({ nw: v.videoWidth, nh: v.videoHeight, dw: v.clientWidth, dh: v.clientHeight });
        if (loggingRef.current) {
          // Log only CONFIRMED objects (seen across frames) — robust SQL data.
          const confirmed = tracked
            .filter((t) => t._confirmed)
            .map(({ _confirmed, _coeffs, ...rest }) => rest);
          pendingRef.current.push({
            frame_number: frameNoRef.current++,
            ts_seconds: +v.currentTime.toFixed(3),
            detections: confirmed,
          });
        }
        if (runningRef.current) requestAnimationFrame(loop);
      })
      .catch((err) => {
        setError(`Inference error: ${err.message}`);
        runningRef.current = false;
      });
  }

  function startFlush() {
    const tick = async () => {
      if (!loggingRef.current) return;
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
          const r = await api.clientDetections(sourceIdRef.current, sampled);
          setLogged((n) => n + (r.logged || 0));
          onLogged?.();
        } catch {
          /* surfaced via health dot */
        }
      }
      if (loggingRef.current) setTimeout(tick, LOG_FLUSH_MS);
    };
    setTimeout(tick, LOG_FLUSH_MS);
  }

  function onEnded() {
    loggingRef.current = false; // stop logging after the first pass; replay still overlays
    onLogged?.();
  }

  // ---- server-side (heavy models) --------------------------------------------------
  async function runServerSide(file) {
    if (isOpenVocab && !prompts.trim()) {
      setError("Enter comma-separated class names to detect (e.g. 'forklift, hard hat').");
      return;
    }
    setBusy(true);
    try {
      const res = await api.upload(file, { model, prompts });
      setServerResult(res);
      onLogged?.();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  const serverWindow = serverResult ? 1 / (serverResult.source.fps || 5) + 0.05 : 0;
  const serverActive = useMemo(() => {
    if (!serverResult) return [];
    return serverResult.detections.filter((d) => Math.abs(d.ts_seconds - now) <= serverWindow);
  }, [serverResult, now, serverWindow]);

  function syncDims() {
    const v = videoRef.current;
    if (v) setDims({ nw: v.videoWidth, nh: v.videoHeight, dw: v.clientWidth, dh: v.clientHeight });
  }

  const overlayDets = clientSide ? liveDets : serverActive;
  const isCpu = backend === "wasm";

  return (
    <div className="panel">
      <h2>Upload a video</h2>
      <p className="muted">
        n/s run <strong>in your browser on your GPU</strong> (free, real-time). m/x and
        open-vocab run on the server. Detections are logged to PostgreSQL.
      </p>

      <div className="controls">
        <label>
          Model{" "}
          <select value={model} onChange={(e) => onModelChange(e.target.value)} disabled={busy}>
            {MODELS.map((m) => (
              <option key={m.id} value={m.id}>{m.label}</option>
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
      </div>

      {!clientSide && (
        <p className="error" style={{ marginTop: 4 }}>
          ⚠ Open-vocab runs server-side, and server GPU inference isn't free — on the free
          tier it's CPU-only, so expect a small fraction of real-time speed. The YOLO26
          n/s/m/x models all run on your own GPU.
        </p>
      )}
      {clientSide && selected.sizeMB >= 80 && (
        <p className="muted" style={{ marginTop: 4 }}>
          ⬇ First run downloads ~{selected.sizeMB}MB (cached after), then <strong>compiles the
          model onto your GPU</strong> — this warm-up takes a while
          {selected.id === "yolo26x" ? " (notably longer for x)" : ""} before the first frame.
          After that it runs ~{selected.id === "yolo26x" ? "10" : "20"} fps; needs a reasonably
          strong GPU{selected.id === "yolo26x" ? " with enough memory" : ""}.
        </p>
      )}
      {isOpenVocab && !clientSide && (
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

      {clientSide && vocab && (
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

      <label
        className={`dropzone${dragging ? " dz-over" : ""}${busy ? " dz-busy" : ""}`}
        onDragOver={(e) => { e.preventDefault(); if (!busy) setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <input
          type="file" accept="video/*" hidden disabled={busy}
          onChange={(e) => handleFiles(e.target.files)}
          aria-label="Upload a video file for detection"
        />
        <span className="dz-icon" aria-hidden="true">⬆</span>
        <span className="dz-main">
          <strong>Click to choose a video</strong> &nbsp;or drag &amp; drop
        </span>
        <span className="muted dz-sub">
          {fileName ? `Selected: ${fileName}` : "MP4 / WebM / MOV"}
        </span>
      </label>

      <div className="controls" style={{ marginTop: 12 }}>
        {!running ? (
          <button className="btn" onClick={start} disabled={!fileRef.current || busy}>
            {busy
              ? (clientSide ? "Loading model…" : "Analysing…")
              : (clientSide ? "Start detection" : "Analyse on server")}
          </button>
        ) : (
          <button className="btn btn-stop" onClick={stop}>Stop</button>
        )}
        {!fileRef.current && <span className="muted">Choose a video, then press Start.</span>}
      </div>
      {error && <p className="error">⚠ {error}</p>}

      {clientSide && running && (
        <div className="stat-row" style={{ marginTop: 10 }}>
          <span className={`chip ${isCpu ? "chip-warn" : "chip-ok"}`}>
            {backend === "webgpu" ? "⚡ WebGPU (your GPU)" : backend === "wasm" ? "CPU (WASM) — not real-time" : "loading…"}
          </span>
          <span className="chip">{fps.toFixed(1)} fps</span>
          <span className="chip">{overlayDets.length} objects</span>
          <span className="chip">{logged} logged</span>
        </div>
      )}

      {videoUrl && (
        <div className="video-wrap" style={{ position: "relative", marginTop: 12 }}>
          <video
            ref={videoRef}
            src={videoUrl}
            controls
            playsInline
            aria-label="Uploaded video with detection overlay"
            style={{ width: "100%", borderRadius: 8, display: "block" }}
            onLoadedMetadata={syncDims}
            onTimeUpdate={clientSide ? undefined : (e) => setNow(e.target.currentTime)}
            onEnded={clientSide ? onEnded : undefined}
          />
          <BoxOverlay
            detections={overlayDets}
            nativeWidth={dims.nw}
            nativeHeight={dims.nh}
            displayWidth={videoRef.current?.clientWidth || 0}
            displayHeight={videoRef.current?.clientHeight || 0}
          />
        </div>
      )}

      {serverResult && (
        <div className="stat-row" style={{ marginTop: 12 }}>
          <span className="chip">{serverResult.frames_analysed} frames analysed</span>
          <span className="chip">{serverResult.detections_logged} detections logged</span>
          <span className="chip">model {serverResult.source.model_version}</span>
          <span className="chip">conf ≥ {serverResult.source.conf_threshold}</span>
        </div>
      )}
    </div>
  );
}
