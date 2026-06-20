import { useEffect, useRef, useState } from "react";
import { api } from "../api";
import { embedText } from "../webgpu/clip";

// Visual / descriptive queries → compute a CLIP text embedding for semantic search.
const VISUAL_RE = /\b(show|see|find|look|looks|picture|photo|image|thumbnail|gallery|wearing|holding|colou?r|red|blue|green|yellow|orange|purple|pink|black|white|grey|gray|brown|dark|light)\b/i;

const SUGGESTIONS = [
  "What's been detected?",
  "What can I ask about this data?",
  "How many unique objects were detected in total?",
  "What are the top 5 most common classes?",
  "Which source has the most detections?",
  "What's the average confidence per class?",
];

const INTENT_LABELS = {
  analytics: "analytics",
  overview: "overview",
  schema: "schema/help",
  out_of_scope: "out of scope",
};

/**
 * Chatbot over the detections database — a LangGraph multi-agent text-to-SQL pipeline
 * (Groq LLM) on the backend. Shows the answer plus the generated SQL + rows for transparency.
 */
export default function Chat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [zoom, setZoom] = useState(null); // crop shown enlarged in the lightbox, or null
  const listRef = useRef(null);

  useEffect(() => {
    if (!zoom) return;
    const onKey = (e) => e.key === "Escape" && setZoom(null);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [zoom]);

  async function send(q) {
    const question = (q ?? input).trim();
    if (!question || busy) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", text: question }]);
    setBusy(true);
    try {
      // For visual queries, embed the text with CLIP (client-side) so the gallery agent can do
      // semantic similarity search. Best-effort: null on failure → falls back to class matching.
      const qvec = VISUAL_RE.test(question) ? await embedText(question) : null;
      const res = await api.chat(question, qvec);
      setMessages((m) => [...m, {
        role: "bot", text: res.answer, sql: res.sql, rows: res.rows, intent: res.intent,
        crops: res.crops,
      }]);
    } catch (err) {
      setMessages((m) => [...m, { role: "bot", text: null, error: err.message }]);
    } finally {
      setBusy(false);
      requestAnimationFrame(() => listRef.current?.scrollTo(0, listRef.current.scrollHeight));
    }
  }

  return (
    <div className="panel chat">
      <h2>Ask the data</h2>
      <p className="muted">
        A LangGraph multi-agent pipeline: an intent router sends each question to a specialized
        agent — analytics (self-correcting text-to-SQL), data overview, schema/help, or
        out-of-scope. Read-only; the routed intent + generated SQL are shown for transparency.
      </p>

      <div className="chat-log" ref={listRef} aria-live="polite">
        {messages.length === 0 && (
          <div className="chat-suggest">
            {SUGGESTIONS.map((s) => (
              <button key={s} type="button" className="fchip" onClick={() => send(s)}>{s}</button>
            ))}
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`bubble ${m.role}`}>
            {m.role === "user" ? (
              <p>{m.text}</p>
            ) : m.error ? (
              <p className="error">⚠ {m.error}</p>
            ) : (
              <>
                {m.intent && (
                  <span className="intent-badge">{INTENT_LABELS[m.intent] || m.intent}</span>
                )}
                <p>{m.text}</p>
                {m.crops?.length > 0 && (
                  <div className="crop-grid">
                    {m.crops.map((c, j) => (
                      <figure key={j} className="crop">
                        <button
                          type="button"
                          className="crop-btn"
                          onClick={() => setZoom(c)}
                          title={`${c.class_label} ${Math.round((c.confidence || 0) * 100)}% — click to enlarge`}
                        >
                          <img src={c.thumb} alt={c.class_label} loading="lazy" />
                        </button>
                        <figcaption>{c.class_label}</figcaption>
                      </figure>
                    ))}
                  </div>
                )}
                {m.sql && (
                  <details>
                    <summary>SQL</summary>
                    <pre>{m.sql}</pre>
                  </details>
                )}
                {m.rows?.length > 0 && (
                  <details>
                    <summary>{m.rows.length} row(s)</summary>
                    <pre>{JSON.stringify(m.rows.slice(0, 20), null, 2)}</pre>
                  </details>
                )}
              </>
            )}
          </div>
        ))}
        {busy && <p className="muted">Thinking…</p>}
      </div>

      <form className="chat-input" onSubmit={(e) => { e.preventDefault(); send(); }}>
        <label htmlFor="chat-q" className="sr-only">Ask a question about the detections</label>
        <input
          id="chat-q" type="text" value={input} placeholder="Ask about the detections…"
          onChange={(e) => setInput(e.target.value)} disabled={busy} autoComplete="off"
        />
        <button className="btn" type="submit" disabled={busy || !input.trim()}>Ask</button>
      </form>

      {zoom && (
        <div className="lightbox" onClick={() => setZoom(null)} role="dialog" aria-modal="true">
          <figure className="lightbox-fig" onClick={(e) => e.stopPropagation()}>
            <img src={zoom.thumb} alt={zoom.class_label} />
            <figcaption>
              {zoom.class_label}
              {zoom.confidence ? ` · ${Math.round(zoom.confidence * 100)}% conf` : ""}
              {zoom.track_id != null ? ` · object #${zoom.track_id}` : ""}
            </figcaption>
          </figure>
          <button className="lightbox-close" onClick={() => setZoom(null)} aria-label="Close">×</button>
        </div>
      )}
    </div>
  );
}
