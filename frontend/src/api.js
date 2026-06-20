const BASE = import.meta.env.VITE_API_BASE || "/api/v1";

async function req(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, options);
  if (!res.ok) {
    const detail = (await res.json().catch(() => ({}))).detail || res.statusText;
    throw new Error(detail);
  }
  return res.json();
}

export const api = {
  health: () => req("/health"),
  // stats scoped to one run when sourceId is given, else all runs.
  stats: (sourceId) => req(sourceId != null ? `/stats?source_id=${sourceId}` : "/stats"),
  sources: () => req("/sources"),
  detections: (sourceId) => req(`/sources/${sourceId}/detections`),
  deleteSource: (sourceId) => req(`/sources/${sourceId}`, { method: "DELETE" }),
  clearAll: () => req("/sources", { method: "DELETE" }),
  upload: (file, { model = "yolo26n", prompts = "" } = {}) => {
    const form = new FormData();
    form.append("file", file);
    form.append("model", model);
    if (prompts) form.append("prompts", prompts);
    return req("/sources", { method: "POST", body: form });
  },
  // Client-side (WebGPU) session logging:
  clientSession: (body) =>
    req("/client-sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  clientDetections: (sourceId, items) =>
    req(`/client-sessions/${sourceId}/detections`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ items }),
    }),
  // CLIP embeddings for crops, sent out-of-band (one batch) — never blocks detection logging.
  setEmbeddings: (sourceId, items) =>
    req(`/client-sessions/${sourceId}/embeddings`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ items }),
    }),
  chat: (question, queryEmbedding = null) =>
    req("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(
        queryEmbedding ? { question, query_embedding: queryEmbedding } : { question }
      ),
    }),
};

// Build the ws:// URL for the webcam stream, honoring the dev proxy.
export function streamUrl() {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${window.location.host}${BASE}/stream`;
}
