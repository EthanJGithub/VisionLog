// Model registry shared by the UI.
//  • `client` models run IN THE BROWSER on the visitor's GPU (WebGPU; WASM/CPU fallback)
//    — used for BOTH the live webcam AND uploaded files. Free and fast.
//  • `server` models are too heavy for the browser, so they run on the API. On the free
//    tier the server is CPU-only (no GPU), so these are much slower — flagged in the UI.
export const MODELS = [
  {
    id: "yolo26n",
    label: "YOLO26n — fastest",
    url: "/models/yolo26n.onnx",
    sizeMB: 9.5,
    runtimes: ["client"],
    family: "yolo26",
    note: "Nano. Runs on your GPU in the browser — real-time.",
  },
  {
    id: "yolo26s",
    label: "YOLO26s — balanced",
    url: "/models/yolo26s.onnx",
    sizeMB: 37,
    runtimes: ["client"],
    family: "yolo26",
    note: "Small. Runs on your GPU in the browser; needs a stronger GPU for real-time.",
  },
  {
    id: "yolo26m",
    label: "YOLO26m — accurate (server)",
    sizeMB: 80,
    runtimes: ["server"],
    family: "yolo26",
    note: "Medium. Server-side; CPU-only on the free tier (slow — not real-time).",
  },
  {
    id: "yolo26x",
    label: "YOLO26x — most accurate (server)",
    sizeMB: 220,
    runtimes: ["server"],
    family: "yolo26",
    note: "Extra-large. Server-side; CPU-only on the free tier (slow — not real-time).",
  },
  {
    id: "yoloe26",
    label: "YOLOE-26 — open-vocabulary (server)",
    runtimes: ["server"],
    family: "yoloe",
    openVocab: true,
    note: "Detect anything via text prompts. Server-side; CPU-only on the free tier (slow).",
  },
];

export const CLIENT_MODELS = MODELS.filter((m) => m.runtimes.includes("client"));
export const SERVER_MODELS = MODELS.filter((m) => m.runtimes.includes("server"));
export const isClientModel = (id) =>
  !!MODELS.find((m) => m.id === id)?.runtimes.includes("client");
