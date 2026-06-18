// Model registry shared by the UI. `client` models run in-browser (WebGPU/WASM);
// `server` models run on the API (uploads + open-vocabulary).
export const MODELS = [
  {
    id: "yolo26n",
    label: "YOLO26n — fastest",
    url: "/models/yolo26n.onnx",
    sizeMB: 9.5,
    runtimes: ["client", "server"],
    family: "yolo26",
    note: "Nano. Best for real-time webcam on most GPUs.",
  },
  {
    id: "yolo26s",
    label: "YOLO26s — balanced",
    url: "/models/yolo26s.onnx",
    sizeMB: 37,
    runtimes: ["client", "server"],
    family: "yolo26",
    note: "Small. More accurate; needs a stronger GPU for real-time.",
  },
  {
    id: "yolo26m",
    label: "YOLO26m — accurate (server)",
    sizeMB: 80,
    runtimes: ["server"],
    family: "yolo26",
    note: "Medium. Server-side upload path only.",
  },
  {
    id: "yolo26x",
    label: "YOLO26x — most accurate (server)",
    sizeMB: 220,
    runtimes: ["server"],
    family: "yolo26",
    note: "Extra-large. Server-side upload path only.",
  },
  {
    id: "yoloe26",
    label: "YOLOE-26 — open-vocabulary (server)",
    runtimes: ["server"],
    family: "yoloe",
    openVocab: true,
    note: "Detect anything via text prompts. Server-side upload path only.",
  },
];

export const CLIENT_MODELS = MODELS.filter((m) => m.runtimes.includes("client"));
export const SERVER_MODELS = MODELS.filter((m) => m.runtimes.includes("server"));
