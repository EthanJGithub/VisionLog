// Model registry shared by the UI.
//  • All YOLO26 detection variants (n/s/m/x) run IN THE BROWSER on the visitor's GPU
//    (onnxruntime-web + WebGPU; WASM/CPU fallback). Bigger = larger download + lower fps,
//    but they all run on your own GPU — measured on an NVIDIA card: n≈48, m≈20, x≈11 fps.
//  • Open-vocabulary (YOLOE) is the only server-side feature: it needs a ~570MB CLIP text
//    encoder to turn typed prompts into vectors, which is impractical to ship to a browser.
export const MODELS = [
  {
    id: "yolo26n",
    label: "YOLO26n — fastest",
    url: "/models/yolo26n.onnx",
    sizeMB: 9.5,
    runtimes: ["client"],
    family: "yolo26",
    note: "Nano. Real-time on most GPUs (~48 fps).",
  },
  {
    id: "yolo26s",
    label: "YOLO26s — balanced",
    url: "/models/yolo26s.onnx",
    sizeMB: 37,
    runtimes: ["client"],
    family: "yolo26",
    note: "Small. Real-time on a decent GPU.",
  },
  {
    id: "yolo26m",
    label: "YOLO26m — accurate",
    url: "/models/yolo26m.onnx",
    sizeMB: 82,
    runtimes: ["client"],
    family: "yolo26",
    note: "Medium (~82MB download). ~20 fps on a strong GPU.",
  },
  {
    id: "yolo26x",
    label: "YOLO26x — most accurate",
    url: "/models/yolo26x.onnx",
    sizeMB: 223,
    runtimes: ["client"],
    family: "yolo26",
    note: "Extra-large (~223MB download). ~10 fps on a strong GPU; needs one.",
  },
  {
    id: "yolo26-ppe",
    label: "YOLO26n-PPE — fine-tuned (your GPU)",
    url: "/models/yolo26-ppe.onnx",
    vocabUrl: "/models/yolo26-ppe.json",
    sizeMB: 9.5,
    runtimes: ["client"],
    family: "yolo26-ft",
    note: "Domain fine-tune on Construction-PPE — detects helmet / vest / gloves / goggles / boots etc. (classes COCO lacks). Runs on your GPU.",
  },
  {
    id: "yoloe26-vocab",
    label: "YOLOE-26 — open-vocabulary (your GPU)",
    url: "/models/yoloe26-vocab.onnx",
    vocabUrl: "/models/yoloe26-vocab.json",
    sizeMB: 41,
    runtimes: ["client"],
    family: "yoloe",
    head: "v8seg",
    openVocab: true,
    note: "Open-vocabulary, fully in-browser on your GPU — no server, no 570MB text encoder. Detects a curated vocabulary (incl. classes COCO lacks: hard hat, forklift, pallet, license plate…).",
  },
];

export const CLIENT_MODELS = MODELS.filter((m) => m.runtimes.includes("client"));
export const SERVER_MODELS = MODELS.filter((m) => m.runtimes.includes("server"));
export const isClientModel = (id) =>
  !!MODELS.find((m) => m.id === id)?.runtimes.includes("client");
