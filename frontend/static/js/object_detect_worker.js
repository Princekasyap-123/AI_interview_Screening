// frontend/static/js/object_detect_worker.js

/**
 * Web Worker: YOLOX object detection with ONNX Runtime Web.
 *
 * Picks the model by hardware:
 *   - WebGPU available (most current Chrome/Edge laptops)
 *       -> YOLOX-S, 640px input, on the GPU   (more accurate)
 *   - otherwise
 *       -> YOLOX-Tiny, 416px input, WASM/CPU  (fast enough on any CPU)
 * If YOLOX-S fails to start on WebGPU for any reason, it falls back to
 * YOLOX-Tiny on the CPU automatically.
 *
 * YOLOX is Apache-2.0 (free for commercial use). Ultralytics YOLOv8 /
 * YOLO11 were avoided on purpose: they are AGPL-3.0.
 *
 * Running in a worker keeps detection isolated from face-api.js's bundled
 * TensorFlow.js and off the main thread (avatar / face mesh stay smooth).
 *
 * Messages in:
 *   { type: "load", id, ortUrl, wasmPaths, models: { accurate: {url, size}, fast: {url, size} }, force }
 *        force: "accurate" | "fast" | undefined (testing override)
 *   { type: "detect", id, data, minScore, maxNumBoxes }
 *        data = size x size RGBA letterboxed frame (ArrayBuffer, transferred)
 * Messages out:
 *   { type: "loaded", id, backend, model, inputSize, warmupMs }
 *   { type: "result", id, predictions, inferMs }   bbox in input-pixel space
 *   { type: "error", id, message }
 */

/* global ort */

const STRIDES = [8, 16, 32];
const NMS_IOU = 0.45;

const COCO_CLASSES = [
  "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
  "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
  "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
  "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
  "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
  "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
  "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
  "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
];

let session = null;
let inputName = null;
let INPUT = 416;
let gridX = null, gridY = null, gridS = null;
let chw = null;

function buildGrids() {
  const n = STRIDES.reduce((acc, s) => acc + (INPUT / s) * (INPUT / s), 0);
  gridX = new Float32Array(n); gridY = new Float32Array(n); gridS = new Float32Array(n);
  let k = 0;
  for (const s of STRIDES) {
    const g = INPUT / s;
    for (let y = 0; y < g; y++) {
      for (let x = 0; x < g; x++) { gridX[k] = x; gridY[k] = y; gridS[k] = s; k++; }
    }
  }
  chw = new Float32Array(3 * INPUT * INPUT);
}

// RGBA (HWC, 0-255) -> BGR planar float32, which is what YOLOX expects.
function toCHW(rgba) {
  const plane = INPUT * INPUT;
  for (let i = 0, p = 0; p < plane; p++, i += 4) {
    chw[p] = rgba[i + 2];
    chw[plane + p] = rgba[i + 1];
    chw[2 * plane + p] = rgba[i];
  }
  return chw;
}

function iou(a, b) {
  const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[0] + a[2], b[0] + b[2]), y2 = Math.min(a[1] + a[3], b[1] + b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9);
}

function decode(out, numAnchors, numAttrs, minScore, maxNumBoxes) {
  const cand = [];
  for (let i = 0; i < numAnchors; i++) {
    const o = i * numAttrs;
    const obj = out[o + 4];
    if (obj < minScore) continue;
    let best = 0, bestC = -1;
    for (let c = 0; c < numAttrs - 5; c++) {
      const v = out[o + 5 + c];
      if (v > best) { best = v; bestC = c; }
    }
    const score = obj * best;
    if (score < minScore) continue;
    const s = gridS[i];
    const cx = (out[o] + gridX[i]) * s;
    const cy = (out[o + 1] + gridY[i]) * s;
    const w = Math.exp(out[o + 2]) * s;
    const h = Math.exp(out[o + 3]) * s;
    cand.push({ class: COCO_CLASSES[bestC] || String(bestC), score, bbox: [cx - w / 2, cy - h / 2, w, h] });
  }
  cand.sort((a, b) => b.score - a.score);
  const kept = [];
  for (const d of cand) {
    if (kept.length >= maxNumBoxes) break;
    if (kept.every((k) => k.class !== d.class || iou(k.bbox, d.bbox) <= NMS_IOU)) kept.push(d);
  }
  return kept;
}

async function hasWebGPU() {
  try {
    if (!self.navigator || !navigator.gpu) return false;
    return !!(await navigator.gpu.requestAdapter());
  } catch (e) {
    return false;
  }
}

async function createSession(model, ep) {
  INPUT = model.size;
  buildGrids();
  session = await ort.InferenceSession.create(model.url, {
    executionProviders: [ep],
    graphOptimizationLevel: "all",
  });
  inputName = session.inputNames[0];
  // Warm-up so the first real check isn't slow (shaders / JIT).
  const t0 = performance.now();
  chw.fill(114);
  await session.run({ [inputName]: new ort.Tensor("float32", chw, [1, 3, INPUT, INPUT]) });
  return Math.round(performance.now() - t0);
}

async function handleLoad(msg) {
  if (session) {
    self.postMessage({ type: "loaded", id: msg.id, backend: self.__backend, model: self.__model, inputSize: INPUT, warmupMs: 0 });
    return;
  }
  importScripts(msg.ortUrl);
  if (msg.wasmPaths) ort.env.wasm.wasmPaths = msg.wasmPaths;
  // Multi-threading needs cross-origin isolation (COOP/COEP headers),
  // which this site doesn't set — one thread avoids a fallback warning.
  ort.env.wasm.numThreads = 1;

  const wantAccurate = msg.force === "accurate" || (msg.force !== "fast" && (await hasWebGPU()));
  let warmupMs;

  if (wantAccurate) {
    const ep = msg.force === "accurate" && !(await hasWebGPU()) ? "wasm" : "webgpu";
    try {
      warmupMs = await createSession(msg.models.accurate, ep);
      self.__backend = ep;
      self.__model = "yolox_s";
    } catch (err) {
      session = null;
      self.postMessage({ type: "log", message: "YOLOX-S on " + ep + " failed, falling back to YOLOX-Tiny/CPU: " + err });
    }
  }
  if (!session) {
    warmupMs = await createSession(msg.models.fast, "wasm");
    self.__backend = "wasm";
    self.__model = "yolox_tiny";
  }

  self.postMessage({ type: "loaded", id: msg.id, backend: self.__backend, model: self.__model, inputSize: INPUT, warmupMs });
}

async function handleDetect(msg) {
  if (!session) throw new Error("model not loaded");
  const input = new ort.Tensor("float32", toCHW(new Uint8ClampedArray(msg.data)), [1, 3, INPUT, INPUT]);
  const t0 = performance.now();
  const results = await session.run({ [inputName]: input });
  const inferMs = Math.round(performance.now() - t0);
  const out = results[session.outputNames[0]];
  const [, numAnchors, numAttrs] = out.dims;
  const data = out.data instanceof Float32Array ? out.data : await out.getData();
  const predictions = decode(data, numAnchors, numAttrs, msg.minScore, msg.maxNumBoxes);
  self.postMessage({ type: "result", id: msg.id, predictions, inferMs });
}

self.onmessage = async (event) => {
  const msg = event.data || {};
  try {
    if (msg.type === "load") await handleLoad(msg);
    else if (msg.type === "detect") await handleDetect(msg);
  } catch (err) {
    self.postMessage({ type: "error", id: msg.id, message: String((err && err.message) || err) });
  }
};