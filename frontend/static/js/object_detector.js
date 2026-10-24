// frontend/static/js/object_detector.js

/**
 * window.ObjectDetector — YOLOX object detection in a Web Worker
 * (object_detect_worker.js) via ONNX Runtime Web.
 *
 *   const model = await ObjectDetector.load();
 *   const predictions = await model.detect(videoElement, 20, 0.3);
 *   // [{ class: "cell phone", score: 0.82, bbox: [x, y, w, h] }, ...]
 *   // bbox in the video's own pixels; model.lastInferMs / model.info for timing
 *
 * The worker uses YOLOX-S on WebGPU when the browser has it, otherwise
 * YOLOX-Tiny on the CPU. Testing override — set this before
 * ObjectDetector.load() runs (e.g. in an inline script above this tag):
 *   window.OBJECT_DETECTOR_FORCE = "accurate"   // always YOLOX-S
 *   window.OBJECT_DETECTOR_FORCE = "fast"       // always YOLOX-Tiny
 *
 * Also exposed as window.cocoSsd for older code.
 *
 * Load with a plain (non-deferred) <script> BEFORE proctoring_client.js:
 *   data-worker          URL of object_detect_worker.js        (required)
 *   data-model-accurate  URL of yolox_s.onnx    (640px input)  (required)
 *   data-model-fast      URL of yolox_tiny.onnx (416px input)  (required)
 *   data-ort             optional override for the ONNX Runtime Web script URL
 */

(function () {
  "use strict";

  const ORT_VERSION = "1.20.1";
  const script = document.currentScript;
  const ds = (script && script.dataset) || {};
  const WORKER_URL = ds.worker;
  const MODELS = {
    accurate: { url: ds.modelAccurate, size: 640 },
    fast: { url: ds.modelFast, size: 416 },
  };
  // WebGPU bundle — also contains the WASM (CPU) backend used as fallback.
  const ORT_URL = ds.ort ||
    `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/ort.webgpu.min.js`;
  const WASM_PATHS = ORT_URL.slice(0, ORT_URL.lastIndexOf("/") + 1);

  const PAD_GRAY = "rgb(114,114,114)";

  let worker = null;
  let nextId = 1;
  const pending = new Map();
  let loadPromise = null;
  let canvas = null;
  let ctx = null;
  let inputSize = 416;
  // If the worker (GPU context loss, OOM, a dropped message, etc.) ever
  // goes silent on a call, that call's promise must still settle —
  // otherwise whatever is awaiting detect() hangs forever and the
  // caller's own poll loop (proctoring_client.js's tick()) never
  // schedules its next check again. CALL_TIMEOUT_MS guarantees that.
  const CALL_TIMEOUT_MS = 8000;
  // "load" downloads the .onnx model and runs a warm-up pass, which can
  // legitimately take far longer than a single detect() call, so it gets
  // its own, longer timeout. Otherwise a slow first load would be killed
  // by the 8s detect timeout and retried in a loop.
  const LOAD_TIMEOUT_MS = 60000;

  function resetWorker(reason) {
    if (worker) {
      try { worker.terminate(); } catch (e) { /* already dead */ }
    }
    worker = null;
    loadPromise = null; // so the next load()/detect() spins up a fresh worker + model
    const err = new Error("[ObjectDetector] worker reset: " + reason);
    pending.forEach((p) => { clearTimeout(p.timer); p.reject(err); });
    pending.clear();
    console.warn("[ObjectDetector] worker reset:", reason);
  }

  function getWorker() {
    if (worker) return worker;
    if (!WORKER_URL || !MODELS.accurate.url || !MODELS.fast.url) {
      throw new Error("[ObjectDetector] data-worker / data-model-accurate / data-model-fast missing on script tag");
    }
    if (typeof Worker === "undefined") throw new Error("[ObjectDetector] Web Workers not supported");
    worker = new Worker(WORKER_URL);
    worker.onmessage = (e) => {
      const msg = e.data || {};
      if (msg.type === "log") { console.warn("[ObjectDetector]", msg.message); return; }
      const p = pending.get(msg.id);
      if (!p) return;
      pending.delete(msg.id);
      clearTimeout(p.timer);
      if (msg.type === "error") p.reject(new Error(msg.message));
      else p.resolve(msg);
    };
    worker.onerror = (e) => {
      resetWorker(e.message || "unknown worker error");
    };
    return worker;
  }

  function call(payload, transfer, timeoutMs) {
    const limit = timeoutMs || CALL_TIMEOUT_MS;
    return new Promise((resolve, reject) => {
      const id = nextId++;
      const entry = {
        resolve,
        reject,
        timer: setTimeout(() => {
          if (!pending.has(id)) return;
          pending.delete(id);
          resetWorker("no response within " + limit + "ms");
          reject(new Error("[ObjectDetector] worker call timed out"));
        }, limit),
      };
      pending.set(id, entry);
      try {
        getWorker().postMessage(Object.assign({ id }, payload), transfer || []);
      } catch (err) {
        clearTimeout(entry.timer);
        pending.delete(id);
        reject(err);
      }
    });
  }

  function sourceSize(src) {
    if (src instanceof HTMLVideoElement) return [src.videoWidth, src.videoHeight];
    if (src instanceof HTMLImageElement) return [src.naturalWidth, src.naturalHeight];
    return [src.width, src.height];
  }

  // Letterboxes the frame into an inputSize x inputSize canvas (top-left,
  // gray padding), exactly like YOLOX's own preprocessing.
  function capture(src) {
    const [sw, sh] = sourceSize(src);
    if (!sw || !sh) return null;
    const r = Math.min(inputSize / sw, inputSize / sh);
    if (!canvas || canvas.width !== inputSize) {
      canvas = document.createElement("canvas");
      canvas.width = inputSize;
      canvas.height = inputSize;
      ctx = canvas.getContext("2d", { willReadFrequently: true });
      ctx.imageSmoothingQuality = "medium";
    }
    ctx.fillStyle = PAD_GRAY;
    ctx.fillRect(0, 0, inputSize, inputSize);
    ctx.drawImage(src, 0, 0, Math.round(sw * r), Math.round(sh * r));
    return { data: ctx.getImageData(0, 0, inputSize, inputSize).data.buffer, ratio: r, sw, sh };
  }

  const model = {
    lastInferMs: 0,
    info: null, // { model, backend, inputSize }

    async detect(input, maxNumBoxes = 20, minScore = 0.3) {
      const frame = capture(input);
      if (!frame) return [];
      try {
        const res = await call({ type: "detect", data: frame.data, minScore, maxNumBoxes }, [frame.data]);
        model.lastInferMs = res.inferMs;
        return (res.predictions || [])
          .map((p) => {
            const x = Math.max(0, p.bbox[0] / frame.ratio);
            const y = Math.max(0, p.bbox[1] / frame.ratio);
            const w = Math.min(frame.sw - x, p.bbox[2] / frame.ratio);
            const h = Math.min(frame.sh - y, p.bbox[3] / frame.ratio);
            return { class: p.class, score: p.score, bbox: [x, y, w, h] };
          })
          .filter((p) => p.bbox[2] > 0 && p.bbox[3] > 0);
      } catch (err) {
        // Worker died or timed out — call() already discarded it and
        // reset loadPromise, so kick off a reload in the background
        // (fire-and-forget) rather than letting every future detect()
        // fail the same way. This single check just reports "nothing".
        if (!loadPromise) api.load().catch(() => {});
        console.warn("[ObjectDetector] detect() failed, reloading worker:", (err && err.message) || err);
        return [];
      }
    },

    dispose() {
      if (worker) worker.terminate();
      worker = null;
      loadPromise = null;
    },
  };

  const api = {
    load() {
      if (!loadPromise) {
        loadPromise = call(
          {
            type: "load",
            ortUrl: ORT_URL,
            wasmPaths: WASM_PATHS,
            models: MODELS,
            force: window.OBJECT_DETECTOR_FORCE,
          },
          [],
          LOAD_TIMEOUT_MS
        )
          .then((res) => {
            inputSize = res.inputSize;
            model.info = { model: res.model, backend: res.backend, inputSize: res.inputSize };
            console.log(
              `[ObjectDetector] ${res.model} ready in worker ` +
              `(${res.backend}, ${res.inputSize}px, warm-up ${res.warmupMs} ms)`
            );
            return model;
          })
          .catch((err) => {
            loadPromise = null; // allow a later retry
            throw err;
          });
      }
      return loadPromise;
    },
  };

  window.ObjectDetector = api;
  window.cocoSsd = api; // backwards compatibility
})();