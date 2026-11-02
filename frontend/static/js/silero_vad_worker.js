/**
 * Silero VAD (speech / not-speech) in a Web Worker, via ONNX Runtime Web.
 *
 * Unlike the plain loudness gate, Silero tells a human voice from music, TV
 * soundtracks, fans and traffic. It receives 512-sample frames of 16 kHz mono
 * audio and answers with the probability that the frame contains speech.
 *
 * Messages in:  { type: "load", ortUrl, wasmPaths, modelUrl }
 *               { type: "frame", data: Float32Array(512) }
 * Messages out: { type: "ready" } | { type: "error", message } | { type: "prob", p }
 */
/* global ort */

const SR = 16000;
const FRAME = 512;          // samples per model call (32 ms)
const CONTEXT = 64;         // v5 model: the last 64 samples of the previous frame are prepended
const MAX_QUEUE = 12;       // a slow device drops frames instead of building up lag

function createRunner(ortLib, session) {
  let state = new Float32Array(2 * 1 * 128);
  let context = new Float32Array(CONTEXT);
  const sr = new ortLib.Tensor("int64", BigInt64Array.from([BigInt(SR)]), []);
  return {
    reset() { state = new Float32Array(2 * 128); context = new Float32Array(CONTEXT); },
    async run(frame) {
      const input = new Float32Array(CONTEXT + FRAME);
      input.set(context, 0);
      input.set(frame, CONTEXT);
      context = frame.slice(FRAME - CONTEXT);
      const out = await session.run({
        input: new ortLib.Tensor("float32", input, [1, CONTEXT + FRAME]),
        state: new ortLib.Tensor("float32", state, [2, 1, 128]),
        sr,
      });
      state = out.stateN.data.slice();
      return out.output.data[0];
    },
  };
}

if (typeof self !== "undefined" && typeof importScripts === "function") {
  let runner = null;
  let busy = false;
  const queue = [];

  const pump = async () => {
    if (busy) return;
    busy = true;
    while (queue.length) {
      const frame = queue.shift();
      try {
        self.postMessage({ type: "prob", p: await runner.run(frame) });
      } catch (err) {
        self.postMessage({ type: "error", message: String((err && err.message) || err) });
        queue.length = 0;
      }
    }
    busy = false;
  };

  self.onmessage = async (e) => {
    const msg = e.data;
    if (msg.type === "load") {
      try {
        importScripts(msg.ortUrl);
        if (msg.wasmPaths) ort.env.wasm.wasmPaths = msg.wasmPaths;
        ort.env.wasm.numThreads = 1;
        const session = await ort.InferenceSession.create(msg.modelUrl, { executionProviders: ["wasm"] });
        runner = createRunner(ort, session);
        self.postMessage({ type: "ready" });
      } catch (err) {
        self.postMessage({ type: "error", message: String((err && err.message) || err) });
      }
    } else if (msg.type === "frame" && runner) {
      if (queue.length >= MAX_QUEUE) queue.shift();
      queue.push(msg.data);
      pump();
    } else if (msg.type === "reset" && runner) {
      runner.reset();
    }
  };
}

if (typeof module !== "undefined") module.exports = { createRunner, SR, FRAME, CONTEXT };
