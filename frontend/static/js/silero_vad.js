/**
 * SileroVAD — main-thread side. Feeds the microphone to silero_vad_worker.js and
 * keeps the latest speech probabilities.
 *
 *   const vad = new SileroVAD({ workerUrl, modelUrl });
 *   vad.start(audioContext, sourceNode);     // resolves false if the model cannot load
 *   vad.isSpeech()                           // true when a human voice was heard in the last ~250 ms
 *
 * If anything fails (offline CDN, old browser) `ready` stays false and the caller keeps
 * using its loudness-based gate, so the interview never depends on this.
 */
(function () {
  "use strict";

  const ORT_VERSION = "1.20.1";
  const ORT_URL = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/ort.min.js`;
  const WASM_PATHS = ORT_URL.slice(0, ORT_URL.lastIndexOf("/") + 1);
  const TARGET_SR = 16000;
  const FRAME = 512;
  const SPEECH_PROB = 0.4;        // Silero suggests 0.5; softer voices score lower and music stays below 0.15
  const WINDOW_FRAMES = 12;      // 12 x 32 ms = the last ~380 ms
  const NON_SPEECH_FRAMES = 10;     // ~320 ms of fresh frames needed before Silero may veto a sound
  const NON_SPEECH_MAX_PROB = 0.25; // every one of them below this = "not a voice" (music peaks near 0.10)
  const STALE_MS = 400;             // probabilities older than this are not trusted (the page was busy)
  const MIN_SPEECH_FRAMES = 1;   // one speech frame in that window keeps the voice "on" (music / fans never reach it)

  /** Linear resampler + 512-sample framer; pure, so it can be tested without a browser. */
  class Framer {
    constructor(inputRate, onFrame) {
      this.ratio = inputRate / TARGET_SR;
      this.onFrame = onFrame;
      this.pos = 0;                   // fractional read position inside `carry`
      this.carry = new Float32Array(0);
      this.out = new Float32Array(FRAME);
      this.n = 0;
    }
    push(input) {
      const buf = new Float32Array(this.carry.length + input.length);
      buf.set(this.carry, 0);
      buf.set(input, this.carry.length);
      let pos = this.pos;
      while (pos + 1 < buf.length) {
        const i = Math.floor(pos);
        const f = pos - i;
        this.out[this.n++] = buf[i] * (1 - f) + buf[i + 1] * f;
        if (this.n === FRAME) {
          this.onFrame(this.out.slice());
          this.n = 0;
        }
        pos += this.ratio;
      }
      const keep = Math.floor(pos);
      this.carry = buf.slice(keep);
      this.pos = pos - keep;
    }
  }

  class SileroVAD {
    constructor(opts = {}) {
      this.workerUrl = opts.workerUrl;
      this.modelUrl = opts.modelUrl;
      this.ortUrl = opts.ortUrl || ORT_URL;
      this.wasmPaths = opts.wasmPaths || WASM_PATHS;
      this.ready = false;
      this.failed = false;
      this.probs = [];
      this.lastProbAt = 0;
      this.worker = null;
      this.node = null;
    }

    start(ctx, source) {
      return new Promise((resolve) => {
        if (!this.workerUrl || !this.modelUrl || !window.Worker) { this.failed = true; resolve(false); return; }
        let settled = false;
        const done = (ok) => { if (!settled) { settled = true; resolve(ok); } };
        const timer = setTimeout(() => { this._fail("model load timed out"); done(false); }, 20000);
        try {
          this.worker = new Worker(this.workerUrl);
        } catch (err) { this._fail(err); clearTimeout(timer); done(false); return; }

        this.worker.onmessage = (e) => {
          const m = e.data;
          if (m.type === "ready") {
            clearTimeout(timer);
            try { this._attach(ctx, source); this.ready = true; console.log("[SileroVAD] ready"); done(true); }
            catch (err) { this._fail(err); done(false); }
          } else if (m.type === "prob") {
            this.probs.push(m.p);
            this.lastProbAt = performance.now();
            if (this.probs.length > WINDOW_FRAMES) this.probs.shift();
          } else if (m.type === "error") {
            clearTimeout(timer);
            this._fail(m.message);
            done(false);
          }
        };
        this.worker.onerror = (e) => { clearTimeout(timer); this._fail(e.message || "worker error"); done(false); };
        this.worker.postMessage({
          type: "load", ortUrl: this.ortUrl, wasmPaths: this.wasmPaths, modelUrl: this.modelUrl,
        });
      });
    }

    _attach(ctx, source) {
      const framer = new Framer(ctx.sampleRate, (frame) => {
        if (this.worker) this.worker.postMessage({ type: "frame", data: frame }, [frame.buffer]);
      });
      // ScriptProcessor is deprecated but works in every browser without extra module files.
      this.node = ctx.createScriptProcessor(2048, 1, 1);
      this.node.onaudioprocess = (e) => framer.push(e.inputBuffer.getChannelData(0));
      const mute = ctx.createGain();       // the node must be connected to run; keep it silent
      mute.gain.value = 0;
      source.connect(this.node);
      this.node.connect(mute);
      mute.connect(ctx.destination);
    }

    _fail(why) {
      console.warn("[SileroVAD] unavailable, using the loudness gate instead:", why);
      this.failed = true;
      this.ready = false;
      this.stop();
    }

    /** True when a human voice (not music / noise) was heard in the last ~380 ms. */
    isSpeech() {
      let n = 0;
      for (const p of this.probs) if (p >= SPEECH_PROB) n++;
      return n >= MIN_SPEECH_FRAMES;
    }

    /**
     * True only when Silero is SURE the sound is not a voice: enough fresh frames, and none of them
     * looks even a little like speech. When the page is busy (frames late or missing) or the voice is
     * soft, this stays false so a candidate is never ignored - music and fans score far below the limit.
     */
    isNonSpeech() {
      if (!this.ready || this.probs.length < NON_SPEECH_FRAMES) return false;
      if (performance.now() - this.lastProbAt > STALE_MS) return false;
      for (let i = this.probs.length - NON_SPEECH_FRAMES; i < this.probs.length; i++) {
        if (this.probs[i] >= NON_SPEECH_MAX_PROB) return false;
      }
      return true;
    }

    /** Highest recent speech probability (for debugging). */
    level() { return this.probs.length ? Math.max(...this.probs) : 0; }

    stop() {
      try { if (this.node) this.node.disconnect(); } catch (e) { /* ignore */ }
      try { if (this.worker) this.worker.terminate(); } catch (e) { /* ignore */ }
      this.node = null;
      this.worker = null;
    }
  }

  if (typeof window !== "undefined") window.SileroVAD = SileroVAD;
  if (typeof module !== "undefined") module.exports = { Framer };
})();
