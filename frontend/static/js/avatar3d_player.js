// frontend/static/js/avatar3d_player.js

/**
 * 3D AI interviewer for interview_room.html (no GPU server, no video frames).
 *
 * Built on TalkingHead (MIT, https://github.com/met4citizen/TalkingHead) and
 * three.js, rendered live in the candidate's browser with WebGL. The avatar
 * is the CC0 "MPFB" character from the TalkingHead repo, compressed to ~4.6 MB
 * (WebP textures + meshopt), with its T-shirt recoloured to a plain dark top
 * at load time.
 *
 * Lip-sync: the interviewer's voice is still Kokoro (server TTS). Each
 * sentence's AudioBuffer is handed to playSpeech(); word timings are
 * estimated from the audio itself (voiced segments + word lengths) and
 * TalkingHead turns the words into mouth shapes (visemes) in real time.
 * Blinking, breathing, eye contact and small head movements are automatic.
 *
 * Same public API as avatar_player.js, so the page wiring stays the same:
 *   mount(stageEl, panelEl) -> Promise
 *   speakStart(text) / speakEnd(text) / onChunk() / setVoiceLevel() / onBoundary()
 *   setListening(on) / stop()
 * Extra (used by tts_player.js / interview_assist.js):
 *   playSpeech(audioBuffer, text) -> Promise (resolves when the audio ends)
 *   stopSpeech()
 *   nod()                       small "yes" nod (backchannel)
 *   listenTo(analyserNode)      reacts to the candidate's voice while listening
 *
 * Stage element data attributes:
 *   data-avatar3d-url     URL of interviewer3d.glb               (required)
 *   data-th-module        URL of talkinghead/talkinghead.mjs     (required)
 * The page must have an import map for "three" and "three/addons/".
 */

(function () {
  "use strict";

  const SHIRT_MATERIAL = "Human.female_casualsuit01";
  const SHIRT_COLOR = 0x2c3440;          // dark charcoal-navy, plain
  const LOAD_TIMEOUT_MS = 60000;

  /**
   * Estimates when each word starts/ends from the audio itself.
   * Kokoro gives no timestamps, so: find the voiced parts of the audio
   * (10 ms RMS envelope, pauses >= 150 ms), then spread the words over the
   * voiced time in proportion to their length. Pauses in speech almost
   * always fall between words, so this lines up well enough for visemes.
   */
  function estimateWordTimings(buffer, text) {
    const words = String(text).replace(/\s+/g, " ").trim().split(" ").filter(Boolean);
    if (!words.length) return { words: [], wtimes: [], wdurations: [] };

    const data = buffer.getChannelData(0);
    const sr = buffer.sampleRate;
    const hop = Math.max(1, Math.round(sr * 0.01));
    const env = [];
    for (let i = 0; i < data.length; i += hop) {
      let s = 0;
      const end = Math.min(data.length, i + hop);
      for (let j = i; j < end; j++) s += data[j] * data[j];
      env.push(Math.sqrt(s / (end - i)));
    }
    const sorted = env.slice().sort((a, b) => a - b);
    const p95 = sorted[Math.floor(sorted.length * 0.95)] || 0;
    const thr = Math.max(0.008, p95 * 0.08);

    // voiced segments in 10 ms frames, merging gaps shorter than 150 ms
    const segs = [];
    let start = -1, lastVoiced = -1;
    for (let f = 0; f < env.length; f++) {
      if (env[f] > thr) {
        if (start < 0) start = f;
        else if (f - lastVoiced > 15) { segs.push([start, lastVoiced + 1]); start = f; }
        lastVoiced = f;
      }
    }
    if (start >= 0) segs.push([start, lastVoiced + 1]);
    if (!segs.length) segs.push([0, env.length]);

    const voicedTotal = segs.reduce((t, [a, b]) => t + (b - a), 0);
    const weights = words.map((w) => w.replace(/[^\w']/g, "").length + 1);
    const wTotal = weights.reduce((a, b) => a + b, 0);

    // map a position on the "voiced-only" timeline back to real time (ms)
    const toMs = (v) => {
      for (const [a, b] of segs) {
        if (v <= b - a) return (a + v) * 10;
        v -= b - a;
      }
      return segs[segs.length - 1][1] * 10;
    };

    const wtimes = [], wdurations = [];
    let acc = 0;
    for (let i = 0; i < words.length; i++) {
      const t0 = toMs((acc / wTotal) * voicedTotal);
      acc += weights[i];
      const t1 = toMs((acc / wTotal) * voicedTotal);
      wtimes.push(Math.round(t0));
      wdurations.push(Math.max(60, Math.round(t1 - t0)));
    }
    return { words, wtimes, wdurations };
  }

  function withTimeout(promise, ms, what) {
    return Promise.race([
      promise,
      new Promise((_, reject) => setTimeout(() => reject(new Error(`[Avatar3DPlayer] ${what} timed out`)), ms)),
    ]);
  }

  class Avatar3DPlayer {
    constructor() {
      this.ready = false;
      this.head = null;
      this.panel = null;
      this.speaking = false;
      this.text = "";
      this.pending = [];          // resolvers of playSpeech() promises
    }

    async mount(stageEl, panelEl) {
      if (!stageEl || !panelEl) throw new Error("[Avatar3DPlayer] stage or panel element not found");
      const ds = stageEl.dataset;
      if (!ds.avatar3dUrl || !ds.thModule) throw new Error("[Avatar3DPlayer] data-avatar3d-url / data-th-module missing");

      // WebGL check first: without it, the page falls back to the video avatar.
      const probe = document.createElement("canvas");
      if (!(probe.getContext("webgl2") || probe.getContext("webgl"))) throw new Error("[Avatar3DPlayer] WebGL not available");

      const { TalkingHead } = await withTimeout(import(ds.thModule), LOAD_TIMEOUT_MS, "loading TalkingHead");

      stageEl.innerHTML = "";
      const host = document.createElement("div");
      host.className = "avatar3d-host";
      stageEl.appendChild(host);

      const head = new TalkingHead(host, {
        lipsyncModules: ["en"],
        lipsyncLang: "en",
        cameraView: "upper",
        cameraDistance: -0.45,           // head and shoulders, like a video call
        cameraY: 0.06,
        cameraRotateEnable: false,
        cameraPanEnable: false,
        cameraZoomEnable: false,
        modelFPS: 30,
        modelPixelRatio: Math.min(window.devicePixelRatio || 1, 2),
        lightAmbientIntensity: 2.2,
        lightDirectColor: 0xfff1e0,
        lightDirectIntensity: 18,
        avatarIdleEyeContact: 0.45,
        avatarIdleHeadMove: 0.4,
        avatarSpeakingEyeContact: 0.75,
        avatarSpeakingHeadMove: 0.45,
        mixerGainSpeech: 1.6,
      });

      await withTimeout(head.showAvatar({
        url: ds.avatar3dUrl,
        body: "F",
        avatarMood: "neutral",
        lipsyncLang: "en",
        avatarListeningEyeContact: 0.8,
        baseline: { headRotateX: -0.01, eyeBlinkLeft: 0.05, eyeBlinkRight: 0.05 },
      }), LOAD_TIMEOUT_MS, "loading the 3D avatar");

      // Plain dark top instead of the logo T-shirt (keeps the fabric folds).
      head.armature.traverse((o) => {
        const mats = o.isMesh ? (Array.isArray(o.material) ? o.material : [o.material]) : [];
        for (const m of mats) {
          if (m && m.name === SHIRT_MATERIAL) {
            m.map = null;
            m.color.setHex(SHIRT_COLOR);
            if ("roughness" in m) m.roughness = 0.85;
            m.needsUpdate = true;
          }
        }
      });

      // Browsers start audio contexts suspended until the user clicks.
      const resume = () => { try { head.audioCtx.resume(); } catch (e) { /* ignore */ } };
      document.addEventListener("click", resume);
      document.addEventListener("keydown", resume);

      document.addEventListener("visibilitychange", () => {
        if (!this.ready) return;
        if (document.hidden) head.stop(); else head.start();
      });

      this.head = head;
      this.panel = panelEl;
      panelEl.classList.add("avatar-ready", "avatar-3d");
      this.ready = true;
      console.log("[Avatar3DPlayer] mounted (3D avatar)");
    }

    // ---------------- speech ----------------

    /** Plays one sentence of Kokoro audio with lip-sync. Resolves when done. */
    playSpeech(buffer, text) {
      if (!this.ready) return Promise.reject(new Error("[Avatar3DPlayer] not ready"));
      const head = this.head;
      try { head.audioCtx.resume(); } catch (e) { /* ignore */ }
      const timing = estimateWordTimings(buffer, text);
      return new Promise((resolve) => {
        const entry = { resolve, timer: null };
        const finish = () => {
          clearTimeout(entry.timer);
          const i = this.pending.indexOf(entry);
          if (i >= 0) this.pending.splice(i, 1);
          resolve();
        };
        entry.resolve = finish;
        // safety net: never hang the interview if the marker is lost
        entry.timer = setTimeout(finish, Math.round(buffer.duration * 1000) + 4000);
        this.pending.push(entry);
        head.speakAudio({ audio: buffer, ...timing }, { lipsyncLang: "en" });
        head.speakMarker(finish);
      });
    }

    stopSpeech() {
      if (!this.ready) return;
      try { this.head.stopSpeaking(); } catch (e) { /* ignore */ }
      this.pending.splice(0).forEach((p) => p.resolve());
    }

    nod() {
      if (!this.ready) return;
      try { this.head.playGesture("yes", 1.2); } catch (e) { /* ignore */ }
    }

    /** While listening, TalkingHead reacts to the candidate's voice level. */
    listenTo(analyser) {
      if (!this.ready || !analyser) return;
      this.listenAnalyser = analyser;
    }

    // ---------------- avatar_player.js compatible hooks ----------------

    speakStart(text) {
      if (!this.ready) return;
      this.text = String(text || "");
      this.speaking = true;
      this.setListening(false);
      this.panel.classList.add("avatar-talking");
      try { this.head.lookAtCamera(800); } catch (e) { /* ignore */ }
    }

    speakEnd(text) {
      if (!this.ready) return;
      if (typeof text === "string" && this.text && text !== this.text) return; // stale end event
      this.speaking = false;
      this.panel.classList.remove("avatar-talking");
    }

    onChunk() { /* lip-sync comes from playSpeech() */ }
    setVoiceLevel() { /* lip-sync comes from playSpeech() */ }
    onBoundary() { /* not needed */ }

    setListening(on) {
      if (!this.ready) return;
      this.panel.classList.toggle("avatar-listening", !!on);
      try {
        if (on && this.listenAnalyser) {
          this.head.startListening(this.listenAnalyser, {}, null);
          this.head.makeEyeContact(3000);
        } else {
          this.head.stopListening();
        }
      } catch (e) { /* ignore */ }
    }

    stop() {
      if (!this.ready) return;
      this.speaking = false;
      this.stopSpeech();
      this.setListening(false);
      this.panel.classList.remove("avatar-talking");
    }
  }

  Avatar3DPlayer.estimateWordTimings = estimateWordTimings;   // exposed for tests
  window.Avatar3DPlayer = Avatar3DPlayer;
})();
