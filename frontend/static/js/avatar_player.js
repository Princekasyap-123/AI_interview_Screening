// frontend/static/js/avatar_player.js

/**
 * Photo-based AI-interviewer avatar for interview_room.html.
 *
 * Free, CPU-only, runs entirely in the browser (no server compute, no paid
 * API, no WebGL). Matches the current no-GPU phase:
 * apps/ai_engine/tts_client.py -> {"mode": "browser_tts"}.
 *
 * Assets (all crops/warps of the same photo, so every pixel is real skin,
 * lips and teeth):
 *   interviewer_base.webp   960 x 720 photo
 *   interviewer_mouth.webp  10 mouth frames stacked vertically, 154 x 144 each,
 *                           drawn at (412, 362):
 *                             N0..N5  jaw openness 0 -> 1 (N0 = lips closed,
 *                                     N1 = original photo)
 *                             R1..R3  rounded lips (o / u / w), openness .4/.7/1
 *                             F       lower lip under upper teeth (f / v)
 *   interviewer_blink.webp  2 eye frames stacked vertically, 206 x 52 each,
 *                           drawn at (382, 224): half-closed, closed
 *
 * How it looks natural on a CPU:
 *   - Mouth is a continuous "openness" value, not frame swapping. Adjacent
 *     frames are alpha-blended on a <canvas>, so the jaw glides between
 *     positions the way it does with a real voice envelope.
 *   - The utterance text is turned into a syllable timeline (vowel peaks,
 *     closures on m/b/p, lip-under-teeth on f/v, pauses on punctuation),
 *     with small random variation so no two sentences move identically.
 *     A critically-damped spring smooths the result (lips have mass).
 *   - Timing self-calibrates: after each utterance the real spoken duration
 *     is compared with the prediction, and the next one is scaled to match
 *     this browser's voice. SpeechSynthesis "boundary" events (Edge natural
 *     voices fire them reliably) re-sync the timeline word by word.
 *   - Head is never frozen: layered slow noise + breathing, small nods on
 *     stressed words, a head tilt on questions, settling at sentence ends,
 *     and occasional "I'm listening" nods while the candidate answers.
 *   - Blinks cluster at pauses and sentence ends, as people do.
 *
 * Public API (unchanged from the previous version):
 *   mount(stageEl, panelEl) -> Promise
 *   speakStart(text, rate) / onBoundary(charIndex) / speakEnd(text)
 *   setListening(on) / stop()
 *
 * If mount() fails for any reason the panel never gets .avatar-ready and
 * the existing WF mark stays visible.
 */

(function () {
  "use strict";

  // ------------------------------------------------------------------
  // Geometry (regenerate the frame sheets if the base photo changes)
  // ------------------------------------------------------------------
  const IMG_W = 960;
  const IMG_H = 720;
  const MOUTH = { x: 412, y: 362, w: 154, h: 144 };
  const MOUTH_FRAMES = ["N0", "N1", "N2", "N3", "N4", "N5", "R1", "R2", "R3", "F"];
  const NEUTRAL = ["N0", "N1", "N2", "N3", "N4", "N5"];           // openness 0, .2, .4, .6, .8, 1
  const ROUND = [["N0", 0], ["R1", 0.4], ["R2", 0.7], ["R3", 1.0]]; // [frame, openness]
  const BLINK = { x: 382, y: 224, w: 206, h: 52, half: 0, closed: 1 };

  const HEAD_PIVOT = { x: 488, y: 600 };   // roughly the base of the neck
  const BASE_SCALE = 1.05;                 // hides photo edges while the head moves

  // ------------------------------------------------------------------
  // Timing
  // ------------------------------------------------------------------
  const BASE_MS_PER_CHAR = 62;  // at utterance.rate = 1.0, before calibration
  const CALIB_KEY = "wfAvatarSpeechCalib";

  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const lerp = (a, b, t) => a + (b - a) * t;
  const ease = (t) => t * t * (3 - 2 * t);
  const rand = (a, b) => a + Math.random() * (b - a);

  // ------------------------------------------------------------------
  // Text -> mouth timeline
  // ------------------------------------------------------------------

  // Peak openness / roundness for each vowel.
  const VOWELS = {
    a: { open: 0.95, round: 0 },
    e: { open: 0.62, round: 0 },
    i: { open: 0.52, round: 0 },
    y: { open: 0.5, round: 0 },
    o: { open: 0.8, round: 0.9 },
    u: { open: 0.45, round: 1 },
  };
  const isVowel = (c) => "aeiouy".includes(c);
  const isLabial = (c) => "mbp".includes(c);
  const isLabioDental = (c) => "fv".includes(c);
  const isRounding = (c) => c === "w" || c === "q";

  /**
   * Builds { keys, wordTimes, total, events } for an utterance.
   * keys:      [{ t, open, round, f }] sorted by t (ms from speech start)
   * wordTimes: [{ index, t }] start time of each word, for boundary resync
   * events:    [{ t, type: "nod" | "sentenceEnd" | "question" | "pause" }]
   */
  function buildTimeline(text, msPerChar) {
    const keys = [];
    const wordTimes = [];
    const events = [];
    let t = 0;
    const push = (k) => keys.push(Object.assign({ open: 0, round: 0, f: 0 }, k));

    push({ t: 0, open: 0 });

    const re = /[A-Za-z0-9']+|[.,!?;:…-]|\s+|./g;
    let m;
    let wordCount = 0;
    while ((m = re.exec(text))) {
      const tok = m[0];
      if (/^\s+$/.test(tok)) {
        // gap between words: lips relax slightly, but don't fully close
        const gap = msPerChar * 0.6;
        push({ t: t + gap * 0.5, open: 0.12 });
        t += gap;
        continue;
      }
      if (/^[.,!?;:…-]$/.test(tok)) {
        const long = /[.!?…]/.test(tok);
        const pause = msPerChar * (long ? 7 : tok === "-" ? 2 : 3.5);
        push({ t: t + 60, open: 0 });
        push({ t: t + pause, open: 0 });
        events.push({ t: t + 40, type: tok === "?" ? "question" : long ? "sentenceEnd" : "pause" });
        t += pause;
        continue;
      }
      if (!/[A-Za-z0-9]/.test(tok)) { t += msPerChar; continue; }

      // ---- a word ----
      wordTimes.push({ index: m.index, t });
      wordCount++;
      const w = tok.toLowerCase().replace(/'/g, "");
      const wordDur = Math.max(w.length, 2) * msPerChar;
      const emphasis = (w.length >= 6 ? 1.12 : 1) * rand(0.86, 1.08);
      if (w.length >= 7 && Math.random() < 0.55) events.push({ t: t + wordDur * 0.25, type: "nod" });

      // split into syllables at vowel groups: [onset][vowels][coda...]
      const sylls = [];
      let cur = { onset: "", nucleus: "", coda: "" };
      for (const c of w) {
        if (isVowel(c)) {
          if (cur.coda) { sylls.push(cur); cur = { onset: cur.coda.slice(-1), nucleus: "", coda: "" }; sylls[sylls.length - 1].coda = sylls[sylls.length - 1].coda.slice(0, -1); }
          cur.nucleus += c;
        } else if (cur.nucleus) {
          cur.coda += c;
        } else {
          cur.onset += c;
        }
      }
      sylls.push(cur);

      const totalChars = sylls.reduce((n, s) => n + s.onset.length + s.nucleus.length + s.coda.length, 0) || 1;
      sylls.forEach((s, si) => {
        const chars = s.onset.length + s.nucleus.length + s.coda.length || 1;
        const dur = wordDur * (chars / totalChars);
        const t0 = t;

        // onset
        const on = s.onset;
        if (on && isLabial(on[on.length - 1])) push({ t: t0 + dur * 0.08, open: 0 });
        else if (on && isLabioDental(on[on.length - 1])) push({ t: t0 + dur * 0.1, open: 0.12, f: 1 });
        else if (on && isRounding(on[on.length - 1])) push({ t: t0 + dur * 0.1, open: 0.25, round: 1 });
        else if (on) push({ t: t0 + dur * 0.1, open: 0.22 });

        // nucleus (vowel peak)
        const v = VOWELS[s.nucleus[0]] || { open: 0.4, round: 0 };
        const stress = si === 0 ? 1.08 : 0.94;
        push({
          t: t0 + dur * 0.48,
          open: clamp(v.open * emphasis * stress, 0.18, 1),
          round: v.round,
        });

        // coda
        const co = s.coda;
        if (co) {
          const last = co[co.length - 1];
          if (isLabial(last)) push({ t: t0 + dur * 0.92, open: 0 });
          else if (isLabioDental(last)) push({ t: t0 + dur * 0.9, open: 0.1, f: 1 });
          else push({ t: t0 + dur * 0.9, open: 0.25 });
        }
        t += dur;
      });
    }

    push({ t: t + 80, open: 0 });
    return { keys, wordTimes, events, total: t + 80, words: wordCount };
  }

  function sampleTimeline(keys, t) {
    if (!keys.length) return { open: 0, round: 0, f: 0 };
    if (t <= keys[0].t) return keys[0];
    // binary search for the segment
    let lo = 0, hi = keys.length - 1;
    if (t >= keys[hi].t) return keys[hi];
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (keys[mid].t <= t) lo = mid; else hi = mid;
    }
    const a = keys[lo], b = keys[hi];
    const u = ease(clamp((t - a.t) / Math.max(b.t - a.t, 1), 0, 1));
    return { open: lerp(a.open, b.open, u), round: lerp(a.round, b.round, u), f: lerp(a.f, b.f, u) };
  }

  // Smooth noise: sum of incommensurate sines (cheap, no allocations).
  function noise(t, seed) {
    return (
      Math.sin(t * 0.00071 + seed) * 0.5 +
      Math.sin(t * 0.00133 + seed * 2.1) * 0.3 +
      Math.sin(t * 0.00271 + seed * 3.7) * 0.2
    );
  }

  // Critically damped spring step.
  function spring(state, target, omega, dt) {
    const f = 1 + 2 * dt * omega;
    const oo = omega * omega;
    const hoo = dt * oo;
    const hhoo = dt * hoo;
    const detInv = 1 / (f + hhoo);
    const x = (f * state.x + dt * state.v + hhoo * target) * detInv;
    const v = (state.v + hoo * (target - state.x)) * detInv;
    state.x = x; state.v = v;
    return x;
  }

  function preload(url) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error(`[AvatarPlayer] failed to load ${url}`));
      img.src = url;
    });
  }

  // ------------------------------------------------------------------
  // Player
  // ------------------------------------------------------------------

  class AvatarPlayer {
    constructor() {
      this.ready = false;
      this.panel = null;
      this.canvas = null;
      this.ctx = null;
      this.img = { base: null, mouth: null, blink: null };

      this.motionScale = 1;
      this.rafId = null;
      this.lastNow = 0;

      // mouth springs
      this.sOpen = { x: 0, v: 0 };
      this.sRound = { x: 0, v: 0 };
      this.sF = { x: 0, v: 0 };

      // head springs
      this.sRot = { x: 0, v: 0 };
      this.sTx = { x: 0, v: 0 };
      this.sTy = { x: 0, v: 0 };
      this.impulses = []; // { start, dur, ty, rot }

      // blink
      this.blinkStart = -1;
      this.nextBlinkAt = 0;

      // speech
      this.speaking = false;
      this.listening = false;
      this.text = "";
      this.timeline = null;
      this.speechStart = 0;
      this.offset = 0;
      this.offsetTarget = 0;
      this.eventIdx = 0;
      this.questionTilt = 0;
      this.calib = 1;
      this.rate = 1;
      this.nextListenNod = 0;

      try {
        const c = parseFloat(window.localStorage.getItem(CALIB_KEY));
        if (c > 0.5 && c < 2) this.calib = c;
      } catch (e) { /* storage unavailable — use default */ }
    }

    /**
     * Reads image URLs from the stage's data attributes
     * (data-avatar-base / data-avatar-mouth / data-avatar-blink),
     * preloads them, starts the render loop and marks the panel ready.
     */
    async mount(stageEl, panelEl) {
      if (!stageEl || !panelEl) throw new Error("[AvatarPlayer] stage or panel element not found");
      const src = {
        base: stageEl.dataset.avatarBase,
        mouth: stageEl.dataset.avatarMouth,
        blink: stageEl.dataset.avatarBlink,
      };
      if (!src.base || !src.mouth || !src.blink) {
        throw new Error("[AvatarPlayer] data-avatar-base/mouth/blink missing on stage element");
      }

      const [base, mouth, blink] = await Promise.all([preload(src.base), preload(src.mouth), preload(src.blink)]);
      this.img = { base, mouth, blink };

      const canvas = document.createElement("canvas");
      canvas.className = "avatar-character";
      canvas.width = IMG_W;
      canvas.height = IMG_H;
      canvas.setAttribute("aria-hidden", "true");
      const ctx = canvas.getContext("2d");
      if (!ctx) throw new Error("[AvatarPlayer] canvas 2D not supported");
      ctx.imageSmoothingQuality = "high";

      stageEl.innerHTML = "";
      stageEl.appendChild(canvas);
      const shade = document.createElement("div");
      shade.className = "avatar-shade";
      stageEl.appendChild(shade);

      this.panel = panelEl;
      this.canvas = canvas;
      this.ctx = ctx;

      try {
        if (window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
          this.motionScale = 0.3;
        }
      } catch (e) { /* ignore */ }

      const now = performance.now();
      this.lastNow = now;
      this.nextBlinkAt = now + rand(1500, 3500);
      this.rafId = requestAnimationFrame((t) => this._frame(t));

      panelEl.classList.add("avatar-ready");
      this.ready = true;
      console.log("[AvatarPlayer] mounted (speech calibration " + this.calib.toFixed(2) + ")");
    }

    // ---------------- speech hooks (called from interview_room.html) ----

    /** text = utterance being spoken; rate = ttsPlayer.rate */
    speakStart(text, rate) {
      if (!this.ready) return;
      this.rate = rate > 0 ? rate : 1;
      this.text = String(text || "");
      const msPerChar = (BASE_MS_PER_CHAR / this.rate) * this.calib;
      this.timeline = buildTimeline(this.text, msPerChar);
      this.speechStart = performance.now();
      this.offset = 0;
      this.offsetTarget = 0;
      this.eventIdx = 0;
      this.speaking = true;
      this.setListening(false);
      this.panel.classList.add("avatar-talking");
      // slight lift of the head as she starts talking
      this._impulse(0, 420, -2.2, -0.25);
    }

    /** Re-sync from SpeechSynthesis "boundary" events (word starts). */
    onBoundary(charIndex) {
      if (!this.ready || !this.speaking || !this.timeline) return;
      if (typeof charIndex !== "number" || charIndex < 0) return;
      const wt = this.timeline.wordTimes;
      if (!wt.length) return;
      let lo = 0, hi = wt.length - 1;
      while (lo < hi) {
        const mid = (lo + hi + 1) >> 1;
        if (wt[mid].index <= charIndex) lo = mid; else hi = mid - 1;
      }
      const elapsed = performance.now() - this.speechStart;
      this.offsetTarget = wt[lo].t - elapsed;
    }

    /**
     * text = the utterance that ended. speak() in tts_player.js calls
     * synth.cancel() before a new utterance, so an old utterance's
     * end/error can arrive after the new one started — ignore those.
     */
    speakEnd(text) {
      if (!this.ready) return;
      if (typeof text === "string" && this.text && text !== this.text) return;
      if (this.speaking && this.timeline) this._calibrate(performance.now() - this.speechStart);
      this.speaking = false;
      this.timeline = null;
      if (this.panel) this.panel.classList.remove("avatar-talking");
      // settle: small nod as she finishes
      this._impulse(80, 520, 2.4, 0.1);
      this._maybeBlinkSoon(0.6, 260);
    }

    /** Attentive pose while the candidate is answering. */
    setListening(on) {
      if (!this.ready) return;
      this.listening = !!on;
      this.panel.classList.toggle("avatar-listening", this.listening);
      if (this.listening) this.nextListenNod = performance.now() + rand(2500, 4500);
    }

    /** Hard stop — used by cleanupAllMedia(). */
    stop() {
      if (!this.ready) return;
      this.speaking = false;
      this.timeline = null;
      this.setListening(false);
      if (this.panel) this.panel.classList.remove("avatar-talking");
    }

    // ---------------- internals ----------------

    _calibrate(actualMs) {
      const predicted = this.timeline.total;
      if (actualMs < 1500 || predicted < 1000) return; // too short to be meaningful
      const ratio = actualMs / predicted;
      if (ratio < 0.5 || ratio > 2) return;            // interrupted / cancelled speech
      this.calib = clamp(this.calib * lerp(1, ratio, 0.5), 0.5, 2);
      try { window.localStorage.setItem(CALIB_KEY, this.calib.toFixed(3)); } catch (e) { /* ignore */ }
    }

    _impulse(delayMs, durMs, ty, rot) {
      this.impulses.push({ start: performance.now() + delayMs, dur: durMs, ty, rot });
    }

    _maybeBlinkSoon(prob, delay) {
      if (Math.random() < prob) {
        const at = performance.now() + delay;
        if (at < this.nextBlinkAt) this.nextBlinkAt = at;
      }
    }

    _speechTarget(now) {
      if (!this.speaking || !this.timeline) return { open: 0, round: 0, f: 0 };
      this.offset = lerp(this.offset, this.offsetTarget, 0.15);
      const t = now - this.speechStart + this.offset;

      // timeline events (nods, sentence ends, questions)
      const ev = this.timeline.events;
      while (this.eventIdx < ev.length && ev[this.eventIdx].t <= t) {
        const e = ev[this.eventIdx++];
        if (e.type === "nod") this._impulse(0, 420, 3.8, 0.28);
        else if (e.type === "sentenceEnd") { this._impulse(0, 600, 2.0, 0); this._maybeBlinkSoon(0.65, 120); this.questionTilt = 0; }
        else if (e.type === "question") { this.questionTilt = 1; this._maybeBlinkSoon(0.4, 200); }
        else if (e.type === "pause") this._maybeBlinkSoon(0.25, 100);
      }

      if (t <= this.timeline.total - 80) return sampleTimeline(this.timeline.keys, t);

      // Voice still going past the prediction: soft generic talking.
      const p = t * 0.0125;
      const open = 0.32 + 0.28 * Math.abs(Math.sin(p)) * (0.7 + 0.3 * Math.sin(p * 0.37));
      return { open, round: Math.max(0, Math.sin(p * 0.21)) * 0.4, f: 0 };
    }

    _frame(now) {
      this.rafId = requestAnimationFrame((t) => this._frame(t));
      const dt = clamp((now - this.lastNow) / 1000, 0, 0.05);
      this.lastNow = now;

      // ---------- mouth ----------
      const tgt = this._speechTarget(now);
      const open = clamp(spring(this.sOpen, tgt.open, 30, dt), 0, 1);
      const round = clamp(spring(this.sRound, tgt.round, 22, dt), 0, 1);
      const fAmt = clamp(spring(this.sF, tgt.f, 34, dt), 0, 1);

      // ---------- head ----------
      const M = this.motionScale;
      const talkAmp = this.speaking ? 1.8 : 1;
      let rot = noise(now, 1.3) * 0.38 * talkAmp;
      let tx = noise(now, 4.7) * 2.6 * talkAmp;
      let ty = noise(now, 7.9) * 1.6 * talkAmp;
      const breath = Math.sin(now * 0.0015);                     // ~4.2 s cycle
      ty += breath * 1.1;

      if (this.listening) {
        rot += 0.75;
        if (now >= this.nextListenNod) {
          this._impulse(0, 560, 4.4, 0);
          if (Math.random() < 0.35) this._impulse(600, 500, 3.2, 0);
          this.nextListenNod = now + rand(3500, 7500);
        }
      }
      if (this.speaking && this.questionTilt) rot += 0.9 * this.questionTilt;
      if (!this.speaking) this.questionTilt = Math.max(0, this.questionTilt - dt * 0.6);

      // nod / lift impulses: smooth down-and-back
      this.impulses = this.impulses.filter((im) => now < im.start + im.dur);
      for (const im of this.impulses) {
        const u = (now - im.start) / im.dur;
        if (u < 0) continue;
        const k = Math.sin(Math.PI * u) * (1 - u * 0.3);
        ty += im.ty * k;
        rot += im.rot * k;
      }

      const hr = spring(this.sRot, rot * M, 6, dt);
      const hx = spring(this.sTx, tx * M, 6, dt);
      const hy = spring(this.sTy, ty * M, 9, dt);

      // ---------- blink ----------
      if (this.blinkStart < 0 && now >= this.nextBlinkAt) {
        this.blinkStart = now;
        const dbl = Math.random() < 0.12;
        this.nextBlinkAt = now + (dbl ? 330 : rand(2600, 6000));
      }
      let blink = 0;
      if (this.blinkStart >= 0) {
        const bt = now - this.blinkStart;
        if (bt < 70) blink = bt / 70;
        else if (bt < 95) blink = 1;
        else if (bt < 200) blink = 1 - (bt - 95) / 105;
        else { blink = 0; this.blinkStart = -1; }
      }

      this._draw(hr, hx, hy, open, round, fAmt, blink);
    }

    _draw(rotDeg, tx, ty, open, round, fAmt, blink) {
      const ctx = this.ctx;
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.globalAlpha = 1;

      // head transform: scale about the centre, rotate about the neck
      ctx.translate(IMG_W / 2, IMG_H / 2);
      ctx.scale(BASE_SCALE, BASE_SCALE);
      ctx.translate(-IMG_W / 2, -IMG_H / 2);
      ctx.translate(HEAD_PIVOT.x + tx, HEAD_PIVOT.y + ty);
      ctx.rotate((rotDeg * Math.PI) / 180);
      ctx.translate(-HEAD_PIVOT.x, -HEAD_PIVOT.y);

      ctx.drawImage(this.img.base, 0, 0, IMG_W, IMG_H);

      // ---- mouth: neutral series blended by openness ----
      const lv = open * (NEUTRAL.length - 1);
      const i0 = Math.min(Math.floor(lv), NEUTRAL.length - 2);
      const fr = lv - i0;
      this._mouth(NEUTRAL[i0], 1);
      if (fr > 0.01) this._mouth(NEUTRAL[i0 + 1], fr);

      // ---- rounded lips blended on top ----
      if (round > 0.02) {
        let j = 0;
        while (j < ROUND.length - 2 && open > ROUND[j + 1][1]) j++;
        const [fa, oa] = ROUND[j];
        const [fb, ob] = ROUND[j + 1];
        const u = clamp((open - oa) / (ob - oa), 0, 1);
        this._mouth(fa, round);
        if (u > 0.01) this._mouth(fb, round * u);
      }

      // ---- f / v ----
      if (fAmt > 0.02) this._mouth("F", fAmt);

      // ---- eyes ----
      if (blink > 0.01) {
        this._blink(BLINK.half, Math.min(1, blink * 2));
        if (blink > 0.5) this._blink(BLINK.closed, (blink - 0.5) * 2);
      }
      ctx.globalAlpha = 1;
    }

    _mouth(name, alpha) {
      const i = MOUTH_FRAMES.indexOf(name);
      this.ctx.globalAlpha = alpha;
      this.ctx.drawImage(this.img.mouth, 0, i * MOUTH.h, MOUTH.w, MOUTH.h, MOUTH.x, MOUTH.y, MOUTH.w, MOUTH.h);
    }

    _blink(i, alpha) {
      this.ctx.globalAlpha = alpha;
      this.ctx.drawImage(this.img.blink, 0, i * BLINK.h, BLINK.w, BLINK.h, BLINK.x, BLINK.y, BLINK.w, BLINK.h);
    }
  }

  // Exposed for tests only.
  AvatarPlayer._buildTimeline = buildTimeline;
  AvatarPlayer._sampleTimeline = sampleTimeline;

  window.AvatarPlayer = AvatarPlayer;
})();