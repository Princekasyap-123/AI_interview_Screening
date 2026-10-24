// frontend/static/js/tts_player.js

/**
 * Speaks the AI interviewer's text.
 *
 * Primary path — SERVER VOICE (Kokoro, free, CPU):
 *   The text is split into sentences; each one is fetched as a WAV from
 *   POST /api/avatar/tts/ (apps/avatar/views.py) and played with the Web
 *   Audio API. The next sentence is fetched while the current one plays,
 *   so speech starts after the first sentence is ready. Every candidate
 *   hears the same natural voice, and because we have the real audio,
 *   its loudness is streamed to the avatar for audio-driven lip-sync.
 *
 * Best path — REAL LIP-SYNC (MuseTalk on a GPU server, optional):
 *   If /api/avatar/lipsync/status/ says the GPU server is up and the page
 *   set ttsPlayer.clipPlayer (the avatar), each sentence is fetched as a
 *   lip-synced MP4 (video + the same Kokoro voice) and played on the
 *   avatar. Any failure drops back to the server voice below — the
 *   interview never waits on the GPU.
 *
 * Fallback — BROWSER VOICE (speechSynthesis):
 *   Used automatically if the server voice is unavailable (model not
 *   installed, network error, old browser). Same events as before.
 *
 * Events (unchanged + new):
 *   speakStart(text)                     speaking began (whole utterance)
 *   speakEnd(text)                       speaking finished (whole utterance)
 *   speakError(error)
 *   wordBoundary({ charIndex, text })    browser voice only
 *   chunkStart({ text, durationMs })     server voice: a sentence starts playing
 *   level(number 0..1)                   server voice: voice loudness, every frame
 *
 *   const ttsPlayer = new TTSPlayer();
 *   interviewSocket.on("turn", (turn) => ttsPlayer.handleTurn(turn));
 */

class TTSPlayer {
  constructor() {
    this.synth = window.speechSynthesis;
    this.voice = null;

    // Browser-voice speaking style (fallback path only).
    this.rate = 0.9;
    this.pitch = 1.0;

    // Server voice.
    this.serverUrl = "/api/avatar/tts/";
    this.warmupUrl = "/api/avatar/tts/warmup/";
    this.useServerVoice = true;      // set false to force the browser voice
    this.audioCtx = null;
    this.analyser = null;
    this.gain = null;
    this.currentSource = null;
    this.levelRAF = null;
    this.token = 0;                  // increments on every speak()/stop()
    this.lastMode = null;            // "lipsync" | "server" | "browser"

    // Real lip-sync (MuseTalk). clipPlayer is set by interview_room.html.
    this.lipsyncUrl = "/api/avatar/lipsync/";
    this.lipsyncStatusUrl = "/api/avatar/lipsync/status/";
    this.lipsyncAvailable = false;
    this.lipsyncTimeoutMs = 120000;  // per sentence; free T4 GPU renders ~1.15x real time
    this.clipPlayer = null;
    // 3D avatar (avatar3d_player.js): plays each sentence with lip-sync.
    // Set by interview_room.html; null = plain Web Audio playback.
    this.speechPlayer = null;
    this.asideCache = new Map();     // text -> AudioBuffer ("Mm-hmm.", etc.)
    this.asideBusy = false;
    this.asideSource = null;         // playing aside (Web Audio), stopped by _halt()
    this.asideCancelled = false;

    this.listeners = {
      speakStart: [],    // fn(text)
      speakEnd: [],      // fn(text)
      speakError: [],    // fn(error)
      wordBoundary: [],  // fn({ charIndex, text })
      chunkStart: [],    // fn({ text, durationMs })
      level: [],         // fn(level 0..1)
    };

    if (!this.synth) {
      console.warn("[TTSPlayer] speechSynthesis not supported — only the server voice is available.");
    }

    this._loadPreferredVoice();
    this._warmUpServerVoice();
    this._checkLipsync();
  }

  // ------------------------------------------------------------------
  // Public event API
  // ------------------------------------------------------------------

  on(eventName, callback) {
    if (!this.listeners[eventName]) {
      throw new Error(`Unknown TTSPlayer event: ${eventName}`);
    }
    this.listeners[eventName].push(callback);
  }

  _emit(eventName, payload) {
    for (const cb of this.listeners[eventName]) {
      try {
        cb(payload);
      } catch (err) {
        console.error(`TTSPlayer listener for "${eventName}" threw:`, err);
      }
    }
  }

  // ------------------------------------------------------------------
  // Main entry point — called from interviewSocket.on("turn", ...)
  // ------------------------------------------------------------------

  handleTurn(turn) {
    const speakableActions = ["speak_intro", "speak_question", "speak_follow_up"];
    if (speakableActions.includes(turn.action) && turn.text) {
      this.speak(turn.text);
    }
  }

  speak(text) {
    this._halt();                       // stop anything still playing
    const token = ++this.token;
    const canServer = this.useServerVoice && this._sessionId() && (window.AudioContext || window.webkitAudioContext);
    if (canServer && this.lipsyncAvailable && this.clipPlayer && this.clipPlayer.playClip) {
      this._speakLipsync(text, token).catch((err) => {
        if (token !== this.token) return;
        console.warn("[TTSPlayer] lip-sync unavailable, using server voice:", err.message || err);
        this._speakServer(text, token).catch((err2) => {
          if (token !== this.token) return;
          console.warn("[TTSPlayer] server voice unavailable, using browser voice:", err2.message || err2);
          this._speakBrowser(text);
        });
      });
    } else if (canServer) {
      this._speakServer(text, token).catch((err) => {
        if (token !== this.token) return;           // superseded — ignore
        console.warn("[TTSPlayer] server voice unavailable, using browser voice:", err.message || err);
        this._speakBrowser(text);
      });
    } else {
      this._speakBrowser(text);
    }
  }

  stop() {
    this.token++;
    this._halt();
  }

  // ------------------------------------------------------------------
  // Asides: short lines spoken WITHOUT speakStart/speakEnd events, so the
  // interview flow (recording, turns) is not touched. Used while the
  // candidate is answering: "Mm-hmm.", "Sure, here is the question again…"
  // ------------------------------------------------------------------

  /** Fetches and caches aside audio in the background (call once at start). */
  prefetchAsides(lines) {
    if (!this.useServerVoice || !this._sessionId()) return;
    for (const t of lines) {
      if (this.asideCache.has(t)) continue;
      this._fetchChunk(t).then((b) => this.asideCache.set(t, b)).catch(() => {});
    }
  }

  /** Speaks text as an aside. Resolves true when done, false if skipped. */
  async sayAside(text) {
    if (this.asideBusy || !this.useServerVoice || !this._sessionId()) return false;
    this.asideBusy = true;
    this.asideCancelled = false;
    try {
      for (const part of TTSPlayer._chunks(text)) {
        if (this.asideCancelled) break;
        let buf = this.asideCache.get(part);
        if (!buf) {
          buf = await this._fetchChunk(part);
          if (part.length < 40) this.asideCache.set(part, buf);
        }
        if (this.asideCancelled) break;     // the next question started meanwhile
        if (this.speechPlayer && this.speechPlayer.playSpeech) {
          await this.speechPlayer.playSpeech(buf, part);
        } else {
          const ctx = this._ensureAudio();
          if (ctx.state === "suspended") await ctx.resume();
          await new Promise((resolve) => {
            const src = ctx.createBufferSource();
            src.buffer = buf;
            src.connect(this.gain);
            src.onended = () => { if (this.asideSource === src) this.asideSource = null; resolve(); };
            this.asideSource = src;
            src.start();
          });
        }
      }
      return true;
    } catch (err) {
      console.warn("[TTSPlayer] aside failed:", err.message || err);
      return false;
    } finally {
      this.asideBusy = false;
    }
  }

  // ------------------------------------------------------------------
  // Server voice (Kokoro via /api/avatar/tts/)
  // ------------------------------------------------------------------

  _sessionId() {
    try {
      // SESSION_ID is a top-level const in interview_room.html's inline script.
      // eslint-disable-next-line no-undef
      return typeof SESSION_ID !== "undefined" ? SESSION_ID : null;
    } catch (e) {
      return null;
    }
  }

  _warmUpServerVoice() {
    if (!this.useServerVoice || !window.fetch) return;
    fetch(this.warmupUrl, { credentials: "same-origin" }).catch(() => {});
  }

  /** Splits text into sentence-sized chunks (<= ~280 chars, short ones merged). */
  static _chunks(text) {
    const sentences = String(text).replace(/\s+/g, " ").trim()
      .split(/(?<=[.!?…])\s+(?=[A-Z0-9"'(])/);
    const out = [];
    for (let s of sentences) {
      while (s.length > 280) {                       // very long sentence: cut at a comma/space
        let cut = s.lastIndexOf(", ", 280);
        if (cut < 80) cut = s.lastIndexOf(" ", 280);
        if (cut < 80) cut = 280;
        out.push(s.slice(0, cut + 1).trim());
        s = s.slice(cut + 1).trim();
      }
      if (!s) continue;
      if (out.length && out[out.length - 1].length < 40) out[out.length - 1] += " " + s;
      else out.push(s);
    }
    return out.length ? out : [String(text)];
  }

  /**
   * Pieces for real lip-sync. The GPU renders one piece at a time, a bit
   * slower than real time (free 2xT4: ~1.15x), and each piece can only play
   * once it is rendered. Short pieces (~2-4 s) keep both the first wait and
   * the gaps between pieces small: while one piece plays, the next one is
   * already rendering. Cuts are made at sentence ends and commas first, and
   * never right after a word that needs the next one ("for this | interview").
   */
  static _lipsyncChunks(text, firstMax = 48, maxLen = 70) {
    const glue = /^(a|an|the|this|that|these|those|to|of|for|with|in|on|at|by|from|and|or|but|your|our|my|their|his|her|its|you|we|i|is|are|was|be|how|what|which|who|would|will|can|could|should)$/i;
    const bare = (w) => w.replace(/[^\w']/g, "");

    const split = (sentence, limit) => {                 // -> [piece, rest] or null
      const words = sentence.split(" ");
      if (sentence.length <= limit + 12 || words.length < 6) return null;
      let len = 0, comma = -1, plain = -1;
      for (let i = 0; i < words.length - 2; i++) {       // leave >= 2 words for the rest
        len += words[i].length + 1;
        if (len > limit) break;
        if (i >= 2 && /[,;:]$/.test(words[i])) comma = i;
        if (i >= 3 && !glue.test(bare(words[i]))) plain = i;
      }
      const cut = comma >= 0 ? comma : plain;
      if (cut < 0) return null;
      return [words.slice(0, cut + 1).join(" "), words.slice(cut + 1).join(" ")];
    };

    const sentences = String(text).replace(/\s+/g, " ").trim()
      .split(/(?<=[.!?\u2026])\s+(?=[A-Z0-9"'(])/).filter(Boolean);
    const out = [];
    for (let s of sentences) {
      for (;;) {
        const parts = split(s, out.length ? maxLen : firstMax);
        if (!parts) break;
        out.push(parts[0]);
        s = parts[1];
      }
      const prev = out[out.length - 1];
      const room = out.length === 1 ? firstMax : maxLen;   // the first piece stays short
      if (prev && prev.length + s.length + 1 <= room && (prev.length < 25 || s.length < 25)) {
        out[out.length - 1] = prev + " " + s;              // merge tiny pieces (not into the first)
      } else {
        out.push(s);
      }
    }
    return out.length ? out : [String(text)];
  }

  _ensureAudio() {
    if (!this.audioCtx) {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      this.audioCtx = new Ctx();
      this.gain = this.audioCtx.createGain();
      this.analyser = this.audioCtx.createAnalyser();
      this.analyser.fftSize = 1024;
      this.gain.connect(this.analyser);
      this.analyser.connect(this.audioCtx.destination);
    }
    return this.audioCtx;
  }

  async _fetchChunk(text) {
    const res = await fetch(this.serverUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "same-origin",
      body: JSON.stringify({ session_id: this._sessionId(), text }),
    });
    if (!res.ok) throw new Error(`TTS HTTP ${res.status}`);
    const data = await res.arrayBuffer();
    return await this._ensureAudio().decodeAudioData(data);
  }

  /**
   * opts.skipStart: speakStart was already emitted (lip-sync handed over mid-way)
   * opts.endText:   text to report in speakEnd (the full original utterance)
   */
  async _speakServer(text, token, opts = {}) {
    const ctx = this._ensureAudio();
    if (ctx.state === "suspended") {
      try { await ctx.resume(); } catch (e) { /* handled below */ }
      if (ctx.state === "suspended") throw new Error("audio is blocked until the user interacts with the page");
    }

    const chunks = TTSPlayer._chunks(text);
    let next = this._fetchChunk(chunks[0]);
    let started = false;

    for (let i = 0; i < chunks.length; i++) {
      let buffer;
      try {
        buffer = await next;
      } catch (err) {
        if (!started) throw err;                    // nothing played yet: fall back to browser voice
        console.warn("[TTSPlayer] a sentence failed to load, ending early:", err);
        break;
      }
      if (token !== this.token) return;             // stopped / superseded
      if (i + 1 < chunks.length) {
        next = this._fetchChunk(chunks[i + 1]);
        next.catch(() => {});                       // handled when awaited
      }

      if (!started) {
        started = true;
        this.lastMode = "server";
        console.log("[TTSPlayer] speaking (server voice):", text.slice(0, 60));
        if (!opts.skipStart) this._emit("speakStart", text);
        this._startLevelMeter(token);
      }
      this._emit("chunkStart", { text: chunks[i], durationMs: Math.round(buffer.duration * 1000) });
      if (this.speechPlayer && this.speechPlayer.playSpeech) {
        await this.speechPlayer.playSpeech(buffer, chunks[i]);   // 3D avatar: audio + lip-sync
      } else {
        await this._play(buffer, token);
      }
      if (token !== this.token) return;
      if (i + 1 < chunks.length) await new Promise((r) => setTimeout(r, 110)); // natural breath between sentences
    }

    if (token !== this.token) return;
    this._stopLevelMeter();
    console.log("[TTSPlayer] finished speaking");
    this._emit("speakEnd", opts.endText || text);
  }

  // ------------------------------------------------------------------
  // Real lip-sync (MuseTalk via /api/avatar/lipsync/)
  // ------------------------------------------------------------------

  _checkLipsync() {
    if (!this.useServerVoice || !window.fetch) return;
    fetch(this.lipsyncStatusUrl, { credentials: "same-origin" })
      .then((r) => (r.ok ? r.json() : { available: false }))
      .then((d) => {
        this.lipsyncAvailable = !!d.available;
        console.log("[TTSPlayer] real lip-sync (GPU):", this.lipsyncAvailable ? "available" : "not available");
      })
      .catch(() => { this.lipsyncAvailable = false; });
  }

  async _fetchClip(text) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), this.lipsyncTimeoutMs);
    const t0 = performance.now();
    try {
      const res = await fetch(this.lipsyncUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "same-origin",
        body: JSON.stringify({ session_id: this._sessionId(), text }),
        signal: ctrl.signal,
      });
      if (!res.ok) throw new Error(`lip-sync HTTP ${res.status}`);
      const data = await res.json();
      console.log(`[TTSPlayer] lip-sync clip ready in ${Math.round(performance.now() - t0)} ms (GPU ${data.infer_ms} ms)`);
      return data.clip_url;
    } finally {
      clearTimeout(timer);
    }
  }

  /** Renders one piece on the GPU and downloads the whole MP4 into memory,
   *  so playback never waits on the network (no stutter at the start). */
  async _fetchClipBlob(text) {
    const url = await this._fetchClip(text);
    const res = await fetch(url, { credentials: "same-origin" });
    if (!res.ok) throw new Error(`lip-sync clip HTTP ${res.status}`);
    return URL.createObjectURL(await res.blob());
  }

  /**
   * Real lip-sync, smooth mode: ALL pieces are rendered and downloaded first,
   * then played back to back with no gaps. While waiting, the avatar stays on
   * its live idle loop with a "Preparing" indicator (clipPlayer.setPreparing).
   * If anything fails before playback starts, the caller falls back to the
   * audio-only voice; if a clip fails during playback, the rest is finished
   * with the audio-only voice.
   */
  async _speakLipsync(text, token) {
    const cp = this.clipPlayer;
    const chunks = TTSPlayer._chunks(text);       // whole sentences: no gaps to hide any more
    const urls = [];
    const release = () => urls.forEach((u) => URL.revokeObjectURL(u));
    const t0 = performance.now();

    if (cp.setPreparing) cp.setPreparing(true);
    try {
      for (const c of chunks) {                   // the GPU renders one piece at a time anyway
        const u = await this._fetchClipBlob(c);
        urls.push(u);
        if (token !== this.token) { release(); return; }
      }
    } catch (err) {
      release();
      throw err;
    } finally {
      if (token === this.token && cp.setPreparing) cp.setPreparing(false);
    }
    console.log(`[TTSPlayer] all ${urls.length} lip-sync clip(s) ready in ${Math.round(performance.now() - t0)} ms`);

    this.lastMode = "lipsync";
    console.log("[TTSPlayer] speaking (real lip-sync):", text.slice(0, 60));
    this._emit("speakStart", text);
    try {
      for (let i = 0; i < urls.length; i++) {
        this._emit("chunkStart", { text: chunks[i], durationMs: 0 });
        try {
          await cp.playClip(urls[i]);
        } catch (err) {
          if (token !== this.token) return;
          console.warn("[TTSPlayer] lip-sync clip could not play, finishing with server voice:", err.message || err);
          if (cp.stopClip) cp.stopClip();
          await this._speakServer(chunks.slice(i).join(" "), token, { skipStart: true, endText: text });
          return;
        }
        if (token !== this.token) return;
      }
    } finally {
      setTimeout(release, 1000);                  // the last frame may still be on screen
    }

    if (token !== this.token) return;
    console.log("[TTSPlayer] finished speaking");
    this._emit("speakEnd", text);
  }

  _play(buffer, token) {
    return new Promise((resolve) => {
      const src = this.audioCtx.createBufferSource();
      src.buffer = buffer;
      src.connect(this.gain);
      src.onended = () => {
        if (this.currentSource === src) this.currentSource = null;
        resolve();
      };
      this.currentSource = src;
      if (token !== this.token) { resolve(); return; }
      src.start();
    });
  }

  _startLevelMeter(token) {
    this._stopLevelMeter();
    const buf = new Float32Array(this.analyser.fftSize);
    const tick = () => {
      if (token !== this.token) return;
      this.analyser.getFloatTimeDomainData(buf);
      let sum = 0;
      for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
      const rms = Math.sqrt(sum / buf.length);
      this._emit("level", Math.min(1, rms * 4));   // ~0.25 RMS speech -> ~1
      this.levelRAF = requestAnimationFrame(tick);
    };
    this.levelRAF = requestAnimationFrame(tick);
  }

  _stopLevelMeter() {
    if (this.levelRAF) cancelAnimationFrame(this.levelRAF);
    this.levelRAF = null;
  }

  _halt() {
    this._stopLevelMeter();
    this.asideCancelled = true;                       // a real turn always wins over an aside
    if (this.asideSource) {
      try { this.asideSource.stop(); } catch (e) { /* already stopped */ }
      this.asideSource = null;
    }
    if (this.speechPlayer && this.speechPlayer.stopSpeech) this.speechPlayer.stopSpeech();
    if (this.clipPlayer && this.clipPlayer.stopClip) this.clipPlayer.stopClip();
    if (this.currentSource) {
      try { this.currentSource.onended = null; this.currentSource.stop(); } catch (e) { /* already stopped */ }
      this.currentSource = null;
    }
    if (this.synth) this.synth.cancel();
  }

  // ------------------------------------------------------------------
  // Browser voice (fallback)
  // ------------------------------------------------------------------

  _loadPreferredVoice() {
    if (!this.synth) return;

    // Female voices only (the avatar is a woman), best-first:
    //   "...Online (Natural)" = Microsoft neural voices (Edge);
    //   "Google ..." = Chrome; the rest are OS voices.
    const preferredNames = [
      "Microsoft Ava Online (Natural) - English (United States)",
      "Microsoft Jenny Online (Natural) - English (United States)",
      "Microsoft Aria Online (Natural) - English (United States)",
      "Microsoft Neerja Online (Natural) - English (India)",
      "Microsoft Sonia Online (Natural) - English (United Kingdom)",
      "Google US English",
      "Google UK English Female",
      "Samantha",
      "Microsoft Zira - English (United States)",
    ];

    const maleHints = [
      "male", "david", "mark", "guy", "ryan", "prabhat", "ravi",
      "george", "daniel", "alex", "fred", "thomas", "james", "christopher", "andrew", "brian",
    ];
    const isMale = (v) => {
      const n = v.name.toLowerCase();
      if (n.includes("female")) return false;
      return maleHints.some((h) => n.includes(h));
    };

    const pickVoice = () => {
      const voices = this.synth.getVoices();
      if (voices.length === 0) return;

      let chosen = null;
      for (const name of preferredNames) {
        chosen = voices.find((v) => v.name === name || v.name.startsWith(name));
        if (chosen) break;
      }
      if (!chosen) {
        const english = voices.filter((v) => /^en[-_]/i.test(v.lang) && !isMale(v));
        chosen =
          english.find((v) => /^en[-_]US/i.test(v.lang)) ||
          english.find((v) => /^en[-_]GB/i.test(v.lang)) ||
          english.find((v) => /^en[-_]IN/i.test(v.lang)) ||
          english[0] ||
          voices[0];
      }
      this.voice = chosen;
      console.log("[TTSPlayer] fallback browser voice:", this.voice ? `${this.voice.name} (${this.voice.lang})` : "none");
    };

    pickVoice();
    this.synth.addEventListener("voiceschanged", pickVoice);
  }

  _speakBrowser(text) {
    if (!this.synth) {
      console.error("[TTSPlayer] cannot speak, no voice available.");
      this._emit("speakEnd", text); // never leave the interview waiting for mic-start
      return;
    }
    this.lastMode = "browser";
    this.synth.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    if (this.voice) utterance.voice = this.voice;
    utterance.rate = this.rate;
    utterance.pitch = this.pitch;

    utterance.addEventListener("start", () => {
      console.log("[TTSPlayer] speaking (browser voice):", text.slice(0, 60));
      this._emit("speakStart", text);
    });
    utterance.addEventListener("end", () => {
      console.log("[TTSPlayer] finished speaking");
      this._emit("speakEnd", text);
    });
    utterance.addEventListener("error", (event) => {
      console.error("[TTSPlayer] speech error:", event.error);
      this._emit("speakError", event.error);
      this._emit("speakEnd", text);
    });
    utterance.addEventListener("boundary", (event) => {
      if (event.name && event.name !== "word") return;
      this._emit("wordBoundary", { charIndex: event.charIndex, text });
    });

    this.synth.speak(utterance);
  }
}

window.TTSPlayer = TTSPlayer;
