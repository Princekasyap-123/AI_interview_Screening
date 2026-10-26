// frontend/static/js/interview_assist.js

/**
 * Makes the AI interviewer conversational while the candidate answers.
 * Everything runs in the browser on the candidate's microphone stream;
 * nothing is stored and no server changes are needed.
 *
 *  1. End-of-answer detection: once the candidate has spoken for a while
 *     and then stays silent for submitSilenceMs (3.5 s), emits "autoSubmit".
 *     A short countdown is shown first ("countdown" events) and any speech
 *     cancels it, so thinking pauses are not cut off.
 *  2. Backchannel: in a natural pause after a stretch of speech, emits
 *     "backchannel" (the page plays "Uh-huh." / "I see." + a nod),
 *     at most once every backchannelGapMs (9 s).
 *  3. No answer: after nudgeMs (5 s) with no speech emits "nudge" once
 *     ("Take your time…"); after noAnswerMs (11 s) emits "autoSubmit".
 *  4. Spoken requests (Chrome / Edge only, Web Speech API): "can you
 *     repeat the question?" -> "repeatRequest"; "that's all" / "I'm done"
 *     at the end of the answer -> "doneSaid"; abusive language (English,
 *     Hinglish, Hindi; same list as apps/interviews/services/abuse_filter.py)
 *     -> "abuse" at once, mid-answer. Other browsers simply skip this part;
 *     1–3 still work, and the server checks every answer for abuse anyway.
 *
 * 5. Barge-in (candidate talks WHILE the interviewer is speaking): after
 *    startBargeMonitor() the same microphone is watched during the
 *    interviewer's question. Sustained speech (>= bargeConfirmMs) louder than
 *    the interviewer's own echo emits "voiceStart" (tentative, start
 *    recording), then "bargeIn" (stop the interviewer's voice and listen), or
 *    "bargeCancel" if it was only a cough / click. The reply is classified
 *    by the server: a question gets an answer and the interviewer then
 *    continues the question.
 *
 * Voice activity is measured with a Web Audio AnalyserNode. The noise floor
 * is the 15th percentile of the last ~6 s of levels, so steady background
 * sound (fan, AC, laptop hum) becomes "silence" within seconds and never
 * blocks end-of-answer detection. If sound stays "loud" for 20 s without a
 * single pause, it is treated as noise too. Debug in the browser console:
 *   window.ASSIST_DEBUG = true   -> prints level / floor / state every second.
 *
 * API:
 *   const assist = new InterviewAssist();
 *   assist.init(mediaStream)          once, after the camera/mic starts
 *   assist.startAnswer()              when the candidate's answer starts
 *   assist.stopAnswer()               when the answer is submitted / ended
 *   assist.pause() / assist.resume()  around interviewer asides
 *   assist.on(event, fn)              autoSubmit, countdown, backchannel,
 *                                     nudge, repeatRequest, doneSaid
 *   assist.avatarAnalyser             AnalyserNode for the 3D avatar
 */

(function () {
  "use strict";

  const TICK_MS = 50;
  const LEVEL_WINDOW = 120;            // ~6 s of 50 ms level samples for the noise floor
  const FLOOR_PERCENTILE = 0.15;
  const SPEECH_ABOVE_FLOOR_DB = 10;    // speech must be this much louder than the floor
  const MIN_SPEECH_DB = -55;           // ...and at least this loud
  const NO_PAUSE_NOISE_MS = 20000;     // "speaking" 20 s without any pause = noise
  const PAUSE_GUARD_MS = 15000;        // an aside can never keep detection paused longer
  const DEFAULTS = {
    minSpeechMs: 500,             // speech needed before auto-submit can happen (short answers like "yes" count)
    submitSilenceMs: 2200,        // silence after a LONG answer that ends it (a half-finished sentence is continued by the server)
    shortSubmitSilenceMs: 1500,   // ...after a SHORT one (< shortAnswerMs of speech): usually done, not thinking
    shortAnswerMs: 2500,
    countdownFromMs: 1000,        // show the countdown after this much silence
    initialSpeechMs: 0,           // speech already heard when the answer window opens (barge-in)
    bargeStartDelayMs: 1200,      // no barge-in during the first moments of the interviewer's voice
    bargeConfirmMs: 450,          // sustained speech needed to cut the interviewer off
    bargeMarginDb: 12,            // must be this much louder than the interviewer's own echo
    bargeMinDb: -45,              // ...and at least this loud
    backchannelMinRunMs: 2500,    // speech stretch before a backchannel is allowed
    backchannelPauseMs: [600, 1100],
    backchannelGapMs: 9000,
    backchannelStartMs: 5000,     // not in the first seconds of an answer
    backchannelChance: 0.6,
    nudgeMs: 7000,                // no speech at all for 5 s -> interviewer says something
    noAnswerMs: 14000,            // still nothing (asides not counted) -> move on
    maxRepeats: 2,
  };

  // Mirror of apps/interviews/services/abuse_filter.py — keep in sync.
  // Patterns have no double letters because the text is collapsed first.
  const ABUSE_LATIN = new RegExp([
    "\\bfuck\\w*", "\\bmotherf\\w*", "\\bmofo\\b", "\\bbitch\\w*", "\\bbastard\\w*",
    "\\bashole\\w*", "\\barsehole\\w*", "\\bcunt\\w*", "\\bwhore\\w*", "\\bslut\\w*", "\\bdickhead\\w*",
    "\\bretard(?:ed|s)?\\b", "\\bfagot\\w*", "\\bsisterf\\w*",
    "\\bma?d[ae]?rchod\\w*", "\\bmadarchd\\w*", "\\bb[ae]?h?[ae]n ?chod\\w*", "\\bbh[ae]nchod\\w*",
    "\\bbhnchod\\w*", "\\bchutiy\\w*", "\\bchutia\\w*", "\\bchut\\b", "\\bbhosd\\w*", "\\bbhosr\\w*",
    "\\bgandu\\b", "\\bgand mar\\w*", "\\bgand\\b", "\\blauda\\b", "\\blavda\\b",
    "\\blawda\\b", "\\bloda\\b", "\\blodu\\b", "\\bharami\\w*",
    "\\bharamzad\\w*", "\\bharamkhor\\w*", "\\bkutiya\\w*", "\\bkamin[ae]\\b", "\\bjhant\\w*",
    "\\bteri ma ki\\b", "\\bteri behen ki\\b", "\\bbsdk\\b", "\\bbkl\\b", "\\bmkc\\b",
  ].join("|"));
  // Matched on the NOT-collapsed text so "Nigeria" / "Niger" never match the slur.
  const ABUSE_RAW = /\bnigg+(?:er|a|ah)s?\b|\bnigga\w*/g;
  const ABUSE_DEVANAGARI = [
    "मादरचोद", "मादरजात", "बहनचोद", "भेनचोद", "भैनचोद", "बहनचो", "चूतिया", "चुतिया", "चूतिये",
    "चुतिये", "चूत", "भोसड़", "भोसड", "भोसडी", "गांडू", "गाँडू", "गांड", "गाँड", "लौड़ा", "लौडा",
    "लवड़ा", "लंड", "रंडी", "हरामी", "हरामज़ादा", "हरामजादा", "हरामखोर", "कुतिया", "कमीना",
    "कमीने", "झांट", "झाँट", "तेरी माँ की", "तेरी मां की", "तेरी बहन की",
  ];
  // Chrome's recognizer masks English profanity as "f***" — that IS a swear word.
  const MASKED_PROFANITY = /\b[a-z]\*{2,}/i;
  const LEET = { "@": "a", "4": "a", "1": "i", "!": "i", "0": "o", "3": "e", "$": "s", "5": "s", "7": "t" };

  function findAbusiveTerms(text) {
    if (!text) return [];
    const found = [];
    const masked = text.match(MASKED_PROFANITY);
    if (masked) found.push(masked[0]);
    let t = text.normalize("NFKC").toLowerCase().replace(/[@4!10357$]/g, (c) => LEET[c] || c);
    t = t.replace(/[^a-z\s]/g, " ").replace(/([a-z])\1+/g, "$1").replace(/\s+/g, " ").trim();
    const re = new RegExp(ABUSE_LATIN.source, "g");
    let m;
    while ((m = re.exec(t))) found.push(m[0]);
    const raw = text.normalize("NFKC").toLowerCase().replace(/[@4!10357$]/g, (c) => LEET[c] || c).replace(/[^a-z\s]/g, " ");
    let r;
    ABUSE_RAW.lastIndex = 0;
    while ((r = ABUSE_RAW.exec(raw))) found.push(r[0]);
    const deva = text.normalize("NFC");
    for (const root of ABUSE_DEVANAGARI) if (deva.includes(root.normalize("NFC"))) found.push(root);
    return [...new Set(found)];
  }

  const REPEAT_RE = /\b(repeat( the| that| your| it)?( question)?|say (that|it) again|come again|pardon|one more time|(didn'?t|did not|couldn'?t|could not) (get|hear|catch|understand) (that|you|it|the question)|what was the question)\b/i;
  const DONE_RE = /\b(that'?s (all|it|my answer)|that is (all|my answer)|i'?m done|i am done|i have (finished|completed)( my answer)?)\W*$/i;

  class InterviewAssist {
    constructor(options = {}) {
      this.cfg = Object.assign({}, DEFAULTS, options);
      this.runCfg = this.cfg;           // per-answer settings (see startAnswer overrides)
      this.handlers = {};
      this.bargeTimer = null;
      this.ctx = null;
      this.analyser = null;
      this.avatarAnalyser = null;
      this.active = false;
      this.paused = false;
      this.timer = null;
      this.rec = null;
      this.recWanted = false;
    }

    on(evt, fn) { (this.handlers[evt] = this.handlers[evt] || []).push(fn); return this; }
    _emit(evt, ...args) {
      (this.handlers[evt] || []).forEach((fn) => {
        try { fn(...args); } catch (e) { console.error("[InterviewAssist] handler error:", e); }
      });
    }

    init(stream) {
      if (!stream || !stream.getAudioTracks || !stream.getAudioTracks().length) {
        console.warn("[InterviewAssist] no microphone track — assist disabled");
        return false;
      }
      const Ctx = window.AudioContext || window.webkitAudioContext;
      if (!Ctx) return false;
      this.ctx = new Ctx();
      const src = this.ctx.createMediaStreamSource(new MediaStream(stream.getAudioTracks()));
      this.analyser = this.ctx.createAnalyser();
      this.analyser.fftSize = 1024;
      src.connect(this.analyser);
      this.avatarAnalyser = this.ctx.createAnalyser();   // separate: the avatar changes its settings
      src.connect(this.avatarAnalyser);
      this.buf = new Float32Array(this.analyser.fftSize);
      const resume = () => { if (this.ctx.state === "suspended") this.ctx.resume().catch(() => {}); };
      document.addEventListener("click", resume);
      document.addEventListener("keydown", resume);
      console.log("[InterviewAssist] ready");
      return true;
    }

    /**
     * overrides: per-answer settings, e.g. after a barge-in
     *   { initialSpeechMs: 600, minSpeechMs: 400, submitSilenceMs: 1600, shortSubmitSilenceMs: 1600 }
     */
    startAnswer(overrides) {
      if (!this.ctx) return;
      if (this.ctx.state === "suspended") this.ctx.resume().catch(() => {});
      this.stopBargeMonitor();
      this.runCfg = Object.assign({}, this.cfg, overrides || {});
      const now = performance.now();
      this.active = true;
      this.paused = false;
      this.startedAt = now;
      this.speechMs = this.runCfg.initialSpeechMs || 0;
      this.runMs = 0;             // current/last continuous speech stretch
      this.voicedTicks = 0;
      this.lastVoiceAt = now;
      this.lastBackchannel = -Infinity;   // first one allowed after backchannelStartMs
      this.pauseHandled = false;
      this.nudged = false;
      this.countdown = null;
      this.repeats = 0;
      this.submitted = false;
      this.levels = this.levels || [];   // kept across answers: the floor is known from the start
      this.floorBoost = 0;
      this.lastQuietAt = now;
      this.lastDebugAt = 0;
      clearInterval(this.timer);
      this.timer = setInterval(() => this._tick(), TICK_MS);
      this._startRecognition();
    }

    stopAnswer() {
      this.active = false;
      clearInterval(this.timer);
      this.timer = null;
      this._setCountdown(null);
      this._stopRecognition();
    }

    /** Ignore the microphone while the interviewer speaks an aside. */
    pause() {
      if (!this.paused) this.pausedAt = performance.now();
      this.paused = true;
      this._setCountdown(null);
      clearTimeout(this.pauseGuard);
      this.pauseGuard = setTimeout(() => this.resume(), PAUSE_GUARD_MS);   // never stay paused
    }

    resume() {
      clearTimeout(this.pauseGuard);
      if (!this.active) return;
      const now = performance.now();
      if (this.paused && this.pausedAt) this.startedAt += now - this.pausedAt;   // asides don't count as "no answer" time
      this.paused = false;
      this.lastVoiceAt = now;      // silence counting starts again after the aside
      this.lastBackchannel = now;
      this.runMs = 0;
    }

    // ---------------- barge-in: candidate talks over the interviewer ----------------

    /** Call when the interviewer starts speaking a QUESTION (not the intro / closing). */
    startBargeMonitor() {
      if (!this.ctx || this.bargeTimer) return;
      if (this.ctx.state === "suspended") this.ctx.resume().catch(() => {});
      this.levels = this.levels || [];
      this.bargeStart = performance.now();
      this.bargeLevels = [];
      this.bargeVoiceMs = 0;
      this.bargeQuietMs = 0;
      this.bargeTentative = false;
      this.bargeTimer = setInterval(() => this._bargeTick(), TICK_MS);
    }

    /** Call when the interviewer stops speaking (or has been cut off). */
    stopBargeMonitor() {
      clearInterval(this.bargeTimer);
      this.bargeTimer = null;
      if (this.bargeTentative) { this.bargeTentative = false; this._emit("bargeCancel"); }
    }

    _bargeTick() {
      if (!this.bargeTimer) return;      // stopped (barge-in already confirmed, or answer window open)
      const now = performance.now();
      const db = this._levelDb();
      this.bargeLevels.push(db);
      if (this.bargeLevels.length > 60) this.bargeLevels.shift();     // ~3 s
      if (now - this.bargeStart < this.cfg.bargeStartDelayMs || this.bargeLevels.length < 20) return;

      // The interviewer's own voice leaking into the mic (speakers) sets the
      // "echo level" = median of the last ~3 s. The candidate's voice is much
      // closer to the mic, so it clears echo + margin; syllable peaks of the
      // interviewer don't last long enough to count (bargeConfirmMs).
      const sorted = this.bargeLevels.slice().sort((a, b) => a - b);
      const echo = sorted[Math.floor(sorted.length / 2)];
      const room = this._noiseFloor();
      const threshold = Math.max(room + this.cfg.bargeMarginDb, echo + this.cfg.bargeMarginDb, this.cfg.bargeMinDb);
      const loud = db > threshold;

      if (window.ASSIST_DEBUG && Math.round(now / 50) % 20 === 0) {
        console.log(`[InterviewAssist] barge: level ${db.toFixed(1)} dB, threshold ${threshold.toFixed(1)} dB, voice ${this.bargeVoiceMs} ms`);
      }

      if (loud) {
        this.bargeVoiceMs += TICK_MS;
        this.bargeQuietMs = 0;
        if (!this.bargeTentative && this.bargeVoiceMs >= 100) {
          this.bargeTentative = true;
          this._emit("voiceStart");
        }
        if (this.bargeVoiceMs >= this.cfg.bargeConfirmMs) {
          const heardMs = this.bargeVoiceMs;
          clearInterval(this.bargeTimer);
          this.bargeTimer = null;
          this.bargeTentative = false;
          this._emit("bargeIn", { heardMs });
        }
      } else {
        this.bargeQuietMs += TICK_MS;
        if (this.bargeQuietMs > 250) {            // a gap longer than 250 ms ends the burst
          this.bargeVoiceMs = 0;
          if (this.bargeTentative) { this.bargeTentative = false; this._emit("bargeCancel"); }
        }
      }
    }

    // ---------------- voice activity ----------------

    _levelDb() {
      this.analyser.getFloatTimeDomainData(this.buf);
      let s = 0;
      for (let i = 0; i < this.buf.length; i++) s += this.buf[i] * this.buf[i];
      return 10 * Math.log10(s / this.buf.length + 1e-12);
    }

    _tick() {
      if (!this.active || this.submitted) return;
      const now = performance.now();
      if (this.paused) { this.lastVoiceAt = now; return; }

      const db = this._levelDb();
      this.levels.push(db);
      if (this.levels.length > LEVEL_WINDOW) this.levels.shift();
      const floor = this._noiseFloor() + this.floorBoost;
      let loud = db > Math.max(floor + SPEECH_ABOVE_FLOOR_DB, MIN_SPEECH_DB);

      // Constant sound with no pause at all for 20 s is noise, not speech:
      // lift the floor above it for the rest of this answer.
      if (!loud) this.lastQuietAt = now;
      else if (now - this.lastQuietAt > NO_PAUSE_NOISE_MS) {
        this.floorBoost += db - floor;
        this.lastQuietAt = now;
        loud = false;
        console.warn("[InterviewAssist] continuous sound for 20 s — treating it as background noise");
      }

      if (window.ASSIST_DEBUG && now - this.lastDebugAt > 1000) {
        this.lastDebugAt = now;
        console.log(`[InterviewAssist] level ${db.toFixed(1)} dB, floor ${floor.toFixed(1)} dB, ` +
          `voice=${loud}, spoken ${Math.round(this.speechMs)} ms, silent ${Math.round(now - this.lastVoiceAt)} ms`);
      }

      this.voicedTicks = loud ? this.voicedTicks + 1 : 0;
      const speaking = this.voicedTicks >= 3;               // >= 150 ms of sound
      if (speaking) {
        this.speechMs += TICK_MS;
        // gaps between words (< the backchannel pause) don't break a speech stretch
        const gap = now - this.lastVoiceAt;
        this.runMs = gap < this.runCfg.backchannelPauseMs[0] ? this.runMs + gap : TICK_MS;
        this.lastVoiceAt = now;
        this.pauseHandled = false;
        this._setCountdown(null);
        return;
      }

      const silence = now - this.lastVoiceAt;
      const elapsed = now - this.startedAt;

      // 1. end of answer — a short utterance ("yes", a quick question) is
      // finished sooner than a long answer, where pauses are thinking time.
      if (this.speechMs >= this.runCfg.minSpeechMs) {
        const needed = this.speechMs < this.runCfg.shortAnswerMs
          ? this.runCfg.shortSubmitSilenceMs
          : this.runCfg.submitSilenceMs;
        if (silence >= needed) { this._submit("silence"); return; }
        if (silence >= this.runCfg.countdownFromMs) {
          this._setCountdown(Math.ceil((needed - silence) / 1000));
        }
      }

      // 2. backchannel in a natural pause
      if (!this.pauseHandled && silence >= this.runCfg.backchannelPauseMs[0] && silence <= this.runCfg.backchannelPauseMs[1]) {
        this.pauseHandled = true;      // one decision per pause
        if (this.runMs >= this.runCfg.backchannelMinRunMs && elapsed >= this.runCfg.backchannelStartMs
            && now - this.lastBackchannel >= this.runCfg.backchannelGapMs && Math.random() < this.runCfg.backchannelChance) {
          this.lastBackchannel = now;
          this._emit("backchannel");
        }
      }

      // 3. no answer at all
      if (this.speechMs < this.runCfg.minSpeechMs) {
        if (!this.nudged && elapsed >= this.runCfg.nudgeMs) { this.nudged = true; this._emit("nudge"); }
        else if (elapsed >= this.runCfg.noAnswerMs) this._submit("no_answer");
    }
    }

    _noiseFloor() {
      if (this.levels.length < 20) return -60;
      const sorted = this.levels.slice().sort((a, b) => a - b);
      return sorted[Math.floor(sorted.length * FLOOR_PERCENTILE)];
    }

    _setCountdown(sec) {
      if (sec === this.countdown) return;
      this.countdown = sec;
      this._emit("countdown", sec);
    }

    _submit(reason) {
      if (this.submitted) return;
      this.submitted = true;
      this._setCountdown(null);
      console.log("[InterviewAssist] auto-submit:", reason);
      this._emit("autoSubmit", reason);
    }

    // ---------------- spoken requests (Web Speech API) ----------------

    _startRecognition() {
      const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SR) return;
      this.recWanted = true;
      this.ignoreBefore = 0;
      if (this.rec) return;                      // still running from before
      const rec = new SR();
      rec.lang = "en-IN";                       // Indian English: better with Hinglish
      rec.continuous = true;
      rec.interimResults = true;
      rec.onresult = (e) => this._onResult(e);
      rec.onerror = (e) => {
        if (e.error === "not-allowed" || e.error === "service-not-allowed") this.recWanted = false;
      };
      rec.onend = () => {
        this.rec = null;
        if (this.recWanted && this.active) this._startRecognition();   // Chrome stops after a while
      };
      try { rec.start(); this.rec = rec; } catch (e) { this.rec = null; }
    }

    _stopRecognition() {
      this.recWanted = false;
      if (this.rec) { try { this.rec.abort(); } catch (e) { /* ignore */ } this.rec = null; }
    }

    _onResult(e) {
      if (!this.active || this.paused || this.submitted) return;
      let text = "";
      for (let i = Math.max(this.ignoreBefore, 0); i < e.results.length; i++) text += " " + e.results[i][0].transcript;
      text = text.trim();
      if (!text) return;

      // Abusive language ends the interview at once (server decides; see page).
      let all = "";
      for (let i = 0; i < e.results.length; i++) all += " " + e.results[i][0].transcript;
      const terms = findAbusiveTerms(all);
      if (terms.length) {
        this._submit("abuse");
        this._emit("abuse", { terms, excerpt: all.trim().slice(-300) });
        return;
      }

      const words = text.split(/\s+/).length;

      if (this.repeats < this.runCfg.maxRepeats && words <= 14 && REPEAT_RE.test(text)) {
        this.repeats++;
        this.ignoreBefore = e.results.length;     // don't match the same words again
        this._emit("repeatRequest", text);
        return;
      }
      const last = e.results[e.results.length - 1];
      if (last.isFinal && this.speechMs >= this.runCfg.minSpeechMs && DONE_RE.test(text)) {
        this._submit("said_done");
        this._emit("doneSaid", text);
      }
    }
  }

  InterviewAssist.findAbusiveTerms = findAbusiveTerms;   // also used for tests
  window.InterviewAssist = InterviewAssist;
})();
