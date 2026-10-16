// frontend/static/js/interview_assist.js

/**
 * Makes the AI interviewer conversational while the candidate answers.
 * Everything runs in the browser on the candidate's microphone stream;
 * nothing is stored and no server changes are needed.
 *
 *  1. End-of-answer detection: once the candidate has spoken for a while
 *     and then stays silent for submitSilenceMs (4 s), emits "autoSubmit".
 *     A short countdown is shown first ("countdown" events) and any speech
 *     cancels it, so thinking pauses are not cut off.
 *  2. Backchannel: in a natural pause after a stretch of speech, emits
 *     "backchannel" (the page plays "Uh-huh." / "I see." + a nod),
 *     at most once every backchannelGapMs (9 s).
 *  3. No answer: after nudgeMs (15 s) with no speech emits "nudge" once
 *     ("Take your time…"); after noAnswerMs (40 s) emits "autoSubmit".
 *  4. Spoken requests (Chrome / Edge only, Web Speech API): "can you
 *     repeat the question?" -> "repeatRequest"; "that's all" / "I'm done"
 *     at the end of the answer -> "doneSaid". Other browsers simply skip
 *     this part; 1–3 still work.
 *
 * Voice activity is measured with a Web Audio AnalyserNode and an adaptive
 * noise floor, so a fan or room noise is not mistaken for speech.
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
  const DEFAULTS = {
    minSpeechMs: 1500,            // speech needed before auto-submit can happen
    submitSilenceMs: 4000,        // silence after speech that ends the answer
    countdownFromMs: 1800,        // show the countdown after this much silence
    backchannelMinRunMs: 2500,    // speech stretch before a backchannel is allowed
    backchannelPauseMs: [600, 1100],
    backchannelGapMs: 9000,
    backchannelStartMs: 5000,     // not in the first seconds of an answer
    backchannelChance: 0.6,
    nudgeMs: 15000,
    noAnswerMs: 40000,
    maxRepeats: 2,
  };

  const REPEAT_RE = /\b(repeat( the| that| your| it)?( question)?|say (that|it) again|come again|pardon|one more time|(didn'?t|did not|couldn'?t|could not) (get|hear|catch|understand) (that|you|it|the question)|what was the question)\b/i;
  const DONE_RE = /\b(that'?s (all|it|my answer)|that is (all|my answer)|i'?m done|i am done|i have (finished|completed)( my answer)?)\W*$/i;

  class InterviewAssist {
    constructor(options = {}) {
      this.cfg = Object.assign({}, DEFAULTS, options);
      this.handlers = {};
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

    startAnswer() {
      if (!this.ctx) return;
      if (this.ctx.state === "suspended") this.ctx.resume().catch(() => {});
      const now = performance.now();
      this.active = true;
      this.paused = false;
      this.startedAt = now;
      this.speechMs = 0;
      this.runMs = 0;             // current/last continuous speech stretch
      this.voicedTicks = 0;
      this.lastVoiceAt = now;
      this.lastBackchannel = -Infinity;   // first one allowed after backchannelStartMs
      this.pauseHandled = false;
      this.nudged = false;
      this.countdown = null;
      this.repeats = 0;
      this.submitted = false;
      this.floor = -60;
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
    }

    resume() {
      if (!this.active) return;
      const now = performance.now();
      if (this.paused && this.pausedAt) this.startedAt += now - this.pausedAt;   // asides don't count as "no answer" time
      this.paused = false;
      this.lastVoiceAt = now;      // silence counting starts again after the aside
      this.lastBackchannel = now;
      this.runMs = 0;
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
      const loud = db > Math.max(this.floor + 12, -52);
      // noise floor: drops fast to quiet levels, rises slowly during silence
      if (db < this.floor) this.floor = db;
      else if (!loud) this.floor += 0.04 * (db - this.floor);

      this.voicedTicks = loud ? this.voicedTicks + 1 : 0;
      const speaking = this.voicedTicks >= 3;               // >= 150 ms of sound
      if (speaking) {
        this.speechMs += TICK_MS;
        // gaps between words (< the backchannel pause) don't break a speech stretch
        const gap = now - this.lastVoiceAt;
        this.runMs = gap < this.cfg.backchannelPauseMs[0] ? this.runMs + gap : TICK_MS;
        this.lastVoiceAt = now;
        this.pauseHandled = false;
        this._setCountdown(null);
        return;
      }

      const silence = now - this.lastVoiceAt;
      const elapsed = now - this.startedAt;

      // 1. end of answer
      if (this.speechMs >= this.cfg.minSpeechMs) {
        if (silence >= this.cfg.submitSilenceMs) { this._submit("silence"); return; }
        if (silence >= this.cfg.countdownFromMs) {
          this._setCountdown(Math.ceil((this.cfg.submitSilenceMs - silence) / 1000));
        }
      }

      // 2. backchannel in a natural pause
      if (!this.pauseHandled && silence >= this.cfg.backchannelPauseMs[0] && silence <= this.cfg.backchannelPauseMs[1]) {
        this.pauseHandled = true;      // one decision per pause
        if (this.runMs >= this.cfg.backchannelMinRunMs && elapsed >= this.cfg.backchannelStartMs
            && now - this.lastBackchannel >= this.cfg.backchannelGapMs && Math.random() < this.cfg.backchannelChance) {
          this.lastBackchannel = now;
          this._emit("backchannel");
        }
      }

      // 3. no answer at all
      if (this.speechMs < 300) {
        if (!this.nudged && elapsed >= this.cfg.nudgeMs) { this.nudged = true; this._emit("nudge"); }
        else if (elapsed >= this.cfg.noAnswerMs) this._submit("no_answer");
      }
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
      rec.lang = "en-US";
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
      const words = text.split(/\s+/).length;

      if (this.repeats < this.cfg.maxRepeats && words <= 14 && REPEAT_RE.test(text)) {
        this.repeats++;
        this.ignoreBefore = e.results.length;     // don't match the same words again
        this._emit("repeatRequest", text);
        return;
      }
      const last = e.results[e.results.length - 1];
      if (last.isFinal && this.speechMs >= this.cfg.minSpeechMs && DONE_RE.test(text)) {
        this._submit("said_done");
        this._emit("doneSaid", text);
      }
    }
  }

  window.InterviewAssist = InterviewAssist;
})();
