// frontend/static/js/avatar_player.js

/**
 * Video-based AI interviewer for interview_room.html.
 *
 * The interviewer is a real recorded person (royalty-free stock footage),
 * shown as two looping clips:
 *   interviewer_idle.mp4   she is quiet and attentive  (listening / pauses)
 *   interviewer_talk.mp4   she is talking              (while the AI speaks)
 * Both are 1280x720, muted, and stacked on top of each other; switching is
 * a short cross-fade, so it looks like one continuous camera shot. No
 * GPU, no server compute — the browser just plays two small videos.
 *
 * Sync with the voice:
 *   - Server voice (Kokoro via tts_player.js): the talking clip plays only
 *     while there is actual sound. During a pause between words/sentences
 *     (voice level near zero for PAUSE_MS) it fades to the idle clip and
 *     the talking clip is paused, then resumes when the voice resumes —
 *     so her mouth moves when the voice moves and stops when it stops.
 *   - Browser voice fallback (no level data): talking clip plays from
 *     speakStart to speakEnd.
 * Lips do not match individual words exactly (it is a recording), but
 * face, jaw, head and expression are fully real.
 *
 * Public API (same as the previous avatar, so interview_room.html works
 * unchanged apart from the data attributes):
 *   mount(stageEl, panelEl) -> Promise
 *   speakStart(text, rate) / speakEnd(text) / onBoundary(charIndex)
 *   onChunk(text, durationMs) / setVoiceLevel(level)
 *   setListening(on) / stop()
 *   playClip(url) -> Promise / stopClip() / setPreparing(on)   (real lip-sync, see below)
 *
 * Real lip-sync (MuseTalk GPU demo):
 *   tts_player.js can hand over a ready-made lip-synced MP4 (with voice)
 *   per sentence. It is played on a third <video> layer on top of the two
 *   loops (same camera framing, so the cross-fade is invisible). Between
 *   sentences the clip stays on its last frame — the GPU server continues
 *   each clip from where the previous one ended. speakEnd()/stop() fade
 *   back to the idle loop. If the next clip is not ready within GAP_IDLE_MS
 *   (the GPU is a bit slower than real time), the idle loop is shown during
 *   the wait so the face never looks frozen; the next clip fades back in.
 *
 * Stage element data attributes:
 *   data-avatar-idle       URL of interviewer_idle.mp4   (required, H.264)
 *   data-avatar-talk       URL of interviewer_talk.mp4   (required, H.264)
 *   data-avatar-idle-webm  URL of interviewer_idle.webm  (optional, VP9 — tried first)
 *   data-avatar-talk-webm  URL of interviewer_talk.webm  (optional, VP9 — tried first)
 *   data-avatar-poster     URL of interviewer_poster.jpg (optional, shown while loading)
 */

(function () {
  "use strict";

  const FADE_MS = 220;          // cross-fade between idle and talking
  const PAUSE_MS = 320;         // silence this long during speech -> idle
  const VOICE_ON = 0.06;        // level above this counts as "voice"
  const LOAD_TIMEOUT_MS = 20000;
  const CLIP_LOAD_TIMEOUT_MS = 15000;
  const GAP_IDLE_MS = 250;      // wait this long for the next clip before showing idle

  function waitForVideo(video) {
    return new Promise((resolve, reject) => {
      if (video.readyState >= 3) { resolve(); return; }
      const timer = setTimeout(() => reject(new Error("[AvatarPlayer] video load timed out: " + video.src)), LOAD_TIMEOUT_MS);
      video.addEventListener("canplay", () => { clearTimeout(timer); resolve(); }, { once: true });
      video.addEventListener("error", () => { clearTimeout(timer); reject(new Error("[AvatarPlayer] video failed to load: " + video.src)); }, { once: true });
      // <source> failures don't fire on the <video>: catch "no source playable".
      const sources = video.querySelectorAll("source");
      if (sources.length) {
        sources[sources.length - 1].addEventListener("error", () => {
          clearTimeout(timer); reject(new Error("[AvatarPlayer] no playable video source"));
        }, { once: true });
      }
    });
  }

  function makeVideo(mp4, webm, poster) {
    const v = document.createElement("video");
    v.className = "avatar-character";
    for (const [url, type] of [[webm, "video/webm"], [mp4, "video/mp4"]]) {
      if (!url) continue;
      const s = document.createElement("source");
      s.src = url;
      s.type = type;
      v.appendChild(s);
    }
    if (poster) v.poster = poster;
    v.muted = true;
    v.defaultMuted = true;
    v.loop = true;
    v.playsInline = true;
    v.autoplay = false;
    v.preload = "auto";
    v.setAttribute("muted", "");
    v.setAttribute("playsinline", "");
    v.setAttribute("aria-hidden", "true");
    v.style.transition = `opacity ${FADE_MS}ms ease`;
    return v;
  }

  class AvatarPlayer {
    constructor() {
      this.ready = false;
      this.panel = null;
      this.idle = null;
      this.talk = null;
      this.speaking = false;     // AI turn in progress (speakStart..speakEnd)
      this.talkingShown = false; // talking clip currently visible
      this.audioDriven = false;
      this.lastVoiceAt = 0;
      this.text = "";
      this.pauseTimer = null;
      this.clip = null;          // lip-sync clip layer
      this.clipActive = false;   // a lip-sync clip is (or was last) on screen
      this.clipDone = null;      // resolves the pending playClip() promise
      this.gapTimer = null;      // shows idle while waiting for the next clip
    }

    async mount(stageEl, panelEl) {
      if (!stageEl || !panelEl) throw new Error("[AvatarPlayer] stage or panel element not found");
      const ds = stageEl.dataset;
      if (!ds.avatarIdle || !ds.avatarTalk) {
        throw new Error("[AvatarPlayer] data-avatar-idle / data-avatar-talk missing on stage element");
      }

      const idle = makeVideo(ds.avatarIdle, ds.avatarIdleWebm, ds.avatarPoster);
      const talk = makeVideo(ds.avatarTalk, ds.avatarTalkWebm, ds.avatarPoster);
      talk.style.opacity = "0";

      stageEl.innerHTML = "";
      stageEl.appendChild(idle);
      stageEl.appendChild(talk);

      // Lip-sync clip layer: NOT muted (the clip carries the voice), no loop.
      const clip = document.createElement("video");
      clip.className = "avatar-character avatar-clip";
      clip.playsInline = true;
      clip.setAttribute("playsinline", "");
      clip.preload = "auto";
      clip.loop = false;
      clip.style.opacity = "0";
      clip.style.transition = `opacity ${FADE_MS}ms ease`;
      stageEl.appendChild(clip);

      const shade = document.createElement("div");
      shade.className = "avatar-shade";
      stageEl.appendChild(shade);
      idle.load();
      talk.load();

      await Promise.all([waitForVideo(idle), waitForVideo(talk)]);

      this.panel = panelEl;
      this.idle = idle;
      this.talk = talk;
      this.clip = clip;
      this._play(idle);

      // Some browsers pause muted background videos when the tab is hidden;
      // resume the right clip when it becomes visible again.
      document.addEventListener("visibilitychange", () => {
        if (!document.hidden && this.ready) this._play(this.talkingShown ? this.talk : this.idle);
      });

      panelEl.classList.add("avatar-ready");
      this.ready = true;
      console.log("[AvatarPlayer] mounted (video avatar)");
    }

    // ---------------- speech hooks ----------------

    speakStart(text) {
      if (!this.ready) return;
      this.text = String(text || "");
      this.speaking = true;
      this.audioDriven = false;
      this.setListening(false);
      this.panel.classList.add("avatar-talking");
      // Browser voice gives no levels: talk for the whole utterance.
      // Server voice switches to level-driven mode on the first level event.
      this._showTalking(true);
    }

    onChunk() {
      if (!this.ready || !this.speaking) return;
      this.audioDriven = true;
    }

    setVoiceLevel(level) {
      if (!this.ready || !this.speaking || this.clipActive) return;
      this.audioDriven = true;
      const now = performance.now();
      if (level > VOICE_ON) {
        this.lastVoiceAt = now;
        if (!this.talkingShown) this._showTalking(true);
        this._armPauseCheck();
      }
    }

    onBoundary() { /* not needed for the video avatar */ }

    speakEnd(text) {
      if (!this.ready) return;
      if (typeof text === "string" && this.text && text !== this.text) return; // stale end event
      this.speaking = false;
      this.audioDriven = false;
      clearTimeout(this.pauseTimer);
      this.panel.classList.remove("avatar-talking");
      this.stopClip();
      this._showTalking(false);
    }

    setListening(on) {
      if (!this.ready) return;
      this.panel.classList.toggle("avatar-listening", !!on);
    }

    stop() {
      if (!this.ready) return;
      this.speaking = false;
      clearTimeout(this.pauseTimer);
      this.setListening(false);
      this.panel.classList.remove("avatar-talking");
      this.stopClip();
      this._showTalking(false);
    }

    // ---------------- real lip-sync clips ----------------

    /**
     * Plays one lip-synced MP4 (video + voice). Resolves when the clip ends
     * (or is stopped); rejects if it cannot load/play, so the caller can
     * fall back to the audio-only voice.
     */
    playClip(url) {
      if (!this.ready || !this.clip) return Promise.reject(new Error("[AvatarPlayer] not ready"));
      this._finishClip();                       // settle any previous promise
      // gapTimer keeps running: if this clip takes long to load, idle shows meanwhile
      const clip = this.clip;
      return new Promise((resolve, reject) => {
        let settled = false;
        const timer = setTimeout(() => fail(new Error("[AvatarPlayer] lip-sync clip load timed out")), CLIP_LOAD_TIMEOUT_MS);
        const cleanup = () => {
          clearTimeout(timer);
          clip.removeEventListener("ended", onEnded);
          clip.removeEventListener("error", onError);
          clip.removeEventListener("canplay", onCanPlay);
        };
        const done = () => { if (settled) return; settled = true; cleanup(); this.clipDone = null; resolve(); };
        const fail = (err) => { if (settled) return; settled = true; cleanup(); this.clipDone = null; reject(err); };
        const onEnded = () => {
          done();                               // stays on its last frame ...
          clearTimeout(this.gapTimer);
          this.gapTimer = setTimeout(() => {    // ... unless the next clip is late
            if (this.clipActive && clip.paused) {   // next clip not playing yet (late or loading)
              clip.style.opacity = "0";
              this._play(this.idle);
            }
          }, GAP_IDLE_MS);
        };
        const onError = () => fail(new Error("[AvatarPlayer] lip-sync clip failed to load"));
        const onCanPlay = () => {
          const p = clip.play();
          const show = () => {
            this.clipActive = true;
            clearTimeout(this.pauseTimer);
            clip.style.opacity = "1";
            // the talking loop is not needed under a real lip-sync clip
            this.talkingShown = false;
            this.talk.style.opacity = "0";
            setTimeout(() => { if (this.clipActive) this.talk.pause(); }, FADE_MS);
          };
          if (p && p.then) p.then(show, (err) => fail(err));
          else show();
        };
        this.clipDone = done;
        clip.addEventListener("ended", onEnded);
        clip.addEventListener("error", onError);
        clip.addEventListener("canplay", onCanPlay, { once: true });
        clip.src = url;
        clip.load();
      });
    }

    /** "Preparing" state while the lip-sync clips render: live idle loop + badge. */
    setPreparing(on) {
      if (!this.ready) return;
      this.panel.classList.toggle("avatar-preparing", !!on);
      if (on) this.setListening(false);
    }

    /** Hides the clip layer (fade) and settles a pending playClip(). */
    stopClip() {
      if (!this.clip) return;
      if (this.panel) this.panel.classList.remove("avatar-preparing");
      clearTimeout(this.gapTimer);
      this._finishClip();
      if (!this.clipActive) { try { this.clip.pause(); } catch (e) { /* ignore */ } return; }
      this.clipActive = false;
      const clip = this.clip;
      try { clip.pause(); } catch (e) { /* ignore */ }
      clip.style.opacity = "0";
      this._play(this.idle);
    }

    _finishClip() {
      if (this.clipDone) { const d = this.clipDone; this.clipDone = null; d(); }
    }

    // ---------------- internals ----------------

    _armPauseCheck() {
      clearTimeout(this.pauseTimer);
      this.pauseTimer = setTimeout(() => {
        if (this.speaking && this.audioDriven && performance.now() - this.lastVoiceAt >= PAUSE_MS) {
          this._showTalking(false);   // natural pause: close the mouth
        }
      }, PAUSE_MS + 20);
    }

    _showTalking(on) {
      if (on === this.talkingShown) return;
      this.talkingShown = on;
      if (on) {
        this._play(this.talk);
        this.talk.style.opacity = "1";
      } else {
        this.talk.style.opacity = "0";
        setTimeout(() => { if (!this.talkingShown) this.talk.pause(); }, FADE_MS);
        this._play(this.idle);
      }
    }

    _play(video) {
      const p = video.play();
      if (p && p.catch) p.catch(() => { /* muted autoplay is allowed; ignore rare interruptions */ });
    }
  }

  window.AvatarPlayer = AvatarPlayer;
})();