// frontend/static/js/tts_player.js

/**
 * Speaks AI interviewer text via the browser's native SpeechSynthesis
 * API — the no-GPU-phase TTS approach, matching apps/ai_engine/tts_client.py's
 * documented seam (which returns {"mode": "browser_tts"} and expects
 * the frontend to speak the text itself, since no backend audio is
 * generated yet).
 *
 * Consumes "turn" events from InterviewSocket directly — wire it via:
 *   const ttsPlayer = new TTSPlayer();
 *   interviewSocket.on("turn", (turn) => ttsPlayer.handleTurn(turn));
 *
 * Exposes onStart/onEnd hooks so interview_room.html can drive UI state
 * (the pulsing "AI is speaking" ring on the avatar panel, per the
 * prototyped interview_room.html) and so the orchestration script knows
 * exactly when speaking ends to start the candidate's mic.
 */

class TTSPlayer {
  constructor() {
    this.synth = window.speechSynthesis;
    this.voice = null;

    // Speaking style. A slightly slower rate sounds more composed and
    // professional, and gives the avatar's lip-sync room to read well.
    // avatar_player.js receives this rate in speakStart(), so if you
    // change it here the mouth pace follows automatically.
    this.rate = 0.9;
    this.pitch = 1.0;
    this.listeners = {
      speakStart: [],    // fn(text)
      speakEnd: [],      // fn(text)
      speakError: [],    // fn(error)
      wordBoundary: [],  // fn({ charIndex, text }) — avatar lip-sync resync
    };

    if (!this.synth) {
      console.error(
        "[TTSPlayer] speechSynthesis not supported in this browser — " +
        "the AI interviewer will not be audible."
      );
    }

    this._loadPreferredVoice();
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
  // Voice selection
  // ------------------------------------------------------------------

  _loadPreferredVoice() {
    if (!this.synth) return;

    // The avatar is a woman, so only female voices are considered.
    // Ordered best-first. Names differ per browser/OS:
    //   - "...Online (Natural)" voices are Microsoft neural voices, available
    //     in Microsoft Edge (most natural and professional sounding).
    //   - "Google ..." voices come with Chrome.
    //   - The rest are OS-installed voices (Windows / macOS).
    // Neerja / Heera are Indian-English voices, a natural fit for
    // candidates in India.
    const preferredNames = [
      "Microsoft Neerja Online (Natural) - English (India)",
      "Microsoft Aria Online (Natural) - English (United States)",
      "Microsoft Jenny Online (Natural) - English (United States)",
      "Microsoft Sonia Online (Natural) - English (United Kingdom)",
      "Google UK English Female",
      "Google US English",
      "Microsoft Heera - English (India)",
      "Samantha",
      "Microsoft Zira - English (United States)",
    ];

    // Voices that are known to be male — never used for this avatar.
    const maleHints = [
      "male", "david", "mark", "guy", "ryan", "prabhat", "ravi",
      "george", "daniel", "alex", "fred", "thomas", "james", "christopher",
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
        // startsWith, because some browsers append extra text to the name.
        chosen = voices.find((v) => v.name === name || v.name.startsWith(name));
        if (chosen) break;
      }

      if (!chosen) {
        const english = voices.filter((v) => /^en[-_]/i.test(v.lang) && !isMale(v));
        chosen =
          english.find((v) => /^en[-_]IN/i.test(v.lang)) ||
          english.find((v) => /^en[-_]GB/i.test(v.lang)) ||
          english.find((v) => /^en[-_]US/i.test(v.lang)) ||
          english[0] ||
          voices[0];
      }

      this.voice = chosen;
      console.log(
        "[TTSPlayer] selected voice:",
        this.voice ? `${this.voice.name} (${this.voice.lang})` : "none"
      );
    };

    // Voices load asynchronously in some browsers (notably Chrome) —
    // "voiceschanged" fires once they're actually available.
    pickVoice();
    this.synth.addEventListener("voiceschanged", pickVoice);
  }

  // ------------------------------------------------------------------
  // Main entry point — called from interviewSocket.on("turn", ...)
  // ------------------------------------------------------------------

  handleTurn(turn) {
    const speakableActions = ["speak_intro", "speak_question", "speak_follow_up"];
    if (speakableActions.includes(turn.action) && turn.text) {
      this.speak(turn.text);
    }
    // "close_interview" turns are NOT spoken via TTS by default — the
    // termination/completion message is shown as text in the UI (per
    // interview_room.html's warning/terminated overlay patterns) rather
    // than spoken, since by that point the interview is over and
    // speaking adds little. If you want it spoken too, call
    // ttsPlayer.speak(turn.message) explicitly from the close_interview
    // handler in interview_room.html's script.
  }

  // ------------------------------------------------------------------
  // Core speak function
  // ------------------------------------------------------------------

  speak(text) {
    if (!this.synth) {
      console.error("[TTSPlayer] cannot speak, speechSynthesis unavailable.");
      this._emit("speakEnd", text); // still fire speakEnd so the
                                     // orchestration script isn't stuck
                                     // waiting forever for mic-start
      return;
    }

    // Cancel any in-progress utterance before starting a new one —
    // prevents overlapping speech if turns arrive faster than expected.
    this.synth.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    if (this.voice) {
      utterance.voice = this.voice;
    }
    utterance.rate = this.rate;
    utterance.pitch = this.pitch;

    utterance.addEventListener("start", () => {
      console.log("[TTSPlayer] speaking:", text.slice(0, 60));
      this._emit("speakStart", text);
    });

    utterance.addEventListener("end", () => {
      console.log("[TTSPlayer] finished speaking");
      this._emit("speakEnd", text);
    });

    utterance.addEventListener("error", (event) => {
      console.error("[TTSPlayer] speech error:", event.error);
      this._emit("speakError", event.error);
      // Still emit speakEnd so downstream logic (e.g. "start recording
      // after AI finishes speaking") doesn't hang forever if TTS fails
      // partway through.
      this._emit("speakEnd", text);
    });

    // Word-level timing for the avatar's lip-sync. Several Chrome
    // network voices never fire this, so avatar_player.js treats it as
    // an optional resync on top of its own text-paced estimate.
    utterance.addEventListener("boundary", (event) => {
      if (event.name && event.name !== "word") return;
      this._emit("wordBoundary", { charIndex: event.charIndex, text });
    });

    this.synth.speak(utterance);
  }

  stop() {
    if (this.synth) {
      this.synth.cancel();
    }
  }
}

window.TTSPlayer = TTSPlayer;