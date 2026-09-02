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
    this.listeners = {
      speakStart: [],  // fn(text)
      speakEnd: [],    // fn(text)
      speakError: [],  // fn(error)
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

    const pickVoice = () => {
      const voices = this.synth.getVoices();
      if (voices.length === 0) return;

      // Prefer a natural-sounding English voice if available — exact
      // names vary by OS/browser (e.g. "Google US English", "Microsoft
      // Aria Online"), so this is a soft preference list, not a
      // guarantee. Falls back to whatever default voice the browser
      // provides if none of these match.
      const preferredNames = [
        "Google US English",
        "Microsoft Aria Online (Natural)",
        "Samantha",
      ];

      this.voice =
        voices.find((v) => preferredNames.includes(v.name)) ||
        voices.find((v) => v.lang === "en-US") ||
        voices[0];

      console.log("[TTSPlayer] selected voice:", this.voice ? this.voice.name : "none");
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
    utterance.rate = 1.0;
    utterance.pitch = 1.0;

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

    this.synth.speak(utterance);
  }

  stop() {
    if (this.synth) {
      this.synth.cancel();
    }
  }
}

window.TTSPlayer = TTSPlayer;