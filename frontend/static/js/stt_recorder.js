// frontend/static/js/stt_recorder.js (UPDATED — no longer calls getUserMedia itself)

/**
 * Captures candidate microphone audio via MediaRecorder, and hands the
 * resulting base64-encoded blob to InterviewSocket.submitAnswer() once
 * recording stops — matching the "answer_submitted" message shape
 * InterviewConsumer expects (audio_base64 + audio_format), per
 * interview_socket.js and apps/interviews/consumers.py.
 *
 * CHANGED: this class no longer calls getUserMedia() itself — it reuses
 * the audio track from WebRTCClient's existing stream, so the candidate
 * is only prompted for mic permission ONCE (via WebRTCClient.init()),
 * not twice. Call sequence is now:
 *
 *   const webrtc = new WebRTCClient();
 *   await webrtc.init(videoElement);              // requests cam+mic once
 *   const recorder = new STTRecorder();
 *   recorder.init(webrtc.getStream());             // reuses that stream, no new prompt
 *   recorder.startRecording();
 *   recorder.stopRecording(questionId, interviewSocket);
 *
 * This file does NOT run its own STT — transcription happens entirely
 * server-side via apps/ai_engine/stt_client.py (Groq Whisper). This
 * class only records, encodes, and sends.
 */

class STTRecorder {
  constructor() {
    this.sourceStream = null;
    this.audioOnlyStream = null;
    this.mediaRecorder = null;
    this.audioChunks = [];
    this.isRecording = false;
    this.mimeType = null;
  }

  // ------------------------------------------------------------------
  // Setup — call once, AFTER WebRTCClient.init() has already obtained
  // camera+mic permission. Takes the shared MediaStream rather than
  // requesting its own.
  // ------------------------------------------------------------------

  init(sharedStream) {
    if (!sharedStream || sharedStream.getAudioTracks().length === 0) {
      throw new Error(
        "STTRecorder.init() requires a MediaStream with at least one " +
        "audio track — pass webrtcClient.getStream()."
      );
    }

    this.sourceStream = sharedStream;

    // MediaRecorder is given an audio-only stream built from the shared
    // stream's existing audio track — this does NOT create a new
    // hardware capture, it just wraps the same live track, so no
    // additional permission prompt occurs and no second microphone
    // session opens.
    const audioTrack = sharedStream.getAudioTracks()[0];
    this.audioOnlyStream = new MediaStream([audioTrack]);

    this.mimeType = this._pickSupportedMimeType();
    if (!this.mimeType) {
      throw new Error(
        "This browser does not support any audio recording format " +
        "compatible with this interview. Please use an up-to-date " +
        "version of Chrome, Firefox, or Edge."
      );
    }

    console.log("[STTRecorder] initialized with mimeType:", this.mimeType);
  }

  _pickSupportedMimeType() {
    const candidates = [
      "audio/webm;codecs=opus",
      "audio/webm",
      "audio/ogg;codecs=opus",
      "audio/mp4",
    ];
    for (const type of candidates) {
      if (MediaRecorder.isTypeSupported(type)) {
        return type;
      }
    }
    return null;
  }

  // ------------------------------------------------------------------
  // Recording lifecycle
  // ------------------------------------------------------------------

  startRecording() {
    if (this.isRecording) {
      console.warn("[STTRecorder] startRecording called while already recording — ignoring.");
      return;
    }
    if (!this.audioOnlyStream) {
      throw new Error("STTRecorder.init(sharedStream) must be called before startRecording().");
    }

    this.audioChunks = [];
    this.mediaRecorder = new MediaRecorder(this.audioOnlyStream, { mimeType: this.mimeType });

    this.mediaRecorder.addEventListener("dataavailable", (event) => {
      if (event.data && event.data.size > 0) {
        this.audioChunks.push(event.data);
      }
    });

    this.mediaRecorder.start();
    this.isRecording = true;
    console.log("[STTRecorder] recording started");
  }

  stopRecording(questionId, interviewSocket) {
    return new Promise((resolve, reject) => {
      if (!this.isRecording || !this.mediaRecorder) {
        reject(new Error("stopRecording called but no recording is in progress."));
        return;
      }

      this.mediaRecorder.addEventListener(
        "stop",
        async () => {
          this.isRecording = false;
          console.log("[STTRecorder] recording stopped, chunks:", this.audioChunks.length);

          if (this.audioChunks.length === 0) {
            console.warn("[STTRecorder] no audio captured — submitting empty answer.");
          }

          try {
            const blob = new Blob(this.audioChunks, { type: this.mimeType });
            const base64Audio = await this._blobToBase64(blob);
            const format = this._extensionFromMimeType(this.mimeType);

            interviewSocket.submitAnswer(questionId, base64Audio, format);
            resolve();
          } catch (err) {
            console.error("[STTRecorder] failed to encode/submit audio:", err);
            reject(err);
          }
        },
        { once: true }
      );

      this.mediaRecorder.stop();
    });
  }

  /**
   * No longer stops any tracks here — the underlying audio track is
   * OWNED by WebRTCClient's stream, not this class. Calling
   * track.stop() here would kill the candidate's mic for the whole
   * call, not just recording. Release the mic exclusively via
   * WebRTCClient.release() when the interview ends.
   */
  release() {
    this.isRecording = false;
    this.mediaRecorder = null;
    console.log("[STTRecorder] recorder released (underlying mic stream untouched — release via WebRTCClient)");
  }

  // ------------------------------------------------------------------
  // Encoding helpers
  // ------------------------------------------------------------------

  _blobToBase64(blob) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64 = reader.result.split(",")[1];
        resolve(base64);
      };
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  }

  _extensionFromMimeType(mimeType) {
    if (mimeType.includes("webm")) return "webm";
    if (mimeType.includes("ogg")) return "ogg";
    if (mimeType.includes("mp4")) return "mp4";
    return "webm";
  }
}

window.STTRecorder = STTRecorder;