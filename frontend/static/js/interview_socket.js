// frontend/static/js/interview_socket.js

/**
 * Owns the WebSocket connection to InterviewConsumer (apps/interviews/consumers.py).
 * Responsible for: connecting, sending "join"/"answer_submitted", and
 * dispatching incoming "turn"/"info"/"error" messages to whatever UI
 * logic needs them (tts_player.js for speaking, interview_room.html's
 * own script for state transitions).
 *
 * This file does NOT touch the DOM directly — it exposes a small event-
 * style API so interview_room.html can wire up UI reactions without this
 * file needing to know about specific element IDs.
 *
 * Message shapes (must match apps/interviews/consumers.py exactly):
 *   OUT: {"type": "join"}
 *   OUT: {"type": "answer_submitted", "question_id": "...", "audio_base64": "...", "audio_format": "webm"}
 *   IN:  {"type": "turn", "action": "speak_intro"|"speak_question"|"speak_follow_up", "question_id": "...", "text": "..."}
 *   IN:  {"type": "turn", "action": "close_interview", "message": "..."}
 *   IN:  {"type": "info", "message": "..."}
 *   IN:  {"type": "error", "message": "..."}
 */

class InterviewSocket {
  constructor(sessionId) {
    this.sessionId = sessionId;
    this.socket = null;
    this.listeners = {
      turn: [],       // fn(turnPayload)
      info: [],       // fn(message)
      error: [],      // fn(message)
      open: [],       // fn()
      close: [],      // fn(closeEvent)
    };
  }

  // ------------------------------------------------------------------
  // Public event API
  // ------------------------------------------------------------------

  on(eventName, callback) {
    if (!this.listeners[eventName]) {
      throw new Error(`Unknown InterviewSocket event: ${eventName}`);
    }
    this.listeners[eventName].push(callback);
  }

  _emit(eventName, payload) {
    for (const cb of this.listeners[eventName]) {
      try {
        cb(payload);
      } catch (err) {
        console.error(`InterviewSocket listener for "${eventName}" threw:`, err);
      }
    }
  }

  // ------------------------------------------------------------------
  // Connection lifecycle
  // ------------------------------------------------------------------

  connect() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${protocol}//${window.location.host}/ws/interview/${this.sessionId}/`;

    this.socket = new WebSocket(url);

    this.socket.addEventListener("open", () => {
      console.log("[InterviewSocket] connected");
      this._emit("open");
      this._sendJoin();
    });

    this.socket.addEventListener("message", (event) => {
      let data;
      try {
        data = JSON.parse(event.data);
      } catch (err) {
        console.error("[InterviewSocket] failed to parse message:", event.data);
        return;
      }
      this._handleMessage(data);
    });

    this.socket.addEventListener("close", (event) => {
      console.log("[InterviewSocket] closed", event.code, event.reason);
      this._emit("close", event);
    });

    this.socket.addEventListener("error", (err) => {
      console.error("[InterviewSocket] socket error:", err);
    });
  }

  disconnect() {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      this.socket.close();
    }
  }

  // ------------------------------------------------------------------
  // Outgoing messages
  // ------------------------------------------------------------------

  _sendJoin() {
    this._send({ type: "join" });
  }

  /**
   * Called by stt_recorder.js once it has a base64-encoded audio blob
   * for the candidate's answer to the current question.
   */
  submitAnswer(questionId, audioBase64, audioFormat) {
    this._send({
      type: "answer_submitted",
      question_id: questionId,
      audio_base64: audioBase64,
      audio_format: audioFormat,
    });
  }

  _send(payload) {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      console.error("[InterviewSocket] cannot send, socket not open:", payload.type);
      return;
    }
    this.socket.send(JSON.stringify(payload));
  }

  // ------------------------------------------------------------------
  // Incoming message dispatch
  // ------------------------------------------------------------------

  _handleMessage(data) {
    switch (data.type) {
      case "turn":
        this._emit("turn", data);
        break;
      case "info":
        this._emit("info", data.message);
        break;
      case "error":
        this._emit("error", data.message);
        break;
      default:
        console.warn("[InterviewSocket] unrecognized message type:", data.type);
    }
  }
}

// Exposed globally for interview_room.html's inline script to use —
// kept as a plain global rather than an ES module export, matching the
// non-bundled <script> tag setup implied by the current static/js/
// folder structure (no build step configured yet).
window.InterviewSocket = InterviewSocket;