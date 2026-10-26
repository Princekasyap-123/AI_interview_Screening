// frontend/static/js/interview_socket.js

/**
 * Owns the WebSocket connection to InterviewConsumer (apps/interviews/consumers.py).
 * Responsible for: connecting (with the access token), sending "join"/
 * "ready_for_question"/"answer_submitted"/"end_interview", keeping the
 * connection alive (ping every 20 s) and RECONNECTING automatically after a
 * network drop (exponential back-off). On every (re)connect it sends "join";
 * the server answers a join on a running interview with the question that
 * is currently open, so the interview carries on where it stopped.
 *
 * This file does NOT touch the DOM directly — it exposes a small event-
 * style API so interview_room.html can wire up UI reactions without this
 * file needing to know about specific element IDs.
 *
 * Message shapes (must match apps/interviews/consumers.py exactly):
 *   OUT: {"type": "join"}
 *   OUT: {"type": "ready_for_question"}
 *   OUT: {"type": "answer_submitted", "question_id": "...", "audio_base64": "...", "audio_format": "webm", "interjected": bool}
 *   OUT: {"type": "end_interview"}
 *   OUT: {"type": "ping"}
 *   IN:  {"type": "turn", "action": "speak_intro"|"speak_question"|"speak_follow_up", "question_id": "...", "text": "..."}
 *   IN:  {"type": "turn", "action": "close_interview", "message": "..."}
 *   IN:  {"type": "ack", "question_id": "..."}       audio arrived, server is working on it
 *   IN:  {"type": "pong"}
 *   IN:  {"type": "info" | "error", "message": "..."}
 *
 * Events: turn, info, error, ack, open, close, reconnecting, reconnected, gaveUp
 */

class InterviewSocket {
  constructor(sessionId, accessToken = "") {
    this.sessionId = sessionId;
    this.accessToken = accessToken;
    this.socket = null;
    this.listeners = {
      turn: [],          // fn(turnPayload)
      info: [],          // fn(message)
      error: [],         // fn(message)
      ack: [],           // fn({question_id})
      open: [],          // fn()
      close: [],         // fn(closeEvent)
      reconnecting: [],  // fn({ attempt, delayMs })
      reconnected: [],   // fn()
      gaveUp: [],        // fn()
    };

    this.manualClose = false;
    this.attempt = 0;
    this.maxAttempts = 8;
    this.hasConnectedBefore = false;
    this.reconnectTimer = null;
    this.pingTimer = null;
    this.pongTimer = null;
    // Close codes that must NOT be retried: 4000 candidate ended, 4001 interview
    // terminated, 4403 bad token, 4404 unknown session.
    this.finalCodes = new Set([4000, 4001, 4403, 4404]);
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
    this.manualClose = false;
    clearTimeout(this.reconnectTimer);

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const token = encodeURIComponent(this.accessToken || "");
    const url = `${protocol}//${window.location.host}/ws/interview/${this.sessionId}/?token=${token}`;

    const socket = new WebSocket(url);
    this.socket = socket;

    socket.addEventListener("open", () => {
      if (this.socket !== socket) return;
      console.log("[InterviewSocket] connected");
      const wasReconnect = this.hasConnectedBefore;
      this.hasConnectedBefore = true;
      this.attempt = 0;
      this._startHeartbeat();
      this._emit("open");
      if (wasReconnect) this._emit("reconnected");
      this._sendJoin();
    });

    socket.addEventListener("message", (event) => {
      let data;
      try {
        data = JSON.parse(event.data);
      } catch (err) {
        console.error("[InterviewSocket] failed to parse message:", event.data);
        return;
      }
      this._handleMessage(data);
    });

    socket.addEventListener("close", (event) => {
      if (this.socket !== socket) return;
      console.log("[InterviewSocket] closed", event.code, event.reason);
      this._stopHeartbeat();
      this._emit("close", event);
      if (this.manualClose || this.finalCodes.has(event.code)) return;
      this._scheduleReconnect();
    });

    socket.addEventListener("error", (err) => {
      console.error("[InterviewSocket] socket error:", err);
    });
  }

  _scheduleReconnect() {
    if (this.attempt >= this.maxAttempts) {
      this._emit("gaveUp");
      return;
    }
    const delayMs = Math.min(8000, 500 * 2 ** this.attempt);
    this.attempt += 1;
    this._emit("reconnecting", { attempt: this.attempt, delayMs });
    clearTimeout(this.reconnectTimer);
    this.reconnectTimer = setTimeout(() => this.connect(), delayMs);
  }

  disconnect() {
    this.manualClose = true;
    clearTimeout(this.reconnectTimer);
    this._stopHeartbeat();
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      this.socket.close();
    }
  }

  /** Coming back online / tab visible again: reconnect now instead of waiting for the timer. */
  reconnectNow() {
    if (this.manualClose) return;
    if (this.socket && (this.socket.readyState === WebSocket.OPEN || this.socket.readyState === WebSocket.CONNECTING)) return;
    this.attempt = 0;
    this.connect();
  }

  _startHeartbeat() {
    this._stopHeartbeat();
    this.pingTimer = setInterval(() => {
      if (!this.socket || this.socket.readyState !== WebSocket.OPEN) return;
      this._send({ type: "ping" });
      clearTimeout(this.pongTimer);
      // No pong in 10 s = a half-dead connection (common on mobile data): drop it and reconnect.
      this.pongTimer = setTimeout(() => {
        console.warn("[InterviewSocket] no pong — reconnecting");
        try { this.socket.close(); } catch (e) { /* ignore */ }
      }, 10000);
    }, 20000);
  }

  _stopHeartbeat() {
    clearInterval(this.pingTimer);
    clearTimeout(this.pongTimer);
    this.pingTimer = this.pongTimer = null;
  }

  // ------------------------------------------------------------------
  // Outgoing messages
  // ------------------------------------------------------------------

  _sendJoin() {
    this._send({ type: "join" });
  }

  sendReadyForQuestion() {
    return this._send({ type: "ready_for_question" });
  }

  /**
   * Called by stt_recorder.js once it has a base64-encoded audio blob
   * for the candidate's answer to the current question. Returns false if the
   * socket is down (the page then asks the candidate to answer again once
   * the connection is back and the server re-asks the question).
   */
  submitAnswer(questionId, audioBase64, audioFormat, extra = {}) {
    return this._send({
      type: "answer_submitted",
      question_id: questionId,
      audio_base64: audioBase64,
      audio_format: audioFormat,
      ...extra,
    });
  }

  /**
   * Called when the candidate clicks "End interview". Server closes
   * the session (status -> TERMINATED, reason -> CANDIDATE_ENDED) and
   * responds with a normal close_interview turn, then closes the
   * socket — see InterviewConsumer._handle_end_interview.
   */
  sendEndInterview() {
    return this._send({ type: "end_interview" });
  }

  isOpen() {
    return !!this.socket && this.socket.readyState === WebSocket.OPEN;
  }

  _send(payload) {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      console.error("[InterviewSocket] cannot send, socket not open:", payload.type);
      return false;
    }
    this.socket.send(JSON.stringify(payload));
    return true;
  }

  // ------------------------------------------------------------------
  // Incoming message dispatch
  // ------------------------------------------------------------------

  _handleMessage(data) {
    switch (data.type) {
      case "turn":
        this._emit("turn", data);
        break;
      case "ack":
        this._emit("ack", data);
        break;
      case "pong":
        clearTimeout(this.pongTimer);
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

window.InterviewSocket = InterviewSocket;
