// frontend/static/js/proctoring_client.js

/**
 * Owns the second WebSocket connection (to ProctoringConsumer,
 * apps/proctoring/consumers.py) and all client-side detection: tab
 * switches, window blur, fullscreen exits, plus face-mesh-based
 * no-face/multiple-faces/gaze-away detection via MediaPipe.
 *
 * This file sends RAW signals only — it does NOT decide warn vs.
 * terminate itself. That decision is entirely server-side
 * (risk_scorer.py), per the security fix made earlier in this build.
 * This file only reports "this happened, for this long" and reacts to
 * whatever the server sends back ("warning" or "terminated").
 *
 * Message shapes (must match apps/proctoring/consumers.py exactly):
 *   OUT: {"type": "flag", "flag_type": "tab_switch", "metadata": {"duration_ms": 2100}}
 *   IN:  {"type": "warning", "message": "..."}
 *   IN:  {"type": "terminated", "message": "..."}
 *   IN:  {"type": "error", "message": "..."}
 *
 * MediaPipe Face Mesh is loaded via CDN script tags in interview_room.html
 * (not bundled here) — this file assumes `FaceMesh` and `Camera` globals
 * exist on window, per the standard MediaPipe JS distribution.
 */

class ProctoringClient {
  constructor(sessionId) {
    this.sessionId = sessionId;
    this.socket = null;

    this.listeners = {
      warning: [],     // fn(message)
      terminated: [],  // fn(message)
      open: [],
    };

    // Tracks in-flight duration timers so we only report a flag once
    // the underlying condition ends (or crosses a "still ongoing"
    // check-in interval) — durations matter for server-side countability
    // (see flag_processor.py's DURATION_GATED_TYPES).
    this._activeTimers = {
      tabSwitch: null,
      windowBlur: null,
      noFace: null,
      gazeAway: null,
    };

    this._faceMesh = null;
    this._camera = null;
    this._videoElement = null;
    this._lastFaceCheckState = { faceCount: 1, gazeAway: false };
  }

  // ------------------------------------------------------------------
  // Public event API
  // ------------------------------------------------------------------

  on(eventName, callback) {
    if (!this.listeners[eventName]) {
      throw new Error(`Unknown ProctoringClient event: ${eventName}`);
    }
    this.listeners[eventName].push(callback);
  }

  _emit(eventName, payload) {
    for (const cb of this.listeners[eventName]) {
      try {
        cb(payload);
      } catch (err) {
        console.error(`ProctoringClient listener for "${eventName}" threw:`, err);
      }
    }
  }

  // ------------------------------------------------------------------
  // Connection lifecycle
  // ------------------------------------------------------------------

  connect() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${protocol}//${window.location.host}/ws/proctoring/${this.sessionId}/`;

    this.socket = new WebSocket(url);

    this.socket.addEventListener("open", () => {
      console.log("[ProctoringClient] connected");
      this._emit("open");
    });

    this.socket.addEventListener("message", (event) => {
      let data;
      try {
        data = JSON.parse(event.data);
      } catch (err) {
        console.error("[ProctoringClient] failed to parse message:", event.data);
        return;
      }
      this._handleMessage(data);
    });

    this.socket.addEventListener("close", (event) => {
      console.log("[ProctoringClient] closed", event.code, event.reason);
    });

    this.socket.addEventListener("error", (err) => {
      console.error("[ProctoringClient] socket error:", err);
    });
  }

  disconnect() {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      this.socket.close();
    }
    this._stopFaceMesh();
  }

  _handleMessage(data) {
    switch (data.type) {
      case "warning":
        this._emit("warning", data.message);
        break;
      case "terminated":
        this._emit("terminated", data.message);
        this.disconnect();
        break;
      case "error":
        console.error("[ProctoringClient] server error:", data.message);
        break;
      default:
        console.warn("[ProctoringClient] unrecognized message type:", data.type);
    }
  }

  // ------------------------------------------------------------------
  // Outgoing flag reporting
  // ------------------------------------------------------------------

  _sendFlag(flagType, metadata = {}) {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      console.warn("[ProctoringClient] cannot send flag, socket not open:", flagType);
      return;
    }
    this.socket.send(JSON.stringify({ type: "flag", flag_type: flagType, metadata }));
  }

  // ------------------------------------------------------------------
  // Tab switch / window blur detection
  // ------------------------------------------------------------------

  startBrowserEventDetection() {
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) {
        this._startTimer("tabSwitch");
      } else {
        this._endTimer("tabSwitch", "tab_switch");
      }
    });

    window.addEventListener("blur", () => {
      this._startTimer("windowBlur");
    });

    window.addEventListener("focus", () => {
      this._endTimer("windowBlur", "window_blur");
    });

    document.addEventListener("fullscreenchange", () => {
      if (!document.fullscreenElement) {
        // Fullscreen exit is always-countable server-side (no duration
        // gate), so report immediately rather than timing it.
        this._sendFlag("fullscreen_exit", {});
      }
    });

    document.addEventListener("copy", () => {
      this._sendFlag("copy_paste", { action: "copy" });
    });
    document.addEventListener("paste", () => {
      this._sendFlag("copy_paste", { action: "paste" });
    });

    // Rough devtools-open heuristic: large delta between outer and inner
    // window dimensions. Not bulletproof (false positives on some
    // browser zoom levels / extensions), but a reasonable free signal.
    this._devtoolsCheckInterval = setInterval(() => {
      const threshold = 160;
      const widthDelta = window.outerWidth - window.innerWidth;
      const heightDelta = window.outerHeight - window.innerHeight;
      if (widthDelta > threshold || heightDelta > threshold) {
        this._sendFlag("devtools_opened", { widthDelta, heightDelta });
      }
    }, 3000);

    console.log("[ProctoringClient] browser event detection active");
  }

  _startTimer(key) {
    if (this._activeTimers[key] !== null) return; // already timing
    this._activeTimers[key] = Date.now();
  }

  _endTimer(key, flagType) {
    const startedAt = this._activeTimers[key];
    if (startedAt === null) return;
    const durationMs = Date.now() - startedAt;
    this._activeTimers[key] = null;
    this._sendFlag(flagType, { duration_ms: durationMs });
  }

  // ------------------------------------------------------------------
  // Fullscreen enforcement helper (called from interview_room.html on
  // interview start — not auto-triggered here since entering fullscreen
  // requires a user gesture in most browsers)
  // ------------------------------------------------------------------

  static async requestFullscreen() {
    try {
      await document.documentElement.requestFullscreen();
    } catch (err) {
      console.error("[ProctoringClient] fullscreen request failed:", err);
    }
  }

  // ------------------------------------------------------------------
  // MediaPipe Face Mesh — no-face / multiple-faces / gaze-away detection
  // ------------------------------------------------------------------

  /**
   * videoElement: the <video> element already showing the candidate's
   * own webcam feed (webrtc_client.js owns creating this stream — this
   * function just attaches MediaPipe's Camera helper to read frames
   * from the same element, it does not request its own separate
   * getUserMedia stream).
   */
  startFaceMesh(videoElement) {
    if (typeof FaceMesh === "undefined" || typeof Camera === "undefined") {
      console.error(
        "[ProctoringClient] MediaPipe FaceMesh/Camera not found on window — " +
        "make sure the CDN scripts are included in interview_room.html."
      );
      return;
    }

    this._videoElement = videoElement;

    this._faceMesh = new FaceMesh({
      locateFile: (file) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
    });

    this._faceMesh.setOptions({
      maxNumFaces: 3, // detect up to 3 so we can distinguish 1 vs 2+ reliably
      refineLandmarks: false,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    this._faceMesh.onResults((results) => this._onFaceMeshResults(results));

    this._camera = new Camera(videoElement, {
      onFrame: async () => {
        await this._faceMesh.send({ image: videoElement });
      },
      width: 640,
      height: 480,
    });

    this._camera.start();
    console.log("[ProctoringClient] face mesh started");
  }

  _stopFaceMesh() {
    if (this._camera) {
      this._camera.stop();
      this._camera = null;
    }
    this._faceMesh = null;
    if (this._devtoolsCheckInterval) {
      clearInterval(this._devtoolsCheckInterval);
    }
  }

  _onFaceMeshResults(results) {
    const faceCount = results.multiFaceLandmarks ? results.multiFaceLandmarks.length : 0;

    if (faceCount === 0) {
      this._startTimer("noFace");
    } else {
      this._endTimer("noFace", "no_face");
    }

    if (faceCount >= 2) {
      // Always-countable server-side, report immediately rather than
      // timing — two faces for even a moment is a real signal.
      this._sendFlag("multiple_faces", { face_count: faceCount });
    }

    if (faceCount === 1) {
      const gazeAway = this._estimateGazeAway(results.multiFaceLandmarks[0]);
      if (gazeAway) {
        this._startTimer("gazeAway");
      } else {
        this._endTimer("gazeAway", "gaze_away");
      }
    }
  }

  /**
   * Very rough gaze estimate using iris/eye landmark horizontal offset
   * relative to face bounding box — NOT a calibrated gaze tracker. This
   * catches "looking sharply left/right/down" (e.g. at a phone or
   * second monitor) but will not catch subtle eye movement. Tune the
   * threshold based on real testing; false positives here just mean an
   * uncounted low-severity flag (see flag_processor.py's generous
   * MIN_GAZE_AWAY_DURATION_MS), not an automatic strike.
   */
  _estimateGazeAway(landmarks) {
    // MediaPipe Face Mesh landmark indices: 33/263 = left/right eye
    // outer corners, 1 = nose tip — a coarse horizontal symmetry check.
    const leftEye = landmarks[33];
    const rightEye = landmarks[263];
    const nose = landmarks[1];

    if (!leftEye || !rightEye || !nose) return false;

    const eyeMidpointX = (leftEye.x + rightEye.x) / 2;
    const horizontalOffset = Math.abs(nose.x - eyeMidpointX);

    const GAZE_AWAY_THRESHOLD = 0.04; // normalized coordinate units, needs real tuning
    return horizontalOffset > GAZE_AWAY_THRESHOLD;
  }
}

window.ProctoringClient = ProctoringClient;