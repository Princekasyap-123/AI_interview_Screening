// frontend/static/js/proctoring_client.js

/**
 * Owns the second WebSocket connection (to ProctoringConsumer,
 * apps/proctoring/consumers.py) and all client-side detection: tab
 * switches, window blur, fullscreen exits, face-mesh-based
 * no-face/multiple-faces/gaze-away detection via MediaPipe, and
 * face-match (identity) detection via face-api.js.
 *
 * This file sends RAW signals only — it does NOT decide warn vs.
 * terminate itself. That decision is entirely server-side
 * (risk_scorer.py). This file only reports "this happened, for this
 * long" and reacts to whatever the server sends back.
 *
 * Message shapes (must match apps/proctoring/consumers.py exactly):
 *   OUT: {"type": "flag", "flag_type": "tab_switch", "metadata": {"duration_ms": 2100}}
 *   OUT: {"type": "reference_photo", "image_base64": "...", "image_format": "jpeg"}
 *   IN:  {"type": "warning", "message": "..."}
 *   IN:  {"type": "terminated", "message": "..."}
 *   IN:  {"type": "error", "message": "..."}
 *
 * MediaPipe Face Mesh and face-api.js are loaded via CDN script tags in
 * interview_room.html — this file assumes `FaceMesh`, `Camera`, and
 * `faceapi` globals exist on window.
 *
 * NOTE on face-api.js model weights: loaded from a CDN mirror below.
 * If that mirror ever becomes unreachable, download the weight files
 * and serve them from your own static/ folder instead, then update
 * FACE_API_MODEL_URL accordingly.
 */

const FACE_API_MODEL_URL =
  "https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js/weights";

// How long to wait after the camera starts before capturing the
// reference face — lets exposure/focus stabilize so the reference
// descriptor isn't computed from a blurry first frame.
const REFERENCE_CAPTURE_DELAY_MS = 3000;
const REFERENCE_CAPTURE_RETRY_MS = 2000;
const REFERENCE_CAPTURE_MAX_ATTEMPTS = 6; // ~12s of retrying before giving up

// Grace window after the reference is captured before live mismatch
// checks begin — avoids a false mismatch racing in immediately.
const FACE_MATCH_GRACE_MS = 5000;
const FACE_MATCH_CHECK_INTERVAL_MS = 4000;

// Euclidean distance threshold between face-api.js descriptors.
// face-api's own docs suggest ~0.6 as a typical same-person cutoff;
// used slightly stricter here since a false mismatch has real
// consequences (proctoring strike). Tune based on real testing.
const FACE_MATCH_DISTANCE_THRESHOLD = 0.55;

class ProctoringClient {
  constructor(sessionId) {
    this.sessionId = sessionId;
    this.socket = null;

    this.listeners = {
      warning: [],
      terminated: [],
      open: [],
      flag: [],
    };

    this._activeTimers = {
      tabSwitch: null,
      windowBlur: null,
      noFace: null,
      gazeAway: null,
      faceMismatch: null,
    };

    this._faceMesh = null;
    this._camera = null;
    this._videoElement = null;
    this._lastFaceCheckState = { faceCount: 1, gazeAway: false };

    // Face-match state
    this._matchVideoElement = null;
    this._referenceDescriptor = null;
    this._faceMatchReady = false;
    this._faceMatchCheckInterval = null;
    this._referenceCaptureAttempts = 0;
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
    this._stopFaceMatch();
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
    this._emit("flag", { flagType, metadata });

    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      console.warn("[ProctoringClient] cannot send flag, socket not open:", flagType);
      return;
    }
    this.socket.send(JSON.stringify({ type: "flag", flag_type: flagType, metadata }));
  }

  _sendReferencePhoto(imageBase64) {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      console.warn("[ProctoringClient] cannot send reference photo, socket not open");
      return;
    }
    this.socket.send(JSON.stringify({
      type: "reference_photo",
      image_base64: imageBase64,
      image_format: "jpeg",
    }));
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
        this._sendFlag("fullscreen_exit", {});
      }
    });

    document.addEventListener("copy", () => {
      this._sendFlag("copy_paste", { action: "copy" });
    });
    document.addEventListener("paste", () => {
      this._sendFlag("copy_paste", { action: "paste" });
    });

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
    if (this._activeTimers[key] !== null) return;
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
  // Fullscreen enforcement helper
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
      maxNumFaces: 3,
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

  _estimateGazeAway(landmarks) {
    const leftEye = landmarks[33];
    const rightEye = landmarks[263];
    const nose = landmarks[1];

    if (!leftEye || !rightEye || !nose) return false;

    const eyeMidpointX = (leftEye.x + rightEye.x) / 2;
    const horizontalOffset = Math.abs(nose.x - eyeMidpointX);

    const GAZE_AWAY_THRESHOLD = 0.04;
    return horizontalOffset > GAZE_AWAY_THRESHOLD;
  }

  // ------------------------------------------------------------------
  // face-api.js — reference capture + live identity match
  // ------------------------------------------------------------------

  /**
   * Loads face-api.js models, then captures a reference face descriptor
   * ~3s after the video feed is available (letting it stabilize first).
   * Call this once, alongside startFaceMesh(), using the SAME video
   * element (the candidate's own webcam feed — no separate camera
   * request here either).
   */
  async startFaceMatch(videoElement) {
    if (typeof faceapi === "undefined") {
      console.error(
        "[ProctoringClient] face-api.js not found on window — make sure " +
        "the CDN script is included in interview_room.html."
      );
      return;
    }

    this._matchVideoElement = videoElement;

    try {
      await Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromUri(FACE_API_MODEL_URL),
        faceapi.nets.faceLandmark68Net.loadFromUri(FACE_API_MODEL_URL),
        faceapi.nets.faceRecognitionNet.loadFromUri(FACE_API_MODEL_URL),
      ]);
    } catch (err) {
      console.error("[ProctoringClient] failed to load face-api models:", err);
      return;
    }

    setTimeout(() => this._captureReferenceFace(), REFERENCE_CAPTURE_DELAY_MS);
  }

  async _captureReferenceFace() {
    const videoElement = this._matchVideoElement;
    if (!videoElement) return;

    this._referenceCaptureAttempts += 1;

    let detection;
    try {
      detection = await faceapi
        .detectSingleFace(videoElement, new faceapi.TinyFaceDetectorOptions())
        .withFaceLandmarks()
        .withFaceDescriptor();
    } catch (err) {
      console.error("[ProctoringClient] face-api detection error during reference capture:", err);
      detection = null;
    }

    if (!detection) {
      if (this._referenceCaptureAttempts >= REFERENCE_CAPTURE_MAX_ATTEMPTS) {
        console.error(
          "[ProctoringClient] gave up capturing reference face after " +
          `${this._referenceCaptureAttempts} attempts — face-match ` +
          "detection will not be active for this session."
        );
        return;
      }
      console.warn("[ProctoringClient] no face found for reference capture, retrying...");
      setTimeout(() => this._captureReferenceFace(), REFERENCE_CAPTURE_RETRY_MS);
      return;
    }

    this._referenceDescriptor = detection.descriptor;

    // Snapshot a still frame for the recruiter dashboard's audit trail
    // — separate from the descriptor used for live comparison above.
    try {
      const canvas = document.createElement("canvas");
      canvas.width = videoElement.videoWidth;
      canvas.height = videoElement.videoHeight;
      canvas.getContext("2d").drawImage(videoElement, 0, 0);
      const imageBase64 = canvas.toDataURL("image/jpeg", 0.85).split(",")[1];
      this._sendReferencePhoto(imageBase64);
    } catch (err) {
      console.error("[ProctoringClient] failed to capture reference snapshot:", err);
    }

    // Grace window before live mismatch checks begin.
    setTimeout(() => {
      this._faceMatchReady = true;
      this._faceMatchCheckInterval = setInterval(
        () => this._checkFaceMatch(),
        FACE_MATCH_CHECK_INTERVAL_MS
      );
      console.log("[ProctoringClient] face-match live checks active");
    }, FACE_MATCH_GRACE_MS);

    console.log("[ProctoringClient] reference face captured");
  }

  async _checkFaceMatch() {
    if (!this._faceMatchReady || !this._referenceDescriptor || !this._matchVideoElement) {
      return;
    }

    let detection;
    try {
      detection = await faceapi
        .detectSingleFace(this._matchVideoElement, new faceapi.TinyFaceDetectorOptions())
        .withFaceLandmarks()
        .withFaceDescriptor();
    } catch (err) {
      console.error("[ProctoringClient] face-api detection error during match check:", err);
      return;
    }

    if (!detection) {
      // No face currently visible — that's already covered by the
      // MediaPipe no_face flag; don't double-flag the same underlying
      // condition as a mismatch.
      this._endTimer("faceMismatch", "face_mismatch");
      return;
    }

    const distance = faceapi.euclideanDistance(this._referenceDescriptor, detection.descriptor);

    if (distance > FACE_MATCH_DISTANCE_THRESHOLD) {
      this._startTimer("faceMismatch");
    } else {
      this._endTimer("faceMismatch", "face_mismatch");
    }
  }

  _stopFaceMatch() {
    if (this._faceMatchCheckInterval) {
      clearInterval(this._faceMatchCheckInterval);
      this._faceMatchCheckInterval = null;
    }
    this._faceMatchReady = false;
  }
}

window.ProctoringClient = ProctoringClient;