// frontend/static/js/proctoring_client.js

/**
 * Second WebSocket connection (ProctoringConsumer) plus all client-side
 * detection:
 *   - tab switch / window blur / fullscreen exit / copy-paste / devtools
 *   - MediaPipe face mesh (with iris landmarks): no-face, multiple-faces,
 *     and LOOKING AWAY: head turn (yaw / pitch) + eye direction (iris
 *     position inside each eye), compared to the candidate's own
 *     screen-facing baseline. Looking away continuously for
 *     LOOK_AWAY_WARN_MS sends "gaze_away" (a strike in flag_processor.py)
 *     and repeats every LOOK_AWAY_REPEAT_MS while it continues.
 *   - face-api.js identity match (averaged reference + voting window +
 *     immediate re-check when a face reappears after being absent)
 *   - YOLOX object detection (object_detector.js, Web Worker; YOLOX-S on
 *     WebGPU, else YOLOX-Tiny on CPU): phone / other electronic device /
 *     notes, book or other large object held up in frame
 *   - "reading suspected": repeated, consistent head turns away from the
 *     calibrated screen-facing pose while the candidate is answering
 *
 * Sends RAW signals only — warn vs. terminate is decided server-side
 * (risk_scorer.py).
 *
 * Message shapes (must match apps/proctoring/consumers.py):
 *   OUT: {"type": "flag", "flag_type": "phone_detected", "metadata": {...}}
 *   OUT: {"type": "reference_photo", "image_base64": "...", "image_format": "jpeg"}
 *   IN:  {"type": "warning", "message": "...", "flag_type": "...", "count": 2, "limit": 4}
 *        (limit 0 = notice only)
 *   IN:  {"type": "terminated", "message": "...", "flag_type": "..."}
 *   IN:  {"type": "error", "message": "..."}
 *
 * Tuning: in the browser console run  window.PROCTOR_DEBUG = true  to
 * print live identity distances, object detections and head-pose
 * deltas. Use those numbers to adjust the constants below.
 *
 * Libraries loaded via <script> in interview_room.html: MediaPipe
 * FaceMesh, face-api.js, and object_detector.js. Each is awaited
 * (up to 20s) so a slow CDN doesn't silently disable a detector.
 *
 * NOTE: there is no TensorFlow.js / COCO-SSD on the page. face-api.js
 * bundles an older TensorFlow.js that crashes other TF builds sharing the
 * page. Object detection is YOLOX via ONNX Runtime inside a Web
 * Worker (window.ObjectDetector from object_detector.js) — no global `tf`
 * exists here, so do not wait for it or call it from this file.
 */

const FACE_API_MODEL_URL =
  "https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js/weights";

// ---- Reference capture (identity) ----
const REFERENCE_CAPTURE_DELAY_MS = 3000;
const REFERENCE_CAPTURE_RETRY_MS = 1500;
const REFERENCE_SAMPLE_GAP_MS = 700;
const REFERENCE_SAMPLES = 5;            // descriptors averaged into one reference
const REFERENCE_CAPTURE_MAX_ATTEMPTS = 20;

// ---- Identity matching ----
const FACE_MATCH_GRACE_MS = 3000;
const FACE_MATCH_CHECK_INTERVAL_MS = 2500;
// Distance between face-api.js descriptors. Same person is typically
// ~0.25-0.45, different people usually >0.55. Check real numbers with
// PROCTOR_DEBUG and adjust: lower = stricter (more mismatches).
const FACE_MATCH_DISTANCE_THRESHOLD = 0.5;
// Report when at least REQUIRED of the last WINDOW checks mismatched
// (not "N in a row" — a borderline face must not keep resetting it).
const FACE_MATCH_WINDOW = 6;
const FACE_MATCH_REQUIRED_MISMATCHES = 4;
// When a face disappears for this long and comes back, re-verify at once.
const IDENTITY_RECHECK_AFTER_ABSENCE_MS = 2000;
const IDENTITY_RECHECK_SAMPLES = 3;
const IDENTITY_RECHECK_INTERVAL_MS = 1000;
const IDENTITY_RECHECK_MAX_ATTEMPTS = 8;
const MISMATCH_REPORT_COOLDOWN_MS = 10000;
const FACE_DETECTOR_INPUT_SIZE = 320;   // smaller = faster on CPU

// ---- Generic timer-based flags ----
const MIN_REPORTABLE_DURATION_MS = 1500;
const MULTI_FACE_CHECKIN_MS = 5000;

// ---- Looking away (head turn + eye direction) ----
// All limits are DEVIATIONS from the candidate's own baseline, measured
// while they look at the screen (recalibratePose()). Tune with
// window.PROCTOR_DEBUG = true (prints the live values every second).
// On a PHONE the candidate holds the device in a hand (it tilts and drifts all the time) and the
// camera sits above the text they read, so the same numbers would fire "looking away" constantly
// and end the interview. Phones therefore get roomier limits.
const _PHONE = /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent);
const LOOK_AWAY_WARN_MS = _PHONE ? 6000 : 3000;     // continuously away this long -> report
const LOOK_AWAY_REPEAT_MS = 10000;    // still away -> report again (matches SAME_TYPE_COOLDOWN_SECONDS)
const LOOK_RECOVER_SAMPLES = 3;       // ~0.6 s back on screen ends an episode
const HEAD_YAW_LIMIT = _PHONE ? 0.30 : 0.20;        // left/right turn (cheek-distance ratio, ~20-25 deg on PC)
const HEAD_PITCH_DOWN_LIMIT = _PHONE ? 0.18 : 0.10; // head tipped down (lap / phone / desk)
const HEAD_PITCH_UP_LIMIT = _PHONE ? 0.22 : 0.14;   // head tipped up
const EYE_H_LIMIT = _PHONE ? 0.20 : 0.14;           // eyes to the side (iris position, 0..1 of eye width)
const EYE_DOWN_LIMIT = _PHONE ? 0.24 : 0.16;        // eyes down (iris position, 0..1 of eye height)
const HEAD_YAW_ABSOLUTE = 0.38;       // before the baseline exists: only a big turn counts

// ---- Face mesh throttle (CPU saver) ----
const FACE_MESH_MIN_INTERVAL_MS = 66;   // ~15 fps is plenty

// ---- Object detection (YOLOX in a worker) ----
// A check starts OBJECT_CHECK_INTERVAL_MS after the previous one finished,
// so a slow device simply checks less often instead of piling up work.
const OBJECT_CHECK_INTERVAL_MS = 500;   // raise to 1000 if the page feels laggy
const OBJECT_DETECT_MIN_SCORE = 0.3;    // worker-side floor; per-class rules below decide
const OBJECT_WINDOW = 3;                // look at the last 3 checks
const OBJECT_CONFIRM_HITS = 2;          // ...and report if 2 of them saw it (~1-2 s)
// While an object stays in view, it is re-reported this often (the server
// counts each report as a warning; limits live in flag_processor.py).
const OBJECT_REPORT_COOLDOWN_MS = 3000;
// COCO class -> rule. kind decides the flag type; minArea is the box's
// share of the frame (filters tiny/far-away background objects).
// Phones are often labelled "remote" when held in a hand, so it counts too.
// A notebook / tablet held up to the camera is usually labelled "laptop"
// with a modest score (COCO has no "notebook" or "paper" class), so laptop
// counts from 0.3 when the box is large (>=4% of the frame) — that is what
// "something big held in front of the candidate" looks like. TV (second
// screen) needs a higher score so a TV in the background doesn't trigger
// warnings. "keyboard" is ignored on purpose: the candidate's own keyboard
// is often in view.
const OBJECT_RULES = {
  "cell phone": { kind: "phone", label: "mobile phone", minScore: 0.4, minArea: 0 },
  "remote":     { kind: "phone", label: "handheld device", minScore: 0.45, minArea: 0 },
  "laptop":     { kind: "notes", label: "notebook / tablet / laptop held up", minScore: 0.3, minArea: 0.04 },
  // "tv" (second screen) removed: a TV or monitor in the background of the room
  // produced false phone_detected strikes. Re-add the line below if you want it back:
  // "tv":      { kind: "phone", label: "second screen", minScore: 0.65, minArea: 0.04 },
  "book":       { kind: "notes", label: "book / notes", minScore: 0.35, minArea: 0 },
};
const OBJECT_FLAG_TYPES = { phone: "phone_detected", notes: "notes_detected" };

// ---- Head pose / reading heuristic ----
const POSE_SAMPLE_INTERVAL_MS = 200;
const POSE_CALIBRATION_SAMPLES = 25;    // ~5s of screen-facing baseline
// Deviation from the calibrated baseline. pitch ratio goes UP when the
// head tips down. Starting values — tune with PROCTOR_DEBUG.
const POSE_DOWN_DELTA = 0.09;
const POSE_SIDE_DELTA = 0.15;
const READING_WINDOW_MS = 25000;
const READING_MIN_SPAN_MS = 15000;
const READING_OFF_FRACTION = 0.35;      // looking away >=35% of the window
const READING_SUSTAINED_FRACTION = 0.6; // ...or >=60% even without repeated glances
const READING_MIN_GLANCES = 3;          // look-away episodes in the window
const READING_DOMINANT_SHARE = 0.7;     // mostly the SAME direction (down or side)
const READING_REPORT_COOLDOWN_MS = 60000;

// ---- Devtools heuristic ----
const DEVTOOLS_SIZE_THRESHOLD_PX = 200;
const DEVTOOLS_CONFIRM_CHECKS = 2;
const DEVTOOLS_CHECK_INTERVAL_MS = 3000;

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

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
      multipleFaces: null,
    };

    // Face mesh
    this._faceMesh = null;
    this._faceMeshLoopActive = false;
    this._faceMeshRAF = null;
    this._lastFaceMeshAt = 0;
    this._videoElement = null;

    // Identity
    this._matchVideoElement = null;
    this._referenceDescriptors = [];
    this._referenceMean = null;
    this._faceMatchReady = false;
    this._faceMatchCheckInterval = null;
    this._referenceCaptureAttempts = 0;
    this._matchHistory = [];
    this._identityBusy = false;
    this._recheckActive = false;
    this._lastMismatchReportAt = 0;

    // Objects
    this._objectModel = null;
    this._objectVideo = null;
    this._objectTimer = null;
    this._objectLoopActive = false;
    this._objectHistory = { phone: [], notes: [] };
    this._objectLastReport = { phone: 0, notes: 0 };
    this._objectLastScore = { phone: 0, notes: 0 };
    this._objectLastLabel = { phone: "", notes: "" };

    // Looking away
    this._awaySince = null;
    this._awayDir = null;
    this._awayRecover = 0;
    this._lastAwayReportAt = 0;

    // Head pose / reading
    this._answering = false;
    this._poseBaseline = null;
    this._poseCal = [];
    this._lastPoseSampleAt = 0;
    this._lastPoseDebugAt = 0;
    this._readingSamples = [];
    this._glanceTimes = [];
    this._prevOff = false;
    this._lastReadingReportAt = 0;

    // Devtools
    this._devtoolsCheckInterval = null;
    this._devtoolsStreak = 0;
    this._devtoolsReported = false;
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

  _emit(eventName, payload, extra) {
    for (const cb of this.listeners[eventName]) {
      try {
        cb(payload, extra);
      } catch (err) {
        console.error(`ProctoringClient listener for "${eventName}" threw:`, err);
      }
    }
  }

  _debug(...args) {
    if (window.PROCTOR_DEBUG) console.log("[Proctoring debug]", ...args);
  }

  async _waitForGlobal(name, timeoutMs = 20000) {
    const start = Date.now();
    while (typeof window[name] === "undefined") {
      if (Date.now() - start > timeoutMs) return false;
      await new Promise((resolve) => setTimeout(resolve, 250));
    }
    return true;
  }

  // ------------------------------------------------------------------
  // Connection lifecycle
  // ------------------------------------------------------------------

  connect() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const token = typeof ACCESS_TOKEN !== "undefined" ? ACCESS_TOKEN : "";
    const url = `${protocol}//${window.location.host}/ws/proctoring/${this.sessionId}/?token=${encodeURIComponent(token)}`;

    this.socket = new WebSocket(url);

    this.socket.addEventListener("open", () => {
      console.log("[ProctoringClient] connected");
      this._emit("open");
      // The face check can finish before this socket opens: send the
      // reference photo it had to hold back.
      if (this._pendingReferencePhoto) {
        const photo = this._pendingReferencePhoto;
        this._pendingReferencePhoto = null;
        this._sendReferencePhoto(photo);
      }
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
    this._stopObjectDetection();
    if (this._devtoolsCheckInterval) {
      clearInterval(this._devtoolsCheckInterval);
      this._devtoolsCheckInterval = null;
    }
  }

  _handleMessage(data) {
    switch (data.type) {
      case "warning":
        this._emit("warning", data.message, {
          flagType: data.flag_type,
          count: data.count,
          limit: data.limit,
        });
        break;
      case "terminated":
        this._emit("terminated", data.message, { flagType: data.flag_type });
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
  // Outgoing messages
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
      console.warn("[ProctoringClient] socket not open yet — holding the reference photo until it connects");
      this._pendingReferencePhoto = imageBase64;
      return;
    }
    this.socket.send(JSON.stringify({
      type: "reference_photo",
      image_base64: imageBase64,
      image_format: "jpeg",
    }));
  }

  // ------------------------------------------------------------------
  // Timer helpers
  // ------------------------------------------------------------------

  _startTimer(key) {
    if (this._activeTimers[key] !== null) return;
    this._activeTimers[key] = Date.now();
  }

  /** Ends a timer and reports its duration — unless it was just a blip. */
  _endTimer(key, flagType, minReportableMs = MIN_REPORTABLE_DURATION_MS) {
    const startedAt = this._activeTimers[key];
    if (startedAt === null) return;
    const durationMs = Date.now() - startedAt;
    this._activeTimers[key] = null;
    if (durationMs < minReportableMs) return;
    this._sendFlag(flagType, { duration_ms: durationMs });
  }

  /** While a condition holds, report a check-in every checkinMs. */
  _tickOngoing(key, flagType, checkinMs, extra = {}) {
    const now = Date.now();
    const startedAt = this._activeTimers[key];
    if (startedAt === null) {
      this._activeTimers[key] = now;
      return;
    }
    const elapsed = now - startedAt;
    if (elapsed >= checkinMs) {
      this._activeTimers[key] = now;
      this._sendFlag(flagType, { ...extra, duration_ms: elapsed });
    }
  }

  // ------------------------------------------------------------------
  // Tab switch / window blur / fullscreen / copy-paste / devtools
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

    // Window-size heuristic for docked devtools; only meaningful in
    // fullscreen (leaving fullscreen is reported separately).
    this._devtoolsCheckInterval = setInterval(() => {
      if (!document.fullscreenElement) {
        this._devtoolsStreak = 0;
        this._devtoolsReported = false;
        return;
      }

      const widthDelta = window.outerWidth - window.innerWidth;
      const heightDelta = window.outerHeight - window.innerHeight;
      const suspicious =
        widthDelta > DEVTOOLS_SIZE_THRESHOLD_PX || heightDelta > DEVTOOLS_SIZE_THRESHOLD_PX;

      if (suspicious) {
        this._devtoolsStreak += 1;
        if (this._devtoolsStreak >= DEVTOOLS_CONFIRM_CHECKS && !this._devtoolsReported) {
          this._devtoolsReported = true;
          this._sendFlag("devtools_opened", { widthDelta, heightDelta });
        }
      } else {
        this._devtoolsStreak = 0;
        this._devtoolsReported = false;
      }
    }, DEVTOOLS_CHECK_INTERVAL_MS);

    console.log("[ProctoringClient] browser event detection active");
  }

  static async requestFullscreen() {
    try {
      await document.documentElement.requestFullscreen();
    } catch (err) {
      console.error("[ProctoringClient] fullscreen request failed:", err);
    }
  }

  // ------------------------------------------------------------------
  // Answering state + pose calibration (used by the reading heuristic)
  // ------------------------------------------------------------------

  /** Call with true when the candidate starts answering, false when done. */
  setAnswering(isAnswering) {
    this._answering = !!isAnswering;
    this._resetReadingWindow();
  }

  /**
   * Re-learn the "looking at the screen" head pose. Call when the
   * interview starts (the candidate is listening to the AI intro and
   * looking at the screen).
   */
  recalibratePose() {
    this._poseBaseline = null;
    this._poseCal = [];
    this._resetReadingWindow();
    this._resetLookAway();
  }

  _resetLookAway() {
    this._awaySince = null;
    this._awayDir = null;
    this._awayRecover = 0;
  }

  _resetReadingWindow() {
    this._readingSamples = [];
    this._glanceTimes = [];
    this._prevOff = false;
  }

  // ------------------------------------------------------------------
  // MediaPipe Face Mesh — no-face / multiple-faces / gaze / head pose
  // ------------------------------------------------------------------

  /**
   * videoElement MUST already have a live stream attached (call after
   * WebRTCClient.init). Frames are pumped from that existing element —
   * MediaPipe's Camera helper is deliberately NOT used because it opens
   * a second getUserMedia stream.
   */
  async startFaceMesh(videoElement) {
    if (!(await this._waitForGlobal("FaceMesh"))) {
      console.error(
        "[ProctoringClient] MediaPipe FaceMesh did not load — check the " +
        "CDN script tag in interview_room.html. Face-based checks are off."
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
      refineLandmarks: true,     // adds iris landmarks (468-477) for eye direction
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    this._faceMesh.onResults((results) => this._onFaceMeshResults(results));

    this._faceMeshLoopActive = true;
    const pump = async () => {
      if (!this._faceMeshLoopActive) return;
      const now = performance.now();
      if (
        now - this._lastFaceMeshAt >= FACE_MESH_MIN_INTERVAL_MS &&
        videoElement.readyState >= 2 &&
        videoElement.videoWidth > 0
      ) {
        this._lastFaceMeshAt = now;
        try {
          await this._faceMesh.send({ image: videoElement });
        } catch (err) {
          console.error("[ProctoringClient] face mesh send failed:", err);
        }
      }
      this._faceMeshRAF = requestAnimationFrame(pump);
    };
    pump();

    console.log("[ProctoringClient] face mesh started (shared stream, no extra camera request)");
  }

  _stopFaceMesh() {
    this._faceMeshLoopActive = false;
    if (this._faceMeshRAF) {
      cancelAnimationFrame(this._faceMeshRAF);
      this._faceMeshRAF = null;
    }
    this._faceMesh = null;
  }

  _onFaceMeshResults(results) {
    const faces = results.multiFaceLandmarks || [];
    const faceCount = faces.length;

    if (faceCount === 0) {
      this._startTimer("noFace");
    } else {
      // If the face was gone for a while and is back, the person may
      // have changed — verify identity immediately instead of waiting.
      const absentMs =
        this._activeTimers.noFace !== null ? Date.now() - this._activeTimers.noFace : 0;
      this._endTimer("noFace", "no_face");
      if (absentMs >= IDENTITY_RECHECK_AFTER_ABSENCE_MS) {
        this._startIdentityRecheck();
      }
    }

    if (faceCount >= 2) {
      this._tickOngoing("multipleFaces", "multiple_faces", MULTI_FACE_CHECKIN_MS, {
        face_count: faceCount,
      });
    } else {
      this._activeTimers.multipleFaces = null;
    }

    if (faceCount === 1) {
      this._samplePose(faces[0]);
    } else {
      this._resetLookAway();      // no face / several faces are reported separately
    }
  }

  // ------------------------------------------------------------------
  // Head pose + "reading suspected"
  // ------------------------------------------------------------------

  /**
   * Two ratios, each using a single axis so the video's aspect ratio
   * cancels out:
   *   pitch = (nose.y - eyeMid.y) / (chin.y - eyeMid.y)   (up when head tips DOWN)
   *   yaw   = (nose.x - eyeMid.x) / eyeDistance
   * Absolute values differ per person/camera, so we always compare to a
   * baseline measured while the candidate looks at the screen.
   */
  _poseFromLandmarks(lm) {
    const nose = lm[1];
    const leftEye = lm[33];
    const rightEye = lm[263];
    const chin = lm[152];
    const cheekL = lm[234];
    const cheekR = lm[454];
    if (!nose || !leftEye || !rightEye || !chin || !cheekL || !cheekR) return null;

    const eyeMidX = (leftEye.x + rightEye.x) / 2;
    const eyeMidY = (leftEye.y + rightEye.y) / 2;
    const eyeDist = Math.abs(rightEye.x - leftEye.x);
    const faceH = chin.y - eyeMidY;
    if (eyeDist < 0.02 || faceH < 0.02) return null;

    // Yaw from the nose's distance to each side of the face: 0 when
    // facing the camera, towards +/-1 as the head turns. Uses real pixel
    // proportions (x scaled by the video's aspect ratio).
    const v = this._videoElement;
    const aspect = v && v.videoHeight ? v.videoWidth / v.videoHeight : 4 / 3;
    const dist = (a, b) => Math.hypot((a.x - b.x) * aspect, a.y - b.y);
    const dL = dist(nose, cheekL);
    const dR = dist(nose, cheekR);

    const pose = {
      pitch: (nose.y - eyeMidY) / faceH,
      yaw: (nose.x - eyeMidX) / eyeDist,
      yawR: (dL - dR) / (dL + dR),
      gx: null,
      gy: null,
    };

    // Eye direction from the iris centre inside each eye (needs
    // refineLandmarks). gx: 0 = image-left corner, 1 = image-right corner.
    // gy: 0 = upper lid, 1 = lower lid.
    if (lm.length >= 478) {
      const eyes = [
        { a: lm[33], b: lm[133], up: lm[159], low: lm[145], iris: lm[468] },
        { a: lm[362], b: lm[263], up: lm[386], low: lm[374], iris: lm[473] },
      ];
      let gx = 0, gy = 0, n = 0;
      for (const e of eyes) {
        const x0 = Math.min(e.a.x, e.b.x), x1 = Math.max(e.a.x, e.b.x);
        const h = e.low.y - e.up.y;
        if (x1 - x0 < 0.005 || h < 0.002) continue;
        gx += (e.iris.x - x0) / (x1 - x0);
        gy += (e.iris.y - e.up.y) / h;
        n++;
      }
      if (n) { pose.gx = gx / n; pose.gy = gy / n; }
    }
    return pose;
  }

  /**
   * Is the candidate looking away from the screen right now?
   * Returns null (on screen) or a direction: "left" / "right" / "down" / "up"
   * plus the measured deviations (sent to the server for the recruiter).
   */
  _lookDirection(pose) {
    const b = this._poseBaseline;
    if (!b) {
      // No baseline yet: only an obvious head turn counts.
      if (Math.abs(pose.yawR) > HEAD_YAW_ABSOLUTE) {
        return { dir: pose.yawR > 0 ? "right" : "left", source: "head", dYaw: pose.yawR };
      }
      return null;
    }
    const dYaw = pose.yawR - b.yawR;
    const dPitch = pose.pitch - b.pitch;
    const eyes = pose.gx !== null && b.gx !== null && b.gx !== undefined;
    const dGx = eyes ? pose.gx - b.gx : 0;
    const dGy = eyes ? pose.gy - b.gy : 0;
    const info = { dYaw: +dYaw.toFixed(3), dPitch: +dPitch.toFixed(3), dGx: +dGx.toFixed(3), dGy: +dGy.toFixed(3) };

    if (Math.abs(dYaw) > HEAD_YAW_LIMIT) return { dir: dYaw > 0 ? "right" : "left", source: "head", ...info };
    if (dPitch > HEAD_PITCH_DOWN_LIMIT) return { dir: "down", source: "head", ...info };
    if (dPitch < -HEAD_PITCH_UP_LIMIT) return { dir: "up", source: "head", ...info };
    if (eyes && Math.abs(dGx) > EYE_H_LIMIT) return { dir: dGx > 0 ? "right" : "left", source: "eyes", ...info };
    if (eyes && dGy > EYE_DOWN_LIMIT) return { dir: "down", source: "eyes", ...info };
    return null;
  }

  _updateLookAway(now, look) {
    if (!look) {
      if (this._awaySince !== null && ++this._awayRecover >= LOOK_RECOVER_SAMPLES) this._resetLookAway();
      return;
    }
    this._awayRecover = 0;
    if (this._awaySince === null) this._awaySince = now;
    this._awayDir = look.dir;
    const durationMs = now - this._awaySince;
    if (durationMs >= LOOK_AWAY_WARN_MS && now - this._lastAwayReportAt >= LOOK_AWAY_REPEAT_MS) {
      this._lastAwayReportAt = now;
      this._sendFlag("gaze_away", { duration_ms: durationMs, direction: look.dir, source: look.source,
        d_yaw: look.dYaw, d_pitch: look.dPitch, d_gaze_x: look.dGx, d_gaze_y: look.dGy });
    }
  }

  _samplePose(landmarks) {
    const now = Date.now();
    if (now - this._lastPoseSampleAt < POSE_SAMPLE_INTERVAL_MS) return;
    this._lastPoseSampleAt = now;

    const pose = this._poseFromLandmarks(landmarks);
    if (!pose) return;

    // Looking away runs all the time (also before the baseline exists).
    const look = this._lookDirection(pose);
    this._updateLookAway(now, look);

    if (!this._poseBaseline) {
      // Calibrate only on frames that are clearly screen-facing.
      if (Math.abs(pose.yawR) < HEAD_YAW_ABSOLUTE / 2) this._poseCal.push(pose);
      if (this._poseCal.length >= POSE_CALIBRATION_SAMPLES) {
        const eyeCal = this._poseCal.filter((p) => p.gx !== null);
        this._poseBaseline = {
          pitch: median(this._poseCal.map((p) => p.pitch)),
          yaw: median(this._poseCal.map((p) => p.yaw)),
          yawR: median(this._poseCal.map((p) => p.yawR)),
          gx: eyeCal.length ? median(eyeCal.map((p) => p.gx)) : null,
          gy: eyeCal.length ? median(eyeCal.map((p) => p.gy)) : null,
        };
        this._poseCal = [];
        console.log("[ProctoringClient] head-pose + eye baseline set", this._poseBaseline);
      }
      return;
    }

    const dPitch = pose.pitch - this._poseBaseline.pitch;
    const dYaw = pose.yaw - this._poseBaseline.yaw;

    if (window.PROCTOR_DEBUG && now - this._lastPoseDebugAt > 1000) {
      this._lastPoseDebugAt = now;
      const b = this._poseBaseline;
      this._debug(
        `look dYaw=${(pose.yawR - b.yawR).toFixed(3)} (>${HEAD_YAW_LIMIT}) ` +
        `dPitch=${dPitch.toFixed(3)} (down>${HEAD_PITCH_DOWN_LIMIT}, up<-${HEAD_PITCH_UP_LIMIT}) ` +
        (pose.gx !== null && b.gx !== null
          ? `eyesX=${(pose.gx - b.gx).toFixed(3)} (>${EYE_H_LIMIT}) eyesY=${(pose.gy - b.gy).toFixed(3)} (>${EYE_DOWN_LIMIT}) `
          : "eyes=n/a ") +
        `away=${look ? look.dir + "/" + look.source : "no"}` +
        (this._awaySince ? ` for ${now - this._awaySince}ms` : "")
      );
    }

    if (!this._answering) return;

    // Looking UP is ignored on purpose: people look up/away while
    // recalling, which is normal. Down or consistently to one side is
    // what reading from a phone/lap/second screen looks like.
    let dir = null;
    if (dPitch > POSE_DOWN_DELTA) dir = "down";
    else if (Math.abs(dYaw) > POSE_SIDE_DELTA) dir = "side";

    this._pushReadingSample(now, dir);
  }

  _pushReadingSample(now, dir) {
    const off = dir !== null;
    this._readingSamples.push({ t: now, dir });
    if (off && !this._prevOff) this._glanceTimes.push(now);
    this._prevOff = off;

    while (this._readingSamples.length && now - this._readingSamples[0].t > READING_WINDOW_MS) {
      this._readingSamples.shift();
    }
    while (this._glanceTimes.length && now - this._glanceTimes[0] > READING_WINDOW_MS) {
      this._glanceTimes.shift();
    }

    const span = now - this._readingSamples[0].t;
    if (span < READING_MIN_SPAN_MS) return;
    if (now - this._lastReadingReportAt < READING_REPORT_COOLDOWN_MS) return;

    const offSamples = this._readingSamples.filter((s) => s.dir !== null);
    const fraction = offSamples.length / this._readingSamples.length;
    if (fraction < READING_OFF_FRACTION) return;

    const down = offSamples.filter((s) => s.dir === "down").length;
    const side = offSamples.length - down;
    const dominantShare = Math.max(down, side) / offSamples.length;
    if (dominantShare < READING_DOMINANT_SHARE) return;

    const glances = this._glanceTimes.length;
    if (glances < READING_MIN_GLANCES && fraction < READING_SUSTAINED_FRACTION) return;

    this._lastReadingReportAt = now;
    this._sendFlag("reading_suspected", {
      off_axis_pct: Math.round(fraction * 100),
      glances,
      direction: down >= side ? "down" : "side",
      window_s: Math.round(span / 1000),
    });
    this._resetReadingWindow();
  }

  // ------------------------------------------------------------------
  // Object detection — phone / notes (COCO-SSD via TensorFlow.js)
  // ------------------------------------------------------------------

  /**
   * Runs entirely in the candidate's browser (YOLOX in a Web Worker).
   * See OBJECT_RULES for what is detected. A sighting only becomes a flag
   * when it is seen in at least OBJECT_CONFIRM_HITS of the last
   * OBJECT_WINDOW checks.
   */
  async startObjectDetection(videoElement) {
    if (!(await this._waitForGlobal("ObjectDetector"))) {
      console.error(
        "[ProctoringClient] object detector did not load — phone detection " +
        "is off. Check the object_detector.js script tag in interview_room.html."
      );
      return;
    }

    this._objectVideo = videoElement;

    try {
      this._objectModel = await window.ObjectDetector.load();
    } catch (err) {
      console.error("[ProctoringClient] failed to load object detection model:", err);
      return;
    }

    this._objectLoopActive = true;
    const tick = async () => {
      if (!this._objectLoopActive) return;
      await this._detectObjectsOnce();
      this._objectTimer = setTimeout(tick, OBJECT_CHECK_INTERVAL_MS);
    };
    tick();

    console.log("[ProctoringClient] object detection started (phone / device / notes)");
  }

  _stopObjectDetection() {
    this._objectLoopActive = false;
    if (this._objectTimer) {
      clearTimeout(this._objectTimer);
      this._objectTimer = null;
    }
  }

  async _detectObjectsOnce() {
    const video = this._objectVideo;
    if (!this._objectModel || !video || video.readyState < 2 || video.videoWidth === 0) return;

    let predictions;
    try {
      predictions = await this._objectModel.detect(video, 20, OBJECT_DETECT_MIN_SCORE);
    } catch (err) {
      console.error("[ProctoringClient] object detection error:", err);
      return;
    }

    const frameArea = video.videoWidth * video.videoHeight;
    const best = { phone: 0, notes: 0 };
    const bestLabel = { phone: "", notes: "" };
    for (const p of predictions) {
      const rule = OBJECT_RULES[p.class];
      if (!rule || p.score < rule.minScore) continue;
      const area = (p.bbox[2] * p.bbox[3]) / frameArea;
      if (area < rule.minArea) continue;
      if (p.score > best[rule.kind]) {
        best[rule.kind] = p.score;
        bestLabel[rule.kind] = rule.label;
      }
    }

    this._debug(
      `objects (${this._objectModel.lastInferMs || "?"} ms):`,
      predictions.map((p) => `${p.class}:${p.score.toFixed(2)}`).join(", ") || "none"
    );

    this._recordObjectResult("phone", best.phone, bestLabel.phone);
    this._recordObjectResult("notes", best.notes, bestLabel.notes);
  }

  _recordObjectResult(kind, score, label = "") {
    const history = this._objectHistory[kind];
    history.push(score > 0);
    if (history.length > OBJECT_WINDOW) history.shift();
    if (score > 0) {
      this._objectLastScore[kind] = score;
      this._objectLastLabel[kind] = label;
    }

    const hits = history.filter(Boolean).length;
    const now = Date.now();
    if (
      hits >= OBJECT_CONFIRM_HITS &&
      now - this._objectLastReport[kind] >= OBJECT_REPORT_COOLDOWN_MS
    ) {
      this._objectLastReport[kind] = now;
      this._sendFlag(OBJECT_FLAG_TYPES[kind], {
        hits,
        window: OBJECT_WINDOW,
        score: Number(this._objectLastScore[kind].toFixed(2)),
        object: this._objectLastLabel[kind],
        model: (this._objectModel.info && this._objectModel.info.model) || "yolox",
      });
    }
  }

  // ------------------------------------------------------------------
  // face-api.js — reference capture + live identity match
  // ------------------------------------------------------------------

  _detectorOptions() {
    return new faceapi.TinyFaceDetectorOptions({
      inputSize: FACE_DETECTOR_INPUT_SIZE,
      scoreThreshold: 0.5,
    });
  }

  async startFaceMatch(videoElement) {
    if (!(await this._waitForGlobal("faceapi"))) {
      console.error(
        "[ProctoringClient] face-api.js did not load — identity matching is off. " +
        "Check the script tag in interview_room.html."
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

  /**
   * Collects REFERENCE_SAMPLES descriptors (a few seconds apart) and
   * averages them. The first successful detection also provides the
   * snapshot saved for the recruiter dashboard.
   */
  async _captureReferenceFace() {
    const videoElement = this._matchVideoElement;
    if (!videoElement) return;

    this._referenceCaptureAttempts += 1;

    let detection = null;
    try {
      detection = await faceapi
        .detectSingleFace(videoElement, this._detectorOptions())
        .withFaceLandmarks()
        .withFaceDescriptor();
    } catch (err) {
      console.error("[ProctoringClient] face-api detection error during reference capture:", err);
    }

    if (detection) {
      if (this._referenceDescriptors.length === 0) {
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
      }
      this._referenceDescriptors.push(detection.descriptor);
    }

    if (this._referenceDescriptors.length >= REFERENCE_SAMPLES) {
      this._finalizeReference();
      return;
    }

    if (this._referenceCaptureAttempts >= REFERENCE_CAPTURE_MAX_ATTEMPTS) {
      if (this._referenceDescriptors.length >= 2) {
        this._finalizeReference();
      } else {
        console.error(
          "[ProctoringClient] could not capture a usable reference face — " +
          "identity matching will not be active for this session."
        );
      }
      return;
    }

    setTimeout(
      () => this._captureReferenceFace(),
      detection ? REFERENCE_SAMPLE_GAP_MS : REFERENCE_CAPTURE_RETRY_MS
    );
  }

  _finalizeReference() {
    const n = this._referenceDescriptors.length;
    const mean = new Float32Array(128);
    for (const d of this._referenceDescriptors) {
      for (let i = 0; i < 128; i++) mean[i] += d[i] / n;
    }
    this._referenceMean = mean;
    console.log(`[ProctoringClient] reference face captured (${n} samples averaged)`);

    setTimeout(() => {
      this._faceMatchReady = true;
      this._faceMatchCheckInterval = setInterval(
        () => this._checkFaceMatch(),
        FACE_MATCH_CHECK_INTERVAL_MS
      );
      console.log("[ProctoringClient] face-match live checks active");
    }, FACE_MATCH_GRACE_MS);
  }

  /**
   * One identity measurement. Returns {distance, mismatch}, or null if
   * there was no face or another measurement is still running.
   */
  async _measureIdentity() {
    if (!this._referenceMean || !this._matchVideoElement) return null;
    if (this._identityBusy) return null;
    this._identityBusy = true;
    try {
      const detection = await faceapi
        .detectSingleFace(this._matchVideoElement, this._detectorOptions())
        .withFaceLandmarks()
        .withFaceDescriptor();
      if (!detection) return null;

      const distance = faceapi.euclideanDistance(this._referenceMean, detection.descriptor);
      return { distance, mismatch: distance > FACE_MATCH_DISTANCE_THRESHOLD };
    } catch (err) {
      console.error("[ProctoringClient] face-api detection error during match check:", err);
      return null;
    } finally {
      this._identityBusy = false;
    }
  }

  /** Regular check: report when REQUIRED of the last WINDOW checks mismatched. */
  async _checkFaceMatch() {
    if (!this._faceMatchReady || this._recheckActive) return;

    const r = await this._measureIdentity();
    if (!r) return; // no face (no_face covers that) or busy

    this._debug("identity distance", r.distance.toFixed(3), r.mismatch ? "MISMATCH" : "match");

    this._matchHistory.push(r);
    if (this._matchHistory.length > FACE_MATCH_WINDOW) this._matchHistory.shift();

    const bad = this._matchHistory.filter((x) => x.mismatch);
    if (bad.length >= FACE_MATCH_REQUIRED_MISMATCHES) {
      const avg = bad.reduce((sum, x) => sum + x.distance, 0) / bad.length;
      this._reportMismatch(avg, "sustained");
      this._matchHistory = [];
    }
  }

  /** Fast path: a face just reappeared after being absent. */
  _startIdentityRecheck() {
    if (!this._faceMatchReady || this._recheckActive) return;
    this._recheckActive = true;

    const results = [];
    let attempts = 0;

    const step = async () => {
      attempts += 1;
      const r = await this._measureIdentity();
      if (r) {
        results.push(r);
        this._debug("identity recheck", r.distance.toFixed(3), r.mismatch ? "MISMATCH" : "match");
      }

      const finished =
        results.length >= IDENTITY_RECHECK_SAMPLES || attempts >= IDENTITY_RECHECK_MAX_ATTEMPTS;
      if (!finished) {
        setTimeout(step, IDENTITY_RECHECK_INTERVAL_MS);
        return;
      }

      const bad = results.filter((x) => x.mismatch);
      if (results.length >= 2 && bad.length >= 2 && bad.length / results.length >= 0.6) {
        const avg = bad.reduce((sum, x) => sum + x.distance, 0) / bad.length;
        this._reportMismatch(avg, "after_absence");
      }
      this._recheckActive = false;
    };
    step();
  }

  _reportMismatch(distance, trigger) {
    const now = Date.now();
    if (now - this._lastMismatchReportAt < MISMATCH_REPORT_COOLDOWN_MS) return;
    this._lastMismatchReportAt = now;
    this._sendFlag("face_mismatch", {
      distance: Number(distance.toFixed(3)),
      trigger,
    });
  }

  _stopFaceMatch() {
    if (this._faceMatchCheckInterval) {
      clearInterval(this._faceMatchCheckInterval);
      this._faceMatchCheckInterval = null;
    }
    this._faceMatchReady = false;
    this._matchHistory = [];
    this._recheckActive = false;
  }
}

window.ProctoringClient = ProctoringClient;