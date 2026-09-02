// frontend/static/js/webrtc_client.js

/**
 * Owns the candidate's own camera/mic preview (self-view panel in
 * interview_room.html). Despite the filename, this is NOT a true peer-
 * to-peer WebRTC call in the current no-GPU phase — there is no remote
 * peer to connect to, since the "AI interviewer" is a static logo panel
 * (per the confirmed frontend design), not a live video participant.
 *
 * What this file actually does:
 *   1. Requests camera+mic permission and renders the local stream into
 *      the candidate's self-view <video> element.
 *   2. Exposes that same MediaStream so ProctoringClient.startFaceMesh()
 *      can read frames from it (shared stream, not a second camera
 *      request — avoids prompting for camera permission twice).
 *   3. Exposes the stream so STTRecorder.init(stream) can reuse the
 *      same audio track rather than requesting its own mic session
 *      (fixed in the current stt_recorder.js).
 *
 * When a real AI avatar video call is built (GPU phase, per
 * apps/avatar/), THIS is the file that would grow actual WebRTC peer
 * connection logic (RTCPeerConnection, signaling via Channels, etc.)
 * to receive a live-generated avatar stream. Right now it's local-only.
 */

class WebRTCClient {
  constructor() {
    this.localStream = null;
    this.videoElement = null;
  }

  /**
   * videoElement: the candidate's self-view <video> element in
   * interview_room.html (e.g. #candidateVideo).
   */
  async init(videoElement) {
    this.videoElement = videoElement;

    try {
      this.localStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
        audio: true,
      });
    } catch (err) {
      console.error("[WebRTCClient] camera/mic permission denied or unavailable:", err);
      throw new Error(
        "Camera and microphone access are required for this interview. " +
        "Please allow permissions and reload the page."
      );
    }

    this.videoElement.srcObject = this.localStream;
    this.videoElement.muted = true; // never echo candidate's own audio back to them
    await this.videoElement.play().catch((err) => {
      console.warn("[WebRTCClient] video autoplay was blocked:", err);
    });

    console.log("[WebRTCClient] local stream initialized");
  }

  getVideoElement() {
    return this.videoElement;
  }

  getStream() {
    return this.localStream;
  }

  toggleMic(enabled) {
    if (!this.localStream) return;
    this.localStream.getAudioTracks().forEach((track) => {
      track.enabled = enabled;
    });
  }

  /**
   * Stops ALL tracks (video + audio) and releases the camera/mic.
   * Since stt_recorder.js no longer owns a separate audio track (it
   * wraps this stream's existing track), this is now the ONLY place
   * that should ever call track.stop() — calling it here correctly
   * turns off both the camera and mic indicators together when the
   * interview ends.
   */
  release() {
    if (this.localStream) {
      this.localStream.getTracks().forEach((track) => track.stop());
      this.localStream = null;
    }
    if (this.videoElement) {
      this.videoElement.srcObject = null;
    }
    console.log("[WebRTCClient] stream released");
  }
}

window.WebRTCClient = WebRTCClient;