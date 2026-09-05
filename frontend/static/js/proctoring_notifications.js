/**
 * Bottom-left live notification toast + persistent activity log for
 * the interview room. Two responsibilities:
 *   1. push() shows a brief auto-dismissing toast (optional)
 *   2. every push() also lands in a persistent, scrollable log list,
 *      opened via a small pill toggle in the bottom-left corner.
 *
 * Fed by two sources (both wired in interview_room.html):
 *   - ProctoringClient's raw client-side detections (via a new 'flag'
 *     event added to proctoring_client.js) — shown as quiet info entries
 *   - ProctoringClient's 'warning'/'terminated' events from the server
 *     — shown as prominent warning entries (in addition to the existing
 *     fullscreen overlay, which is untouched)
 */

const ProctoringNotifications = (() => {
  const TOAST_DURATION_MS = 5000;
  const MAX_LOG_ENTRIES = 200;

  let logEntries = [];
  let logOpen = false;
  let container = null;
  let toastStack = null;
  let logPanel = null;
  let logList = null;
  let logToggle = null;
  let logBadge = null;
  let unreadCount = 0;

  function init() {
    if (container) return; // idempotent

    container = document.createElement("div");
    container.id = "proctoring-notif-root";
    container.innerHTML = `
      <div id="proctoring-toast-stack"></div>
      <div id="proctoring-log-panel" class="collapsed">
        <div id="proctoring-log-list"></div>
      </div>
      <button id="proctoring-log-toggle" type="button">
        <span class="dot"></span>
        Activity Log
        <span id="proctoring-log-badge" class="badge hidden">0</span>
      </button>
    `;
    document.body.appendChild(container);

    toastStack = document.getElementById("proctoring-toast-stack");
    logPanel = document.getElementById("proctoring-log-panel");
    logList = document.getElementById("proctoring-log-list");
    logToggle = document.getElementById("proctoring-log-toggle");
    logBadge = document.getElementById("proctoring-log-badge");

    logToggle.addEventListener("click", () => {
      logOpen = !logOpen;
      logPanel.classList.toggle("collapsed", !logOpen);
      if (logOpen) {
        unreadCount = 0;
        updateBadge();
      }
    });

    injectStyles();
  }

  /**
   * push({ level: "info"|"warning", title, detail, showToast })
   * level "warning" = countable/server-issued events (tab_switch,
   * window_blur, no_face, multiple_faces, fullscreen_exit, copy_paste,
   * devtools_opened, plus server warning/terminated messages).
   * level "info" = logged-only signals (gaze_away) — showToast defaults
   * to false for these so the log doesn't spam toasts on a noisy
   * heuristic, per the earlier gaze-detection discussion.
   */
  function push({ level = "info", title, detail = "", showToast = level === "warning" }) {
    if (!container) init();

    const entry = { level, title, detail, timestamp: new Date() };

    logEntries.unshift(entry);
    if (logEntries.length > MAX_LOG_ENTRIES) logEntries.length = MAX_LOG_ENTRIES;
    renderLogEntry(entry);

    if (!logOpen) {
      unreadCount += 1;
      updateBadge();
    }

    if (showToast) renderToast(entry);
  }

  function renderToast(entry) {
    const toast = document.createElement("div");
    toast.className = `proctoring-toast ${entry.level}`;
    toast.innerHTML = `
      <span class="icon">${entry.level === "warning" ? "⚠️" : "ℹ️"}</span>
      <div class="text">
        <div class="title">${escapeHtml(entry.title)}</div>
        ${entry.detail ? `<div class="detail">${escapeHtml(entry.detail)}</div>` : ""}
      </div>
    `;
    toastStack.appendChild(toast);
    requestAnimationFrame(() => toast.classList.add("visible"));
    setTimeout(() => {
      toast.classList.remove("visible");
      toast.addEventListener("transitionend", () => toast.remove(), { once: true });
    }, TOAST_DURATION_MS);
  }

  function renderLogEntry(entry) {
    const row = document.createElement("div");
    row.className = `proctoring-log-row ${entry.level}`;
    row.innerHTML = `
      <span class="time">${formatTime(entry.timestamp)}</span>
      <span class="dot ${entry.level}"></span>
      <span class="row-title">${escapeHtml(entry.title)}</span>
      ${entry.detail ? `<span class="row-detail">${escapeHtml(entry.detail)}</span>` : ""}
    `;
    logList.insertBefore(row, logList.firstChild);
  }

  function updateBadge() {
    if (unreadCount > 0) {
      logBadge.textContent = unreadCount > 99 ? "99+" : String(unreadCount);
      logBadge.classList.remove("hidden");
    } else {
      logBadge.classList.add("hidden");
    }
  }

  function formatTime(date) {
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  function injectStyles() {
    if (document.getElementById("proctoring-notif-styles")) return;
    const style = document.createElement("style");
    style.id = "proctoring-notif-styles";
    style.textContent = `
      #proctoring-notif-root {
        position: fixed;
        left: 16px;
        bottom: 16px;
        z-index: 9999;
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        gap: 8px;
        font-family: 'Inter', system-ui, sans-serif;
        font-size: 13px;
      }
      #proctoring-toast-stack {
        display: flex;
        flex-direction: column-reverse;
        gap: 6px;
        margin-bottom: 4px;
        max-width: 300px;
      }
      .proctoring-toast {
        background: #1B1E24;
        border: 1px solid #2A2D34;
        color: #F5F3EF;
        border-radius: 6px;
        padding: 10px 12px;
        display: flex;
        gap: 8px;
        align-items: flex-start;
        box-shadow: 0 4px 16px rgba(0,0,0,0.35);
        opacity: 0;
        transform: translateY(8px);
        transition: opacity 0.25s ease, transform 0.25s ease;
      }
      .proctoring-toast.visible { opacity: 1; transform: translateY(0); }
      .proctoring-toast.warning { border-left: 3px solid #C9432B; }
      .proctoring-toast.info { border-left: 3px solid #5B8A76; }
      .proctoring-toast .title { font-weight: 500; font-size: 13px; }
      .proctoring-toast .detail { opacity: 0.7; margin-top: 2px; font-size: 12px; }

      #proctoring-log-toggle {
        background: #1B1E24;
        border: 1px solid #2A2D34;
        color: #F5F3EF;
        border-radius: 20px;
        padding: 8px 14px;
        display: flex;
        align-items: center;
        gap: 6px;
        cursor: pointer;
        font-size: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.25);
      }
      #proctoring-log-toggle:hover { border-color: #8A8F98; }
      #proctoring-log-toggle .dot {
        width: 6px; height: 6px; border-radius: 50%;
        background: #5B8A76; display: inline-block;
      }
      #proctoring-log-toggle .badge {
        background: #C9432B;
        border-radius: 10px;
        padding: 1px 6px;
        font-size: 11px;
      }
      #proctoring-log-toggle .badge.hidden { display: none; }

      #proctoring-log-panel {
        width: 300px;
        max-height: 320px;
        background: #14161A;
        border: 1px solid #2A2D34;
        border-radius: 8px;
        overflow-y: auto;
        transition: max-height 0.2s ease, opacity 0.2s ease;
      }
      #proctoring-log-panel.collapsed {
        max-height: 0;
        opacity: 0;
        overflow: hidden;
        pointer-events: none;
        border: none;
      }
      #proctoring-log-list { padding: 6px; }
      .proctoring-log-row {
        display: flex;
        gap: 8px;
        align-items: baseline;
        padding: 6px 8px;
        border-radius: 6px;
        color: #D5D8E0;
      }
      .proctoring-log-row:hover { background: rgba(255,255,255,0.03); }
      .proctoring-log-row .time { color: #8A8F98; font-size: 11px; flex-shrink: 0; }
      .proctoring-log-row .dot {
        width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0;
      }
      .proctoring-log-row .dot.warning { background: #C9432B; }
      .proctoring-log-row .dot.info { background: #5B8A76; }
      .proctoring-log-row .row-title { font-weight: 500; font-size: 12.5px; }
      .proctoring-log-row .row-detail { color: #8A8F98; font-size: 11.5px; }
    `;
    document.head.appendChild(style);
  }

  return { init, push };
})();

document.addEventListener("DOMContentLoaded", () => ProctoringNotifications.init());