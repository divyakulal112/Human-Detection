"""Smart Human Detection System - Streamlit dashboard.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from camera_worker import CameraWorker, STATUS_ERROR, STATUS_RUNNING, STATUS_STARTING
from detector import AVAILABLE_MODELS, PersonDetector

REFRESH_SECONDS = 0.05  # UI refresh cadence while the camera is live
BASE_DIR = Path(__file__).resolve().parent
CAPTURE_DIR = BASE_DIR / "captures"

st.set_page_config(
    page_title="Smart Human Detection",
    page_icon="🧍",
    
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .stApp { background: #0e1420; }
    html, body, [class*="css"] { color: #e6edf3; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }

    h1, h2, h3 { color: #e6edf3 !important; }

    .hd-title { font-size: 2.1rem; font-weight: 700; margin: 0; letter-spacing: .5px; }
    .hd-sub { color: #8b98a9; font-size: .95rem; margin-top: .25rem; }
    .hd-pill { display: inline-block; padding: .3rem .8rem; border-radius: 999px;
               font-weight: 600; font-size: .82rem; letter-spacing: 1px; }
    .hd-live    { background: rgba(78,205,165,.15); color: #4ecda5; border: 1px solid #4ecda5; }
    .hd-stopped { background: rgba(139,152,169,.15); color: #8b98a9; border: 1px solid #8b98a9; }
    .hd-error   { background: rgba(255,107,107,.15); color: #ff6b6b; border: 1px solid #ff6b6b; }
    .hd-busy    { background: rgba(255,209,102,.15); color: #ffd166; border: 1px solid #ffd166; }

    div[data-testid="stMetric"] {
        background: #161d2b; border: 1px solid #232c3d; border-radius: 14px;
        padding: 14px 18px;
    }
    div[data-testid="stMetricLabel"] { color: #8b98a9; text-transform: uppercase;
        font-size: .72rem; letter-spacing: 1.2px; }
    div[data-testid="stMetricValue"] { color: #4ecda5; font-weight: 700; }

    .stButton > button {
        background: #1b2434; color: #e6edf3; border: 1px solid #2c3850;
        border-radius: 10px; padding: .5rem 1.4rem; font-weight: 600; width: 100%;
        transition: all .15s ease;
    }
    .stButton > button:hover { border-color: #4ecda5; color: #4ecda5; }
    .stButton > button[kind="primary"] { background: #4ecda5; color: #0e1420; border: none; }
    .stButton > button[kind="primary"]:hover { background: #3fbb94; color: #0e1420; }

    .hd-errorbox {
        background: rgba(255,107,107,.08); border: 1px solid #ff6b6b;
        color: #ffb3b3; border-radius: 12px; padding: .9rem 1.1rem; font-size: .92rem;
    }
    .hd-hint {
        background: rgba(139,168,255,.07); border: 1px solid #3d4c6b;
        color: #a9b8cf; border-radius: 12px; padding: .9rem 1.1rem; font-size: .9rem;
    }
    .hd-frame {
        background: #0a0f17; border: 1px solid #232c3d; border-radius: 16px;
        overflow: hidden; text-align: center;
    }
    footer { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------- session state
def init_state() -> None:
    ss = st.session_state
    ss.setdefault("detector", None)
    ss.setdefault("detector_error", None)
    ss.setdefault("worker", None)
    ss.setdefault("model_name", "yolov8n.pt")
    ss.setdefault("source_kind", "Webcam")
    ss.setdefault("last_capture", None)
    ss.setdefault("session_started", time.time())
    ss.setdefault("frames_captured", 0)


def ensure_detector(model_name: str) -> PersonDetector | None:
    """(Re)create + load the detector only when the model choice changes."""
    ss = st.session_state
    detector: PersonDetector | None = ss.get("detector")
    if detector is not None and detector.model_name == model_name and detector.loaded:
        return detector
    if detector is not None and ss.get("loaded_model") == model_name and detector.loaded:
        return detector

    detector = PersonDetector(model_name=model_name, confidence=ss.get("confidence", 0.4))
    with st.spinner(f"Loading {model_name} (downloads ~6 MB on first run)..."):
        ok = detector.load()
    ss["detector"] = detector
    ss["loaded_model"] = model_name if ok else None
    ss["detector_error"] = detector.error
    return detector if ok else None


def stop_worker() -> None:
    worker = st.session_state.get("worker")
    if worker is not None:
        worker.stop(timeout=3.0)
    st.session_state["worker"] = None


def start_worker(detector: PersonDetector, kind: str, value: str) -> None:
    # Prevent starting a new worker if one is already active.
    existing = st.session_state.get("worker")
    if existing is not None and getattr(existing, "active", False):
        # Worker already running – nothing to do.
        return
    # Clean up any stale worker before starting a new one.
    stop_worker()
    source = "webcam" if kind == "Webcam" else value
    worker = CameraWorker(
        detector=detector,
        source=source,
        camera_index=int(st.session_state.get("camera_index", 0)),
        save_dir=CAPTURE_DIR,
    )
    st.session_state["worker"] = worker
    worker.start()


def handle_pending_commands(detector: PersonDetector | None) -> None:
    ss = st.session_state
    cmd = ss.pop("cmd", None)
    if cmd:
        print(f"[app] processing cmd={cmd!r}", flush=True)
    if cmd == "start" and detector is not None:
        kind = ss.get("source_kind", "Webcam")
        value = ss.get("media_path", "")
        if kind != "Webcam" and not value:
            ss["notice"] = ("error", "Choose an image or video file first.")
            return
        start_worker(detector, kind, value)
    elif cmd == "stop":
        stop_worker()
    elif cmd == "capture":
        worker = ss.get("worker")
        if worker is None or worker.latest_frame() is None:
            ss["notice"] = ("error", "Start the camera before capturing an image.")
        else:
            path = worker.save_capture()
            if path:
                ss["frames_captured"] += 1
                ss["last_capture"] = path
                ss["notice"] = ("success", f"Photo saved: {path}")
            else:
                ss["notice"] = ("error", "Could not encode the frame.")


def status_pill(status: str) -> str:
    if status == "live":
        return '<span class="hd-pill hd-live">● LIVE</span>'
    if status == "starting":
        return '<span class="hd-pill hd-busy">◔ STARTING</span>'
    if status == "error":
        return '<span class="hd-pill hd-error">✕ ERROR</span>'
    return '<span class="hd-pill hd-stopped">■ STOPPED</span>'


def fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def idle_panel(container) -> None:
    container.markdown(
        '<div class="hd-frame" style="padding:5.5rem 2rem;">'
        "<p style='color:#8b98a9; font-size:1.05rem; margin:0;'>"
        "Camera is off. Press <b>Start camera</b> to begin detection."
        "</p></div>",
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------------ layout
init_state()

header_left, header_right = st.columns([4, 1], vertical_alignment="top")

st.markdown(
    '<p class="hd-title">Smart Human Detection System</p>'
    '<p class="hd-sub">Real-time person detection with YOLO · powered by OpenCV + Streamlit</p>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### ⚙️ Detection settings")
    st.session_state["confidence"] = st.slider(
        "Confidence threshold",
        min_value=0.05, max_value=0.95,
        value=float(st.session_state.get("confidence", 0.40)), step=0.05,
        help="Lower = more detections (but more false alarms).",
    )
    model_choice = st.selectbox(
        "YOLO model",
        options=list(AVAILABLE_MODELS.keys()),
        format_func=lambda k: AVAILABLE_MODELS[k],
        index=list(AVAILABLE_MODELS).index(st.session_state["model_name"]),
        help="Smaller models are faster on a laptop CPU.",
    )
    if model_choice != st.session_state["model_name"]:
        if active := (st.session_state.get("worker") or None) and st.session_state["worker"].active:
            st.warning("Stop the camera before switching models.")
            model_choice = st.session_state["model_name"]
        else:
            st.session_state["model_name"] = model_choice

    st.markdown("### 📹 Video source")
    source_kind = st.radio(
        "Source",
        ["Webcam", "Image file", "Video file"],
        index=["Webcam", "Image file", "Video file"].index(st.session_state["source_kind"]),
        horizontal=True,
    )
    st.session_state["source_kind"] = source_kind

    if source_kind == "Webcam":
        st.session_state["camera_index"] = st.selectbox(
            "Camera index", options=[0, 1, 2, 3],
            index=int(st.session_state.get("camera_index", 0)),
            help="0 is usually the built-in laptop webcam.",
        )
    else:
        accepted = "jpg jpeg png bmp webp" if source_kind == "Image file" else "mp4 avi mov mkv mjpeg"
        uploaded = st.file_uploader("Choose a file", type=accepted.split())
        if uploaded is not None:
            tmp_dir = BASE_DIR / "temp_media"
            tmp_dir.mkdir(exist_ok=True)
            tmp_path = tmp_dir / uploaded.name
            tmp_path.write_bytes(uploaded.getbuffer())
            st.session_state["media_path"] = str(tmp_path)
        elif "media_path" not in st.session_state:
            st.info("No file chosen yet. A source file is required to start.")
            st.session_state["media_path"] = ""

    st.markdown("---")
    st.markdown("### ℹ️ How to use")
    st.markdown(
        "1. Pick a model & confidence.\n"
        "2. Press **Start camera**.\n"
        "3. People get boxed + counted live.\n"
        "4. Press **Capture photo** to save a frame.\n"
        "5. Press **Stop camera** to release the webcam."
    )
    st.caption(
        "Camera stays exclusive to this app while streaming — "
        "click **Stop camera** before reopening it in Zoom/Teams."
    )

# ------------------------------------------------------------------- commands
detector = ensure_detector(st.session_state["model_name"])
detector_error = st.session_state.get("detector_error")
if detector is not None:
    detector.set_confidence(st.session_state["confidence"])

handle_pending_commands(detector)

worker: CameraWorker | None = st.session_state.get("worker")
snap = worker.snapshot() if worker is not None else None

# ------------------------------------------------------------------ control bar
c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
worker_alive = bool(snap and snap["is_alive"])
with c1:
    if st.button("▶️ Start camera", type="primary", disabled=detector is None or worker_alive):
        print("[app] Start button pressed", flush=True)
        st.session_state["cmd"] = "start"
        st.rerun()
with c2:
    if st.button("⏹ Stop camera", disabled=not worker_alive):
        print("[app] Stop button pressed", flush=True)
        st.session_state["cmd"] = "stop"
        st.rerun()
with c3:
    if st.button("📸 Capture photo", disabled=not worker_alive):
        print("[app] Capture button pressed", flush=True)
        st.session_state["cmd"] = "capture"
        st.rerun()
with c4:
    status_text = (snap or {}).get("status", "stopped") if detector_error is None else "error"
    pill_key = {"running": "live", "starting": "starting", "error": "error"}.get(status_text, "stopped")
    st.markdown(
        f"<div style='padding-top:.55rem'>{status_pill(pill_key)}</div>",
        unsafe_allow_html=True,
    )
with header_right:
    st.markdown(status_pill(pill_key), unsafe_allow_html=True)

# ------------------------------------------------------------------ notices
if detector_error:
    st.markdown(f'<div class="hd-errorbox">⚠️ {detector_error}</div>', unsafe_allow_html=True)
notice = st.session_state.pop("notice", None)
if notice:
    kind, text = notice
    (st.error if kind == "error" else st.success)(text)

friendly = (snap or {}).get("friendly_error") if snap else None
raw_error = (snap or {}).get("error_message") if snap else None
if friendly:
    st.markdown(f'<div class="hd-errorbox">⚠️ {friendly}</div>', unsafe_allow_html=True)
    if raw_error and raw_error != friendly:
        st.caption(f"Technical detail: {raw_error}")

# ---------------------------------------------------------------- main monitoring
main_col, side_col = st.columns([3, 1], gap="large")

with main_col:
    st.markdown("#### Live feed")
    frame_placeholder = st.empty()
    if worker_alive and worker is not None:
        frame = worker.latest_frame()
        if frame is not None:
            frame_placeholder.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                channels="RGB",
                width="stretch",
            )
        else:
            frame_placeholder.markdown(
                '<div class="hd-frame" style="padding:4.5rem 2rem;">'
                "<p style='color:#8b98a9;margin:0;'>Waiting for the first frame…</p></div>",
                unsafe_allow_html=True,
            )
    else:
        idle_panel(frame_placeholder)

with side_col:
    st.markdown("#### Live metrics")
    snap = snap or {}
    people = snap.get("people_now", 0)
    st.metric("People now", people)
    m1, m2 = st.columns(2)
    m1.metric("Peak people", snap.get("people_max", 0))
    m2.metric("Frames", snap.get("frame_count", 0))
    fps = snap.get("fps", 0.0)
    st.metric("Processing FPS", f"{fps:.1f}" if fps else "—")
    inference = snap.get("avg_inference_ms", 0.0)
    st.metric("Inference", f"{inference:.0f} ms" if inference else "—")

st.markdown("---")

# ---------------------------------------------------------------- statistics
st.markdown("#### Session statistics")
s1, s2, s3, s4 = st.columns(4)
frames = snap.get("frame_count", 0) if snap else 0
with_people = snap.get("frames_with_people", 0) if snap else 0
total_people = snap.get("total_people", 0) if snap else 0
session_sec = snap.get("session_seconds", 0.0) if snap else max(0, time.time() - st.session_state["session_started"])

occupancy = (with_people / frames * 100) if frames else 0.0
s1.metric("Session time", fmt_duration(session_sec))
s2.metric("Frames processed", frames)
s3.metric("Frames with people", f"{occupancy:.0f}%")
s4.metric("Photos saved this session", st.session_state["frames_captured"])

info = snap or {}
with st.expander("🔎 Technical details & session info", expanded=False):
    d1, d2 = st.columns(2)
    d1.markdown(
        f"- **Model:** `{st.session_state['model_name']}`\n"
        f"- **Confidence:** {st.session_state['confidence']:.2f}\n"
        f"- **Source:** {st.session_state['source_kind']}\n"
        f"- **Requested camera index:** {st.session_state.get('camera_index', '—')}"
    )
    d2.markdown(
        f"- **Resolution:** {info.get('resolution') or '—'}\n"
        f"- **Camera index used:** {info.get('used_camera_index') if info.get('used_camera_index') is not None else '—'}\n"
        f"- **Worker alive:** {'yes' if info.get('is_alive') else 'no'}\n"
        f"- **Status:** {info.get('status', 'stopped')}\n"
        f"- **Total people counted:** {total_people}\n"
    )
    if st.session_state.get("last_capture"):
        st.markdown(f"- **Last photo:** `{st.session_state['last_capture']}`")

if not worker_alive:
    with st.expander("📁 Saved captures", expanded=False):
        saved = sorted(CAPTURE_DIR.glob("*.jpg")) if CAPTURE_DIR.exists() else []
        if saved:
            cols = st.columns(4)
            for i, p in enumerate(saved[-8:]):
                cols[i % 4].image(str(p), width="stretch", caption=p.name)
        else:
            st.info("No photos saved yet. Start the camera and press Capture photo.")

# --------------------------------------------------------------------- auto-refresh
if worker_alive:
    time.sleep(REFRESH_SECONDS)
    st.rerun()