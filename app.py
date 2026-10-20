"""Smart Human Detection System - Streamlit dashboard.

Supports real-time webcam detection, single image upload detection, and video upload detection.
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
from media_processor import process_image, process_video

REFRESH_SECONDS = 0.05  # UI refresh cadence while the camera is live
BASE_DIR = Path(__file__).resolve().parent
CAPTURE_DIR = BASE_DIR / "captures"
TEMP_DIR = BASE_DIR / "temp_media"
SAMPLE_BUS_IMAGE = BASE_DIR / "tests" / "data" / "bus.jpg"
SAMPLE_VIDEO = BASE_DIR / "tests" / "data" / "people_pan.avi"

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
    .stButton > button[kind="primary"] { background: #4ecda5; color: #0e1420; border: none; font-weight: 700; }
    .stButton > button[kind="primary"]:hover { background: #3fbb94; color: #0e1420; }

    .hd-card {
        background: #161d2b; border: 1px solid #232c3d; border-radius: 14px;
        padding: 1.2rem 1.4rem; margin-bottom: 1rem;
    }
    .hd-errorbox {
        background: rgba(255,107,107,.08); border: 1px solid #ff6b6b;
        color: #ffb3b3; border-radius: 12px; padding: .9rem 1.1rem; font-size: .92rem;
        margin-bottom: 1rem;
    }
    .hd-successbox {
        background: rgba(78,205,165,.08); border: 1px solid #4ecda5;
        color: #4ecda5; border-radius: 12px; padding: .9rem 1.1rem; font-size: .92rem;
        margin-bottom: 1rem;
    }
    .hd-hint {
        background: rgba(139,168,255,.07); border: 1px solid #3d4c6b;
        color: #a9b8cf; border-radius: 12px; padding: .9rem 1.1rem; font-size: .9rem;
        margin-bottom: 1rem;
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

    # Media detection state
    ss.setdefault("image_bytes", None)
    ss.setdefault("image_name", None)
    ss.setdefault("image_result", None)

    ss.setdefault("video_bytes", None)
    ss.setdefault("video_name", None)
    ss.setdefault("video_path", None)
    ss.setdefault("video_result", None)


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
    existing = st.session_state.get("worker")
    if existing is not None and getattr(existing, "active", False):
        return
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

with header_left:
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
        min_value=0.05,
        max_value=0.95,
        value=float(st.session_state.get("confidence", 0.40)),
        step=0.05,
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
        if (active := (st.session_state.get("worker") or None)) and st.session_state["worker"].active:
            st.warning("Stop the camera before switching models.")
            model_choice = st.session_state["model_name"]
        else:
            st.session_state["model_name"] = model_choice

    st.markdown("### 🎯 Detection mode")
    source_kind = st.radio(
        "Select Mode",
        ["Webcam", "Image file", "Video file"],
        index=["Webcam", "Image file", "Video file"].index(st.session_state["source_kind"]),
        horizontal=False,
    )

    # When switching away from Webcam, stop camera worker automatically
    if source_kind != st.session_state["source_kind"]:
        if source_kind != "Webcam":
            stop_worker()
        st.session_state["source_kind"] = source_kind
        st.rerun()

    if source_kind == "Webcam":
        st.session_state["camera_index"] = st.selectbox(
            "Camera index",
            options=[0, 1, 2, 3],
            index=int(st.session_state.get("camera_index", 0)),
            help="0 is usually the built-in laptop webcam.",
        )
        st.markdown("---")
        st.markdown("### ℹ️ How to use (Webcam)")
        st.markdown(
            "1. Pick a model & confidence.\n"
            "2. Press **Start camera**.\n"
            "3. People get boxed + counted live.\n"
            "4. Press **Capture photo** to save a frame.\n"
            "5. Press **Stop camera** to release webcam."
        )
    elif source_kind == "Image file":
        st.markdown("---")
        st.markdown("### ℹ️ How to use (Image)")
        st.markdown(
            "1. Upload a JPG, PNG, or WEBP image.\n"
            "2. Review the image preview.\n"
            "3. Click **Detect Humans**.\n"
            "4. View bounding boxes and person count.\n"
            "5. Download the annotated image."
        )
    elif source_kind == "Video file":
        st.markdown("---")
        st.markdown("### ℹ️ How to use (Video)")
        st.markdown(
            "1. Upload an MP4, AVI, or MOV video.\n"
            "2. Review the video preview.\n"
            "3. Click **Detect Humans in Video**.\n"
            "4. Watch frame-by-frame progress.\n"
            "5. View & download the annotated video."
        )

# ------------------------------------------------------------------- load detector
detector = ensure_detector(st.session_state["model_name"])
detector_error = st.session_state.get("detector_error")
if detector is not None:
    detector.set_confidence(st.session_state["confidence"])

handle_pending_commands(detector)

worker: CameraWorker | None = st.session_state.get("worker")
snap = worker.snapshot() if worker is not None else None
worker_alive = bool(snap and snap["is_alive"])

# ------------------------------------------------------------------ notices
if detector_error:
    st.markdown(f'<div class="hd-errorbox">⚠️ {detector_error}</div>', unsafe_allow_html=True)
notice = st.session_state.pop("notice", None)
if notice:
    kind, text = notice
    (st.error if kind == "error" else st.success)(text)

# ==============================================================================
# MODE 1: WEBCAM
# ==============================================================================
if source_kind == "Webcam":
    # Control bar
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    with c1:
        if st.button("▶️ Start camera", type="primary", disabled=detector is None or worker_alive):
            st.session_state["cmd"] = "start"
            st.rerun()
    with c2:
        if st.button("⏹ Stop camera", disabled=not worker_alive):
            st.session_state["cmd"] = "stop"
            st.rerun()
    with c3:
        if st.button("📸 Capture photo", disabled=not worker_alive):
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

    friendly = (snap or {}).get("friendly_error") if snap else None
    raw_error = (snap or {}).get("error_message") if snap else None
    if friendly:
        st.markdown(f'<div class="hd-errorbox">⚠️ {friendly}</div>', unsafe_allow_html=True)
        if raw_error and raw_error != friendly:
            st.caption(f"Technical detail: {raw_error}")

    # Main monitoring
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

    # Statistics
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

    if worker_alive:
        time.sleep(REFRESH_SECONDS)
        st.rerun()

# ==============================================================================
# MODE 2: IMAGE FILE UPLOAD
# ==============================================================================
elif source_kind == "Image file":
    with header_right:
        st.markdown('<span class="hd-pill hd-live">🖼️ IMAGE MODE</span>', unsafe_allow_html=True)

    st.markdown("### 🖼️ Image Human Detection")
    st.markdown(
        '<p style="color:#8b98a9; margin-top:-0.5rem; margin-bottom:1.5rem;">'
        "Upload any photo to detect people, highlight them with bounding boxes, and count total humans."
        "</p>",
        unsafe_allow_html=True,
    )

    upload_col, info_col = st.columns([3, 1])

    with upload_col:
        uploaded_image = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            help="Supported formats: JPG, JPEG, PNG, BMP, WEBP",
            key="image_file_uploader",
        )

    with info_col:
        st.markdown("<div style='height: 1.8rem'></div>", unsafe_allow_html=True)
        if SAMPLE_BUS_IMAGE.exists():
            if st.button("🧪 Use Sample Photo", help="Load sample bus.jpg containing pedestrians"):
                st.session_state["image_bytes"] = SAMPLE_BUS_IMAGE.read_bytes()
                st.session_state["image_name"] = "bus.jpg"
                st.session_state["image_result"] = None
                st.rerun()

    if uploaded_image is not None:
        if st.session_state.get("image_name") != uploaded_image.name:
            st.session_state["image_bytes"] = uploaded_image.getvalue()
            st.session_state["image_name"] = uploaded_image.name
            st.session_state["image_result"] = None

    active_image_bytes = st.session_state.get("image_bytes")
    active_image_name = st.session_state.get("image_name")

    if active_image_bytes:
        btn_col, clear_col, _ = st.columns([2, 1, 3])
        with btn_col:
            detect_clicked = st.button("🔍 Detect Humans", type="primary", disabled=detector is None)
        with clear_col:
            if st.button("🗑️ Clear"):
                st.session_state["image_bytes"] = None
                st.session_state["image_name"] = None
                st.session_state["image_result"] = None
                st.rerun()

        if detect_clicked and detector is not None:
            with st.spinner("Processing image with YOLO..."):
                try:
                    res = process_image(
                        detector=detector,
                        image_input=active_image_bytes,
                        confidence=st.session_state["confidence"],
                    )
                    st.session_state["image_result"] = res
                except Exception as exc:
                    st.session_state["image_result"] = None
                    st.markdown(
                        f'<div class="hd-errorbox">⚠️ Detection failed: {exc}</div>',
                        unsafe_allow_html=True,
                    )

        result = st.session_state.get("image_result")

        # Metric cards if result is available
        if result is not None:
            people_count = result["people"]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Humans Detected", people_count)
            m2.metric("Inference Time", f"{result['inference_ms']:.0f} ms")
            m3.metric("Confidence Used", f"{st.session_state['confidence']:.2f}")
            m4.metric("Resolution", f"{result['resolution'][0]}x{result['resolution'][1]}")

            if people_count > 0:
                st.markdown(
                    f'<div class="hd-successbox">✅ Successfully detected <b>{people_count}</b> human(s) in <code>{active_image_name}</code>.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="hd-hint">ℹ️ No humans detected with confidence ≥ {st.session_state["confidence"]:.2f}. Try lowering the confidence slider in the sidebar.</div>',
                    unsafe_allow_html=True,
                )

            # Display images side by side or main
            col_orig, col_annot = st.columns(2)
            with col_orig:
                st.markdown("#### Original Image")
                st.image(result["original_rgb"], caption=f"Original: {active_image_name}", width="stretch")
            with col_annot:
                st.markdown("#### Detected Result")
                st.image(result["annotated_rgb"], caption=f"Annotated ({people_count} people)", width="stretch")

            # Download button
            if result.get("annotated_bytes"):
                dl_name = f"detected_{Path(active_image_name).stem}.jpg"
                st.download_button(
                    label="⬇️ Download Annotated Image",
                    data=result["annotated_bytes"],
                    file_name=dl_name,
                    mime="image/jpeg",
                    key="download_annotated_img",
                )

            # Technical details
            if result.get("boxes"):
                with st.expander("📋 Detection details (bounding boxes)", expanded=False):
                    for i, (box, conf) in enumerate(zip(result["boxes"], result["confidences"]), start=1):
                        st.markdown(f"- **Person {i}:** Confidence `{conf * 100:.1f}%` — Bounding Box `(x1={box[0]}, y1={box[1]}, x2={box[2]}, y2={box[3]})`")

        else:
            # Preview before detection
            st.markdown("#### Image Preview")
            try:
                arr = np.frombuffer(active_image_bytes, np.uint8)
                preview_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if preview_bgr is not None:
                    preview_rgb = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB)
                    st.image(preview_rgb, caption=f"Selected: {active_image_name}", width="stretch")
                    st.caption("Press **Detect Humans** above to run the YOLO model.")
                else:
                    st.markdown(
                        '<div class="hd-errorbox">⚠️ Could not decode this image file. Please upload a valid JPG, PNG, BMP, or WEBP image.</div>',
                        unsafe_allow_html=True,
                    )
            except Exception as exc:
                st.markdown(
                    f'<div class="hd-errorbox">⚠️ Error loading preview: {exc}</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.markdown(
            '<div class="hd-frame" style="padding:4.5rem 2rem;">'
            "<p style='color:#8b98a9; font-size:1.05rem; margin:0;'>"
            "No image selected. Drag and drop a photo or click <b>Browse files</b> above."
            "</p></div>",
            unsafe_allow_html=True,
        )

# ==============================================================================
# MODE 3: VIDEO FILE UPLOAD
# ==============================================================================
elif source_kind == "Video file":
    with header_right:
        st.markdown('<span class="hd-pill hd-live">🎬 VIDEO MODE</span>', unsafe_allow_html=True)

    st.markdown("### 🎬 Video Human Detection")
    st.markdown(
        '<p style="color:#8b98a9; margin-top:-0.5rem; margin-bottom:1.5rem;">'
        "Upload a video file to detect humans frame-by-frame, watch live detections, and download the annotated video."
        "</p>",
        unsafe_allow_html=True,
    )

    upload_col, info_col = st.columns([3, 1])

    with upload_col:
        uploaded_video = st.file_uploader(
            "Choose a video file",
            type=["mp4", "avi", "mov", "mkv", "mjpeg"],
            help="Supported formats: MP4, AVI, MOV, MKV, MJPEG",
            key="video_file_uploader",
        )

    with info_col:
        st.markdown("<div style='height: 1.8rem'></div>", unsafe_allow_html=True)
        if SAMPLE_VIDEO.exists():
            if st.button("🧪 Use Sample Video", help="Load sample people_pan.avi"):
                TEMP_DIR.mkdir(parents=True, exist_ok=True)
                sample_copy = TEMP_DIR / "sample_people_pan.avi"
                sample_copy.write_bytes(SAMPLE_VIDEO.read_bytes())
                st.session_state["video_path"] = str(sample_copy)
                st.session_state["video_name"] = "people_pan.avi"
                st.session_state["video_bytes"] = sample_copy.read_bytes()
                st.session_state["video_result"] = None
                st.rerun()

    if uploaded_video is not None:
        if st.session_state.get("video_name") != uploaded_video.name:
            TEMP_DIR.mkdir(parents=True, exist_ok=True)
            saved_path = TEMP_DIR / f"upload_{uploaded_video.name}"
            saved_path.write_bytes(uploaded_video.getvalue())
            st.session_state["video_path"] = str(saved_path)
            st.session_state["video_name"] = uploaded_video.name
            st.session_state["video_bytes"] = uploaded_video.getvalue()
            st.session_state["video_result"] = None

    active_video_path = st.session_state.get("video_path")
    active_video_name = st.session_state.get("video_name")
    active_video_bytes = st.session_state.get("video_bytes")

    if active_video_path and Path(active_video_path).exists():
        st.markdown(f"**Selected Video:** `{active_video_name}` ({Path(active_video_path).stat().st_size / (1024 * 1024):.1f} MB)")

        btn_col, max_f_col, clear_col = st.columns([2, 2, 1])
        with btn_col:
            detect_video_clicked = st.button("🎬 Detect Humans in Video", type="primary", disabled=detector is None)
        with max_f_col:
            max_frames_choice = st.selectbox(
                "Frames to process",
                options=[0, 30, 60, 120, 300],
                format_func=lambda x: "All frames" if x == 0 else f"First {x} frames (faster)",
                index=0,
                help="Limit frames for faster processing on laptop CPUs.",
            )
        with clear_col:
            if st.button("🗑️ Clear", key="clear_video_btn"):
                st.session_state["video_path"] = None
                st.session_state["video_name"] = None
                st.session_state["video_bytes"] = None
                st.session_state["video_result"] = None
                st.rerun()

        # Processing loop when user clicks the button
        if detect_video_clicked and detector is not None:
            st.markdown("#### ⏳ Processing Video...")
            prog_bar = st.progress(0.0)
            status_txt = st.empty()
            live_frame_placeholder = st.empty()

            def on_progress(cur: int, total: int, res, fps_val: float):
                pct = min(max(cur / max(total, 1), 0.0), 1.0)
                prog_bar.progress(pct)
                status_txt.markdown(
                    f"**Processing frame {cur} of {total}** ({pct * 100:.0f}%) · "
                    f"People in frame: `{res.people}` · Speed: `{fps_val:.1f} FPS`"
                )
                if cur % 2 == 1 or cur == total:
                    rgb = cv2.cvtColor(res.frame, cv2.COLOR_BGR2RGB)
                    live_frame_placeholder.image(
                        rgb,
                        caption=f"Frame {cur} ({res.people} people detected)",
                        width="stretch",
                    )

            try:
                out_path = TEMP_DIR / f"annotated_{Path(active_video_name).stem}.mp4"
                res = process_video(
                    detector=detector,
                    video_input=active_video_path,
                    output_path=out_path,
                    confidence=st.session_state["confidence"],
                    max_frames=max_frames_choice if max_frames_choice > 0 else None,
                    progress_callback=on_progress,
                )
                st.session_state["video_result"] = res
                prog_bar.progress(1.0)
                status_txt.markdown("**✅ Video processing completed!**")
                time.sleep(0.5)
                st.rerun()
            except Exception as exc:
                st.markdown(
                    f'<div class="hd-errorbox">⚠️ Video processing failed: {exc}</div>',
                    unsafe_allow_html=True,
                )

        video_result = st.session_state.get("video_result")

        # Results display
        if video_result is not None:
            st.markdown(
                f'<div class="hd-successbox">🎉 Completed processing <b>{video_result["frames_processed"]}</b> frames in {video_result["elapsed_seconds"]:.1f}s.</div>',
                unsafe_allow_html=True,
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Peak People", video_result["people_max"])
            m2.metric("Frames Processed", video_result["frames_processed"])
            pct_people = (video_result["frames_with_people"] / max(video_result["frames_processed"], 1)) * 100
            m3.metric("Frames with People", f"{pct_people:.0f}%")
            m4.metric("Processing Speed", f"{video_result['processing_fps']:.1f} FPS")

            # Video playback
            st.markdown("#### Annotated Video Result")
            output_file_path = video_result.get("output_path")

            if output_file_path and Path(output_file_path).exists():
                try:
                    st.video(output_file_path)
                except Exception:
                    st.info("Browser preview not supported for this video container. Please download the file below to view.")

                if video_result.get("output_bytes"):
                    dl_name = f"detected_{Path(active_video_name).stem}.mp4"
                    st.download_button(
                        label="⬇️ Download Processed Video",
                        data=video_result["output_bytes"],
                        file_name=dl_name,
                        mime="video/mp4",
                        key="download_annotated_video",
                    )

            # Key frames gallery
            key_frames = video_result.get("key_frames", [])
            if key_frames:
                with st.expander(f"📸 Key frames with detections ({len(key_frames)} snapshots)", expanded=True):
                    cols = st.columns(min(len(key_frames), 3))
                    for i, (f_idx, count, rgb_frame) in enumerate(key_frames[:6]):
                        cols[i % len(cols)].image(
                            rgb_frame,
                            caption=f"Frame #{f_idx} · {count} people",
                            width="stretch",
                        )

        else:
            # Video preview before detection
            st.markdown("#### Video Preview")
            try:
                st.video(active_video_path)
                st.caption("Press **Detect Humans in Video** above to process the full video.")
            except Exception as exc:
                st.caption(f"Note: Direct preview not supported for this container ({exc}). You can still click 'Detect Humans in Video'.")

    else:
        st.markdown(
            '<div class="hd-frame" style="padding:4.5rem 2rem;">'
            "<p style='color:#8b98a9; font-size:1.05rem; margin:0;'>"
            "No video selected. Drag and drop a video file or click <b>Browse files</b> above."
            "</p></div>",
            unsafe_allow_html=True,
        )