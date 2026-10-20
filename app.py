
"""Smart Human Detection System — AI Dashboard.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from camera_worker import (
    CameraWorker,
    get_active_webcam_worker,
    diag_log,
)
from detector import AVAILABLE_MODELS, PersonDetector
from media_processor import process_image, process_video


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Vision AI | Human Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
CAPTURE_DIR = BASE_DIR / "captures"
TEMP_DIR = BASE_DIR / "temp_media"
SAMPLE_BUS_IMAGE = BASE_DIR / "tests" / "data" / "bus.jpg"
SAMPLE_VIDEO = BASE_DIR / "tests" / "data" / "people_pan.avi"

REFRESH_SECONDS = 0.15


# ============================================================
# PREMIUM DASHBOARD DESIGN
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --bg: #0a100d;
        --panel: #111a15;
        --panel2: #16221a;
        --border: #293b30;
        --text: #eef6f0;
        --muted: #91a69a;
        --green: #42df98;
        --green-dark: #16875a;
    }

    .stApp {
        background: var(--bg);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(10, 16, 13, 0.95);
    }

    [data-testid="stSidebar"] {
        background: #0e1712;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.4rem;
    }

    .block-container {
        max-width: 1600px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4 {
        color: var(--text) !important;
        letter-spacing: -0.45px;
    }

    p, label, li {
        color: #c4d3c9;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #16221a, #101813);
        border: 1px solid var(--border);
        padding: 18px;
        border-radius: 15px;
        min-height: 118px;
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted) !important;
    }

    [data-testid="stMetricValue"] {
        color: var(--green) !important;
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 10px;
        min-height: 42px;
        font-weight: 600;
        transition: all 0.18s ease;
    }

    .stButton > button[kind="primary"] {
        background: var(--green-dark);
        color: white;
        border: 1px solid var(--green);
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        border-color: var(--green);
        color: var(--green);
    }

    [data-testid="stFileUploader"] {
        background: var(--panel);
        border: 1px dashed #42644f;
        border-radius: 14px;
        padding: 12px;
    }

    [data-testid="stImage"] img {
        border-radius: 12px;
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    [data-testid="stExpander"] {
        border: 1px solid var(--border);
        border-radius: 12px;
    }

    hr {
        border-color: var(--border);
    }

    .brand {
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: 1.2px;
        color: #f3fff7;
    }

    .brand span {
        color: var(--green);
    }

    .muted {
        color: var(--muted);
        font-size: 0.88rem;
    }

    .eyebrow {
        color: var(--green);
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .hero {
        background: linear-gradient(125deg, #16281d, #101813 70%);
        border: 1px solid var(--border);
        padding: 26px;
        border-radius: 18px;
        margin-bottom: 22px;
    }

    .hero h1 {
        margin: 8px 0;
        font-size: 2.1rem;
    }

    .hero p {
        color: var(--muted);
        margin-bottom: 0;
    }

    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .preview-empty {
        background: #0d1510;
        border: 1px dashed #34513d;
        border-radius: 16px;
        padding: 65px 20px;
        text-align: center;
    }

    .preview-empty .symbol {
        font-size: 2.4rem;
        color: var(--green);
    }

    .preview-empty h3 {
        margin-top: 12px;
    }

    .preview-empty p {
        color: var(--muted);
    }

    .status {
        display: inline-block;
        border: 1px solid #34513d;
        background: #17271d;
        color: var(--green);
        border-radius: 999px;
        padding: 6px 12px;
        font-size: 0.76rem;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

def init_state() -> None:
    ss = st.session_state

    defaults = {
        "detector": None,
        "detector_error": None,
        "loaded_model": None,
        "worker": None,
        "model_name": "yolov8n.pt",
        "source_kind": "Webcam",
        "confidence": 0.40,
        "camera_index": 0,
        "last_capture": None,
        "session_started": time.time(),
        "frames_captured": 0,
        "image_bytes": None,
        "image_name": None,
        "image_result": None,
        "video_bytes": None,
        "video_name": None,
        "video_path": None,
        "video_result": None,
        "notice": None,
        "cmd": None,
        "navigation": "Dashboard",
    }

    for key, value in defaults.items():
        ss.setdefault(key, value)

    if ss.get("worker") is None:
        active = get_active_webcam_worker()
        if active is not None and active.active:
            ss["worker"] = active


init_state()


# ============================================================
# DETECTOR
# ============================================================

def ensure_detector(model_name: str) -> PersonDetector | None:
    ss = st.session_state
    detector = ss.get("detector")

    if (
        detector is not None
        and detector.model_name == model_name
        and detector.loaded
    ):
        return detector

    detector = PersonDetector(
        model_name=model_name,
        confidence=ss.get("confidence", 0.4),
    )

    with st.spinner(f"Loading {model_name} and warming up the AI model..."):
        try:
            ok = detector.load()
        except Exception as exc:
            ok = False
            ss["detector_error"] = str(exc)

    ss["detector"] = detector
    ss["loaded_model"] = model_name if ok else None
    ss["detector_error"] = None if ok else (
        getattr(detector, "error", None)
        or ss.get("detector_error")
        or f"Could not load model {model_name}."
    )

    return detector if ok else None


# ============================================================
# CAMERA WORKER
# ============================================================

def stop_worker(reason: str = "user_request") -> None:
    worker = st.session_state.get("worker")

    diag_log(
        f"[UI] stop_worker called: reason={reason}, "
        f"status={getattr(worker, 'status', None)}"
    )

    if worker is not None:
        worker.stop(timeout=3.0)

    st.session_state["worker"] = None


def start_worker(detector: PersonDetector) -> None:
    existing = st.session_state.get("worker")

    if existing is not None and (
        getattr(existing, "active", False)
        or getattr(existing, "is_alive", False)
    ):
        return

    active = get_active_webcam_worker()

    if active is not None and (
        getattr(active, "active", False)
        or getattr(active, "is_alive", False)
    ):
        st.session_state["worker"] = active
        return

    # Preserve the original camera worker and its threading logic.
    stop_worker(reason="start_worker_fallthrough")

    worker = CameraWorker(
        detector=detector,
        source="webcam",
        camera_index=int(st.session_state.get("camera_index", 0)),
        save_dir=CAPTURE_DIR,
    )

    st.session_state["worker"] = worker
    worker.start()


def handle_pending_commands(detector: PersonDetector | None) -> None:
    ss = st.session_state
    cmd = ss.pop("cmd", None)

    if cmd == "start":
        if detector is not None:
            start_worker(detector)

    elif cmd == "stop":
        stop_worker(reason="stop_button")

    elif cmd == "capture":
        worker = ss.get("worker")

        if worker is None or worker.latest_frame() is None:
            ss["notice"] = (
                "error",
                "Start the camera before capturing a photo.",
            )
        else:
            path = worker.save_capture()

            if path:
                ss["frames_captured"] += 1
                ss["last_capture"] = path
                ss["notice"] = ("success", f"Photo saved: {path}")
            else:
                ss["notice"] = ("error", "Could not save the current frame.")


# ============================================================
# REUSABLE UI
# ============================================================

def hero(eyebrow: str, heading: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">{eyebrow}</div>
            <h1>{heading}</h1>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def empty_preview(title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="preview-empty">
            <div class="symbol">⌁</div>
            <h3>{title}</h3>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def fmt_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def show_notice() -> None:
    notice = st.session_state.pop("notice", None)

    if notice:
        kind, message = notice
        if kind == "error":
            st.error(message)
        else:
            st.success(message)


def get_snapshot() -> dict:
    worker = st.session_state.get("worker")
    if worker is None:
        return {}
    try:
        return worker.snapshot() or {}
    except Exception as exc:
        diag_log(f"[UI] snapshot error: {exc}")
        return {}


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    st.markdown(
        '<div class="brand">VISION<span>AI</span></div>',
        unsafe_allow_html=True,
    )
    st.caption("SMART HUMAN DETECTION")
    st.divider()

    st.markdown("##### WORKSPACE")

    pages = [
        "Dashboard",
        "Live Detection",
        "Image Analysis",
        "Video Analysis",
        "Detection History",
        "Settings",
    ]

    current_page = st.session_state.get("navigation", "Dashboard")
    page_index = pages.index(current_page) if current_page in pages else 0

    page = st.radio(
        "Navigation",
        pages,
        index=page_index,
        label_visibility="collapsed",
        key="navigation_radio",
    )

    st.session_state["navigation"] = page

    st.divider()
    st.markdown("##### MODEL CONFIGURATION")

    model_names = list(AVAILABLE_MODELS.keys())
    current_model = st.session_state["model_name"]

    if current_model not in model_names:
        current_model = model_names[0]

    model_choice = st.selectbox(
        "YOLO model",
        options=model_names,
        index=model_names.index(current_model),
        format_func=lambda key: AVAILABLE_MODELS[key],
        key="model_selector",
    )

    worker_now = st.session_state.get("worker")
    worker_is_active = bool(
        worker_now is not None
        and (
            getattr(worker_now, "active", False)
            or getattr(worker_now, "is_alive", False)
        )
    )

    if model_choice != st.session_state["model_name"]:
        if worker_is_active:
            st.warning("Stop the webcam before changing models.")
        else:
            st.session_state["model_name"] = model_choice
            st.session_state["detector"] = None
            st.session_state["loaded_model"] = None
            st.session_state["detector_error"] = None
            st.rerun()

    st.session_state["confidence"] = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=float(st.session_state["confidence"]),
        step=0.05,
        help="Higher confidence can reduce false detections.",
    )

    st.divider()
    st.markdown(
        '<span class="status">● AI WORKSPACE</span>',
        unsafe_allow_html=True,
    )
    st.caption("YOLO · OpenCV · Streamlit")


# ============================================================
# INITIALIZE DETECTOR AND HANDLE COMMANDS
# ============================================================

detector = ensure_detector(st.session_state["model_name"])

if detector is not None:
    detector.set_confidence(st.session_state["confidence"])

handle_pending_commands(detector)
show_notice()

if st.session_state.get("detector_error"):
    st.error(f"Model error: {st.session_state['detector_error']}")

snap = get_snapshot()
worker = st.session_state.get("worker")
worker_alive = bool(snap.get("is_alive", False))


# ============================================================
# DASHBOARD OVERVIEW
# ============================================================

if page == "Dashboard":
    hero(
        "AI MONITORING PLATFORM",
        "Human Detection Dashboard",
        "Monitor people, analyze images, and process video using your YOLO detection model.",
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "People in current frame",
        snap.get("people_now", 0),
    )
    c2.metric(
        "Processing speed",
        f"{snap.get('fps', 0):.1f} FPS"
        if snap.get("fps")
        else "—",
    )
    c3.metric(
        "Frames processed",
        snap.get("frame_count", 0),
    )
    c4.metric(
        "Photos captured",
        st.session_state["frames_captured"],
    )

    st.markdown("### Detection workspace")

    left, right = st.columns([1.65, 1], gap="large")

    with left:
        st.markdown("#### Live monitoring")

        if worker_alive and worker is not None:
            frame = worker.latest_frame()

            if frame is not None:
                st.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    width="stretch",
                )
            else:
                empty_preview(
                    "Waiting for camera",
                    "The camera is starting and the first frame is loading.",
                )
        else:
            empty_preview(
                "No active camera",
                "Open Live Detection to start your webcam.",
            )

    with right:
        st.markdown("#### Quick actions")

        if st.button(
            "Open Live Detection",
            use_container_width=True,
            type="primary",
        ):
            st.session_state["navigation"] = "Live Detection"
            st.rerun()

        if st.button("Analyze an image", use_container_width=True):
            st.session_state["navigation"] = "Image Analysis"
            st.rerun()

        if st.button("Analyze a video", use_container_width=True):
            st.session_state["navigation"] = "Video Analysis"
            st.rerun()

        st.markdown("#### Active configuration")
        st.write(f"**Model:** `{st.session_state['model_name']}`")
        st.write(f"**Confidence:** {st.session_state['confidence']:.2f}")
        st.write(
            f"**Camera:** {'Running' if worker_alive else 'Stopped'}"
        )

        if st.session_state.get("last_capture"):
            st.caption(
                f"Last capture: {st.session_state['last_capture']}"
            )

    st.markdown("### Session overview")

    a, b, c = st.columns(3)
    a.metric(
        "Session duration",
        fmt_duration(
            snap.get(
                "session_seconds",
                time.time() - st.session_state["session_started"],
            )
        ),
    )
    b.metric(
        "Peak people",
        snap.get("people_max", 0),
    )
    frames = snap.get("frame_count", 0)
    frames_with_people = snap.get("frames_with_people", 0)
    occupancy = (
        frames_with_people / frames * 100
        if frames
        else 0
    )
    c.metric("Frames containing people", f"{occupancy:.0f}%")


# ============================================================
# LIVE DETECTION
# ============================================================

elif page == "Live Detection":
    hero(
        "REAL-TIME MONITORING",
        "Live Detection",
        "Start your webcam to detect and count people in real time.",
    )

    top_left, top_mid, top_right = st.columns([1, 1, 2])

    with top_left:
        if st.button(
            "▶ Start camera",
            type="primary",
            use_container_width=True,
            disabled=detector is None or worker_alive,
            key="live_start",
        ):
            st.session_state["cmd"] = "start"
            st.rerun()

    with top_mid:
        if st.button(
            "■ Stop camera",
            use_container_width=True,
            disabled=not worker_alive,
            key="live_stop",
        ):
            st.session_state["cmd"] = "stop"
            st.rerun()

    with top_right:
        camera_status = "● LIVE" if worker_alive else "○ STOPPED"
        st.markdown(f"**Camera status:** {camera_status}")

    if not worker_alive:
        st.session_state["camera_index"] = st.selectbox(
            "Camera index",
            options=[0, 1, 2, 3],
            index=int(st.session_state.get("camera_index", 0)),
            help="Most built-in webcams use index 0.",
        )

    left, right = st.columns([1.7, 1], gap="large")

    with left:
        st.markdown("#### Camera feed")

        frame_placeholder = st.empty()
        frame = worker.latest_frame() if worker is not None else None

        if worker_alive and frame is not None:
            frame_placeholder.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                channels="RGB",
                width="stretch",
            )
        else:
            empty_preview(
                "Camera preview",
                "Press Start camera to begin live human detection.",
            )

        if st.button(
            "📸 Capture photo",
            disabled=not worker_alive,
            use_container_width=True,
            key="live_capture",
        ):
            st.session_state["cmd"] = "capture"
            st.rerun()

    with right:
        st.markdown("#### Live metrics")
        st.metric("People now", snap.get("people_now", 0))
        st.metric("Peak people", snap.get("people_max", 0))
        st.metric(
            "Processing FPS",
            f"{snap.get('fps', 0):.1f}"
            if snap.get("fps")
            else "—",
        )
        st.metric(
            "Inference time",
            f"{snap.get('avg_inference_ms', 0):.0f} ms"
            if snap.get("avg_inference_ms")
            else "—",
        )

    st.markdown("### Session statistics")
    a, b, c, d = st.columns(4)

    a.metric("Frames", snap.get("frame_count", 0))
    b.metric("Frames with people", snap.get("frames_with_people", 0))
    c.metric("Total people counted", snap.get("total_people", 0))
    d.metric("Photos saved", st.session_state["frames_captured"])

    if snap.get("friendly_error"):
        st.error(snap["friendly_error"])
        if snap.get("error_message"):
            st.caption(snap["error_message"])

    with st.expander("Technical details"):
        st.write(f"**Model:** {st.session_state['model_name']}")
        st.write(f"**Confidence:** {st.session_state['confidence']:.2f}")
        st.write(f"**Status:** {snap.get('status', 'stopped')}")
        st.write(f"**Resolution:** {snap.get('resolution', '—')}")
        st.write(f"**Camera used:** {snap.get('used_camera_index', '—')}")

    if not worker_alive:
        saved = (
            sorted(CAPTURE_DIR.glob("*.jpg"))
            if CAPTURE_DIR.exists()
            else []
        )

        if saved:
            st.markdown("### Recent captures")
            cols = st.columns(min(4, len(saved)))

            for i, path in enumerate(saved[-8:]):
                cols[i % len(cols)].image(
                    str(path),
                    caption=path.name,
                    width="stretch",
                )

    # Preserve the original live-refresh approach.
    # If the camera is already flickering in the old version, investigate
    # its stop caller separately before changing camera_worker.py.
    if worker_alive:
        time.sleep(REFRESH_SECONDS)
        st.rerun()


# ============================================================
# IMAGE ANALYSIS
# ============================================================

elif page == "Image Analysis":
    hero(
        "IMAGE INTELLIGENCE",
        "Image Analysis",
        "Upload a photo to detect people, inspect bounding boxes, and download the result.",
    )

    upload_col, sample_col = st.columns([3, 1])

    with upload_col:
        uploaded_image = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            key="image_file_uploader",
        )

    with sample_col:
        st.markdown("#### Try a sample")

        if SAMPLE_BUS_IMAGE.exists() and st.button(
            "Load sample photo",
            use_container_width=True,
        ):
            st.session_state["image_bytes"] = SAMPLE_BUS_IMAGE.read_bytes()
            st.session_state["image_name"] = "bus.jpg"
            st.session_state["image_result"] = None
            st.rerun()

    if uploaded_image is not None:
        if st.session_state.get("image_name") != uploaded_image.name:
            st.session_state["image_bytes"] = uploaded_image.getvalue()
            st.session_state["image_name"] = uploaded_image.name
            st.session_state["image_result"] = None

    image_bytes = st.session_state.get("image_bytes")
    image_name = st.session_state.get("image_name")

    if image_bytes:
        st.caption(f"Selected file: {image_name}")

        detect_col, clear_col, _ = st.columns([1, 1, 3])

        with detect_col:
            detect_clicked = st.button(
                "🔍 Detect humans",
                type="primary",
                disabled=detector is None,
                use_container_width=True,
            )

        with clear_col:
            if st.button("Clear image", use_container_width=True):
                st.session_state["image_bytes"] = None
                st.session_state["image_name"] = None
                st.session_state["image_result"] = None
                st.rerun()

        if detect_clicked and detector is not None:
            try:
                with st.spinner("Analyzing image with YOLO..."):
                    result = process_image(
                        detector=detector,
                        image_input=image_bytes,
                        confidence=st.session_state["confidence"],
                    )
                st.session_state["image_result"] = result
            except Exception as exc:
                st.error(f"Image detection failed: {exc}")

        result = st.session_state.get("image_result")

        if result is None:
            try:
                arr = np.frombuffer(image_bytes, np.uint8)
                preview = cv2.imdecode(arr, cv2.IMREAD_COLOR)

                if preview is not None:
                    st.markdown("#### Image preview")
                    st.image(
                        cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                        caption=image_name,
                        width="stretch",
                    )
                else:
                    st.error("This image could not be decoded.")
            except Exception as exc:
                st.error(f"Could not display the image: {exc}")

        else:
            st.markdown("### Detection results")
            m1, m2, m3, m4 = st.columns(4)

            m1.metric("People detected", result["people"])
            m2.metric("Inference", f"{result['inference_ms']:.0f} ms")
            m3.metric("Confidence threshold", f"{st.session_state['confidence']:.2f}")
            m4.metric(
                "Resolution",
                f"{result['resolution'][0]} × {result['resolution'][1]}",
            )

            original_col, result_col = st.columns(2, gap="large")

            with original_col:
                st.markdown("#### Original image")
                st.image(
                    result["original_rgb"],
                    width="stretch",
                )

            with result_col:
                st.markdown("#### Annotated image")
                st.image(
                    result["annotated_rgb"],
                    width="stretch",
                )

            if result.get("people", 0) > 0:
                st.success(
                    f"Detected {result['people']} person(s) in {image_name}."
                )
            else:
                st.info(
                    "No people detected. Try lowering the confidence threshold."
                )

            if result.get("annotated_bytes"):
                st.download_button(
                    "⬇ Download annotated image",
                    data=result["annotated_bytes"],
                    file_name=f"detected_{Path(image_name).stem}.jpg",
                    mime="image/jpeg",
                    key="download_annotated_img",
                )

            if result.get("boxes"):
                with st.expander("Bounding boxes and confidence details"):
                    for i, (box, conf) in enumerate(
                        zip(result["boxes"], result["confidences"]),
                        start=1,
                    ):
                        st.write(
                            f"Person {i}: {conf * 100:.1f}% confidence — "
                            f"(x1={box[0]}, y1={box[1]}, "
                            f"x2={box[2]}, y2={box[3]})"
                        )
    else:
        empty_preview(
            "Upload an image to get started",
            "Supported formats: JPG, JPEG, PNG, BMP and WEBP.",
        )


# ============================================================
# VIDEO ANALYSIS
# ============================================================

elif page == "Video Analysis":
    hero(
        "VIDEO INTELLIGENCE",
        "Video Analysis",
        "Process videos frame by frame, monitor progress, and export annotated results.",
    )

    upload_col, sample_col = st.columns([3, 1])

    with upload_col:
        uploaded_video = st.file_uploader(
            "Upload a video",
            type=["mp4", "avi", "mov", "mkv", "mjpeg"],
            key="video_file_uploader",
        )

    with sample_col:
        st.markdown("#### Try a sample")

        if SAMPLE_VIDEO.exists() and st.button(
            "Load sample video",
            use_container_width=True,
        ):
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
            safe_name = Path(uploaded_video.name).name
            saved_path = TEMP_DIR / f"upload_{safe_name}"
            saved_path.write_bytes(uploaded_video.getvalue())

            st.session_state["video_path"] = str(saved_path)
            st.session_state["video_name"] = uploaded_video.name
            st.session_state["video_bytes"] = uploaded_video.getvalue()
            st.session_state["video_result"] = None

    video_path = st.session_state.get("video_path")
    video_name = st.session_state.get("video_name")

    if video_path and Path(video_path).exists():
        st.markdown(f"**Selected video:** `{video_name}`")
        st.caption(
            f"File size: {Path(video_path).stat().st_size / (1024 * 1024):.1f} MB"
        )

        control_col, clear_col = st.columns([2, 1])

        with control_col:
            max_frames = st.selectbox(
                "Processing limit",
                options=[0, 30, 60, 120, 300],
                format_func=lambda n: "All frames" if n == 0 else f"First {n} frames",
                key="video_max_frames",
            )

        with clear_col:
            if st.button(
                "Clear video",
                use_container_width=True,
                key="clear_video_btn",
            ):
                st.session_state["video_path"] = None
                st.session_state["video_name"] = None
                st.session_state["video_bytes"] = None
                st.session_state["video_result"] = None
                st.rerun()

        detect_video = st.button(
            "🎬 Analyze video",
            type="primary",
            disabled=detector is None,
            use_container_width=True,
        )

        if detect_video and detector is not None:
            st.markdown("### Processing")
            progress = st.progress(0.0)
            status = st.empty()
            frame_placeholder = st.empty()

            def on_progress(cur: int, total: int, res, fps_val: float):
                pct = min(max(cur / max(total, 1), 0.0), 1.0)
                progress.progress(pct)
                status.write(
                    f"Frame {cur}/{total} · "
                    f"{res.people} people · {fps_val:.1f} FPS"
                )

                if cur % 2 == 1 or cur == total:
                    frame_placeholder.image(
                        cv2.cvtColor(res.frame, cv2.COLOR_BGR2RGB),
                        caption=f"Frame {cur} · {res.people} people",
                        width="stretch",
                    )

            try:
                output_path = TEMP_DIR / (
                    f"annotated_{Path(video_name).stem}.mp4"
                )

                with st.spinner("Processing video..."):
                    result = process_video(
                        detector=detector,
                        video_input=video_path,
                        output_path=output_path,
                        confidence=st.session_state["confidence"],
                        max_frames=max_frames if max_frames > 0 else None,
                        progress_callback=on_progress,
                    )

                st.session_state["video_result"] = result
                st.rerun()

            except Exception as exc:
                st.error(f"Video processing failed: {exc}")

        result = st.session_state.get("video_result")

        if result is None:
            st.markdown("#### Video preview")
            try:
                st.video(video_path)
            except Exception as exc:
                st.info(f"Preview unavailable for this format: {exc}")

        else:
            st.success(
                f"Completed {result['frames_processed']} frames in "
                f"{result['elapsed_seconds']:.1f} seconds."
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Peak people", result["people_max"])
            m2.metric("Frames processed", result["frames_processed"])

            percentage = (
                result["frames_with_people"]
                / max(result["frames_processed"], 1)
                * 100
            )
            m3.metric("Frames with people", f"{percentage:.0f}%")
            m4.metric("Processing speed", f"{result['processing_fps']:.1f} FPS")

            st.markdown("### Annotated video")
            output_path = result.get("output_path")

            if output_path and Path(output_path).exists():
                st.video(output_path)

                if result.get("output_bytes"):
                    st.download_button(
                        "⬇ Download processed video",
                        data=result["output_bytes"],
                        file_name=f"detected_{Path(video_name).stem}.mp4",
                        mime="video/mp4",
                        key="download_annotated_video",
                    )

            key_frames = result.get("key_frames", [])
            if key_frames:
                with st.expander(
                    f"Key frames with detections ({len(key_frames)})",
                    expanded=True,
                ):
                    cols = st.columns(min(3, len(key_frames)))

                    for i, (frame_idx, count, rgb_frame) in enumerate(key_frames[:6]):
                        cols[i % len(cols)].image(
                            rgb_frame,
                            caption=f"Frame {frame_idx} · {count} people",
                            width="stretch",
                        )
    else:
        empty_preview(
            "Upload a video to get started",
            "Supported formats: MP4, AVI, MOV, MKV and MJPEG.",
        )


# ============================================================
# DETECTION HISTORY
# ============================================================

elif page == "Detection History":
    hero(
        "SESSION RECORDS",
        "Detection History",
        "Review photos captured by your current application.",
    )

    st.markdown("### Saved captures")

    captures = (
        sorted(CAPTURE_DIR.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
        if CAPTURE_DIR.exists()
        else []
    )

    if captures:
        st.caption(f"{len(captures)} saved photo(s)")
        cols = st.columns(3)

        for i, path in enumerate(captures[:30]):
            with cols[i % 3]:
                st.image(
                    str(path),
                    caption=path.name,
                    width="stretch",
                )

                with open(path, "rb") as image_file:
                    st.download_button(
                        "Download",
                        data=image_file.read(),
                        file_name=path.name,
                        mime="image/jpeg",
                        key=f"history_download_{path.name}",
                        use_container_width=True,
                    )
    else:
        empty_preview(
            "No captures yet",
            "Start the webcam and capture a photo to build your history.",
        )


# ============================================================
# SETTINGS
# ============================================================

elif page == "Settings":
    hero(
        "CONFIGURATION",
        "Detection Settings",
        "Configure your AI model and detection sensitivity.",
    )

    st.markdown("### Model configuration")
    st.write(f"**Current model:** `{st.session_state['model_name']}`")
    st.write(f"**Confidence threshold:** {st.session_state['confidence']:.2f}")

    st.markdown("### Camera configuration")

    worker_now = st.session_state.get("worker")
    camera_active = bool(
        worker_now is not None
        and (
            getattr(worker_now, "active", False)
            or getattr(worker_now, "is_alive", False)
        )
    )

    if camera_active:
        st.warning("Stop the webcam before changing the camera index.")

    st.session_state["camera_index"] = st.selectbox(
        "Preferred camera index",
        options=[0, 1, 2, 3],
        index=int(st.session_state.get("camera_index", 0)),
        disabled=camera_active,
        key="settings_camera_index",
    )

    st.markdown("### Current session")
    a, b, c = st.columns(3)
    a.metric("Photos captured", st.session_state["frames_captured"])
    b.metric("Camera active", "Yes" if camera_active else "No")
    c.metric(
        "Session duration",
        fmt_duration(time.time() - st.session_state["session_started"]),
    )

    st.info(
        "Changing the model or confidence affects subsequent detection. "
        "Your camera worker and detection engine remain in their existing modules."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.markdown(
    """
    <div style="text-align:center; color:#71877a; font-size:0.8rem;">
        VISION AI · SMART HUMAN DETECTION SYSTEM
        <br>
        Powered by YOLO, OpenCV and Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
