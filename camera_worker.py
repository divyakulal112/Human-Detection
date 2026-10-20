"""Background capture + detection worker.

Decouples high-speed camera frame capture from asynchronous YOLO inference
so the live camera feed remains silky smooth (~30 FPS) with zero frame queue latency.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from detector import PersonDetector, draw_detections

STATUS_STOPPED = "stopped"
STATUS_STARTING = "starting"
STATUS_RUNNING = "running"
STATUS_ERROR = "error"

# Consecutive failed cap.read() calls tolerated before declaring STATUS_ERROR.
# 150 * 0.02 s = 3 s — covers Windows MSMF auto-exposure/focus renegotiation delays.
MAX_READ_FAILURES = 150
CAMERA_READ_WAIT = 0.02          # seconds to sleep after each failed read
CAMERA_INDEX_PROBES = (0, 1, 2, 3)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# How many times to retry cap.read() right after isOpened() before giving up on
# a camera index.  Windows MSMF can take up to ~2 s to start streaming even
# after isOpened() returns True.  30 × 0.1 s = 3 s warm-up budget.
CAMERA_WARMUP_RETRIES = 30
CAMERA_WARMUP_WAIT = 0.10        # seconds between warm-up read attempts

# Process-wide lock & tracker to guarantee the webcam is never opened by multiple threads/sessions.
# NOTE: This must NOT be held while calling stop() / _release() on another worker — doing so
# creates a cross-thread deadlock because _release() also acquires this lock.
_ACTIVE_WEBCAM_LOCK = threading.Lock()
_ACTIVE_WEBCAM_WORKER: Optional["CameraWorker"] = None
DIAG_LOG_PATH = Path(__file__).resolve().parent / "debug_trace.log"


def diag_log(msg: str) -> None:
    print(msg, flush=True)
    try:
        with open(DIAG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {msg}\n")
    except Exception:
        pass


def get_active_webcam_worker() -> Optional["CameraWorker"]:
    """Return the currently active webcam worker (or None).  Thread-safe."""
    with _ACTIVE_WEBCAM_LOCK:
        return _ACTIVE_WEBCAM_WORKER


class _SingleImageSource:
    """Mimics ``cv2.VideoCapture`` so a still image can feed the same loop."""

    def __init__(self, frame: np.ndarray):
        self._frame = frame

    def read(self):
        return True, self._frame.copy()

    def isOpened(self) -> bool:
        return True

    def set(self, *_args) -> bool:
        return False

    def get(self, *_args) -> float:
        return 0.0

    def release(self) -> None:
        pass


class CameraWorker:
    """Owns the ``cv2.VideoCapture`` handle and the detection loop."""

    def __init__(
        self,
        detector: PersonDetector,
        source: str = "webcam",
        camera_index: int = 0,
        auto_probe: bool = True,
        save_dir: str | Path = "captures",
        loop_media: bool = True,
    ):
        self.detector = detector
        self.source = source
        self.camera_index = camera_index
        self.auto_probe = auto_probe
        self.save_dir = Path(save_dir)
        self.loop_media = loop_media

        self.status = STATUS_STOPPED
        self.error_message: Optional[str] = None
        self.friendly_error: Optional[str] = None
        self.resolution: Optional[tuple] = None
        self.used_camera_index: Optional[int] = None

        self.people_now = 0
        self.people_max = 0
        self.frame_count = 0
        self.frames_with_people = 0
        self.total_people = 0
        self.avg_inference_ms = 0.0
        self.fps = 0.0
        self.started_at: Optional[float] = None
        self.last_capture_path: Optional[str] = None

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._infer_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._infer_thread: Optional[threading.Thread] = None
        self._latest: Optional[np.ndarray] = None
        self._latest_raw: Optional[np.ndarray] = None
        self._latest_boxes: List[Tuple[int, int, int, int]] = []
        self._latest_confidences: List[float] = []
        self._cap: Optional[cv2.VideoCapture] = None

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            print("[camera] start() called but worker thread is already alive — ignoring")
            return
        self._stop.clear()
        self._infer_event.clear()
        self.status = STATUS_STARTING
        self.error_message = None
        self.friendly_error = None
        self.started_at = time.time()
        if self.source == "webcam":
            global _ACTIVE_WEBCAM_WORKER
            with _ACTIVE_WEBCAM_LOCK:
                _ACTIVE_WEBCAM_WORKER = self
        self._thread = threading.Thread(target=self._run, name="camera-worker", daemon=True)
        print(f"[camera] start(): launching capture thread (source={self.source!r}, index={self.camera_index})")
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        """Signal the worker to exit and wait briefly for a clean release.

        _release() is called here (and only here from the *outside*).  The
        background thread (_run) does NOT call _release() on its normal exit
        path so there is no double-release race.
        """
        import traceback
        stack_str = "".join(traceback.format_stack(limit=8))
        diag_log(f"[camera] stop(): signalling stop (self={id(self)}, status={self.status}, is_alive={self.is_alive})\nStack:\n{stack_str}")
        self._stop.set()
        self._infer_event.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=timeout)
            if thread.is_alive():
                print(f"[camera] stop(): thread did not exit within {timeout}s — forcing release")
        infer_thread = self._infer_thread
        if infer_thread and infer_thread.is_alive():
            infer_thread.join(timeout=timeout)
        if self.status in (STATUS_STARTING, STATUS_RUNNING):
            self.status = STATUS_STOPPED
        self._release()
        print("[camera] stop(): completed")

    @property
    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    @property
    def active(self) -> bool:
        """True while capture+inference is actually running."""

        return self.is_alive and self.status in (STATUS_STARTING, STATUS_RUNNING)

    @property
    def session_seconds(self) -> float:
        if self.started_at is None:
            return 0.0
        if self.is_alive:
            return time.time() - self.started_at
        return getattr(self, "_ended_at", time.time()) - self.started_at

    # ------------------------------------------------------------------- frames
    def latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "status": self.status,
                "error_message": self.error_message,
                "friendly_error": self.friendly_error,
                "used_camera_index": self.used_camera_index,
                "people_now": self.people_now,
                "people_max": self.people_max,
                "frame_count": self.frame_count,
                "frames_with_people": self.frames_with_people,
                "total_people": self.total_people,
                "avg_inference_ms": self.avg_inference_ms,
                "fps": self.fps,
                "resolution": self.resolution,
                "session_seconds": self.session_seconds,
                "is_alive": self.is_alive,
            }

    # ------------------------------------------------------------------ capture
    def save_capture(self, prefix: str = "capture") -> Optional[str]:
        """Save the latest annotated frame to ``captures/`` and return its path."""
        frame = self.latest_frame()
        if frame is None:
            return None
        self.save_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        path = self.save_dir / f"{prefix}_{stamp}.jpg"
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            return None
        path.write_bytes(buf.tobytes())
        with self._lock:
            self.last_capture_path = str(path)
        return str(path)

    # -------------------------------------------------------------------- thread

    def _run(self) -> None:
        """Main entry point for the capture thread.

        Both _run() and stop() call _release(), but _release() is fully
        idempotent: self._cap is set to None on the first call so the
        cv2.VideoCapture is only ever released once.  stop() always calls
        _release() after joining the thread as a safety net (e.g. if the
        thread was forcibly killed or died before reaching its own _release).
        """
        try:
            print(f"[camera] _run(): opening source={self.source!r}")
            cap = self._open_source()
        except Exception as exc:
            print(f"[camera] _run(): _open_source() raised: {exc}")
            self.error_message = str(exc)
            self.friendly_error = self._friendly(str(exc))
            self.status = STATUS_ERROR
            self._ended_at = time.time()
            # _cap was never assigned — nothing to release here.
            # stop() will call _release() as a safe no-op if needed.
            return

        self._cap = cap
        print("[camera] _run(): reading initial frame")
        ok, frame = cap.read()
        if not ok or frame is None:
            print("[camera] _run(): initial frame read failed")
            self.error_message = "Initial frame read failed."
            self.friendly_error = "Could not read frames from the camera."
            self.status = STATUS_ERROR
            self._ended_at = time.time()
            # _cap is assigned — release it now so the camera is freed even if
            # the user never clicks Stop.  stop() calling _release() later is safe
            # because _release() is idempotent (self._cap is already None).
            self._release()
            return

        self.resolution = (int(frame.shape[1]), int(frame.shape[0]))
        print(f"[camera] _run(): status → RUNNING  resolution={self.resolution}")
        self.status = STATUS_RUNNING

        # Make the initial frame available immediately so the preview appears without delay
        with self._lock:
            self._latest_raw = frame.copy()
            self._latest = draw_detections(frame, [], [], 0)

        if self.source == "webcam":
            self._run_webcam_mode(cap, frame)
        else:
            self._run_sync_mode(cap, frame)

        # Normal/stop-requested exit: release now.  stop() calling _release()
        # after thread.join() is safe because self._cap will already be None.
        self._ended_at = time.time()
        if self.status in (STATUS_STARTING, STATUS_RUNNING):
            self.status = STATUS_STOPPED
        print(f"[camera] _run(): thread exiting (status={self.status}) — releasing camera")
        self._release()

    def _run_webcam_mode(self, cap: cv2.VideoCapture, initial_frame: np.ndarray) -> None:
        """Decoupled webcam mode: capture runs at full camera FPS, inference runs asynchronously."""
        if self._infer_thread and self._infer_thread.is_alive():
            self._infer_event.set()
            self._infer_thread.join(timeout=1.0)

        self._infer_thread = threading.Thread(
            target=self._webcam_inference_loop,
            name="camera-inference",
            daemon=True,
        )
        self._infer_thread.start()
        self._infer_event.set()

        failures = 0
        tick = time.time()

        while not self._stop.is_set():
            try:
                ok, frame = cap.read()
            except Exception as exc:
                print(f"[camera] cap.read() raised an exception: {exc}")
                ok, frame = False, None

            if not ok or frame is None:
                failures += 1
                if failures >= MAX_READ_FAILURES:
                    print(
                        f"[camera] {failures} consecutive read failures — "
                        "declaring STATUS_ERROR and stopping capture loop."
                    )
                    self.friendly_error = (
                        "The webcam stopped sending frames. It may have been "
                        "disconnected or taken over by another application."
                    )
                    self.error_message = "Frame read failed."
                    self.status = STATUS_ERROR
                    break
                if failures % 10 == 1:
                    print(f"[camera] read failure #{failures}/{MAX_READ_FAILURES} — retrying…")
                self._stop.wait(CAMERA_READ_WAIT)
                continue

            failures = 0
            try:
                with self._lock:
                    self._latest_raw = frame.copy()
                    boxes = list(self._latest_boxes)
                    confs = list(self._latest_confidences)
                    people = self.people_now

                # Render overlay onto the fresh live frame
                annotated = draw_detections(frame, boxes, confs, people)

                now = time.time()
                with self._lock:
                    self._latest = annotated
                    self.frame_count += 1
                    delta = now - tick
                    if delta > 0:
                        self.fps += 0.15 * ((1.0 / delta) - self.fps)
                    tick = now

                # Signal inference thread that a fresh frame is ready
                self._infer_event.set()
            except Exception:
                pass

            # Wait briefly so stop event is noticed within ~10 ms and prevent pegging CPU
            self._stop.wait(0.01)

        # Ensure inference thread exits cleanly
        self._infer_event.set()
        if self._infer_thread and self._infer_thread.is_alive():
            self._infer_thread.join(timeout=1.0)

    def _webcam_inference_loop(self) -> None:
        """Background inference loop: detects people on the freshest available camera frame."""
        while not self._stop.is_set():
            self._infer_event.wait(timeout=0.1)
            self._infer_event.clear()
            if self._stop.is_set():
                break

            with self._lock:
                if self._latest_raw is None:
                    continue
                raw_frame = self._latest_raw.copy()

            h, w = raw_frame.shape[:2]
            scale_x = scale_y = 1.0
            infer_input = raw_frame

            # Resize if frame is larger than 640 for faster inference while maintaining aspect ratio
            max_dim = max(h, w)
            if max_dim > 640:
                factor = 640.0 / max_dim
                nw, nh = int(w * factor), int(h * factor)
                infer_input = cv2.resize(raw_frame, (nw, nh))
                scale_x = w / float(nw)
                scale_y = h / float(nh)

            try:
                result = self.detector.detect(infer_input)
                scaled_boxes = []
                for x1, y1, x2, y2 in result.boxes:
                    scaled_boxes.append((
                        int(x1 * scale_x),
                        int(y1 * scale_y),
                        int(x2 * scale_x),
                        int(y2 * scale_y),
                    ))

                with self._lock:
                    self._latest_boxes = scaled_boxes
                    self._latest_confidences = result.confidences
                    self.people_now = result.people
                    if result.people > self.people_max:
                        self.people_max = result.people
                    if result.people > 0:
                        self.frames_with_people += 1
                        self.total_people += result.people
                    alpha = 0.15
                    self.avg_inference_ms += alpha * (result.inference_ms - self.avg_inference_ms)
            except Exception:
                pass

    def _run_sync_mode(self, cap: cv2.VideoCapture, initial_frame: np.ndarray) -> None:
        """Synchronous mode for video files and still images (ensures every frame is analyzed)."""
        failures = 0
        tick = time.time()
        ok = True
        frame = initial_frame

        while not self._stop.is_set():
            if not ok or frame is None:
                failures += 1
                if failures >= MAX_READ_FAILURES:
                    self.friendly_error = "The selected media source ended."
                    self.error_message = "Frame read failed."
                    self.status = STATUS_ERROR
                    break
                if self.loop_media and self.source != "webcam":
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
                continue

            failures = 0
            try:
                result = self.detector.detect(frame)
            except Exception as exc:
                self.error_message = f"Detection failed: {exc}"
                self.friendly_error = (
                    "Detection failed. Try lowering the confidence or restarting "
                    "the camera."
                )
                self.status = STATUS_ERROR
                break

            now = time.time()
            with self._lock:
                self._latest = result.frame
                self.frame_count += 1
                self.people_now = result.people
                if result.people > self.people_max:
                    self.people_max = result.people
                if result.people > 0:
                    self.frames_with_people += 1
                    self.total_people += result.people
                alpha = 0.15
                self.avg_inference_ms += alpha * (result.inference_ms - self.avg_inference_ms)
                delta = now - tick
                if delta > 0:
                    self.fps += alpha * ((1.0 / delta) - self.fps)
                tick = now


            self._stop.wait(0.02)
            ok, frame = cap.read()

  
    def _open_source(self):
        if self.source == "webcam":
            return self._open_webcam()
        path = Path(str(self.source))
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            frame = cv2.imread(str(path))
            if frame is None:
                raise RuntimeError(f"Could not open media file: {self.source}")
            return _SingleImageSource(frame)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Could not open media file: {self.source}")
        return cap

    def _open_webcam(self) -> cv2.VideoCapture:
        """Open the physical webcam, evicting any previously active worker first.

        IMPORTANT: The previous worker is stopped *before* we acquire
        _ACTIVE_WEBCAM_LOCK.  Calling stop() inside the lock would deadlock:
        stop() → _release() also acquires _ACTIVE_WEBCAM_LOCK, and because
        _ACTIVE_WEBCAM_LOCK is a plain (non-reentrant) Lock the old worker thread
        would block forever waiting for the lock that the new worker thread is
        already holding.
        """
        global _ACTIVE_WEBCAM_WORKER

        # --- Step 1: grab the previous worker reference WITHOUT holding the lock ---
        with _ACTIVE_WEBCAM_LOCK:
            prev = _ACTIVE_WEBCAM_WORKER if _ACTIVE_WEBCAM_WORKER is not self else None

        # --- Step 2: stop previous worker OUTSIDE the lock ---
        if prev is not None and prev.is_alive:
            diag_log(f"[camera] evicting previous worker (self={id(self)}, prev={id(prev)}, status={prev.status}) before opening camera")
            prev.stop(timeout=3.0)          # _release() acquires the lock — safe here
            diag_log("[camera] previous worker stopped")

        # --- Step 3: register self as the active worker ---
        with _ACTIVE_WEBCAM_LOCK:
            _ACTIVE_WEBCAM_WORKER = self

        # --- Step 4: open the hardware camera ---
        index = self.camera_index
        candidates = [index]
        if self.auto_probe:
            candidates += [i for i in CAMERA_INDEX_PROBES if i != index]

        print(f"[camera] probing camera indices: {candidates}")
        for idx in candidates:
            cap = self._try_open_index(idx)
            if cap is not None:
                return cap

        raise RuntimeError(f"No webcam responded on camera indexes {candidates}.")

    def _try_open_index(self, idx: int) -> "Optional[cv2.VideoCapture]":
        """Try to open a single camera index; return a ready VideoCapture or None.

        Windows MSMF often reports isOpened()=True but returns empty frames for
        up to ~2-3 seconds while the driver initialises.  We retry cap.read()
        for CAMERA_WARMUP_RETRIES × CAMERA_WARMUP_WAIT seconds before giving up.
        Without this wait the probe loop abandons the real camera and tries dummy
        indices 1-3, which all fail, producing the repeated ON/OFF cycle seen in
        the logs.
        """
        cap: Optional[cv2.VideoCapture] = None

        # Prefer CAP_DSHOW (lower latency on Windows); fall through to MSMF/default.
        if hasattr(cv2, "CAP_DSHOW"):
            print(f"[camera] index {idx}: trying CAP_DSHOW")
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                print(f"[camera] index {idx}: CAP_DSHOW isOpened()=False")
                cap.release()
                cap = None
            else:
                print(f"[camera] index {idx}: CAP_DSHOW isOpened()=True")

        if cap is None:
            if hasattr(cv2, "CAP_MSMF"):
                print(f"[camera] index {idx}: trying CAP_MSMF")
                cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
            else:
                print(f"[camera] index {idx}: trying default backend")
                cap = cv2.VideoCapture(idx)

        if not cap.isOpened():
            print(f"[camera] index {idx}: isOpened()=False after all backends — skipping")
            cap.release()
            return None

        print(f"[camera] index {idx}: isOpened()=True — warming up driver (up to {CAMERA_WARMUP_RETRIES * CAMERA_WARMUP_WAIT:.1f}s)")

        # Warm-up loop: Windows MSMF/DSHOW drivers need time to start streaming
        # even after isOpened() returns True.  Bail out early if stop was requested.
        warmed = False
        warm_frame = None
        for attempt in range(CAMERA_WARMUP_RETRIES):
            if self._stop.is_set():
                print(f"[camera] index {idx}: stop requested during warm-up — aborting")
                cap.release()
                return None
            ok, warm_frame = cap.read()
            if ok and warm_frame is not None:
                print(f"[camera] index {idx}: warm-up succeeded after {attempt + 1} attempt(s)")
                warmed = True
                break
            time.sleep(CAMERA_WARMUP_WAIT)

        if not warmed:
            print(f"[camera] index {idx}: no frames after {CAMERA_WARMUP_RETRIES} warm-up attempts — releasing")
            cap.release()
            return None

        # Camera is streaming.  Try to negotiate resolution without re-reading;
        # just set the preferred size and trust the driver (no extra read call).
        for (w, h) in [(640, 480), (1280, 720)]:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if actual_w > 0 and actual_h > 0:
                print(f"[camera] index {idx}: resolution set to {actual_w}×{actual_h} (requested {w}×{h})")
                break

        actual = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print(f"[camera] index {idx}: ready — resolution={actual}")
        self.used_camera_index = idx
        return cap

    def _release(self) -> None:
        """Release the hardware camera exactly once.  Fully idempotent — safe to
        call multiple times (subsequent calls are no-ops because self._cap is
        set to None on the first call)."""
        global _ACTIVE_WEBCAM_WORKER
        # Deregister self BEFORE releasing the hardware so another opener can
        # safely acquire the camera index immediately after.
        with _ACTIVE_WEBCAM_LOCK:
            if _ACTIVE_WEBCAM_WORKER is self:
                _ACTIVE_WEBCAM_WORKER = None
                print("[camera] _release(): deregistered from _ACTIVE_WEBCAM_WORKER")

        cap = self._cap
        self._cap = None          # guard against concurrent/double-release
        if cap is not None:
            try:
                cap.release()
                print("[camera] _release(): cv2.VideoCapture.release() called")
            except Exception as exc:
                print(f"[camera] _release(): cap.release() raised: {exc}")
        else:
            print("[camera] _release(): cap was already None — no-op")
        with self._lock:
            self._latest = None
            self._latest_raw = None

    @staticmethod
    def _friendly(raw: str) -> str:
        lowered = raw.lower()
        if "no webcam" in lowered:
            return (
                "No webcam was found. Check that a camera is connected, then "
                "close other apps that may be using it (Zoom, Teams, ...). On "
                "Windows also check Settings > Privacy & security > Camera."
            )
        if "could not open media" in lowered:
            return "That file could not be opened. Pick another image/video."
        return f"The camera could not be started: {raw}"