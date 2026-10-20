"""Background capture + detection worker.

The Streamlit UI never reads the camera directly: it starts this worker,
reads its latest annotated frame and asks it to stop. The worker checks a
``threading.Event`` every iteration, so stopping is never blocked by a long
infinite loop, and the camera handle is always released.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from detector import PersonDetector

STATUS_STOPPED = "stopped"
STATUS_STARTING = "starting"
STATUS_RUNNING = "running"
STATUS_ERROR = "error"

MAX_READ_FAILURES = 45  # ~1.5 s of consecutive failed reads before erroring
CAMERA_INDEX_PROBES = (0, 1, 2, 3)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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
        self._thread: Optional[threading.Thread] = None
        self._latest: Optional[np.ndarray] = None
        self._cap: Optional[cv2.VideoCapture] = None

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self.status = STATUS_STARTING
        self.error_message = None
        self.friendly_error = None
        self.started_at = time.time()
        self._thread = threading.Thread(target=self._run, name="camera-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        """Signal the worker to exit and wait briefly for a clean release."""
        self._stop.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=timeout)
        if self.status in (STATUS_STARTING, STATUS_RUNNING):
            self.status = STATUS_STOPPED
        self._release()

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
        failures = 0
        try:
            cap = self._open_source()
        except Exception as exc:
            self.error_message = str(exc)
            self.friendly_error = self._friendly(str(exc))
            self.status = STATUS_ERROR
            self._ended_at = time.time()
            return

        self._cap = cap
        ok, frame = cap.read()
        if ok and frame is not None:
            self.resolution = (int(frame.shape[1]), int(frame.shape[0]))
        self.status = STATUS_RUNNING

        tick = time.time()
        while not self._stop.is_set():
            if not ok or frame is None:
                failures += 1
                if failures >= MAX_READ_FAILURES:
                    if self.source == "webcam":
                        self.friendly_error = (
                            "The webcam stopped sending frames. It may have been "
                            "disconnected or taken over by another application."
                        )
                    else:
                        self.friendly_error = "The selected media source ended."
                    self.error_message = "Frame read failed."
                    self.status = STATUS_ERROR
                    break
                if self.source != "webcam":
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop video/file
                ok, frame = cap.read()
                continue

            failures = 0
            try:
                result = self.detector.detect(frame)
            except Exception as exc:  # keep the UI alive on inference errors
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
                if result.people:
                    self.frames_with_people += 1
                    self.total_people += result.people
                alpha = 0.15
                self.avg_inference_ms += alpha * (result.inference_ms - self.avg_inference_ms)
                delta = now - tick
                if delta > 0:
                    self.fps += alpha * ((1.0 / delta) - self.fps)
                tick = now

            # Wake briefly so the stop flag is noticed within ~60 ms.
            self._stop.wait(0.02)
            ok, frame = cap.read()

        self._ended_at = time.time()
        if self.status in (STATUS_STARTING, STATUS_RUNNING):
            self.status = STATUS_STOPPED
        self._release()

    # ------------------------------------------------------------------- source
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
        index = self.camera_index
        if self.auto_probe:
            candidates = [index] + [i for i in CAMERA_INDEX_PROBES if i != index]
        else:
            candidates = [index]
        for idx in candidates:
            # Try DirectShow (Windows) first, then Media Foundation, then default backend.
            cap = None
            if hasattr(cv2, "CAP_DSHOW"):
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap.release()
                    cap = None
            if cap is None:
                if hasattr(cv2, "CAP_MSMF"):
                    cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
                else:
                    cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                continue
            # Attempt high resolution first, then fall back to a lower resolution for performance.
            for (w, h) in [(1280, 720), (640, 480)]:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                ok, _ = cap.read()
                if ok:
                    self.used_camera_index = idx
                    return cap
            cap.release()
        raise RuntimeError(f"No webcam responded on camera indexes {candidates}.")

    def _release(self) -> None:
        cap = self._cap
        self._cap = None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        with self._lock:
            self._latest = None

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