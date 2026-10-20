"""Tests for the background camera worker: lifecycle, stop responsiveness,
counting, capture, and friendly errors. No physical webcam is required -
these run against media files and an unreachable camera index."""

from __future__ import annotations

import shutil
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from camera_worker import STATUS_ERROR, STATUS_RUNNING, CameraWorker
from detector import PersonDetector
from tests.test_detector import ensure_bus_image

DATA_DIR = Path(__file__).parent / "data"
VIDEO_PATH = DATA_DIR / "people_pan.avi"


def ensure_test_video() -> Path:
    """Build an ~25 fps clip from the sample photo (contains real people)."""
    if VIDEO_PATH.exists():
        return VIDEO_PATH
    ensure_bus_image()
    src = cv2.imread(str(str(ensure_bus_image())))
    h, w = src.shape[:2]
    vw, vh = 640, 480
    writer = cv2.VideoWriter(
        str(VIDEO_PATH), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (vw, vh)
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create the test video.")
    for i in range(30):
        # Gentle pan across the photo so consecutive frames differ.
        x0 = int((w - vw) * (i / 29))
        y0 = (h - vh) // 2
        frame = src[y0 : y0 + vh, x0 : x0 + vw]
        frame = cv2.resize(frame, (vw, vh))
        writer.write(frame)
    writer.release()
    return VIDEO_PATH


class WorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        ensure_test_video()
        cls.detector = PersonDetector("yolov8n.pt", confidence=0.4)
        if not cls.detector.load():
            raise AssertionError(f"Model failed to load: {cls.detector.error}")

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="hdtest_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.workers: list[CameraWorker] = []

    def _make(self, **kwargs) -> CameraWorker:
        defaults = dict(detector=self.detector, save_dir=self.tmp)
        defaults.update(kwargs)
        worker = CameraWorker(**defaults)
        self.workers.append(worker)
        self.addCleanup(worker.stop)
        return worker

    def _wait_until(self, predicate, timeout: float = 30.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if predicate():
                return True
            time.sleep(0.2)
        return False

    # ----------------------------------------------------------------- tests
    def test_worker_processes_video_and_counts_people(self) -> None:
        worker = self._make(source=str(VIDEO_PATH))
        worker.start()
        self.assertTrue(
            self._wait_until(lambda: worker.frame_count >= 15),
            f"worker stalled: {worker.snapshot()}",
        )
        snap = worker.snapshot()
        self.assertEqual(snap["status"], STATUS_RUNNING)
        self.assertGreaterEqual(snap["people_max"], 1, "people counter never saw anyone")
        self.assertGreater(snap["avg_inference_ms"], 0)
        self.assertEqual(snap["resolution"], (640, 480))

    def test_stop_is_quick_and_releases_camera(self) -> None:
        worker = self._make(source=str(VIDEO_PATH))
        worker.start()
        self.assertTrue(self._wait_until(lambda: worker.frame_count >= 5))
        t0 = time.time()
        worker.stop(timeout=3.0)
        elapsed = time.time() - t0
        self.assertLess(elapsed, 3.5, f"stop() took {elapsed:.2f}s")
        self.assertFalse(worker.is_alive, "worker thread still alive after stop()")
        self.assertFalse(worker.active)
        self.assertIsNone(worker.latest_frame(), "frame buffer not cleared on stop")
        self.assertEqual(worker.snapshot()["status"], "stopped")

    def test_restart_after_stop(self) -> None:
        worker = self._make(source=str(VIDEO_PATH))
        worker.start()
        self.assertTrue(self._wait_until(lambda: worker.frame_count >= 3))
        worker.stop()
        worker.start()
        self.assertTrue(self._wait_until(lambda: worker.frame_count >= 6))
        self.assertEqual(worker.snapshot()["status"], STATUS_RUNNING)
        worker.stop()

    def test_capture_saves_readable_jpg(self) -> None:
        worker = self._make(source=str(VIDEO_PATH))
        worker.start()
        self.assertTrue(self._wait_until(lambda: worker.frame_count >= 3))
        # Wait for a processed (annotated) frame to exist.
        self.assertTrue(self._wait_until(lambda: worker.latest_frame() is not None))
        path = worker.save_capture()
        self.assertIsNotNone(path)
        saved = Path(path)
        self.assertTrue(saved.exists())
        self.assertGreater(saved.stat().st_size, 5000)
        img = cv2.imread(str(saved))
        self.assertIsNotNone(img, "saved capture is not a valid image")
        self.assertEqual(img.shape[2], 3)
        worker.stop()

    def test_capture_without_frame_returns_none(self) -> None:
        worker = self._make(source=str(VIDEO_PATH))  # never started
        self.assertIsNone(worker.save_capture())

    def test_unreachable_camera_index_reports_friendly_error(self) -> None:
        worker = self._make(source="webcam", camera_index=99, auto_probe=False)
        worker.start()
        self.assertTrue(
            self._wait_until(lambda: worker.snapshot()["status"] == STATUS_ERROR, timeout=60),
            f"expected error status, got {worker.snapshot()}",
        )
        snap = worker.snapshot()
        self.assertIn("webcam", (snap["friendly_error"] or "").lower())
        self.assertIsNone(worker.latest_frame())
        worker.stop()
        self.assertFalse(worker.is_alive)

    def test_real_webcam_if_available(self) -> None:
        """Exercises the physical webcam when one is connected, else skips."""
        probe = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        ok = probe.isOpened() and probe.read()[0]
        probe.release()
        if not ok:
            self.skipTest("No physical webcam available in this environment.")

        worker = self._make(source="webcam", camera_index=0)
        worker.start()
        self.assertTrue(
            self._wait_until(lambda: worker.frame_count >= 5, timeout=45),
            f"webcam never produced processed frames: {worker.snapshot()}",
        )
        snap = worker.snapshot()
        self.assertEqual(snap["status"], STATUS_RUNNING)
        self.assertIsNotNone(worker.latest_frame())
        self.assertGreaterEqual(snap["people_max"], 0)
        self.assertIsNotNone(snap["used_camera_index"])
        print(
            f"\n[webcam] index={snap['used_camera_index']} res={snap['resolution']} "
            f"fps={snap['fps']:.1f} people_now={snap['people_now']}"
        )
        t0 = time.time()
        worker.stop(timeout=3.0)
        self.assertLess(time.time() - t0, 3.5)
        self.assertFalse(worker.is_alive)

    def test_missing_media_file_reports_friendly_error(self) -> None:
        worker = self._make(source=str(self.tmp / "nope.mp4"))
        worker.start()
        self.assertTrue(
            self._wait_until(lambda: worker.snapshot()["status"] == STATUS_ERROR, timeout=30),
            f"expected error status, got {worker.snapshot()}",
        )
        self.assertIn("file", (worker.snapshot()["friendly_error"] or "").lower())
        worker.stop()

    def test_single_image_source_starts(self) -> None:
        ensure_bus_image()
        worker = self._make(source=str(ensure_bus_image()))
        worker.start()
        self.assertTrue(
            self._wait_until(lambda: worker.frame_count >= 2),
            f"image source never produced frames: {worker.snapshot()}",
        )
        self.assertGreaterEqual(worker.snapshot()["people_max"], 1)
        worker.stop()
        self.assertFalse(worker.is_alive)


if __name__ == "__main__":
    unittest.main(verbosity=2)
