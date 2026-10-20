"""Unit tests for media_processor (image and video upload processing)."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from detector import PersonDetector
from media_processor import decode_image, process_image, process_video
from tests.test_detector import ensure_bus_image
from tests.test_worker import ensure_test_video


class MediaProcessorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bus_path = ensure_bus_image()
        cls.video_path = ensure_test_video()
        cls.detector = PersonDetector("yolov8n.pt", confidence=0.4)
        if not cls.detector.load():
            raise AssertionError(f"Model failed to load: {cls.detector.error}")

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="mp_test_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_decode_image_from_path(self) -> None:
        img = decode_image(self.bus_path)
        self.assertIsInstance(img, np.ndarray)
        self.assertEqual(len(img.shape), 3)

    def test_decode_image_from_bytes(self) -> None:
        data = self.bus_path.read_bytes()
        img = decode_image(data)
        self.assertIsInstance(img, np.ndarray)
        self.assertEqual(img.shape[2], 3)

    def test_decode_image_invalid_bytes(self) -> None:
        with self.assertRaises(ValueError):
            decode_image(b"not an image at all")

    def test_decode_image_empty_bytes(self) -> None:
        with self.assertRaises(ValueError):
            decode_image(b"")

    def test_process_image_detects_people(self) -> None:
        res = process_image(self.detector, self.bus_path)
        self.assertGreaterEqual(res["people"], 1)
        self.assertGreater(len(res["annotated_bytes"]), 1000)
        self.assertEqual(res["original_rgb"].shape, res["annotated_rgb"].shape)
        self.assertGreater(res["inference_ms"], 0)

    def test_process_image_blank(self) -> None:
        blank = np.zeros((200, 200, 3), dtype=np.uint8)
        res = process_image(self.detector, blank)
        self.assertEqual(res["people"], 0)
        self.assertEqual(len(res["confidences"]), 0)

    def test_process_video_frames(self) -> None:
        out_video = self.tmp / "annotated.mp4"
        progress_calls = []

        def on_progress(cur, total, result, fps):
            progress_calls.append((cur, result.people))

        res = process_video(
            detector=self.detector,
            video_input=self.video_path,
            output_path=out_video,
            max_frames=5,
            progress_callback=on_progress,
        )

        self.assertEqual(res["frames_processed"], 5)
        self.assertGreaterEqual(res["people_max"], 1)
        self.assertTrue(out_video.exists())
        self.assertGreater(out_video.stat().st_size, 1000)
        self.assertEqual(len(progress_calls), 5)

    def test_process_video_invalid_file(self) -> None:
        invalid_path = self.tmp / "corrupt.mp4"
        invalid_path.write_bytes(b"garbage data")
        with self.assertRaises(ValueError):
            process_video(self.detector, invalid_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
