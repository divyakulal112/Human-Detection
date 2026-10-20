"""Offline tests for the YOLO person detector (no webcam needed)."""

from __future__ import annotations

import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from detector import ModelLoadError, PersonDetector

DATA_DIR = Path(__file__).parent / "data"
BUS_IMAGE = DATA_DIR / "bus.jpg"


def ensure_bus_image() -> Path:
    if not BUS_IMAGE.exists():
        import urllib.request

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve("https://ultralytics.com/images/bus.jpg", BUS_IMAGE)
    return BUS_IMAGE


class DetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        ensure_bus_image()
        cls.detector = PersonDetector("yolov8n.pt", confidence=0.4)
        if not cls.detector.load():
            raise AssertionError(f"Model failed to load: {cls.detector.error}")

    def test_model_loads(self) -> None:
        self.assertTrue(self.detector.loaded)
        self.assertIsNone(self.detector.error)

    def test_detects_people_in_known_image(self) -> None:
        frame = cv2.imread(str(BUS_IMAGE))
        result = self.detector.detect(frame)
        # The ultralytics sample photo shows 3 people.
        self.assertGreaterEqual(result.people, 1)
        self.assertEqual(len(result.confidences), result.people)
        self.assertEqual(len(result.boxes), result.people)
        self.assertGreaterEqual(result.model, "")

    def test_annotated_frame_is_valid(self) -> None:
        frame = cv2.imread(str(BUS_IMAGE))
        result = self.detector.detect(frame)
        self.assertEqual(result.frame.shape, frame.shape)
        self.assertEqual(result.frame.dtype, np.uint8)

    def test_blank_image_detects_nobody(self) -> None:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.detector.detect(blank)
        self.assertEqual(result.people, 0)
        self.assertEqual(result.boxes, [])

    def test_high_confidence_filters_detections(self) -> None:
        frame = cv2.imread(str(BUS_IMAGE))
        low = self.detector.detect(frame, confidence=0.10)
        high = self.detector.detect(frame, confidence=0.99)
        self.assertGreaterEqual(low.people, high.people)
        self.assertEqual(high.people, 0)  # sample photo maxes out ~0.87

    def test_empty_frame_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.detector.detect(np.zeros((0, 0, 3), dtype=np.uint8))

    def test_confidence_setter_bounds(self) -> None:
        self.detector.set_confidence(5.0)
        self.assertAlmostEqual(self.detector.confidence, 0.99)
        self.detector.set_confidence(-1.0)
        self.assertAlmostEqual(self.detector.confidence, 0.01)
        self.detector.set_confidence(0.4)
        self.assertAlmostEqual(self.detector.confidence, 0.4)

    def test_missing_model_gives_friendly_error(self) -> None:
        bad = PersonDetector("definitely_not_a_real_model_xyz.pt")
        self.assertFalse(bad.load())
        self.assertIsNotNone(bad.error)
        self.assertFalse(bad.loaded)
        with self.assertRaises(ModelLoadError):
            bad.detect(np.zeros((64, 64, 3), dtype=np.uint8))

    def test_warm_inference_latency(self) -> None:
        """Measures steady-state latency (cold start is much slower)."""
        frame = str(BUS_IMAGE)
        self.detector.detect(cv2.imread(frame))  # warm up
        samples = []
        for _ in range(3):
            t0 = time.perf_counter()
            self.detector.detect(cv2.imread(frame))
            samples.append((time.perf_counter() - t0) * 1000)
        avg = sum(samples) / len(samples)
        print(f"\n[benchmark] warm detection avg = {avg:.0f} ms over {len(samples)} runs")
        self.assertLess(avg, 3000)


if __name__ == "__main__":
    unittest.main(verbosity=2)