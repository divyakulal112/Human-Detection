
"""Person detection using pretrained Ultralytics YOLO models.

Only the official ``person`` class (COCO id 0) is detected. Weights are
loaded through the Ultralytics API, which downloads them from the official
Ultralytics GitHub release the first time they are used.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

PERSON_CLASS_ID = 0

# Palette picked to match the dashboard CSS accents.
BOX_COLOR = (78, 205, 165)
BADGE_COLOR = (78, 205, 165)

AVAILABLE_MODELS = {
    "yolov8n.pt": "YOLOv8n - fastest (recommended on CPU)",
    "yolov8s.pt": "YOLOv8s - small, more accurate",
    "yolov8m.pt": "YOLOv8m - medium, slowest on CPU",
}


class ModelLoadError(RuntimeError):
    """Raised when the YOLO weights cannot be loaded."""


@dataclass
class DetectionResult:
    """Outcome of one detection pass."""

    frame: np.ndarray
    people: int
    confidences: List[float] = field(default_factory=list)
    boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    inference_ms: float = 0.0
    model: str = ""


class PersonDetector:
    """Thread-friendly wrapper around an Ultralytics YOLO model."""

    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.4):
        self.model_name = model_name
        self.loaded = False
        self.error: Optional[str] = None
        self._model = None
        self._confidence = float(confidence)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ setup
    def load(self) -> bool:
        """Download/instanciate the model. Returns True on success."""
        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover - import env dependent
            self.error = (
                "The 'ultralytics' package is not installed correctly "
                f"({exc}). Run: pip install -r requirements.txt"
            )
            return False

        try:
            self._model = YOLO(self.model_name)
        except Exception as exc:
            self.error = (
                f"Could not load model '{self.model_name}'. "
                "The first run downloads the weights (~6 MB) from the official "
                f"Ultralytics release, so a working internet connection is needed. "
                f"Details: {exc}"
            )
            self._model = None
            return False

        self.loaded = True
        self.error = None
        return True

    def _require_model(self) -> None:
        if self._model is None:
            raise ModelLoadError(self.error or "Model is not loaded.")

    # ------------------------------------------------------------- parameters
    @property
    def confidence(self) -> float:
        with self._lock:
            return self._confidence

    def set_confidence(self, value: float) -> None:
        with self._lock:
            self._confidence = float(min(max(value, 0.01), 0.99))

    # -------------------------------------------------------------- inference
    def detect(self, frame: np.ndarray, confidence: Optional[float] = None) -> DetectionResult:
        """Detect people in ``frame`` and return an annotated copy."""
        self._require_model()
        if frame is None or frame.size == 0:
            raise ValueError("Empty frame passed to the detector.")

        conf = self.confidence if confidence is None else float(confidence)
        conf = min(max(conf, 0.01), 0.99)

        started = time.perf_counter()
        results = self._model.predict(
            frame,
            conf=conf,
            classes=[PERSON_CLASS_ID],
            verbose=False,
        )
        inference_ms = (time.perf_counter() - started) * 1000.0

        annotated = frame.copy()
        people = 0
        confidences: List[float] = []
        boxes: List[Tuple[int, int, int, int]] = []

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                score = float(box.conf[0])
                people += 1
                confidences.append(score)
                boxes.append((x1, y1, x2, y2))
                _draw_box(annotated, x1, y1, x2, y2, score)

        _draw_badge(annotated, people)

        return DetectionResult(
            frame=annotated,
            people=people,
            confidences=confidences,
            boxes=boxes,
            inference_ms=inference_ms,
            model=self.model_name,
        )


# --------------------------------------------------------------------- drawing
def _draw_box(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, score: float) -> None:
    import cv2

    h, w = frame.shape[:2]
    x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
    y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))
    if x2 <= x1 or y2 <= y1:
        return

    cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)

    label = f"Person {score * 100:.0f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    (tw, th), _ = cv2.getTextSize(label, font, scale, 2)
    top = max(0, y1 - th - 8)
    cv2.rectangle(frame, (x1, top), (min(w - 1, x1 + tw + 8), y1), BOX_COLOR, -1)
    cv2.putText(
        frame,
        label,
        (x1 + 4, max(th + 2, y1 - 5)),
        font,
        scale,
        (12, 24, 20),
        2,
        cv2.LINE_AA,
    )


def _draw_badge(frame: np.ndarray, people: int) -> None:
    import cv2

    text = f"PEOPLE: {people}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.9
    (tw, th), _ = cv2.getTextSize(text, font, scale, 2)
    cv2.rectangle(frame, (12, 12), (12 + tw + 20, 12 + th + 18), BADGE_COLOR, -1)
    cv2.putText(
        frame,
        text,
        (22, 12 + th + 6),
        font,
        scale,
        (12, 24, 20),
        2,
        cv2.LINE_AA,
    )