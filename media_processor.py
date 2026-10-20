"""Media processing utilities for single images and video files.

Extracts frames, performs person detection using PersonDetector, and builds
annotated outputs with bounding boxes, badges, and progress tracking.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from detector import DetectionResult, PersonDetector

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".mjpeg"}


def decode_image(image_input: Union[bytes, str, Path, np.ndarray]) -> np.ndarray:
    """Decode raw image bytes, a file path, or validate an existing ndarray into a BGR image."""
    if isinstance(image_input, np.ndarray):
        if image_input.size == 0:
            raise ValueError("Empty image array provided.")
        return image_input

    if isinstance(image_input, (str, Path)):
        path = Path(image_input)
        if not path.exists():
            raise FileNotFoundError(f"Image file does not exist: {path}")
        frame = cv2.imread(str(path))
        if frame is None or frame.size == 0:
            raise ValueError(f"Could not decode image file: {path.name}")
        return frame

    if isinstance(image_input, (bytes, bytearray)):
        if len(image_input) == 0:
            raise ValueError("Uploaded image file is empty (0 bytes).")
        arr = np.frombuffer(image_input, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            raise ValueError("Could not decode image. Unsupported or corrupted format.")
        return frame

    raise TypeError(f"Unsupported image input type: {type(image_input)}")


def process_image(
    detector: PersonDetector,
    image_input: Union[bytes, str, Path, np.ndarray],
    confidence: Optional[float] = None,
) -> Dict:
    """Run human detection on a single image and return structured results and preview bytes."""
    frame_bgr = decode_image(image_input)
    original_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    result: DetectionResult = detector.detect(frame_bgr, confidence=confidence)
    annotated_bgr = result.frame
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    ok, buf = cv2.imencode(".jpg", annotated_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    annotated_bytes = buf.tobytes() if ok else b""

    return {
        "people": result.people,
        "confidences": result.confidences,
        "boxes": result.boxes,
        "inference_ms": result.inference_ms,
        "model": result.model,
        "original_rgb": original_rgb,
        "annotated_rgb": annotated_rgb,
        "annotated_bgr": annotated_bgr,
        "annotated_bytes": annotated_bytes,
        "resolution": (frame_bgr.shape[1], frame_bgr.shape[0]),
    }


def process_video(
    detector: PersonDetector,
    video_input: Union[str, Path, bytes],
    output_path: Optional[Union[str, Path]] = None,
    confidence: Optional[float] = None,
    max_frames: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int, DetectionResult, float], None]] = None,
    stop_check: Optional[Callable[[], bool]] = None,
    save_key_frames: int = 6,
) -> Dict:
    """Process a video frame-by-frame, write an annotated output video, and report metrics."""
    temp_in_path: Optional[Path] = None

    if isinstance(video_input, (bytes, bytearray)):
        if len(video_input) == 0:
            raise ValueError("Uploaded video file is empty (0 bytes).")
        temp_dir = Path("temp_media")
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_in_path = temp_dir / f"temp_in_{int(time.time() * 1000)}.mp4"
        temp_in_path.write_bytes(video_input)
        video_src = str(temp_in_path)
    else:
        video_src = str(video_input)
        if not Path(video_src).exists():
            raise FileNotFoundError(f"Video file does not exist: {video_src}")

    cap = cv2.VideoCapture(video_src)
    if not cap.isOpened():
        cap.release()
        if temp_in_path and temp_in_path.exists():
            temp_in_path.unlink(missing_ok=True)
        raise ValueError("Could not open video file. The file may be corrupted or in an unsupported format.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        cap.release()
        if temp_in_path and temp_in_path.exists():
            temp_in_path.unlink(missing_ok=True)
        raise ValueError("Video has invalid dimensions (width or height is 0).")

    if output_path is None:
        out_dir = Path("temp_media")
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"processed_{int(time.time() * 1000)}.mp4"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        # Fallback to MJPG if mp4v fails on some systems
        alt_path = output_path.with_suffix(".avi")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(alt_path), fourcc, fps, (width, height))
        if writer.isOpened():
            output_path = alt_path

    frames_processed = 0
    frames_with_people = 0
    people_max = 0
    people_counts: List[int] = []
    inference_times: List[float] = []
    key_frames: List[Tuple[int, int, np.ndarray]] = []  # (frame_idx, people_count, rgb_frame)

    t_start = time.perf_counter()

    try:
        limit = total_frames if max_frames is None or max_frames <= 0 else min(total_frames, max_frames)
        if limit <= 0:
            limit = max_frames or 100000

        while True:
            if stop_check and stop_check():
                break

            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frames_processed += 1
            result = detector.detect(frame, confidence=confidence)
            writer.write(result.frame)

            people_counts.append(result.people)
            inference_times.append(result.inference_ms)

            if result.people > people_max:
                people_max = result.people

            if result.people > 0:
                frames_with_people += 1
                if len(key_frames) < save_key_frames:
                    rgb = cv2.cvtColor(result.frame, cv2.COLOR_BGR2RGB)
                    key_frames.append((frames_processed, result.people, rgb))

            if progress_callback:
                progress_callback(frames_processed, limit, result, fps)

            if max_frames and frames_processed >= max_frames:
                break

    finally:
        cap.release()
        writer.release()
        if temp_in_path and temp_in_path.exists():
            temp_in_path.unlink(missing_ok=True)

    elapsed = max(time.perf_counter() - t_start, 0.001)
    avg_inference = sum(inference_times) / len(inference_times) if inference_times else 0.0
    processing_fps = frames_processed / elapsed if elapsed > 0 else 0.0

    output_bytes: Optional[bytes] = None
    if output_path.exists() and output_path.stat().st_size > 0:
        try:
            output_bytes = output_path.read_bytes()
        except Exception:
            output_bytes = None

    return {
        "output_path": str(output_path),
        "output_bytes": output_bytes,
        "total_frames": total_frames,
        "frames_processed": frames_processed,
        "frames_with_people": frames_with_people,
        "people_max": people_max,
        "fps": fps,
        "processing_fps": processing_fps,
        "elapsed_seconds": elapsed,
        "avg_inference_ms": avg_inference,
        "key_frames": key_frames,
        "resolution": (width, height),
    }
