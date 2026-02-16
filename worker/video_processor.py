"""
Video processing module for traffic sign detection.

Extracts frames from video files at configurable intervals, runs the
detection pipeline on each frame, and deduplicates detections of the
same sign across consecutive frames using IoU-based tracking.
"""

import logging
import os
import tempfile
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from detector import BoundingBox, SignDetection, detect_signs_in_image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FrameDetection:
    """A detection tied to a specific frame/timestamp."""

    detection: SignDetection
    frame_number: int
    timestamp_s: float

    def to_dict(self) -> dict:
        d = self.detection.to_dict()
        d["frame_number"] = self.frame_number
        d["timestamp_s"] = round(self.timestamp_s, 3)
        return d


@dataclass
class TrackedSign:
    """A unique sign tracked across multiple frames."""

    sign_id: int
    category: str
    description: str
    best_confidence: float
    best_bbox: BoundingBox
    first_seen_s: float
    last_seen_s: float
    first_frame: int
    last_frame: int
    frame_count: int
    best_clip_scores: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "sign_id": self.sign_id,
            "category": self.category,
            "description": self.description,
            "best_confidence": round(self.best_confidence, 4),
            "best_bbox": self.best_bbox.to_dict(),
            "first_seen_s": round(self.first_seen_s, 3),
            "last_seen_s": round(self.last_seen_s, 3),
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "frame_count": self.frame_count,
            "duration_s": round(self.last_seen_s - self.first_seen_s, 3),
            "top_scores": {
                k: round(v, 4) for k, v in self.best_clip_scores.items()
            },
        }


@dataclass
class VideoDetectionResult:
    """Full result for a processed video."""

    video_source: str
    duration_s: float
    fps: float
    total_frames: int
    frames_processed: int
    frame_interval: float
    tracked_signs: list[TrackedSign]
    per_frame_detections: list[FrameDetection]
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "video_source": self.video_source,
            "video_info": {
                "duration_s": round(self.duration_s, 2),
                "fps": round(self.fps, 2),
                "total_frames": self.total_frames,
                "frames_processed": self.frames_processed,
                "frame_interval_s": round(self.frame_interval, 3),
            },
            "num_unique_signs": len(self.tracked_signs),
            "tracked_signs": [s.to_dict() for s in self.tracked_signs],
            "num_frame_detections": len(self.per_frame_detections),
            "per_frame_detections": [d.to_dict() for d in self.per_frame_detections],
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# IoU-based deduplication
# ---------------------------------------------------------------------------


def _iou(a: BoundingBox, b: BoundingBox) -> float:
    """Compute Intersection over Union between two bounding boxes."""
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0

    area_a = a.area
    area_b = b.area
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _deduplicate_tracks(
    frame_detections: list[FrameDetection],
    iou_threshold: float = 0.3,
    max_gap_frames: int = 15,
) -> list[TrackedSign]:
    """
    Merge per-frame detections into unique tracked signs.

    Two detections in nearby frames are considered the same sign if:
    1. They have the same category.
    2. Their bounding boxes overlap above iou_threshold.
    3. They are within max_gap_frames of each other.
    """
    tracks: list[dict] = []  # active tracks

    for fd in sorted(frame_detections, key=lambda x: x.frame_number):
        det = fd.detection
        matched = False

        for track in tracks:
            if track["category"] != det.category:
                continue
            if fd.frame_number - track["last_frame"] > max_gap_frames:
                continue
            if _iou(det.bbox, track["last_bbox"]) >= iou_threshold:
                # Update existing track
                track["last_frame"] = fd.frame_number
                track["last_seen_s"] = fd.timestamp_s
                track["frame_count"] += 1
                track["last_bbox"] = det.bbox
                if det.confidence > track["best_confidence"]:
                    track["best_confidence"] = det.confidence
                    track["best_bbox"] = det.bbox
                    track["best_clip_scores"] = det.clip_scores
                matched = True
                break

        if not matched:
            tracks.append({
                "category": det.category,
                "description": det.category_description,
                "best_confidence": det.confidence,
                "best_bbox": det.bbox,
                "last_bbox": det.bbox,
                "first_seen_s": fd.timestamp_s,
                "last_seen_s": fd.timestamp_s,
                "first_frame": fd.frame_number,
                "last_frame": fd.frame_number,
                "frame_count": 1,
                "best_clip_scores": det.clip_scores,
            })

    return [
        TrackedSign(
            sign_id=i + 1,
            category=t["category"],
            description=t["description"],
            best_confidence=t["best_confidence"],
            best_bbox=t["best_bbox"],
            first_seen_s=t["first_seen_s"],
            last_seen_s=t["last_seen_s"],
            first_frame=t["first_frame"],
            last_frame=t["last_frame"],
            frame_count=t["frame_count"],
            best_clip_scores=t["best_clip_scores"],
        )
        for i, t in enumerate(tracks)
    ]


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------


def extract_frames(
    video_path: str,
    interval_s: float = 0.5,
    max_frames: int = 300,
) -> list[tuple[int, float, Image.Image]]:
    """
    Extract frames from a video file at regular intervals.

    Args:
        video_path: Path to the video file.
        interval_s: Seconds between extracted frames.
        max_frames: Maximum number of frames to extract.

    Returns:
        List of (frame_number, timestamp_seconds, PIL.Image) tuples.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(fps * interval_s))

    frames = []
    frame_num = 0

    while len(frames) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR (OpenCV) to RGB (PIL)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        timestamp = frame_num / fps

        frames.append((frame_num, timestamp, pil_image))
        frame_num += frame_interval

    cap.release()
    logger.info(
        "Extracted %d frames from %s (total: %d, interval: %d frames / %.1fs)",
        len(frames), video_path, total_frames, frame_interval, interval_s,
    )
    return frames


def get_video_info(video_path: str) -> dict:
    """Get basic video metadata."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    cap.release()
    return {
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration_s": duration,
    }


# ---------------------------------------------------------------------------
# Video detection pipeline
# ---------------------------------------------------------------------------


def detect_signs_in_video(
    video_path: str,
    interval_s: float = 0.5,
    max_frames: int = 300,
    yolo_conf: float | None = None,
    clip_conf: float | None = None,
    iou_threshold: float = 0.3,
    progress_callback=None,
) -> VideoDetectionResult:
    """
    Run the full sign detection pipeline on a video.

    1. Extract frames at the specified interval.
    2. Run YOLOv8 + CLIP detection on each frame.
    3. Deduplicate detections across frames using IoU tracking.

    Args:
        video_path: Path to the video file.
        interval_s: Seconds between sampled frames (default 0.5).
        max_frames: Maximum frames to process (default 300 = ~2.5 min at 0.5s).
        yolo_conf: YOLO confidence threshold (uses env default if None).
        clip_conf: CLIP confidence threshold (uses env default if None).
        iou_threshold: IoU threshold for deduplication (default 0.3).
        progress_callback: Optional callable(current, total, message).

    Returns:
        VideoDetectionResult with tracked signs and per-frame detections.
    """
    info = get_video_info(video_path)
    frames = extract_frames(video_path, interval_s=interval_s, max_frames=max_frames)

    all_frame_detections: list[FrameDetection] = []
    kwargs = {}
    if yolo_conf is not None:
        kwargs["yolo_conf"] = yolo_conf
    if clip_conf is not None:
        kwargs["clip_conf"] = clip_conf

    for idx, (frame_num, timestamp, pil_image) in enumerate(frames):
        if progress_callback:
            progress_callback(idx, len(frames), f"Processing frame {idx + 1}/{len(frames)}")

        detections = detect_signs_in_image(pil_image, **kwargs)

        for det in detections:
            all_frame_detections.append(
                FrameDetection(
                    detection=det,
                    frame_number=frame_num,
                    timestamp_s=timestamp,
                )
            )

    # Deduplicate across frames
    tracked = _deduplicate_tracks(
        all_frame_detections,
        iou_threshold=iou_threshold,
    )

    # Sort tracked signs by first appearance
    tracked.sort(key=lambda s: s.first_seen_s)

    return VideoDetectionResult(
        video_source=video_path,
        duration_s=info["duration_s"],
        fps=info["fps"],
        total_frames=info["total_frames"],
        frames_processed=len(frames),
        frame_interval=interval_s,
        tracked_signs=tracked,
        per_frame_detections=all_frame_detections,
    )
