"""
Video processing pipeline for YOLOv7-based traffic sign detection.

Workflow:
  1. Extract frames from the uploaded simulation video at a configurable
     sampling interval (default: every 0.5 s).
  2. Run YOLOv7 detection on each sampled frame.
  3. Merge per-frame detections into tracked sign events using IoU-based
     temporal deduplication (same sign seen across frames = one record).
  4. Re-read the original video and write an annotated copy with bounding
     boxes drawn on every frame (not just sampled ones).

Future: depth data extraction can be added as a post-processing step
on top of the per-frame detections returned here.
"""

import logging
import os
import subprocess
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from detector import BoundingBox, SignDetection, detect_signs_in_image
from visualizer import draw_detections

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FrameDetection:
    """A detection tied to a specific video frame / timestamp."""

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
    """A unique sign instance tracked across multiple video frames."""

    sign_id: int
    category: str
    description: str
    best_confidence: float
    best_bbox: BoundingBox
    raw_class_name: str
    first_seen_s: float
    last_seen_s: float
    first_frame: int
    last_frame: int
    frame_count: int

    def to_dict(self) -> dict:
        return {
            "sign_id": self.sign_id,
            "category": self.category,
            "description": self.description,
            "best_confidence": round(self.best_confidence, 4),
            "best_bbox": self.best_bbox.to_dict(),
            "raw_class_name": self.raw_class_name,
            "first_seen_s": round(self.first_seen_s, 3),
            "last_seen_s": round(self.last_seen_s, 3),
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "frame_count": self.frame_count,
            "duration_s": round(self.last_seen_s - self.first_seen_s, 3),
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
# IoU-based temporal deduplication
# ---------------------------------------------------------------------------


def _iou(a: BoundingBox, b: BoundingBox) -> float:
    """Compute Intersection over Union of two bounding boxes."""
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def _deduplicate_tracks(
    frame_detections: list[FrameDetection],
    iou_threshold: float = 0.3,
    max_gap_frames: int = 15,
) -> list[TrackedSign]:
    """
    Merge per-frame detections into unique tracked sign events.

    Two detections in nearby frames are the same sign if:
      - Same taxonomy category
      - Bounding box overlap ≥ iou_threshold
      - Frame gap ≤ max_gap_frames
    """
    tracks: list[dict] = []

    for fd in sorted(frame_detections, key=lambda x: x.frame_number):
        det = fd.detection
        matched = False

        for track in tracks:
            if track["category"] != det.category:
                continue
            if fd.frame_number - track["last_frame"] > max_gap_frames:
                continue
            if _iou(det.bbox, track["last_bbox"]) >= iou_threshold:
                track["last_frame"] = fd.frame_number
                track["last_seen_s"] = fd.timestamp_s
                track["frame_count"] += 1
                track["last_bbox"] = det.bbox
                if det.confidence > track["best_confidence"]:
                    track["best_confidence"] = det.confidence
                    track["best_bbox"] = det.bbox
                    track["raw_class_name"] = det.raw_class_name
                matched = True
                break

        if not matched:
            tracks.append({
                "category": det.category,
                "description": det.category_description,
                "best_confidence": det.confidence,
                "best_bbox": det.bbox,
                "last_bbox": det.bbox,
                "raw_class_name": det.raw_class_name,
                "first_seen_s": fd.timestamp_s,
                "last_seen_s": fd.timestamp_s,
                "first_frame": fd.frame_number,
                "last_frame": fd.frame_number,
                "frame_count": 1,
            })

    return [
        TrackedSign(
            sign_id=i + 1,
            category=t["category"],
            description=t["description"],
            best_confidence=t["best_confidence"],
            best_bbox=t["best_bbox"],
            raw_class_name=t["raw_class_name"],
            first_seen_s=t["first_seen_s"],
            last_seen_s=t["last_seen_s"],
            first_frame=t["first_frame"],
            last_frame=t["last_frame"],
            frame_count=t["frame_count"],
        )
        for i, t in enumerate(tracks)
    ]


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------


def get_video_info(video_path: str) -> dict:
    """Return basic metadata for a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration_s": total_frames / fps if fps > 0 else 0.0,
    }


def extract_frames(
    video_path: str,
    interval_s: float = 0.5,
    max_frames: int = 300,
) -> list[tuple[int, float, Image.Image]]:
    """
    Extract frames from a video at regular time intervals.

    Returns a list of (frame_number, timestamp_s, PIL.Image.RGB) tuples.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = max(1, int(fps * interval_s))

    frames: list[tuple[int, float, Image.Image]] = []
    frame_num = 0

    while len(frames) < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append((frame_num, frame_num / fps, Image.fromarray(rgb)))
        frame_num += frame_step

    cap.release()
    logger.info(
        "Extracted %d frames from %s (total=%d, step=%d frames / %.2fs)",
        len(frames), video_path, total_frames, frame_step, interval_s,
    )
    return frames


# ---------------------------------------------------------------------------
# Main video detection pipeline
# ---------------------------------------------------------------------------


def detect_signs_in_video(
    video_path: str,
    interval_s: float = 0.5,
    max_frames: int = 300,
    conf_threshold: float | None = None,
    iou_threshold: float | None = None,
    img_size: int | None = None,
    dedup_iou_threshold: float = 0.3,
    progress_callback=None,
) -> VideoDetectionResult:
    """
    Run YOLOv7 detection across all sampled frames of a video.

    Args:
        video_path: Path to input video file.
        interval_s: Seconds between sampled frames (default 0.5).
        max_frames: Maximum number of frames to sample (default 300).
        conf_threshold: YOLOv7 confidence threshold (env default if None).
        iou_threshold: YOLOv7 NMS IoU threshold (env default if None).
        img_size: Input resolution for YOLOv7 inference (default 640).
        dedup_iou_threshold: IoU threshold for cross-frame deduplication.
        progress_callback: Optional callable(current, total, message).

    Returns:
        VideoDetectionResult with tracked signs and per-frame detections.
    """
    info = get_video_info(video_path)
    frames = extract_frames(video_path, interval_s=interval_s, max_frames=max_frames)

    # Build kwargs for detect_signs_in_image (only non-None values)
    detect_kwargs: dict = {}
    if conf_threshold is not None:
        detect_kwargs["conf_threshold"] = conf_threshold
    if iou_threshold is not None:
        detect_kwargs["iou_threshold"] = iou_threshold
    if img_size is not None:
        detect_kwargs["img_size"] = img_size

    all_frame_detections: list[FrameDetection] = []

    for idx, (frame_num, timestamp, pil_image) in enumerate(frames):
        if progress_callback:
            progress_callback(
                idx, len(frames),
                f"Detecting signs: frame {idx + 1}/{len(frames)} (t={timestamp:.1f}s)",
            )

        detections = detect_signs_in_image(pil_image, **detect_kwargs)

        for det in detections:
            all_frame_detections.append(
                FrameDetection(
                    detection=det,
                    frame_number=frame_num,
                    timestamp_s=timestamp,
                )
            )

    if progress_callback:
        progress_callback(len(frames), len(frames), "Deduplicating detections across frames…")

    tracked = _deduplicate_tracks(
        all_frame_detections,
        iou_threshold=dedup_iou_threshold,
    )
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


# ---------------------------------------------------------------------------
# Annotated video generation
# ---------------------------------------------------------------------------


def generate_annotated_video(
    video_path: str,
    frame_detections: list[FrameDetection],
    output_path: str,
    progress_callback=None,
) -> str:
    """
    Write an annotated copy of the video with bounding boxes drawn on every
    frame.  Detections from the nearest sampled keyframe are propagated
    forward so boxes appear on all intermediate frames too.

    Re-encodes to H.264 via ffmpeg for browser playback (falls back to
    mp4v if ffmpeg is not available).

    Args:
        video_path: Path to the original video.
        frame_detections: Per-frame detections from detect_signs_in_video.
        output_path: Destination path for the annotated video (.mp4).
        progress_callback: Optional callable(current, total, message).

    Returns:
        output_path (the file that was written).
    """
    # Build frame-indexed lookup
    detections_by_frame: dict[int, list[SignDetection]] = {}
    for fd in frame_detections:
        detections_by_frame.setdefault(fd.frame_number, []).append(fd.detection)

    sampled_frames = sorted(detections_by_frame.keys())
    # Half-interval tolerance for clearing carried-forward detections
    # (cleared only after we've moved well past a keyframe)
    hold_frames_after_last = 15

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise ValueError(f"Cannot open VideoWriter for: {output_path}")

    active_dets: list[SignDetection] = []
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Update active detections at each keyframe
        if frame_num in detections_by_frame:
            active_dets = detections_by_frame[frame_num]
        else:
            # Clear detections after we've passed the last keyframe by a margin
            if sampled_frames and frame_num > sampled_frames[-1] + hold_frames_after_last:
                active_dets = []

        if active_dets:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            annotated_pil = draw_detections(Image.fromarray(rgb), active_dets)
            frame = cv2.cvtColor(np.array(annotated_pil), cv2.COLOR_RGB2BGR)

        writer.write(frame)
        frame_num += 1

        if progress_callback and frame_num % 150 == 0:
            progress_callback(
                frame_num, total_frames,
                f"Rendering annotated video: {frame_num}/{total_frames} frames",
            )

    cap.release()
    writer.release()
    logger.info("Wrote annotated video (%d frames): %s", frame_num, output_path)

    # Re-encode to H.264 for web playback
    _reencode_h264(output_path)
    return output_path


def _reencode_h264(path: str) -> None:
    """In-place re-encode mp4v → H.264 using ffmpeg, if available."""
    tmp = path + ".h264.mp4"
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", path,
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-movflags", "+faststart",
                "-an",          # no audio needed
                tmp,
            ],
            check=True,
            capture_output=True,
        )
        os.replace(tmp, path)
        logger.info("Re-encoded to H.264: %s", path)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.warning("H.264 re-encode skipped (mp4v kept): %s", exc)
        # Remove the partial temp file if it exists
        if os.path.exists(tmp):
            os.remove(tmp)
