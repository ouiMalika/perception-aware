"""
Celery tasks for asynchronous traffic sign detection.

Mirrors the VisionBoard-AI worker architecture:
- Heavy ML models are loaded once per process (lazy singletons).
- Each task downloads images, runs the two-stage pipeline, and stores
  structured results in the Celery result backend (Redis).
"""

import json
import logging
import os
import time

from celery import Celery

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Celery app
# ---------------------------------------------------------------------------

app = Celery("perception_worker")
app.config_from_object({
    "broker_url": os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    "result_backend": os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/0"),
    "task_serializer": "json",
    "result_serializer": "json",
    "accept_content": ["json"],
    "task_track_started": True,
    "task_time_limit": 600,       # 10 min hard limit per task
    "task_soft_time_limit": 540,  # 9 min soft limit
    "worker_prefetch_multiplier": 1,  # process one task at a time (GPU)
})


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

@app.task(bind=True, name="detect_signs_batch")
def detect_signs_batch(self, image_urls: list[str], options: dict | None = None):
    """
    Process a batch of image URLs through the sign detection pipeline.

    Args:
        image_urls: List of publicly-accessible image URLs.
        options: Optional dict with keys:
            - yolo_conf: float (default 0.25)
            - clip_conf: float (default 0.15)
            - max_detections: int (default 50)

    Returns:
        dict with "results" key containing per-image detection results.
    """
    from detector import process_image_url

    options = options or {}
    total = len(image_urls)
    results = []

    for idx, url in enumerate(image_urls):
        # Update progress
        self.update_state(
            state="PROGRESS",
            meta={
                "current": idx,
                "total": total,
                "status": f"Processing image {idx + 1}/{total}",
            },
        )

        logger.info("Processing image %d/%d: %s", idx + 1, total, url)
        start = time.time()

        result = process_image_url(url, **options)
        elapsed = time.time() - start

        result_dict = result.to_dict()
        result_dict["processing_time_s"] = round(elapsed, 2)
        results.append(result_dict)

    return {
        "status": "completed",
        "total_images": total,
        "total_detections": sum(r["num_detections"] for r in results),
        "results": results,
    }


@app.task(bind=True, name="detect_signs_single")
def detect_signs_single(self, image_url: str, options: dict | None = None):
    """
    Process a single image URL through the sign detection pipeline.

    Convenience wrapper around detect_signs_batch for single images.
    """
    from detector import process_image_url

    options = options or {}

    self.update_state(
        state="PROGRESS",
        meta={"status": "Processing image..."},
    )

    start = time.time()
    result = process_image_url(image_url, **options)
    elapsed = time.time() - start

    result_dict = result.to_dict()
    result_dict["processing_time_s"] = round(elapsed, 2)

    return {
        "status": "completed",
        "result": result_dict,
    }


@app.task(bind=True, name="detect_signs_local")
def detect_signs_local(self, image_path: str, options: dict | None = None):
    """
    Process a local image file through the sign detection pipeline.

    Useful for testing and CLI usage.
    """
    from detector import process_image_path

    options = options or {}

    self.update_state(
        state="PROGRESS",
        meta={"status": f"Processing {image_path}..."},
    )

    start = time.time()
    result = process_image_path(image_path, **options)
    elapsed = time.time() - start

    result_dict = result.to_dict()
    result_dict["processing_time_s"] = round(elapsed, 2)

    return {
        "status": "completed",
        "result": result_dict,
    }


# ---------------------------------------------------------------------------
# Video tasks
# ---------------------------------------------------------------------------


@app.task(bind=True, name="detect_signs_video")
def detect_signs_video(self, video_path: str, options: dict | None = None):
    """
    Process a video file through the sign detection pipeline.

    Extracts frames at a configurable interval, runs YOLOv8 + CLIP on each,
    and deduplicates detections across frames via IoU-based tracking.

    Args:
        video_path: Path to the video file on the worker filesystem.
        options: Optional dict with keys:
            - yolo_conf: float (default 0.25)
            - clip_conf: float (default 0.15)
            - interval_s: float, seconds between frames (default 0.5)
            - max_frames: int, max frames to sample (default 300)
            - iou_threshold: float, dedup IoU threshold (default 0.3)

    Returns:
        dict with tracked signs and per-frame detections.
    """
    from video_processor import detect_signs_in_video, generate_annotated_video

    options = options or {}

    # Separate video-specific options from detection options
    interval_s = options.pop("interval_s", 0.5)
    max_frames = options.pop("max_frames", 300)
    iou_threshold = options.pop("iou_threshold", 0.3)

    def progress_callback(current, total, message):
        self.update_state(
            state="PROGRESS",
            meta={
                "current": current,
                "total": total,
                "status": message,
            },
        )

    self.update_state(
        state="PROGRESS",
        meta={"status": f"Starting video processing: {video_path}"},
    )

    start = time.time()
    result = detect_signs_in_video(
        video_path=video_path,
        interval_s=interval_s,
        max_frames=max_frames,
        iou_threshold=iou_threshold,
        progress_callback=progress_callback,
        **options,
    )
    elapsed = time.time() - start

    result_dict = result.to_dict()
    result_dict["processing_time_s"] = round(elapsed, 2)

    # Generate annotated video with bounding boxes
    annotated_path = None
    if result.per_frame_detections:
        self.update_state(
            state="PROGRESS",
            meta={"status": "Generating annotated video with bounding boxes..."},
        )
        base, ext = os.path.splitext(video_path)
        annotated_path = f"{base}_annotated.mp4"
        try:
            generate_annotated_video(
                video_path=video_path,
                frame_detections=result.per_frame_detections,
                output_path=annotated_path,
                progress_callback=progress_callback,
            )
            result_dict["annotated_video_path"] = annotated_path
            logger.info("Annotated video saved: %s", annotated_path)
        except Exception as exc:
            logger.warning("Failed to generate annotated video: %s", exc)
            result_dict["annotated_video_path"] = None

    return {
        "status": "completed",
        "result": result_dict,
    }
