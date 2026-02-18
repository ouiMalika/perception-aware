"""
Celery tasks for asynchronous YOLOv7-based traffic sign detection.

Each task runs inside the Celery worker process where the YOLOv7 model
is loaded once (lazy singleton in detector.py) and reused across tasks.
"""

import logging
import os
import sys
import time

if "/app" not in sys.path:
    sys.path.insert(0, "/app")

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
    "task_time_limit": 600,       # 10 min hard limit
    "task_soft_time_limit": 540,  # 9 min soft limit
    "worker_prefetch_multiplier": 1,  # one task at a time (GPU/CPU)
})


# ---------------------------------------------------------------------------
# Image tasks
# ---------------------------------------------------------------------------

@app.task(bind=True, name="detect_signs_single")
def detect_signs_single(self, image_url: str, options: dict | None = None):
    """
    Process a single image URL through the YOLOv7 detection pipeline.

    Args:
        image_url: Publicly accessible image URL.
        options: Optional overrides:
            - conf_threshold: float (default from env, 0.25)
            - iou_threshold:  float (default from env, 0.45)
            - img_size:       int   (default 640)
            - max_detections: int   (default 50)
    """
    from detector import process_image_url

    options = options or {}
    self.update_state(state="PROGRESS", meta={"status": "Processing image…"})

    start = time.time()
    result = process_image_url(image_url, **options)
    elapsed = time.time() - start

    result_dict = result.to_dict()
    result_dict["processing_time_s"] = round(elapsed, 2)
    return {"status": "completed", "result": result_dict}


@app.task(bind=True, name="detect_signs_local")
def detect_signs_local(self, image_path: str, options: dict | None = None):
    """
    Process a local image file through the YOLOv7 detection pipeline.
    """
    from detector import process_image_path

    options = options or {}
    self.update_state(state="PROGRESS", meta={"status": f"Processing {image_path}…"})

    start = time.time()
    result = process_image_path(image_path, **options)
    elapsed = time.time() - start

    result_dict = result.to_dict()
    result_dict["processing_time_s"] = round(elapsed, 2)
    return {"status": "completed", "result": result_dict}


@app.task(bind=True, name="detect_signs_batch")
def detect_signs_batch(self, image_urls: list[str], options: dict | None = None):
    """
    Process a batch of image URLs.

    Returns a dict with per-image results under "results".
    """
    from detector import process_image_url

    options = options or {}
    total = len(image_urls)
    results = []

    for idx, url in enumerate(image_urls):
        self.update_state(
            state="PROGRESS",
            meta={"current": idx, "total": total, "status": f"Processing {idx + 1}/{total}"},
        )
        start = time.time()
        result = process_image_url(url, **options)
        elapsed = time.time() - start

        r = result.to_dict()
        r["processing_time_s"] = round(elapsed, 2)
        results.append(r)

    return {
        "status": "completed",
        "total_images": total,
        "total_detections": sum(r["num_detections"] for r in results),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Video task
# ---------------------------------------------------------------------------

@app.task(bind=True, name="detect_signs_video")
def detect_signs_video(self, video_path: str, options: dict | None = None):
    """
    Process a simulation video through the YOLOv7 traffic sign pipeline.

    Steps:
      1. Sample frames at `interval_s` intervals.
      2. Run YOLOv7 on each frame.
      3. Deduplicate detections across frames (IoU tracking).
      4. Generate an annotated video with bounding boxes.

    Args:
        video_path: Absolute path to the video file on the worker filesystem.
        options: Optional overrides:
            - interval_s:          float  Seconds between sampled frames (0.5)
            - max_frames:          int    Max frames to sample (300)
            - conf_threshold:      float  YOLOv7 detection confidence (0.25)
            - iou_threshold:       float  YOLOv7 NMS IoU threshold (0.45)
            - img_size:            int    Input resolution for YOLOv7 (640)
            - dedup_iou_threshold: float  Cross-frame dedup IoU (0.3)

    Returns:
        dict with tracked signs, per-frame detections, and annotated video path.
    """
    from video_processor import detect_signs_in_video, generate_annotated_video

    options = options or {}

    # Pull video-specific options; leave the rest for detect_signs_in_video
    interval_s = options.pop("interval_s", 0.5)
    max_frames = options.pop("max_frames", 300)
    dedup_iou = options.pop("dedup_iou_threshold", 0.3)

    def _progress(current, total, message):
        self.update_state(
            state="PROGRESS",
            meta={"current": current, "total": total, "status": message},
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
        dedup_iou_threshold=dedup_iou,
        progress_callback=_progress,
        **options,
    )
    elapsed = time.time() - start

    result_dict = result.to_dict()
    result_dict["processing_time_s"] = round(elapsed, 2)

    # Generate annotated video
    annotated_path = None
    if result.per_frame_detections:
        _progress(0, 1, "Generating annotated video with bounding boxes…")
        base, _ = os.path.splitext(video_path)
        annotated_path = f"{base}_annotated.mp4"
        try:
            generate_annotated_video(
                video_path=video_path,
                frame_detections=result.per_frame_detections,
                output_path=annotated_path,
                progress_callback=_progress,
            )
            result_dict["annotated_video_path"] = annotated_path
            logger.info("Annotated video: %s", annotated_path)
        except Exception as exc:
            logger.warning("Annotated video generation failed: %s", exc)
            result_dict["annotated_video_path"] = None

    return {"status": "completed", "result": result_dict}
