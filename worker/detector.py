"""
YOLOv7-based traffic sign detection pipeline.

Single-stage detection: YOLOv7 localises AND classifies traffic signs
in one forward pass. This is the approach described in:

  "Traffic Sign Detection and Recognition Using YOLOv7"
  Applied Sciences 13(20), 11402 (2023)
  https://www.mdpi.com/2076-3417/13/20/11402

Model options (controlled via YOLOV7_MODEL_TYPE env var):
  - "coco"    : Official yolov7.pt weights (COCO 80 classes).
                Only detects stop signs & traffic lights out of the box.
                Good for quick smoke-testing.
  - "traffic" : Custom weights trained on a US traffic sign dataset
                (LISA, MTSD, or your own). Set YOLOV7_WEIGHTS_PATH to
                point at the .pt file.  Detects 40-400+ sign classes.

Class name → taxonomy mapping is handled by sign_taxonomy.py so the
downstream pipeline stays the same regardless of which model is loaded.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded model singleton
# ---------------------------------------------------------------------------

_yolo_model = None
_model_class_names: list[str] = []


def _download_weights(url: str) -> str:
    """
    Download weights from *url* to a local cache directory and return the
    local file path.  Re-uses the cached file on subsequent calls.
    """
    cache_dir = Path(
        os.environ.get("YOLOV7_WEIGHTS_CACHE_DIR", "/tmp/yolov7_weights")
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Derive a sane filename from the URL (strip query-string)
    raw_name = url.split("/")[-1].split("?")[0]
    filename = raw_name if raw_name.endswith(".pt") else (raw_name or "weights") + ".pt"
    local_path = cache_dir / filename

    if local_path.exists():
        logger.info("Using cached weights: %s", local_path)
        return str(local_path)

    logger.info("Downloading YOLOv7 weights from %s → %s", url, local_path)
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(local_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                logger.debug("Download progress: %.1f%%", downloaded / total * 100)

    logger.info("Weights downloaded (%d bytes) → %s", downloaded, local_path)
    return str(local_path)


def _resolve_weights() -> str:
    """
    Return the local path for the YOLOv7 weights file.

    Priority:
      1. YOLOV7_WEIGHTS_PATH   – absolute path to a custom .pt file
      2. YOLOV7_WEIGHTS_URL    – URL to download weights from (cached locally)
      3. Default               – 'yolov7.pt' (auto-downloaded by torch.hub)
    """
    if p := os.environ.get("YOLOV7_WEIGHTS_PATH", "").strip():
        if not Path(p).exists():
            raise FileNotFoundError(f"YOLOV7_WEIGHTS_PATH not found: {p}")
        return p

    if url := os.environ.get("YOLOV7_WEIGHTS_URL", "").strip():
        return _download_weights(url)

    # Default: let torch.hub download official yolov7.pt on first run
    return "yolov7.pt"


def _get_model():
    """
    Load YOLOv7 via torch.hub (downloads code + weights once, then cached).

    Returns the model and populates _model_class_names.
    """
    global _yolo_model, _model_class_names

    if _yolo_model is not None:
        return _yolo_model

    weights = _resolve_weights()
    logger.info("Loading YOLOv7 model from: %s", weights)

    # torch.hub will clone WongKinYiu/yolov7 on first call (cached in
    # ~/.cache/torch/hub/).  'custom' lets us load any .pt file.
    model = torch.hub.load(
        "WongKinYiu/yolov7",
        "custom",
        path_or_model=weights,
        force_reload=False,
        trust_repo=True,
        verbose=False,
    )

    # Move to GPU if available, else CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    _model_class_names = model.names if hasattr(model, "names") else []
    logger.info(
        "YOLOv7 loaded on %s with %d classes: %s",
        device,
        len(_model_class_names),
        _model_class_names[:10],
    )

    _yolo_model = model
    return model


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def to_dict(self) -> dict:
        return {
            "x1": round(self.x1, 1),
            "y1": round(self.y1, 1),
            "x2": round(self.x2, 1),
            "y2": round(self.y2, 1),
        }


@dataclass
class SignDetection:
    """A single detected traffic sign from YOLOv7."""

    bbox: BoundingBox
    category: str                   # mapped taxonomy category
    category_description: str
    confidence: float               # YOLOv7 detection confidence
    raw_class_id: int               # original model class index
    raw_class_name: str             # original model class label

    def to_dict(self) -> dict:
        return {
            "bbox": self.bbox.to_dict(),
            "category": self.category,
            "description": self.category_description,
            "confidence": round(self.confidence, 4),
            "raw_class_id": self.raw_class_id,
            "raw_class_name": self.raw_class_name,
        }


@dataclass
class DetectionResult:
    """Full result for one image."""

    image_url: str
    image_width: int
    image_height: int
    detections: list[SignDetection]
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "image_url": self.image_url,
            "image_size": {
                "width": self.image_width,
                "height": self.image_height,
            },
            "num_detections": len(self.detections),
            "detections": [d.to_dict() for d in self.detections],
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------


def load_image_from_url(url: str, timeout: int = 30) -> Image.Image:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    from io import BytesIO
    return Image.open(BytesIO(resp.content)).convert("RGB")


def load_image_from_path(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


# ---------------------------------------------------------------------------
# YOLOv7 inference
# ---------------------------------------------------------------------------


def _run_yolov7(
    image: Image.Image,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    img_size: int = 640,
    max_detections: int = 50,
) -> list[tuple[BoundingBox, float, int, str]]:
    """
    Run a single YOLOv7 forward pass on a PIL image.

    Returns list of (BoundingBox, confidence, class_id, class_name).
    """
    model = _get_model()

    # YOLOv7 hub model accepts PIL images directly
    model.conf = conf_threshold
    model.iou = iou_threshold
    model.max_det = max_detections

    # Resize hint passed via augment=False; hub model handles letterboxing
    results = model(image, size=img_size)

    detections: list[tuple[BoundingBox, float, int, str]] = []

    # results.xyxy[0] is a tensor: [x1, y1, x2, y2, conf, cls]
    pred = results.xyxy[0].cpu().numpy()
    names = results.names  # dict {id: name}

    for row in pred:
        x1, y1, x2, y2, conf, cls_id = row
        cls_id = int(cls_id)
        cls_name = names.get(cls_id, str(cls_id))
        bbox = BoundingBox(float(x1), float(y1), float(x2), float(y2))
        detections.append((bbox, float(conf), cls_id, cls_name))

    detections.sort(key=lambda x: x[1], reverse=True)
    return detections


# ---------------------------------------------------------------------------
# Class-name → taxonomy mapping
# ---------------------------------------------------------------------------


def _map_to_taxonomy(
    class_id: int,
    class_name: str,
) -> tuple[str, str] | None:
    """
    Map a YOLOv7 raw class to a taxonomy category.

    Returns (category_key, description) or None if the detection is not
    a traffic sign (e.g. a car, pedestrian, etc. from a COCO model).
    """
    from sign_taxonomy import map_class_to_category, CATEGORY_DESCRIPTIONS

    category = map_class_to_category(class_id, class_name)
    if category is None:
        return None

    desc = CATEGORY_DESCRIPTIONS.get(category, category.replace("_", " ").title())
    return category, desc


# ---------------------------------------------------------------------------
# Full detection pipeline (single image)
# ---------------------------------------------------------------------------


def detect_signs_in_image(
    image: Image.Image,
    conf_threshold: float | None = None,
    iou_threshold: float | None = None,
    img_size: int | None = None,
    max_detections: int | None = None,
) -> list[SignDetection]:
    """
    Run YOLOv7 on a single PIL image and return traffic sign detections.

    Non-sign classes (cars, people, etc.) are filtered out via the
    taxonomy mapping so only actual traffic signs are returned.
    """
    conf_threshold = conf_threshold or float(
        os.environ.get("DETECTION_CONFIDENCE_THRESHOLD", "0.25")
    )
    iou_threshold = iou_threshold or float(
        os.environ.get("NMS_IOU_THRESHOLD", "0.45")
    )
    img_size = img_size or int(os.environ.get("YOLOV7_IMG_SIZE", "640"))
    max_detections = max_detections or int(
        os.environ.get("MAX_DETECTIONS_PER_IMAGE", "50")
    )

    raw = _run_yolov7(
        image,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        img_size=img_size,
        max_detections=max_detections,
    )

    sign_detections: list[SignDetection] = []
    for bbox, conf, cls_id, cls_name in raw:
        mapped = _map_to_taxonomy(cls_id, cls_name)
        if mapped is None:
            logger.debug("Skipping non-sign class: %s (%d)", cls_name, cls_id)
            continue
        category, description = mapped
        sign_detections.append(
            SignDetection(
                bbox=bbox,
                category=category,
                category_description=description,
                confidence=conf,
                raw_class_id=cls_id,
                raw_class_name=cls_name,
            )
        )

    logger.info(
        "YOLOv7 detected %d raw boxes → %d traffic signs",
        len(raw),
        len(sign_detections),
    )
    return sign_detections


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------


def process_image_url(url: str, **kwargs) -> DetectionResult:
    """Process a single image URL."""
    try:
        image = load_image_from_url(url)
        detections = detect_signs_in_image(image, **kwargs)
        return DetectionResult(
            image_url=url,
            image_width=image.size[0],
            image_height=image.size[1],
            detections=detections,
        )
    except Exception as e:
        logger.exception("Failed to process image: %s", url)
        return DetectionResult(
            image_url=url,
            image_width=0,
            image_height=0,
            detections=[],
            error=str(e),
        )


def process_image_path(path: str, **kwargs) -> DetectionResult:
    """Process a local image file."""
    try:
        image = load_image_from_path(path)
        detections = detect_signs_in_image(image, **kwargs)
        return DetectionResult(
            image_url=path,
            image_width=image.size[0],
            image_height=image.size[1],
            detections=detections,
        )
    except Exception as e:
        logger.exception("Failed to process image: %s", path)
        return DetectionResult(
            image_url=path,
            image_width=0,
            image_height=0,
            detections=[],
            error=str(e),
        )
