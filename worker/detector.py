"""
Two-stage traffic sign detection pipeline.

Stage 1: YOLOv8 localises candidate sign regions (bounding boxes).
Stage 2: CLIP zero-shot classifies each crop into the sign taxonomy.

This architecture decouples *where* signs are from *what* they mean,
giving us flexible coverage of 25+ sign categories without needing
category-specific training data.
"""

import io
import logging
import os
from dataclasses import dataclass, field

import numpy as np
import requests
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded singletons (heavy models loaded once per worker process)
# ---------------------------------------------------------------------------
_yolo_model = None
_clip_model = None
_clip_processor = None
_clip_tokenizer = None


def _get_yolo():
    """Load YOLOv8 model (downloads on first call)."""
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO

        model_size = os.environ.get("YOLO_MODEL_SIZE", "yolov8m")
        logger.info("Loading YOLO model: %s", model_size)
        _yolo_model = YOLO(f"{model_size}.pt")
    return _yolo_model


def _get_clip():
    """Load CLIP model + processor (downloads on first call)."""
    global _clip_model, _clip_processor, _clip_tokenizer
    if _clip_model is None:
        from transformers import CLIPModel, CLIPProcessor

        model_name = "openai/clip-vit-base-patch32"
        logger.info("Loading CLIP model: %s", model_name)
        _clip_model = CLIPModel.from_pretrained(model_name)
        _clip_processor = CLIPProcessor.from_pretrained(model_name)
        _clip_tokenizer = _clip_processor.tokenizer
        _clip_model.eval()
    return _clip_model, _clip_processor


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
    """A single detected and classified traffic sign."""
    bbox: BoundingBox
    category: str
    category_description: str
    confidence: float          # combined YOLO * CLIP confidence
    yolo_confidence: float     # raw YOLO detection confidence
    clip_confidence: float     # raw CLIP classification confidence
    clip_scores: dict = field(default_factory=dict)  # top-5 category scores

    def to_dict(self) -> dict:
        return {
            "bbox": self.bbox.to_dict(),
            "category": self.category,
            "description": self.category_description,
            "confidence": round(self.confidence, 4),
            "yolo_confidence": round(self.yolo_confidence, 4),
            "clip_confidence": round(self.clip_confidence, 4),
            "top_scores": {
                k: round(v, 4) for k, v in self.clip_scores.items()
            },
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
    """Download an image from a URL and return as PIL Image."""
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def load_image_from_path(path: str) -> Image.Image:
    """Load a local image file and return as PIL Image."""
    return Image.open(path).convert("RGB")


# ---------------------------------------------------------------------------
# Stage 1: YOLO sign localisation
# ---------------------------------------------------------------------------

# COCO class IDs that are directly sign-related.
# 9 = traffic light, 11 = stop sign.
_SIGN_RELATED_COCO_IDS = {9, 11}

# COCO classes that are clearly NOT signs — skip these entirely to avoid
# wasting CLIP inference on objects that can never be traffic signs.
_NEVER_SIGN_COCO_IDS = {
    0,   # person
    1,   # bicycle
    2,   # car
    3,   # motorcycle
    5,   # bus
    6,   # train
    7,   # truck
    14,  # bird
    15,  # cat
    16,  # dog
    24,  # backpack
    25,  # umbrella
    26,  # handbag
    28,  # suitcase
    39,  # bottle
    56,  # chair
    57,  # couch
    59,  # bed
    60,  # dining table
    62,  # tv
    63,  # laptop
    64,  # mouse
    66,  # keyboard
    67,  # cell phone
    72,  # refrigerator
}


def _stage1_detect(
    image: Image.Image,
    conf_threshold: float = 0.25,
    max_detections: int = 50,
) -> list[tuple[BoundingBox, float, int]]:
    """
    Run YOLOv8 on an image and return candidate bounding boxes.

    Returns list of (BoundingBox, confidence, class_id) tuples.
    We keep ALL detections and let Stage 2 filter by sign-ness,
    but we favour sign-related COCO classes with a lower threshold.
    """
    model = _get_yolo()
    results = model.predict(
        source=np.array(image),
        conf=conf_threshold,
        max_det=max_detections,
        verbose=False,
    )

    candidates = []
    for result in results:
        boxes = result.boxes
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy()
            conf = float(boxes.conf[i].cpu().numpy())
            cls_id = int(boxes.cls[i].cpu().numpy())

            bbox = BoundingBox(
                x1=float(xyxy[0]),
                y1=float(xyxy[1]),
                x2=float(xyxy[2]),
                y2=float(xyxy[3]),
            )

            # Skip objects that are clearly never signs
            if cls_id in _NEVER_SIGN_COCO_IDS:
                continue

            # Keep sign-related classes always; keep others only at high conf
            if cls_id in _SIGN_RELATED_COCO_IDS or conf >= 0.5:
                candidates.append((bbox, conf, cls_id))

    # Sort by confidence descending
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:max_detections]


# ---------------------------------------------------------------------------
# Stage 2: CLIP zero-shot classification
# ---------------------------------------------------------------------------

def _stage2_classify(
    image: Image.Image,
    bbox: BoundingBox,
    prompts: list[str],
    prompt_to_category: dict[str, str],
    category_descriptions: dict[str, str],
) -> tuple[str, float, dict]:
    """
    Crop the image to the bounding box region and run CLIP zero-shot
    classification against all sign prompts.

    Returns (best_category, clip_confidence, top5_scores).
    """
    model, processor = _get_clip()

    # Crop with a small margin for context
    w, h = image.size
    margin_x = (bbox.x2 - bbox.x1) * 0.1
    margin_y = (bbox.y2 - bbox.y1) * 0.1
    crop_box = (
        max(0, bbox.x1 - margin_x),
        max(0, bbox.y1 - margin_y),
        min(w, bbox.x2 + margin_x),
        min(h, bbox.y2 + margin_y),
    )
    crop = image.crop(crop_box)

    # Minimum crop size check
    if crop.size[0] < 10 or crop.size[1] < 10:
        return "unknown", 0.0, {}

    # Run CLIP
    inputs = processor(
        text=prompts,
        images=crop,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_image[0]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    # Map prompt-level scores back to categories (take max per category)
    category_scores: dict[str, float] = {}
    for idx, prompt in enumerate(prompts):
        cat = prompt_to_category[prompt]
        score = float(probs[idx])
        if cat not in category_scores or score > category_scores[cat]:
            category_scores[cat] = score

    # Sort by score
    sorted_cats = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
    best_cat, best_score = sorted_cats[0]

    # Top-5 for diagnostics (exclude not_a_sign from display)
    top5 = dict(
        (k, v) for k, v in sorted_cats[:6] if k != "not_a_sign"
    )
    # Keep only the top 5 actual sign categories
    top5 = dict(list(top5.items())[:5])

    return best_cat, best_score, top5


# ---------------------------------------------------------------------------
# Combined pipeline
# ---------------------------------------------------------------------------

def detect_signs(
    image: Image.Image,
    prompts: list[str],
    prompt_to_category: dict[str, str],
    category_descriptions: dict[str, str],
    yolo_conf: float = 0.25,
    clip_conf: float = 0.15,
    max_detections: int = 50,
) -> list[SignDetection]:
    """
    Full two-stage detection pipeline on a single image.

    1. YOLO finds candidate regions.
    2. CLIP classifies each region against the sign taxonomy.
    3. Low-confidence CLIP results are filtered out.
    """
    # Stage 1
    candidates = _stage1_detect(image, conf_threshold=yolo_conf, max_detections=max_detections)
    logger.info("Stage 1 (YOLO): %d candidates found", len(candidates))

    detections = []
    for bbox, yolo_conf_val, cls_id in candidates:
        # Stage 2
        category, clip_conf_val, top5 = _stage2_classify(
            image, bbox, prompts, prompt_to_category, category_descriptions
        )

        # Reject objects that CLIP thinks are not signs
        if category == "not_a_sign":
            continue

        # Filter by CLIP confidence
        if clip_conf_val < clip_conf:
            continue

        combined_conf = yolo_conf_val * clip_conf_val
        desc = category_descriptions.get(category, category)

        detections.append(SignDetection(
            bbox=bbox,
            category=category,
            category_description=desc,
            confidence=combined_conf,
            yolo_confidence=yolo_conf_val,
            clip_confidence=clip_conf_val,
            clip_scores=top5,
        ))

    # Sort by combined confidence
    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections[:max_detections]


def detect_signs_in_image(
    image: Image.Image,
    yolo_conf: float | None = None,
    clip_conf: float | None = None,
    max_detections: int | None = None,
) -> list[SignDetection]:
    """
    High-level convenience function. Uses the full taxonomy automatically.
    """
    from sign_taxonomy import (
        ALL_PROMPTS,
        PROMPT_TO_CATEGORY,
        CATEGORY_DESCRIPTIONS,
    )

    yolo_conf = yolo_conf or float(os.environ.get("DETECTION_CONFIDENCE_THRESHOLD", 0.25))
    clip_conf = clip_conf or float(os.environ.get("CLASSIFICATION_CONFIDENCE_THRESHOLD", 0.30))
    max_detections = max_detections or int(os.environ.get("MAX_DETECTIONS_PER_IMAGE", 50))

    return detect_signs(
        image=image,
        prompts=ALL_PROMPTS,
        prompt_to_category=PROMPT_TO_CATEGORY,
        category_descriptions=CATEGORY_DESCRIPTIONS,
        yolo_conf=yolo_conf,
        clip_conf=clip_conf,
        max_detections=max_detections,
    )


def process_image_url(url: str, **kwargs) -> DetectionResult:
    """Process a single image URL through the full pipeline."""
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
    """Process a single local image file through the full pipeline."""
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
