"""
utils.py — Shared utilities for LISA All-Classes SSD
=====================================================
Adapted from: https://github.com/georgesung/ssd_tensorflow_traffic_sign_detection

Provides:
  * IoU calculation (vectorised NumPy)
  * Non-Maximum Suppression
  * Bounding-box drawing / visualisation helpers
  * Image loading/saving wrappers

All box coordinates are in normalised [0, 1] space unless noted otherwise.
"""

import csv
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np

from settings import CLASSES, CONF_THRESH, NMS_IOU_THRESH


# ---------------------------------------------------------------------------
# Bounding-box IoU
# ---------------------------------------------------------------------------

def calc_iou(box_a: Tuple[float, float, float, float],
             box_b: Tuple[float, float, float, float]) -> float:
    """Intersection-over-Union between two boxes.

    Args:
        box_a, box_b: (x1, y1, x2, y2) normalised coordinates.

    Returns:
        IoU in [0, 1].
    """
    # Intersection rectangle
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter   = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union  = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def calc_iou_batch(boxes: np.ndarray,
                   query: np.ndarray) -> np.ndarray:
    """Vectorised IoU between one query box and an array of boxes.

    Args:
        boxes : (N, 4) array of (x1, y1, x2, y2).
        query : (4,)   array of (x1, y1, x2, y2).

    Returns:
        (N,) IoU values.
    """
    ix1 = np.maximum(boxes[:, 0], query[0])
    iy1 = np.maximum(boxes[:, 1], query[1])
    ix2 = np.minimum(boxes[:, 2], query[2])
    iy2 = np.minimum(boxes[:, 3], query[3])

    inter_w = np.maximum(0.0, ix2 - ix1)
    inter_h = np.maximum(0.0, iy2 - iy1)
    inter   = inter_w * inter_h

    area_boxes = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * \
                 np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    area_query = max(0.0, query[2] - query[0]) * max(0.0, query[3] - query[1])
    union      = area_boxes + area_query - inter

    return np.where(union > 0, inter / union, 0.0)


# ---------------------------------------------------------------------------
# Non-Maximum Suppression
# ---------------------------------------------------------------------------

def nms(boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        iou_thresh: float = NMS_IOU_THRESH,
        conf_thresh: float = CONF_THRESH,
        sign_map: Dict[int, str] = None
        ) -> List[Dict]:
    """Non-Maximum Suppression over raw SSD predictions.

    Adapted from the original tutorial's nms() function but works across
    all 47 LISA classes rather than just 2.

    Args:
        boxes     : (N, 4) predicted box coordinates (normalised).
        scores    : (N,)   maximum confidence per anchor.
        class_ids : (N,)   predicted class index per anchor.
        iou_thresh: overlap threshold for suppression.
        conf_thresh: minimum confidence to keep a detection.
        sign_map  : optional {class_idx: class_name} dict for labelling.

    Returns:
        List of dicts: {box, class_id, class_name, confidence}.
    """
    if sign_map is None:
        sign_map = {i: name for i, name in enumerate(CLASSES)}

    # Filter background and low-confidence predictions
    keep_mask = (scores >= conf_thresh) & (class_ids > 0)
    boxes     = boxes[keep_mask]
    scores    = scores[keep_mask]
    class_ids = class_ids[keep_mask]

    if len(boxes) == 0:
        return []

    # Sort descending by confidence
    order = np.argsort(scores)[::-1]
    boxes     = boxes[order]
    scores    = scores[order]
    class_ids = class_ids[order]

    suppressed = np.zeros(len(boxes), dtype=bool)
    results: List[Dict] = []

    for i in range(len(boxes)):
        if suppressed[i]:
            continue
        results.append({
            "box"        : boxes[i].tolist(),
            "class_id"   : int(class_ids[i]),
            "class_name" : sign_map.get(int(class_ids[i]), "unknown"),
            "confidence" : float(scores[i]),
        })
        # Suppress boxes with high IoU that predict the same class
        for j in range(i + 1, len(boxes)):
            if suppressed[j]:
                continue
            if class_ids[j] == class_ids[i]:
                iou = calc_iou(boxes[i], boxes[j])
                if iou > iou_thresh:
                    suppressed[j] = True

    return results


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

# Distinct BGR colours for sign-class groups (cycling if > 47 classes)
_PALETTE = [
    (0,   0,   255),  # red
    (0,   128, 255),  # orange
    (0,   255, 255),  # yellow
    (0,   255, 0),    # green
    (255, 0,   0),    # blue
    (255, 0,   255),  # magenta
    (128, 0,   255),  # purple
    (0,   255, 128),  # teal
]


def _class_color(class_id: int) -> Tuple[int, int, int]:
    return _PALETTE[class_id % len(_PALETTE)]


def draw_detections(image: np.ndarray,
                    detections: List[Dict],
                    img_h: int = None,
                    img_w: int = None) -> np.ndarray:
    """Draw bounding boxes and labels on a BGR image.

    Args:
        image      : HxWxC NumPy array (BGR, uint8).
        detections : output of nms().
        img_h, img_w: image dimensions (inferred from image if None).

    Returns:
        Annotated image (same array, modified in-place).
    """
    h = img_h or image.shape[0]
    w = img_w or image.shape[1]

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        # Convert normalised → pixel
        px1 = int(x1 * w)
        py1 = int(y1 * h)
        px2 = int(x2 * w)
        py2 = int(y2 * h)

        color = _class_color(det["class_id"])
        cv2.rectangle(image, (px1, py1), (px2, py2), color, 2)

        label = f"{det['class_name']} {det['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (px1, py1 - th - 4), (px1 + tw, py1), color, -1)
        cv2.putText(image, label, (px1, py1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return image


# ---------------------------------------------------------------------------
# Sign-name map loader (reads signnames.csv)
# ---------------------------------------------------------------------------

def load_sign_map(csv_path: str = "signnames.csv") -> Dict[int, str]:
    """Load {class_id: name} mapping from signnames.csv."""
    mapping: Dict[int, str] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[int(row["ClassId"])] = row["SignName"]
    return mapping


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def load_image(path: str,
               target_h: int,
               target_w: int,
               as_float: bool = True) -> np.ndarray:
    """Load and resize an image to (target_h, target_w, 3).

    Returns normalised float32 in [-1, 1] if as_float=True, else uint8.
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img = cv2.resize(img, (target_w, target_h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if as_float:
        img = img.astype(np.float32) / 127.5 - 1.0
    return img


def save_annotated(image: np.ndarray, detections: List[Dict], out_path: str):
    """Draw detections and save to out_path (creates dirs as needed)."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    annotated = draw_detections(image.copy(), detections)
    # Convert RGB → BGR for OpenCV save
    cv2.imwrite(out_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
