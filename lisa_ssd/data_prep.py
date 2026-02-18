"""
data_prep.py — Anchor-Box Matching & Label Encoding
====================================================
Adapted from: https://github.com/georgesung/ssd_tensorflow_traffic_sign_detection

The original data_prep.py matched ground-truth boxes to the 4 default boxes
in a 2-class setup.  This module keeps the same algorithm but:

  * Operates on ALL 47 LISA sign classes (class index 1-47)
  * Accepts the normalised Annotation list from dataset.py
  * Is TF-framework-agnostic (pure NumPy) so it can feed any training loop

Output arrays (stored in a prepared pickle):
  conf_labels : (N, A)      int32  — class index per anchor (0 = background)
  loc_labels  : (N, A, 4)  float32 — GT box [x1,y1,x2,y2] for positive anchors
  loc_mask    : (N, A)     float32 — 1 for positive anchors, 0 otherwise
  image_paths : (N,)                — absolute paths to the N images

Where A = NUM_ANCHORS = 7812  (31×48 + 15×23 + 8×12 + 4×6) × 4 defaults

Algorithm (identical to original find_gt_boxes):
  For each image, for each GT box, iterate all feature-map cells and default
  boxes, compute IoU, and mark the anchor with the highest overlap (≥ IOU_THRESH)
  as positive with the GT class label and coordinates.
"""

from __future__ import annotations

import os
import pickle
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from settings import (
    CLASSES, CLASS_TO_IDX, NUM_CLASSES,
    DEFAULT_BOXES, NUM_DEFAULT_BOXES,
    FEATURE_MAPS, NUM_ANCHORS,
    IOU_THRESH, IMG_H, IMG_W,
    PICKLE_DIR,
)
from utils import calc_iou


# ---------------------------------------------------------------------------
# Default (anchor) box generation
# ---------------------------------------------------------------------------

def generate_default_boxes() -> np.ndarray:
    """Generate all anchor boxes for the model.

    Mirrors the default-box logic scattered through the original data_prep.py.

    Returns:
        anchors : (NUM_ANCHORS, 4) float32 — [x1, y1, x2, y2] normalised [0,1].
    """
    anchors: List[List[float]] = []

    for fm_h, fm_w in FEATURE_MAPS:
        for row in range(fm_h):
            for col in range(fm_w):
                # Cell centre in normalised coordinates
                cx = (col + 0.5) / fm_w
                cy = (row + 0.5) / fm_h

                for dx1, dy1, dx2, dy2 in DEFAULT_BOXES:
                    x1 = cx + dx1
                    y1 = cy + dy1
                    x2 = cx + dx2
                    y2 = cy + dy2
                    anchors.append([x1, y1, x2, y2])

    return np.array(anchors, dtype=np.float32)  # (NUM_ANCHORS, 4)


# Pre-compute once at import time
ANCHORS = generate_default_boxes()
assert ANCHORS.shape == (NUM_ANCHORS, 4), (
    f"Expected {NUM_ANCHORS} anchors, got {ANCHORS.shape[0]}"
)


# ---------------------------------------------------------------------------
# Per-image label encoding
# ---------------------------------------------------------------------------

def encode_image(
    gt_boxes: List[List[float]],
    gt_class_ids: List[int],
    anchors: np.ndarray = ANCHORS,
    iou_thresh: float = IOU_THRESH,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode one image's GT annotations into per-anchor training labels.

    Adapted from find_gt_boxes() in the original data_prep.py with the same
    greedy-assignment strategy:
      • For each GT box, find the anchor with max IoU
      • If max IoU ≥ iou_thresh, assign that GT to the anchor
      • Any anchor already assigned keeps its first assignment

    Args:
        gt_boxes     : list of [x1,y1,x2,y2] normalised GT boxes for this image
        gt_class_ids : corresponding class indices (1-47, never 0 = background)
        anchors      : (A, 4) array of anchor coordinates
        iou_thresh   : minimum IoU for a positive match

    Returns:
        conf_labels : (A,)    int32  — class index (0 = background)
        loc_labels  : (A, 4) float32 — GT box coords (meaningful only for pos.)
        loc_mask    : (A,)   float32 — 1 for positive anchors, 0 otherwise
    """
    A = len(anchors)
    conf_labels = np.zeros(A, dtype=np.int32)
    loc_labels  = np.zeros((A, 4), dtype=np.float32)
    loc_mask    = np.zeros(A, dtype=np.float32)

    for gt_box, cls_id in zip(gt_boxes, gt_class_ids):
        gt_box = np.array(gt_box, dtype=np.float32)

        # IoU between this GT box and every anchor
        ix1 = np.maximum(anchors[:, 0], gt_box[0])
        iy1 = np.maximum(anchors[:, 1], gt_box[1])
        ix2 = np.minimum(anchors[:, 2], gt_box[2])
        iy2 = np.minimum(anchors[:, 3], gt_box[3])

        inter_w = np.maximum(0.0, ix2 - ix1)
        inter_h = np.maximum(0.0, iy2 - iy1)
        inter   = inter_w * inter_h

        area_anchors = (np.maximum(0.0, anchors[:, 2] - anchors[:, 0]) *
                        np.maximum(0.0, anchors[:, 3] - anchors[:, 1]))
        area_gt = max(0.0, gt_box[2] - gt_box[0]) * max(0.0, gt_box[3] - gt_box[1])
        union   = area_anchors + area_gt - inter

        ious = np.where(union > 0, inter / union, 0.0)  # (A,)

        # Candidate: anchors exceeding threshold
        pos_indices = np.where(ious >= iou_thresh)[0]

        # Fallback: always assign the best-overlap anchor (SSD convention)
        best_idx = np.argmax(ious)
        if ious[best_idx] > 0:
            pos_indices = np.union1d(pos_indices, [best_idx])

        for idx in pos_indices:
            # First-come assignment: don't overwrite already-matched anchors
            if loc_mask[idx] == 0:
                conf_labels[idx] = cls_id
                loc_labels[idx]  = gt_box
                loc_mask[idx]    = 1.0

    return conf_labels, loc_labels, loc_mask


# ---------------------------------------------------------------------------
# Full dataset preparation
# ---------------------------------------------------------------------------

def prepare_dataset(
    annotations_by_image: Dict[str, List[dict]],
    anchors: np.ndarray = ANCHORS,
    iou_thresh: float = IOU_THRESH,
    verbose: bool = True,
) -> dict:
    """Encode all images' annotations into training arrays.

    Mirrors do_data_prep() in the original data_prep.py but extended to all
    47 sign classes and using pure NumPy for framework independence.

    Args:
        annotations_by_image : {image_path: [Annotation, ...]}
                               Produced by group_by_image() below.
        anchors   : (A, 4) anchor array — use ANCHORS constant by default.
        iou_thresh: IoU threshold for positive anchor assignment.
        verbose   : print progress every 500 images.

    Returns:
        dict with keys:
          "image_paths" : list[str]          — N image paths
          "conf_labels" : np.ndarray (N,A)   int32
          "loc_labels"  : np.ndarray (N,A,4) float32
          "loc_mask"    : np.ndarray (N,A)   float32
    """
    image_paths_out = []
    conf_labels_out = []
    loc_labels_out  = []
    loc_mask_out    = []

    total = len(annotations_by_image)
    for i, (img_path, anns) in enumerate(annotations_by_image.items()):
        gt_boxes    = [a["norm_box"] for a in anns]
        gt_class_ids = [a["class_idx"] for a in anns]

        conf, loc, mask = encode_image(gt_boxes, gt_class_ids, anchors, iou_thresh)

        # Only keep images that have at least one positive anchor (same filter
        # as the original do_data_prep which skips images with no matches)
        if mask.sum() == 0:
            continue

        image_paths_out.append(img_path)
        conf_labels_out.append(conf)
        loc_labels_out.append(loc)
        loc_mask_out.append(mask)

        if verbose and (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{total} images …")

    N = len(image_paths_out)
    if verbose:
        print(f"\nPrepared {N} images with at least one positive anchor "
              f"(discarded {total - N} images with no matching anchors).")

    return {
        "image_paths": image_paths_out,
        "conf_labels": np.stack(conf_labels_out).astype(np.int32),    # (N, A)
        "loc_labels" : np.stack(loc_labels_out).astype(np.float32),   # (N, A, 4)
        "loc_mask"   : np.stack(loc_mask_out).astype(np.float32),     # (N, A)
    }


# ---------------------------------------------------------------------------
# Helper: group flat Annotation list by image path
# ---------------------------------------------------------------------------

def group_by_image(annotations: List[dict]) -> Dict[str, List[dict]]:
    """Group a flat annotation list by image path.

    Some images have multiple bounding boxes (different sign instances).
    encode_image() expects all GT boxes for a single image at once.

    Returns:
        {image_path: [ann, ann, …]}
    """
    groups: Dict[str, List[dict]] = defaultdict(list)
    for ann in annotations:
        groups[ann["image_path"]].append(ann)
    return dict(groups)


# ---------------------------------------------------------------------------
# CLI entry-point: python data_prep.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from dataset import load_pickle, save_pickle

    parser = argparse.ArgumentParser(
        description="Match GT boxes to anchors and save prepared training data."
    )
    parser.add_argument(
        "--raw", default=None,
        help=f"Input raw pickle. Default: {PICKLE_DIR}/data_raw_{IMG_W}x{IMG_H}.p",
    )
    parser.add_argument(
        "--out", default=None,
        help=f"Output prepared pickle. Default: {PICKLE_DIR}/data_prep_{IMG_W}x{IMG_H}.p",
    )
    parser.add_argument(
        "--iou", type=float, default=IOU_THRESH,
        help=f"IoU threshold for positive anchor matching (default {IOU_THRESH}).",
    )
    args = parser.parse_args()

    raw_path = args.raw or os.path.join(
        PICKLE_DIR, f"data_raw_{IMG_W}x{IMG_H}.p"
    )
    out_path = args.out or os.path.join(
        PICKLE_DIR, f"data_prep_{IMG_W}x{IMG_H}.p"
    )

    print(f"Loading raw annotations from: {raw_path}")
    raw_annotations = load_pickle(raw_path)

    print(f"Grouping {len(raw_annotations)} annotations by image …")
    by_image = group_by_image(raw_annotations)
    print(f"  → {len(by_image)} unique images")

    print("Matching GT boxes to anchor boxes …")
    prepared = prepare_dataset(by_image, iou_thresh=args.iou)

    save_pickle(prepared, out_path)
    print(f"\nDone. Saved prepared data → {out_path}")
    print(f"  conf_labels : {prepared['conf_labels'].shape}")
    print(f"  loc_labels  : {prepared['loc_labels'].shape}")
    print(f"  loc_mask    : {prepared['loc_mask'].shape}")
