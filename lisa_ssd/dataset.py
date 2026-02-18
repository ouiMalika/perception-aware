"""
dataset.py — LISA All-Classes Dataset Loader
============================================
Adapted from: https://github.com/georgesung/ssd_tensorflow_traffic_sign_detection
              (data_gathering/create_pickle.py)

The original create_pickle.py loaded only stop signs and pedestrian-crossing
annotations from the LISA dataset.  This module generalises that to ALL 47
sign categories by:

  1. Walking every per-class subdirectory under LISA_DIR
  2. Parsing the semicolon-delimited frameAnnotations.csv in each
  3. Returning a unified annotation list compatible with data_prep.py

Expected LISA directory layout
-------------------------------
lisa/
  stop/
    frameAnnotations.csv       ← semicolon-separated, 1 header row
    frames/
      stop_1/
        frame000.png
        ...
  pedestrianCrossing/
    frameAnnotations.csv
    frames/
      ...
  <signClass>/
    ...

frameAnnotations.csv columns (0-indexed, semicolon delimiter):
  0  Filename          relative path from the class dir, e.g. frames/stop_1/frame0.png
  1  Annotation tag    sign class name (should match the directory name)
  2  Upper left X      pixel, absolute
  3  Upper left Y      pixel, absolute
  4  Lower right X     pixel, absolute
  5  Lower right Y     pixel, absolute
  6  Origin file
  7  Origin frame number
  8  Origin track
  9  Origin track frame number

Usage
-----
    from dataset import load_lisa_dataset, save_pickle, load_pickle

    raw = load_lisa_dataset("./data/lisa")
    save_pickle(raw, "./data/pickles/data_raw_400x260.p")

    # Later:
    raw = load_pickle("./data/pickles/data_raw_400x260.p")
"""

from __future__ import annotations

import csv
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

from settings import (
    CLASSES, CLASS_TO_IDX,
    IMG_H, IMG_W,
    LISA_DIR, PICKLE_DIR,
    LISA_ANNOT_FILENAME, LISA_ANNOT_DELIMITER,
)


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

# One annotation entry:
# {
#   "image_path" : str        — absolute path to image file
#   "class_idx"  : int        — index into CLASSES list (1-47)
#   "class_name" : str        — e.g. "stop"
#   "box"        : [x1,y1,x2,y2]  — absolute pixel coords in *original* image
#   "img_w"      : int        — original image width  (set during load)
#   "img_h"      : int        — original image height (set during load)
# }
Annotation = Dict


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_delimiter(annot_file: str) -> str:
    """Sniff the delimiter of a LISA annotation CSV.

    Some LISA releases use semicolons, others commas.  This auto-detects.
    """
    with open(annot_file, newline="") as f:
        sample = f.read(2048)
    if ";" in sample:
        return ";"
    return ","


def _parse_class_dir(
    cls_dir: Path,
    cls_name: str,
    class_idx: int,
    missing_ok: bool = True,
) -> List[Annotation]:
    """Parse one LISA sign-class directory.

    Returns a list of Annotation dicts.  Images that cannot be found on disk
    are silently skipped (mirrors the behaviour of the original create_pickle.py).
    """
    annot_path = cls_dir / LISA_ANNOT_FILENAME
    if not annot_path.exists():
        if missing_ok:
            return []
        raise FileNotFoundError(f"Annotation file not found: {annot_path}")

    delimiter = _detect_delimiter(str(annot_path))
    annotations: List[Annotation] = []

    with open(annot_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        header = next(reader, None)  # skip header row

        for row in reader:
            if len(row) < 6:
                continue  # malformed row

            rel_filename = row[0].strip()
            ann_tag      = row[1].strip()

            try:
                x1 = int(float(row[2]))
                y1 = int(float(row[3]))
                x2 = int(float(row[4]))
                y2 = int(float(row[5]))
            except (ValueError, IndexError):
                continue

            # Resolve absolute image path
            img_path = cls_dir / rel_filename
            if not img_path.exists():
                # Some datasets store paths relative to the LISA root
                img_path = cls_dir.parent / rel_filename
            if not img_path.exists():
                continue  # image not found — skip silently

            annotations.append({
                "image_path": str(img_path),
                "class_idx" : class_idx,
                "class_name": cls_name,
                "box"       : [x1, y1, x2, y2],
                "img_w"     : None,  # filled later if needed
                "img_h"     : None,
            })

    return annotations


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_lisa_dataset(
    lisa_dir: str = LISA_DIR,
    classes: Optional[List[str]] = None,
    verbose: bool = True,
) -> List[Annotation]:
    """Load ALL sign-class annotations from a LISA dataset directory.

    Replaces the original create_pickle.py which was hard-coded to two classes.

    Args:
        lisa_dir : path to the root of the extracted LISA dataset.
        classes  : subset of CLASSES to load; loads all 47 if None.
        verbose  : print per-class statistics.

    Returns:
        List of Annotation dicts (one per bounding-box annotation).
    """
    lisa_root = Path(lisa_dir)
    if not lisa_root.exists():
        raise FileNotFoundError(
            f"LISA directory not found: {lisa_dir}\n"
            "Download from: http://cvrr-nas.ucsd.edu/LISA/lisa-traffic-sign-dataset.html"
        )

    target_classes = classes if classes is not None else CLASSES[1:]  # skip bg

    all_annotations: List[Annotation] = []
    found_classes: List[str] = []

    for cls_name in target_classes:
        class_idx = CLASS_TO_IDX.get(cls_name)
        if class_idx is None:
            continue  # unknown class, skip

        cls_dir = lisa_root / cls_name
        if not cls_dir.exists():
            if verbose:
                print(f"  [skip] {cls_name:30s} — directory not found")
            continue

        anns = _parse_class_dir(cls_dir, cls_name, class_idx)
        all_annotations.extend(anns)
        found_classes.append(cls_name)

        if verbose:
            print(f"  [ok]   {cls_name:30s}  {len(anns):5d} annotations")

    if verbose:
        print(f"\nLoaded {len(all_annotations)} annotations "
              f"across {len(found_classes)}/{len(target_classes)} classes.")

    return all_annotations


def load_merged_csv(
    merged_csv: str,
    lisa_dir: str = LISA_DIR,
    verbose: bool = True,
) -> List[Annotation]:
    """Parse a pre-merged allAnnotations.csv (alternative to per-class dirs).

    Some LISA releases supply a single merged CSV.  The delimiter may be mixed
    (semicolons AND commas); this function normalises it on the fly.

    Args:
        merged_csv: path to the merged CSV file.
        lisa_dir  : LISA root (used to resolve relative image paths).
        verbose   : print statistics.

    Returns:
        List of Annotation dicts.
    """
    lisa_root = Path(lisa_dir)
    annotations: List[Annotation] = []
    unknown_classes: set = set()

    with open(merged_csv, newline="", encoding="utf-8") as f:
        # Normalise mixed semicolon/comma delimiters
        content = f.read().replace(";", ",")

    from io import StringIO
    reader = csv.reader(StringIO(content))
    header = next(reader, None)

    for row in reader:
        if len(row) < 6:
            continue

        rel_filename = row[0].strip()
        cls_name     = row[1].strip()
        class_idx    = CLASS_TO_IDX.get(cls_name)

        if class_idx is None:
            unknown_classes.add(cls_name)
            continue

        try:
            x1 = int(float(row[2]))
            y1 = int(float(row[3]))
            x2 = int(float(row[4]))
            y2 = int(float(row[5]))
        except (ValueError, IndexError):
            continue

        img_path = lisa_root / rel_filename
        if not img_path.exists():
            continue

        annotations.append({
            "image_path": str(img_path),
            "class_idx" : class_idx,
            "class_name": cls_name,
            "box"       : [x1, y1, x2, y2],
            "img_w"     : None,
            "img_h"     : None,
        })

    if verbose:
        print(f"Loaded {len(annotations)} annotations from {merged_csv}")
        if unknown_classes:
            print(f"  Skipped unknown classes: {sorted(unknown_classes)}")

    return annotations


# ---------------------------------------------------------------------------
# Normalise box coordinates
# ---------------------------------------------------------------------------

def normalise_boxes(
    annotations: List[Annotation],
    target_w: int = IMG_W,
    target_h: int = IMG_H,
) -> List[Annotation]:
    """Convert absolute pixel boxes to normalised [0, 1] coordinates.

    This mirrors the box normalisation done inside the original find_gt_boxes()
    but as a separate preprocessing step.

    Also stores target_w / target_h into each annotation dict.

    Args:
        annotations: list of raw Annotation dicts from load_lisa_dataset().
        target_w, target_h: model input dimensions.

    Returns:
        New list of Annotation dicts with a "norm_box" key added.
        norm_box = [x1_norm, y1_norm, x2_norm, y2_norm]
    """
    result = []
    for ann in annotations:
        x1, y1, x2, y2 = ann["box"]

        # Guard against malformed boxes
        if x2 <= x1 or y2 <= y1:
            continue

        # We don't know the original image size without loading it; the LISA
        # dataset uses various resolutions.  We assume the boxes are already
        # in the original image pixel space.  If img_w / img_h are present
        # (from a prior imread call), use them; otherwise fall back to target.
        orig_w = ann.get("img_w") or target_w
        orig_h = ann.get("img_h") or target_h

        norm = [
            x1 / orig_w,
            y1 / orig_h,
            x2 / orig_w,
            y2 / orig_h,
        ]
        norm = [max(0.0, min(1.0, v)) for v in norm]  # clamp to [0,1]

        new_ann = dict(ann)
        new_ann["norm_box"] = norm
        result.append(new_ann)

    return result


# ---------------------------------------------------------------------------
# Pickle helpers (compatible with original create_pickle.py output)
# ---------------------------------------------------------------------------

def save_pickle(data: object, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved pickle → {path}")


def load_pickle(path: str) -> object:
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# CLI entry-point: python dataset.py --lisa_dir ./data/lisa
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Load all LISA sign annotations and save as pickle."
    )
    parser.add_argument("--lisa_dir", default=LISA_DIR,
                        help="Root of extracted LISA dataset.")
    parser.add_argument("--out", default=None,
                        help="Output pickle path. Default: PICKLE_DIR/data_raw_WxH.p")
    parser.add_argument("--merged_csv", default=None,
                        help="Path to a merged allAnnotations.csv (optional).")
    args = parser.parse_args()

    print(f"Loading LISA dataset from: {args.lisa_dir}")
    if args.merged_csv:
        raw = load_merged_csv(args.merged_csv, lisa_dir=args.lisa_dir)
    else:
        raw = load_lisa_dataset(args.lisa_dir)

    raw = normalise_boxes(raw)

    out_path = args.out or os.path.join(
        PICKLE_DIR, f"data_raw_{IMG_W}x{IMG_H}.p"
    )
    save_pickle(raw, out_path)
    print(f"\nDone. {len(raw)} annotated images → {out_path}")
