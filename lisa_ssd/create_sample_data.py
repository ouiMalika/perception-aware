"""
create_sample_data.py — Synthetic LISA-Compatible Dataset Generator
====================================================================
Generates a small, fully-synthetic dataset in the exact directory layout
that dataset.py and data_prep.py expect.  No real LISA download required.

Use this to:
  1. Verify the full pipeline (dataset → data_prep → train → inference) runs
  2. Develop and debug without the 7.7 GB LISA download

Generated layout
----------------
<out_dir>/
  stop/
    frameAnnotations.csv
    frames/stop_1/frame000.png  ...
  speedLimit35/
    frameAnnotations.csv
    frames/speedLimit35_1/frame000.png  ...
  pedestrianCrossing/
    ...
  ... (N_CLASSES_SAMPLE classes, each with N_IMAGES_PER_CLASS images)

Each synthetic image is a 640×480 BGR frame with a randomly coloured
rectangle representing a "traffic sign".

Usage
-----
  python lisa_ssd/create_sample_data.py --out ./data/lisa
  python lisa_ssd/dataset.py --lisa_dir ./data/lisa
  python lisa_ssd/data_prep.py
  python lisa_ssd/train.py --epochs 3
  python lisa_ssd/inference.py -m image -i ./data/lisa/stop/frames/stop_1/
"""

import argparse
import csv
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np

# Use a subset of classes so generation is fast.  Expand this list to
# generate data for more classes.
SAMPLE_CLASSES = [
    "stop",
    "speedLimit35",
    "pedestrianCrossing",
    "yield",
    "doNotEnter",
    "keepRight",
    "signalAhead",
    "noLeftTurn",
]

# Generation parameters
N_IMAGES_PER_CLASS = 40   # images per sign class
IMG_W, IMG_H = 640, 480   # synthetic frame size (pixels)
SIGN_W_RANGE = (40, 120)  # sign width  in pixels
SIGN_H_RANGE = (40, 120)  # sign height in pixels

ANNOT_HEADER = (
    "Filename;Annotation tag;"
    "Upper left corner X;Upper left corner Y;"
    "Lower right corner X;Lower right corner Y;"
    "Origin file;Origin frame number;Origin track;Origin track frame number"
)

# Repeatable randomness
RNG = random.Random(42)
NP_RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------

def _random_colour() -> tuple:
    return (
        RNG.randint(50, 255),
        RNG.randint(50, 255),
        RNG.randint(50, 255),
    )


def _make_frame(cls_name: str) -> tuple[np.ndarray, list[int]]:
    """Create one synthetic BGR frame with a sign rectangle.

    Returns:
        image  : (H, W, 3) uint8 BGR image
        box    : [x1, y1, x2, y2] pixel coords of the sign
    """
    # Noisy background
    bg = NP_RNG.integers(60, 200, size=(IMG_H, IMG_W, 3), dtype=np.uint8)
    img = bg.copy()

    # Sign dimensions and position
    sw = RNG.randint(*SIGN_W_RANGE)
    sh = RNG.randint(*SIGN_H_RANGE)
    x1 = RNG.randint(10, IMG_W - sw - 10)
    y1 = RNG.randint(10, IMG_H - sh - 10)
    x2 = x1 + sw
    y2 = y1 + sh

    # Draw filled sign rectangle + label
    cv2.rectangle(img, (x1, y1), (x2, y2), _random_colour(), -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 2)
    font_scale = 0.35
    cv2.putText(img, cls_name[:8], (x1 + 3, y1 + sh // 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1,
                cv2.LINE_AA)

    return img, [x1, y1, x2, y2]


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate(out_dir: str, classes: list[str], n_per_class: int) -> None:
    root = Path(out_dir)

    for cls_name in classes:
        cls_dir   = root / cls_name
        frames_dir = cls_dir / "frames" / f"{cls_name}_1"
        frames_dir.mkdir(parents=True, exist_ok=True)

        annot_rows: list[tuple] = []

        for i in range(n_per_class):
            img, box = _make_frame(cls_name)
            fname     = f"frame{i:05d}.png"
            rel_path  = f"frames/{cls_name}_1/{fname}"
            abs_path  = frames_dir / fname

            cv2.imwrite(str(abs_path), img)

            x1, y1, x2, y2 = box
            annot_rows.append((
                rel_path, cls_name,
                x1, y1, x2, y2,
                fname, i, f"{cls_name}_1", i,
            ))

        # Write frameAnnotations.csv
        annot_path = cls_dir / "frameAnnotations.csv"
        with open(annot_path, "w", newline="") as f:
            f.write(ANNOT_HEADER + "\n")
            writer = csv.writer(f, delimiter=";")
            writer.writerows(annot_rows)

        print(f"  [ok] {cls_name:30s}  {n_per_class} images")

    print(f"\nGenerated {len(classes) * n_per_class} synthetic frames → {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a synthetic LISA-compatible dataset for pipeline testing."
    )
    parser.add_argument(
        "--out", default="./data/lisa",
        help="Output root directory (default: ./data/lisa).",
    )
    parser.add_argument(
        "--classes", nargs="+", default=SAMPLE_CLASSES,
        help="Sign classes to generate (default: 8-class sample).",
    )
    parser.add_argument(
        "--n", type=int, default=N_IMAGES_PER_CLASS,
        help=f"Images per class (default: {N_IMAGES_PER_CLASS}).",
    )
    args = parser.parse_args()

    # Validate requested classes
    from settings import CLASSES as ALL_CLASSES
    bad = [c for c in args.classes if c not in ALL_CLASSES]
    if bad:
        sys.exit(f"Unknown class(es): {bad}\nValid classes: {ALL_CLASSES[1:]}")

    print(f"Generating synthetic LISA dataset → {args.out}")
    print(f"  Classes : {args.classes}")
    print(f"  Images  : {args.n} per class\n")

    generate(args.out, args.classes, args.n)

    print("\nNext steps:")
    print(f"  python lisa_ssd/dataset.py  --lisa_dir {args.out}")
    print( "  python lisa_ssd/data_prep.py")
    print( "  python lisa_ssd/train.py --epochs 5")
