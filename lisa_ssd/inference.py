"""
inference.py — Inference Script for LISA All-Classes AlexNet-SSD (TF2)
=======================================================================
Adapted from: https://github.com/georgesung/ssd_tensorflow_traffic_sign_detection

Original used tf.Session.run() and a saved TF1 graph.
This port uses tf.train.Checkpoint to restore the Keras model and provides
the same three modes as the original:

  image  — annotate one or more image files in an input directory
  video  — process a video file frame-by-frame
  demo   — run on the built-in sample_images/ folder

Usage
-----
  # Single image / batch
  python inference.py -m image -i ./input_images/

  # Video
  python inference.py -m video -i ./my_video.mp4 -o ./output/annotated.mp4

  # Quick demo on sample images bundled with this repo
  python inference.py -m demo
"""

from __future__ import annotations

import argparse
import os
import glob
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from settings import (
    IMG_H, IMG_W, NUM_CHANNELS,
    CONF_THRESH, NMS_IOU_THRESH,
    CHECKPOINT_DIR, OUTPUT_DIR,
)
from model import build_alexnet_ssd, decode_predictions
from utils import nms, draw_detections, load_sign_map, load_image


# ---------------------------------------------------------------------------
# Load model from checkpoint
# ---------------------------------------------------------------------------

def load_model(checkpoint_dir: str = CHECKPOINT_DIR) -> tf.keras.Model:
    """Restore the trained model from the latest checkpoint.

    Mirrors the TF1 saver.restore() call in the original inference.py.
    """
    model = build_alexnet_ssd()
    ckpt  = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=1)

    if manager.latest_checkpoint is None:
        raise FileNotFoundError(
            f"No checkpoint found in {checkpoint_dir}.\n"
            "Train the model first: python train.py"
        )

    ckpt.restore(manager.latest_checkpoint).expect_partial()
    print(f"Restored model from {manager.latest_checkpoint}")
    return model


# ---------------------------------------------------------------------------
# Single-image inference (mirrors run_inference() in original inference.py)
# ---------------------------------------------------------------------------

def run_inference(
    image_bgr: np.ndarray,
    model: tf.keras.Model,
    sign_map: dict,
    conf_thresh: float = CONF_THRESH,
    iou_thresh: float = NMS_IOU_THRESH,
) -> tuple[np.ndarray, list]:
    """Run detection on one BGR image and return annotated image + detections.

    Adapted from run_inference() in the original inference.py.

    Args:
        image_bgr : (H, W, 3) uint8 BGR image (from cv2.imread).
        model     : loaded Keras model.
        sign_map  : {class_id: class_name} from load_sign_map().
        conf_thresh, iou_thresh: detection / NMS thresholds.

    Returns:
        annotated : BGR uint8 image with drawn boxes.
        detections: list of dicts from nms() — one per detected sign.
    """
    orig_h, orig_w = image_bgr.shape[:2]

    # Pre-process: resize → RGB → [-1, 1]
    img_rgb = cv2.cvtColor(
        cv2.resize(image_bgr, (IMG_W, IMG_H)), cv2.COLOR_BGR2RGB
    ).astype(np.float32) / 127.5 - 1.0

    input_tensor = tf.constant(img_rgb[np.newaxis])  # (1, H, W, 3)

    # Forward pass
    conf_logits, loc_pred = model(input_tensor, training=False)

    # Decode: softmax + argmax
    boxes, scores, class_ids = decode_predictions(
        conf_logits[0].numpy(),
        loc_pred[0].numpy(),
    )

    # NMS across all 47 classes
    detections = nms(
        boxes, scores, class_ids,
        iou_thresh=iou_thresh,
        conf_thresh=conf_thresh,
        sign_map=sign_map,
    )

    # Draw on original-size image
    annotated = image_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        px1 = int(x1 * orig_w)
        py1 = int(y1 * orig_h)
        px2 = int(x2 * orig_w)
        py2 = int(y2 * orig_h)

        color = (0, 255, 0)  # green default (overridden by draw_detections)
        label = f"{det['class_name']} {det['confidence']:.2f}"

        from utils import _class_color
        color = _class_color(det["class_id"])
        cv2.rectangle(annotated, (px1, py1), (px2, py2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (px1, py1 - th - 4), (px1 + tw, py1), color, -1)
        cv2.putText(annotated, label, (px1, py1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)

    return annotated, detections


# ---------------------------------------------------------------------------
# Process modes
# ---------------------------------------------------------------------------

def process_images(
    input_dir: str,
    model: tf.keras.Model,
    sign_map: dict,
    output_dir: str = OUTPUT_DIR,
    conf_thresh: float = CONF_THRESH,
) -> None:
    """Process all images in input_dir, save annotated versions to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(input_dir, ext)))
        files.extend(glob.glob(os.path.join(input_dir, ext.upper())))

    if not files:
        print(f"No image files found in {input_dir}")
        return

    print(f"Processing {len(files)} images …")
    t0 = time.time()

    for path in files:
        img = cv2.imread(path)
        if img is None:
            print(f"  [skip] Cannot read {path}")
            continue

        annotated, detections = run_inference(img, model, sign_map, conf_thresh)

        out_name = f"annotated_{Path(path).name}"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, annotated)

        sign_str = ", ".join(d["class_name"] for d in detections) or "—"
        print(f"  {Path(path).name:40s}  →  {sign_str}")

    print(f"\nDone in {time.time()-t0:.1f}s. Annotated images saved to: {output_dir}")


def process_video(
    input_path: str,
    model: tf.keras.Model,
    sign_map: dict,
    output_path: str = None,
    conf_thresh: float = CONF_THRESH,
) -> None:
    """Process a video file, optionally saving an annotated output video.

    Mirrors the video-mode logic in the original inference.py.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")

    fps     = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Writing annotated video → {output_path}")

    frame_idx = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, detections = run_inference(frame, model, sign_map, conf_thresh)

        if writer:
            writer.write(annotated)

        if detections:
            signs = ", ".join(d["class_name"] for d in detections)
            print(f"  Frame {frame_idx:5d}/{n_frames}  →  {signs}")

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()

    elapsed = time.time() - t0
    print(f"\nProcessed {frame_idx} frames in {elapsed:.1f}s "
          f"({frame_idx/elapsed:.1f} fps)")


def process_demo(
    model: tf.keras.Model,
    sign_map: dict,
    sample_dir: str = "./sample_images",
    conf_thresh: float = CONF_THRESH,
) -> None:
    """Display detection results on bundled sample images (no save).

    Mirrors the 'demo' mode in the original inference.py.
    Requires a graphical display (cv2.imshow).
    """
    extensions = ["*.jpg", "*.jpeg", "*.png"]
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(sample_dir, ext)))

    if not files:
        print(f"No sample images found in {sample_dir}")
        return

    print(f"Demo mode: showing {len(files)} images. Press any key to advance.")

    for path in files:
        img = cv2.imread(path)
        if img is None:
            continue

        annotated, detections = run_inference(img, model, sign_map, conf_thresh)

        title = f"{Path(path).name} — detected: " + \
                (", ".join(d["class_name"] for d in detections) or "nothing")
        cv2.imshow(title, annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AlexNet-SSD inference on LISA traffic signs."
    )
    parser.add_argument(
        "-m", "--mode", choices=["image", "video", "demo"], default="demo",
        help="Inference mode: image | video | demo  (default: demo)",
    )
    parser.add_argument(
        "-i", "--input_dir", default="./sample_images",
        help="Input directory (images) or file path (video).",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output directory (image mode) or file path (video mode).",
    )
    parser.add_argument(
        "--checkpoint_dir", default=CHECKPOINT_DIR,
        help="Directory with saved model checkpoints.",
    )
    parser.add_argument(
        "--signnames", default="./signnames.csv",
        help="Path to signnames.csv (class id → name mapping).",
    )
    parser.add_argument(
        "--conf", type=float, default=CONF_THRESH,
        help=f"Confidence threshold (default {CONF_THRESH}).",
    )
    parser.add_argument(
        "--nms_iou", type=float, default=NMS_IOU_THRESH,
        help=f"NMS IoU threshold (default {NMS_IOU_THRESH}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    sign_map = load_sign_map(args.signnames)
    model    = load_model(args.checkpoint_dir)

    if args.mode == "image":
        process_images(
            args.input_dir, model, sign_map,
            output_dir=args.output or OUTPUT_DIR,
            conf_thresh=args.conf,
        )
    elif args.mode == "video":
        process_video(
            args.input_dir, model, sign_map,
            output_path=args.output,
            conf_thresh=args.conf,
        )
    elif args.mode == "demo":
        process_demo(
            model, sign_map,
            sample_dir=args.input_dir,
            conf_thresh=args.conf,
        )
