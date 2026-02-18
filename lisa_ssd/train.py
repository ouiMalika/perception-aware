"""
train.py — Training Script for LISA All-Classes AlexNet-SSD (TF2)
==================================================================
Adapted from: https://github.com/georgesung/ssd_tensorflow_traffic_sign_detection

Original used tf.Session + tf.train.AdadeltaOptimizer in TF 0.12.
This module ports the same training loop to TF2 / Keras with:

  * tf.GradientTape for explicit gradient computation
  * Hard-negative mining implemented in numpy (same ratio as original)
  * Periodic checkpoint saving via tf.train.Checkpoint
  * Training / validation loss history saved to JSON
  * Support for resuming from a checkpoint (--resume flag)

Workflow
--------
  # 1. Prepare data (run once)
  python dataset.py --lisa_dir ./data/lisa
  python data_prep.py

  # 2. Train
  python train.py

  # 3. Resume from last checkpoint
  python train.py --resume
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, Tuple

import cv2
import numpy as np
import tensorflow as tf

from settings import (
    IMG_H, IMG_W, NUM_CHANNELS, NUM_CLASSES, NUM_ANCHORS,
    BATCH_SIZE, NUM_EPOCHS, VALIDATION_SIZE, LEARNING_RATE,
    NEG_POS_RATIO, LOC_LOSS_WEIGHT, REG_SCALE,
    PICKLE_DIR, CHECKPOINT_DIR,
)
from model import (
    build_model_and_optimiser,
    SSDLoss,
    build_hard_neg_mask,
    decode_predictions,
)
from dataset import load_pickle


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_prepared_data(
    prep_path: str,
    val_size: float = VALIDATION_SIZE,
) -> Tuple[dict, dict]:
    """Load the prepared pickle and split into train / val sets.

    Returns:
        (train_data, val_data) each with keys:
          image_paths, conf_labels, loc_labels, loc_mask
    """
    data = load_pickle(prep_path)
    N = len(data["image_paths"])
    n_val = max(1, int(N * val_size))
    n_train = N - n_val

    # Shuffle once with a fixed seed for reproducibility
    rng = np.random.default_rng(seed=42)
    idx = rng.permutation(N)

    def _split(arr, train_idx, val_idx):
        return arr[train_idx], arr[val_idx]

    train_idx = idx[:n_train]
    val_idx   = idx[n_train:]

    train_data = {
        "image_paths": [data["image_paths"][i] for i in train_idx],
        "conf_labels": data["conf_labels"][train_idx],
        "loc_labels" : data["loc_labels"][train_idx],
        "loc_mask"   : data["loc_mask"][train_idx],
    }
    val_data = {
        "image_paths": [data["image_paths"][i] for i in val_idx],
        "conf_labels": data["conf_labels"][val_idx],
        "loc_labels" : data["loc_labels"][val_idx],
        "loc_mask"   : data["loc_mask"][val_idx],
    }

    print(f"Dataset: {N} total → {n_train} train, {n_val} val")
    return train_data, val_data


# ---------------------------------------------------------------------------
# Batch generator (mirrors next_batch() in original train.py)
# ---------------------------------------------------------------------------

def load_image_as_tensor(path: str) -> np.ndarray:
    """Read one image, resize to (IMG_H, IMG_W, 3), normalise to [-1, 1]."""
    img = cv2.imread(path)
    if img is None:
        # Return a blank image if file is missing (defensive)
        return np.zeros((IMG_H, IMG_W, NUM_CHANNELS), dtype=np.float32)
    img = cv2.resize(img, (IMG_W, IMG_H))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    return img / 127.5 - 1.0  # [-1, 1]


def next_batch(
    data: dict,
    batch_size: int,
    start: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Yield one mini-batch of (images, conf_labels, loc_labels, loc_mask).

    Mirrors next_batch() from the original train.py.
    Image loading is done on the fly (same as original).

    Args:
        data      : dict with image_paths, conf_labels, loc_labels, loc_mask.
        batch_size: number of samples in the batch.
        start     : starting index into the dataset.

    Returns:
        images     : (B, IMG_H, IMG_W, 3) float32
        conf_labels: (B, A) int32
        loc_labels : (B, A, 4) float32
        loc_mask   : (B, A) float32
    """
    end = min(start + batch_size, len(data["image_paths"]))
    indices = list(range(start, end))

    images = np.stack([
        load_image_as_tensor(data["image_paths"][i]) for i in indices
    ])  # (B, H, W, C)

    return (
        images,
        data["conf_labels"][indices],
        data["loc_labels"][indices],
        data["loc_mask"][indices],
    )


# ---------------------------------------------------------------------------
# One training step
# ---------------------------------------------------------------------------

@tf.function
def train_step(
    model: tf.keras.Model,
    optimiser: tf.keras.optimizers.Optimizer,
    loss_fn: SSDLoss,
    images: tf.Tensor,
    conf_labels: tf.Tensor,
    loc_labels: tf.Tensor,
    conf_mask: tf.Tensor,
    loc_mask: tf.Tensor,
) -> Dict[str, tf.Tensor]:
    """Single gradient-update step.

    Wraps forward pass + loss + backward pass in tf.function for speed.
    """
    with tf.GradientTape() as tape:
        conf_logits, loc_pred = model(images, training=True)

        losses = loss_fn(
            conf_logits=conf_logits,
            loc_pred=loc_pred,
            conf_labels=conf_labels,
            loc_labels=loc_labels,
            conf_mask=conf_mask,
            loc_mask=loc_mask,
        )

        # Add L2 regularisation losses from kernel_regularizer on each layer
        reg_loss = tf.add_n(model.losses) if model.losses else 0.0
        total    = losses["total_loss"] + reg_loss

    grads = tape.gradient(total, model.trainable_variables)
    optimiser.apply_gradients(zip(grads, model.trainable_variables))

    return {**losses, "reg_loss": reg_loss, "total_with_reg": total}


# ---------------------------------------------------------------------------
# Validation (no gradient)
# ---------------------------------------------------------------------------

def validate(
    model: tf.keras.Model,
    loss_fn: SSDLoss,
    val_data: dict,
    batch_size: int,
) -> Dict[str, float]:
    """Compute average validation loss over the full val set."""
    total_losses = {"conf_loss": 0.0, "loc_loss": 0.0, "total_loss": 0.0}
    n_batches = 0

    for start in range(0, len(val_data["image_paths"]), batch_size):
        images, conf_labels, loc_labels, loc_mask = next_batch(
            val_data, batch_size, start
        )
        images_t       = tf.constant(images,       dtype=tf.float32)
        conf_labels_t  = tf.constant(conf_labels,  dtype=tf.int32)
        loc_labels_t   = tf.constant(loc_labels,   dtype=tf.float32)
        loc_mask_t     = tf.constant(loc_mask,     dtype=tf.float32)

        conf_logits, loc_pred = model(images_t, training=False)

        # For validation: use loc_mask directly as conf_mask (no hard-neg mining)
        losses = loss_fn(
            conf_logits=conf_logits,
            loc_pred=loc_pred,
            conf_labels=conf_labels_t,
            loc_labels=loc_labels_t,
            conf_mask=loc_mask_t,
            loc_mask=loc_mask_t,
        )
        for k in total_losses:
            total_losses[k] += float(losses[k])
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in total_losses.items()}


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def run_training(
    prep_path: str,
    checkpoint_dir: str = CHECKPOINT_DIR,
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    val_size: float = VALIDATION_SIZE,
    resume: bool = False,
) -> None:
    """Full training loop — direct port of run_training() from original train.py."""

    os.makedirs(checkpoint_dir, exist_ok=True)

    # ---- Load & split data ------------------------------------------------
    train_data, val_data = load_prepared_data(prep_path, val_size)
    n_train = len(train_data["image_paths"])

    # ---- Build model & optimiser ------------------------------------------
    model, optimiser = build_model_and_optimiser(
        num_classes=NUM_CLASSES,
        learning_rate=learning_rate,
    )
    model.summary(line_length=90)

    loss_fn = SSDLoss(
        num_classes=NUM_CLASSES,
        neg_pos_ratio=NEG_POS_RATIO,
        loc_loss_weight=LOC_LOSS_WEIGHT,
    )

    # ---- Checkpoint -------------------------------------------------------
    ckpt = tf.train.Checkpoint(model=model, optimiser=optimiser)
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, checkpoint_dir, max_to_keep=3
    )

    start_epoch = 0
    if resume and ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        # Infer epoch number from checkpoint filename
        try:
            start_epoch = int(
                ckpt_manager.latest_checkpoint.rsplit("-", 1)[-1]
            )
        except ValueError:
            start_epoch = 0
        print(f"Resumed from checkpoint at epoch {start_epoch}")

    # ---- History ----------------------------------------------------------
    history_path = os.path.join(checkpoint_dir, "loss_history.json")
    if resume and os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
    else:
        history = {"train_loss": [], "val_loss": []}

    # ---- Training loop ----------------------------------------------------
    rng = np.random.default_rng(seed=0)

    for epoch in range(start_epoch, num_epochs):
        t0 = time.time()

        # Shuffle training data each epoch
        perm = rng.permutation(n_train)
        epoch_data = {
            "image_paths": [train_data["image_paths"][i] for i in perm],
            "conf_labels" : train_data["conf_labels"][perm],
            "loc_labels"  : train_data["loc_labels"][perm],
            "loc_mask"    : train_data["loc_mask"][perm],
        }

        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, n_train, batch_size):
            images, conf_labels, loc_labels, loc_mask = next_batch(
                epoch_data, batch_size, start
            )

            # ---- Hard-negative mining (same as original next_batch) -------
            # We need a forward pass to get logits for mining, then train.
            images_t      = tf.constant(images,      dtype=tf.float32)
            conf_labels_t = tf.constant(conf_labels, dtype=tf.int32)
            loc_labels_t  = tf.constant(loc_labels,  dtype=tf.float32)
            loc_mask_t    = tf.constant(loc_mask,    dtype=tf.float32)

            conf_logits_pre, _ = model(images_t, training=False)
            conf_mask_t = build_hard_neg_mask(
                conf_logits_pre, conf_labels_t, loc_mask_t, NEG_POS_RATIO
            )

            # ---- Gradient update -----------------------------------------
            step_losses = train_step(
                model, optimiser, loss_fn,
                images_t, conf_labels_t, loc_labels_t,
                conf_mask_t, loc_mask_t,
            )

            epoch_loss += float(step_losses["total_loss"])
            n_batches  += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # ---- Validation ---------------------------------------------------
        val_losses = validate(model, loss_fn, val_data, batch_size)
        avg_val_loss = val_losses["total_loss"]

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch+1:4d}/{num_epochs}  "
            f"train_loss={avg_train_loss:.4f}  "
            f"val_loss={avg_val_loss:.4f}  "
            f"({elapsed:.1f}s)"
        )

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        # ---- Checkpoint ---------------------------------------------------
        ckpt_manager.save(checkpoint_number=epoch + 1)

        # ---- Save history -------------------------------------------------
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    print("\nTraining complete.")
    print(f"Checkpoints → {checkpoint_dir}")
    print(f"Loss history → {history_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train AlexNet-SSD on all 47 LISA traffic sign classes."
    )
    parser.add_argument(
        "--data", default=None,
        help=f"Path to prepared data pickle. "
             f"Default: {PICKLE_DIR}/data_prep_{IMG_W}x{IMG_H}.p",
    )
    parser.add_argument(
        "--checkpoint_dir", default=CHECKPOINT_DIR,
        help="Directory for model checkpoints.",
    )
    parser.add_argument(
        "--epochs", type=int, default=NUM_EPOCHS,
        help=f"Number of training epochs (default {NUM_EPOCHS}).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE,
        help=f"Mini-batch size (default {BATCH_SIZE}).",
    )
    parser.add_argument(
        "--lr", type=float, default=LEARNING_RATE,
        help=f"Adadelta learning rate (default {LEARNING_RATE}).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from the latest checkpoint.",
    )
    args = parser.parse_args()

    prep_path = args.data or os.path.join(
        PICKLE_DIR, f"data_prep_{IMG_W}x{IMG_H}.p"
    )

    if not os.path.exists(prep_path):
        raise SystemExit(
            f"Prepared data not found: {prep_path}\n"
            "Run first:\n"
            "  python dataset.py --lisa_dir ./data/lisa\n"
            "  python data_prep.py"
        )

    run_training(
        prep_path       = prep_path,
        checkpoint_dir  = args.checkpoint_dir,
        num_epochs      = args.epochs,
        batch_size      = args.batch_size,
        learning_rate   = args.lr,
        resume          = args.resume,
    )
