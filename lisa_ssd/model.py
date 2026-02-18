"""
model.py — AlexNet-SSD for all 47 LISA traffic-sign classes (TF2 / Keras)
=========================================================================
Adapted from: https://github.com/georgesung/ssd_tensorflow_traffic_sign_detection

Original used TF 0.12 low-level ops and a tf.Session training loop.
This module is a 1-to-1 architectural port to TF2 / Keras with three changes:
  1. NUM_CLASSES 3  →  48  (47 LISA sign types + background)
  2. NUM_CHANNELS 1 →   3  (grayscale → RGB; weight-init still random)
  3. All TF1 primitives replaced by tf.keras equivalents

Architecture is identical to the original:
  Input → AlexNet-like conv stack → 4 × SSD prediction hooks → concat outputs

Prediction heads produce:
  conf_logits : (batch, NUM_ANCHORS, NUM_CLASSES)   — class scores
  loc_pred    : (batch, NUM_ANCHORS, 4)             — box coordinates

SSD loss combines:
  • Focal cross-entropy confidence loss  (hard-negative mining, NEG_POS_RATIO)
  • Smooth-L1 localisation loss
  • L2 weight regularisation
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from settings import (
    IMG_H, IMG_W, NUM_CHANNELS, NUM_CLASSES, NUM_DEFAULT_BOXES,
    FEATURE_MAPS, NUM_ANCHORS,
    NEG_POS_RATIO, LOC_LOSS_WEIGHT, REG_SCALE, LEARNING_RATE,
)


# ---------------------------------------------------------------------------
# SSD Prediction Hook
# ---------------------------------------------------------------------------

def ssd_hook(x: tf.Tensor,
             num_defaults: int,
             num_classes: int,
             name: str) -> tuple[tf.Tensor, tf.Tensor]:
    """Apply one SSD prediction head to a feature map.

    Mirrors SSDHook() from the original model.py.

    Returns:
        conf_flat : (batch, H*W*num_defaults, num_classes) — raw logits
        loc_flat  : (batch, H*W*num_defaults, 4)           — box coords
    """
    # 3×3 conv for confidence scores
    conf = layers.Conv2D(
        num_defaults * num_classes, (3, 3), padding="same",
        activation=None, use_bias=True,
        name=f"{name}_conf",
    )(x)

    # 3×3 conv for localisation offsets
    loc = layers.Conv2D(
        num_defaults * 4, (3, 3), padding="same",
        activation=None, use_bias=True,
        name=f"{name}_loc",
    )(x)

    # Reshape: (B, H, W, D*K) → (B, H*W*D, K)
    conf_flat = layers.Reshape((-1, num_classes), name=f"{name}_conf_flat")(conf)
    loc_flat  = layers.Reshape((-1, 4),           name=f"{name}_loc_flat")(loc)

    return conf_flat, loc_flat


# ---------------------------------------------------------------------------
# AlexNet backbone + SSD heads
# ---------------------------------------------------------------------------

def build_alexnet_ssd(
    num_classes: int = NUM_CLASSES,
    img_h: int = IMG_H,
    img_w: int = IMG_W,
    num_channels: int = NUM_CHANNELS,
    num_defaults: int = NUM_DEFAULT_BOXES,
    reg_scale: float = REG_SCALE,
) -> keras.Model:
    """Build the AlexNet-SSD model (TF2/Keras).

    The backbone follows the original tutorial exactly.  The only structural
    change is that all tf.Variable / session-style code is replaced by the
    Keras functional API.

    Feature maps and SSD hooks match FEATURE_MAPS in settings.py:
        Hook 1 → 31×48  (after maxpool-1)
        Hook 2 → 15×23  (after maxpool-2)
        Hook 3 →  8×12  (after maxpool-3)
        Hook 4 →  4×6   (after maxpool-4)
    """
    reg = regularizers.l2(reg_scale)
    inputs = keras.Input(shape=(img_h, img_w, num_channels), name="image")

    # Normalise [0, 255] → [-1, 1]  (identical to original's normalisation)
    x = layers.Lambda(lambda t: t / 127.5 - 1.0, name="normalise")(inputs)

    # ------------------------------------------------------------------
    # Backbone — AlexNet-style convolutions
    # ------------------------------------------------------------------

    # Conv1: 11×11, stride 4, VALID  →  63×98  (H×W for 260×400 input)
    x = layers.Conv2D(64, (11, 11), strides=4, padding="valid",
                      activation="relu", kernel_regularizer=reg,
                      name="conv1")(x)

    # MaxPool 3×3 s=2 VALID  →  31×48   ←  Hook 1
    x = layers.MaxPool2D((3, 3), strides=2, padding="valid",
                         name="maxpool1")(x)
    hook1_conf, hook1_loc = ssd_hook(x, num_defaults, num_classes, "hook1")

    # Conv2: 5×5, SAME  →  31×48
    x = layers.Conv2D(192, (5, 5), padding="same",
                      activation="relu", kernel_regularizer=reg,
                      name="conv2")(x)

    # MaxPool 3×3 s=2 VALID  →  15×23   ←  Hook 2
    x = layers.MaxPool2D((3, 3), strides=2, padding="valid",
                         name="maxpool2")(x)
    hook2_conf, hook2_loc = ssd_hook(x, num_defaults, num_classes, "hook2")

    # Conv3-5  (all SAME, no spatial downsample)
    x = layers.Conv2D(384, (3, 3), padding="same",
                      activation="relu", kernel_regularizer=reg,
                      name="conv3")(x)
    x = layers.Conv2D(384, (3, 3), padding="same",
                      activation="relu", kernel_regularizer=reg,
                      name="conv4")(x)
    x = layers.Conv2D(256, (3, 3), padding="same",
                      activation="relu", kernel_regularizer=reg,
                      name="conv5")(x)

    # Conv6-7 (extra SSD capacity layers, same as original)
    x = layers.Conv2D(1024, (3, 3), padding="same",
                      activation="relu", kernel_regularizer=reg,
                      name="conv6")(x)
    x = layers.Conv2D(1024, (1, 1), padding="same",
                      activation="relu", kernel_regularizer=reg,
                      name="conv7")(x)

    # MaxPool 2×2 s=2 SAME  →  8×12   ←  Hook 3
    x = layers.MaxPool2D((2, 2), strides=2, padding="same",
                         name="maxpool3")(x)
    hook3_conf, hook3_loc = ssd_hook(x, num_defaults, num_classes, "hook3")

    # Conv8, 8_2
    x = layers.Conv2D(256, (1, 1), padding="same",
                      activation="relu", kernel_regularizer=reg,
                      name="conv8")(x)
    x = layers.Conv2D(512, (3, 3), padding="same",
                      activation="relu", kernel_regularizer=reg,
                      name="conv8_2")(x)

    # MaxPool 2×2 s=2 SAME  →  4×6   ←  Hook 4
    x = layers.MaxPool2D((2, 2), strides=2, padding="same",
                         name="maxpool4")(x)
    hook4_conf, hook4_loc = ssd_hook(x, num_defaults, num_classes, "hook4")

    # ------------------------------------------------------------------
    # Concatenate all hook outputs
    # ------------------------------------------------------------------
    conf_out = layers.Concatenate(axis=1, name="conf_concat")(
        [hook1_conf, hook2_conf, hook3_conf, hook4_conf]
    )  # (B, NUM_ANCHORS, NUM_CLASSES)

    loc_out = layers.Concatenate(axis=1, name="loc_concat")(
        [hook1_loc, hook2_loc, hook3_loc, hook4_loc]
    )  # (B, NUM_ANCHORS, 4)

    model = keras.Model(inputs=inputs, outputs=[conf_out, loc_out],
                        name="AlexNet_SSD_LISA47")
    return model


# ---------------------------------------------------------------------------
# SSD Multi-box Loss (with hard-negative mining)
# ---------------------------------------------------------------------------

class SSDLoss:
    """Computes the combined SSD loss used to train the detector.

    Mirrors ModelHelper() from the original model.py but implemented as a
    plain Python class so it can be used with tf.GradientTape.

    Conf loss : cross-entropy with hard-negative mining (NEG_POS_RATIO)
    Loc  loss : smooth-L1 on positive anchors only
    Reg  loss : L2 weight decay (applied separately via model.losses)
    """

    def __init__(self,
                 num_classes: int = NUM_CLASSES,
                 neg_pos_ratio: int = NEG_POS_RATIO,
                 loc_loss_weight: float = LOC_LOSS_WEIGHT):
        self.num_classes    = num_classes
        self.neg_pos_ratio  = neg_pos_ratio
        self.loc_weight     = loc_loss_weight

    def __call__(self,
                 conf_logits: tf.Tensor,
                 loc_pred: tf.Tensor,
                 conf_labels: tf.Tensor,
                 loc_labels: tf.Tensor,
                 conf_mask: tf.Tensor,
                 loc_mask: tf.Tensor) -> dict[str, tf.Tensor]:
        """
        Args:
            conf_logits : (B, A, C)  raw class logits
            loc_pred    : (B, A, 4)  predicted box coords
            conf_labels : (B, A)     int32 class index (0 = background)
            loc_labels  : (B, A, 4)  GT box coords (only used for positives)
            conf_mask   : (B, A)     float32: 1=pos/hard-neg, 0=ignored
            loc_mask    : (B, A)     float32: 1=positive anchor, 0=rest

        Returns:
            dict with keys: conf_loss, loc_loss, total_loss
        """
        # ---- Confidence loss (sparse softmax CE) ----
        conf_loss_raw = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=conf_labels,       # (B, A)
            logits=conf_logits,       # (B, A, C)
        )  # → (B, A)

        conf_loss = tf.reduce_sum(conf_loss_raw * conf_mask) / (
            tf.reduce_sum(conf_mask) + 1e-6
        )

        # ---- Localisation loss (Smooth-L1) ----
        diff = loc_pred - loc_labels        # (B, A, 4)
        abs_diff = tf.abs(diff)
        smooth_l1 = tf.where(abs_diff < 1.0,
                             0.5 * tf.square(diff),
                             abs_diff - 0.5)  # (B, A, 4)

        loc_loss_raw = tf.reduce_sum(smooth_l1, axis=-1)  # (B, A)
        loc_loss = tf.reduce_sum(loc_loss_raw * loc_mask) / (
            tf.reduce_sum(loc_mask) + 1e-6
        )

        total = conf_loss + self.loc_weight * loc_loss

        return {
            "conf_loss" : conf_loss,
            "loc_loss"  : loc_loss,
            "total_loss": total,
        }


# ---------------------------------------------------------------------------
# Hard-Negative Mining mask builder
# ---------------------------------------------------------------------------

def build_hard_neg_mask(
    conf_logits: tf.Tensor,
    conf_labels: tf.Tensor,
    loc_mask: tf.Tensor,
    neg_pos_ratio: int = NEG_POS_RATIO,
) -> tf.Tensor:
    """Select hard negatives for training.

    Mirrors the hard-negative mining in the original train.py's next_batch().

    Returns:
        conf_mask : (B, A) float32 — 1 for positives + hard negatives, else 0.
    """
    # Confidence of the "background" class (index 0) for all anchors
    bg_logit = conf_logits[:, :, 0]           # (B, A)
    pos_mask = tf.cast(loc_mask, tf.float32)  # (B, A)

    # Negatives = anchors labelled background that are NOT positives
    neg_candidates = (1.0 - pos_mask)         # (B, A)

    # Sort negatives by ascending bg confidence (hardest = least confident bg)
    neg_conf = bg_logit * neg_candidates + 1e9 * pos_mask  # push positives to end
    sorted_idx = tf.argsort(neg_conf, direction="ASCENDING", axis=-1)

    # Number of negatives to keep = min(neg_pos_ratio * num_pos, total_neg)
    num_pos    = tf.reduce_sum(pos_mask, axis=-1, keepdims=True)  # (B, 1)
    max_negs   = tf.cast(neg_pos_ratio * num_pos, tf.int32)       # (B, 1)

    A = tf.shape(conf_logits)[1]
    rank = tf.cast(tf.argsort(sorted_idx, axis=-1), tf.float32)   # (B, A)

    neg_mask = tf.cast(
        rank < tf.cast(max_negs, tf.float32), tf.float32
    ) * neg_candidates  # (B, A)

    return pos_mask + neg_mask  # union


# ---------------------------------------------------------------------------
# Convenience: build model + optimiser
# ---------------------------------------------------------------------------

def build_model_and_optimiser(
    num_classes: int = NUM_CLASSES,
    learning_rate: float = LEARNING_RATE,
) -> tuple[keras.Model, keras.optimizers.Optimizer]:
    """Return (model, optimiser) ready for training."""
    model = build_alexnet_ssd(num_classes=num_classes)
    # Adadelta with default parameters — same choice as the original tutorial
    opt = keras.optimizers.Adadelta(learning_rate=learning_rate)
    return model, opt


# ---------------------------------------------------------------------------
# Decode raw model output → (boxes, scores, class_ids)
# ---------------------------------------------------------------------------

def decode_predictions(
    conf_logits: np.ndarray,
    loc_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert raw model output for a single image into detectable form.

    Args:
        conf_logits : (A, C) float32
        loc_pred    : (A, 4) float32

    Returns:
        boxes     : (A, 4)  normalised box coordinates
        scores    : (A,)    maximum class confidence (after softmax)
        class_ids : (A,)    argmax class index
    """
    probs     = tf.nn.softmax(conf_logits, axis=-1).numpy()  # (A, C)
    class_ids = np.argmax(probs, axis=-1)                    # (A,)
    scores    = probs[np.arange(len(class_ids)), class_ids]  # (A,)
    boxes     = np.clip(loc_pred, 0.0, 1.0)                  # (A, 4)
    return boxes, scores, class_ids
