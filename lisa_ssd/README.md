# LISA All-Classes AlexNet-SSD Traffic Sign Detector

Adaptation of the [SSD TensorFlow traffic sign detection tutorial][orig] by
George Sung to support **all 47 US traffic sign categories** in the
[LISA Traffic Sign Dataset][lisa] instead of the original two (stop +
pedestrian crossing).

---

## What changed from the original tutorial

| Dimension | Original | This adaptation |
|---|---|---|
| Framework | TensorFlow 0.12 (tf.Session) | TensorFlow 2.x / Keras |
| Sign classes | 2 (stop, pedestrianCrossing) | **47** (all LISA categories) |
| Image channels | 1 (grayscale) | **3 (RGB)** |
| Dataset loader | `data_gathering/create_pickle.py` | `dataset.py` (all-class) |
| Model output | 3-class softmax | **48-class softmax** (47 + background) |
| Training loop | `tf.Session.run` | `tf.GradientTape` |
| Checkpoint format | TF1 Saver | `tf.train.CheckpointManager` |

The **SSD architecture** (AlexNet backbone + 4 SSD prediction hooks),
**anchor-box matching** logic, **hard-negative mining** ratio, and
**Adadelta optimiser** are kept identical to the original tutorial for a
clean 1-to-1 comparison.

---

## All 47 LISA sign classes

```
addedLane          curveLeft          curveRight         dip
doNotEnter         doNotPass          intersection       keepRight
laneEnds           merge              noLeftTurn         noRightTurn
pedestrianCrossing rampSpeedAdvisory20 rampSpeedAdvisory35 rampSpeedAdvisory40
rampSpeedAdvisory45 rampSpeedAdvisory50 rampSpeedAdvisoryUrdbl rightLaneMustTurn
roundabout         school             schoolSpeedLimit25 signalAhead
slow               speedLimit15       speedLimit25       speedLimit30
speedLimit35       speedLimit40       speedLimit45       speedLimit50
speedLimit55       speedLimit65       speedLimitUrdbl    stop
stopAhead          thruMergeLeft      thruMergeRight     thruTrafficMergeLeft
truckSpeedLimit55  turnLeft           turnRight          yield
yieldAhead         zoneAhead25        zoneAhead45
```

---

## File structure

```
lisa_ssd/
├── settings.py      Configuration: all 47 classes, anchor boxes, hyper-params
├── signnames.csv    ClassId → SignName mapping (all 48 rows incl. background)
├── model.py         AlexNet-SSD Keras model + SSD loss + hard-neg mining
├── dataset.py       LISA all-class CSV loader (replaces create_pickle.py)
├── data_prep.py     Anchor-box matching and label encoding
├── train.py         TF2 training loop with checkpointing
├── inference.py     Image / video / demo inference modes
├── utils.py              IoU, NMS, visualisation helpers
├── create_sample_data.py Synthetic LISA-format dataset generator (no download needed)
└── requirements.txt      Python dependencies
```

---

## Setup

### 1. Install dependencies

**macOS Apple Silicon (M1/M2/M3)** — `tensorflow` is not on PyPI for arm64;
use the Apple-maintained port instead:

```bash
pip install tensorflow-macos tensorflow-metal
pip install opencv-python numpy
```

**macOS Intel / Linux / Windows:**

```bash
pip install -r lisa_ssd/requirements.txt
```

### 2. Get a dataset

#### Option A — Synthetic data (fastest, no download)

`create_sample_data.py` generates a small synthetic dataset in the exact
LISA directory format.  Use this to verify the full pipeline works before
committing to the real download.

```bash
python lisa_ssd/create_sample_data.py --out ./data/lisa   # 8 classes × 40 images
# optionally generate more classes or images:
# python lisa_ssd/create_sample_data.py --n 100 --classes stop yield signalAhead
```

#### Option B — Real LISA dataset

The full LISA Traffic Sign Dataset (~7.7 GB, 47 classes) requires a free
academic registration:
<http://cvrr-nas.ucsd.edu/LISA/lisa-traffic-sign-dataset.html>

> **Note:** Third-party "Tiny LISA" repacks on Kaggle dump all images into
> a flat folder without per-class subdirectories and are **not** compatible
> with this loader.  Use the official release or the synthetic generator above.

After extracting the official dataset the layout must look like:

```
data/
└── lisa/
    ├── stop/
    │   ├── frameAnnotations.csv      ← semicolon-delimited, 1 header row
    │   └── frames/
    │       └── stop_1/
    │           ├── frame000.png
    │           └── ...
    ├── pedestrianCrossing/
    │   ├── frameAnnotations.csv
    │   └── frames/...
    ├── speedLimit35/
    │   ├── frameAnnotations.csv
    │   └── frames/...
    └── ... (one directory per sign class)
```

> **Tip:** The dataset loader silently skips missing class directories, so
> you can start training on any partial subset of the 47 classes.

### 3. Preprocess

```bash
# Step 1: parse all CSVs → normalised annotation pickle
python lisa_ssd/dataset.py --lisa_dir ./data/lisa

# Step 2: match GT boxes to anchor boxes → training-ready pickle
python lisa_ssd/data_prep.py
```

Both scripts accept `--help` for full option lists.

---

## Training

```bash
# Train from scratch (200 epochs, batch 32, Adadelta lr=1.0)
python lisa_ssd/train.py

# Resume from latest checkpoint
python lisa_ssd/train.py --resume

# Custom options
python lisa_ssd/train.py --epochs 100 --batch_size 16 --lr 0.5
```

Checkpoints and loss history are saved to `./checkpoints/` by default.

---

## Inference

```bash
# Demo: show detections on ./sample_images/ (requires display)
python lisa_ssd/inference.py -m demo

# Batch images: annotate all images in a directory
python lisa_ssd/inference.py -m image -i ./test_images/ -o ./output/

# Video: annotate a video file
python lisa_ssd/inference.py -m video -i ./my_video.mp4 -o ./output/annotated.mp4

# Adjust confidence threshold
python lisa_ssd/inference.py -m image -i ./test_images/ --conf 0.7
```

---

## Architecture summary

```
Input (400 × 260 × 3, RGB)
│
├─ Conv1  11×11 s=4 VALID → 98×63 × 64
├─ MaxPool 3×3 s=2 VALID  → 48×31      ──► SSD Hook 1  (31×48×4 = 5 952 anchors)
├─ Conv2   5×5 SAME       → 48×31 × 192
├─ MaxPool 3×3 s=2 VALID  → 23×15      ──► SSD Hook 2  (15×23×4 = 1 380 anchors)
├─ Conv3-5 3×3 SAME       → 23×15 × 384/384/256
├─ Conv6-7 3×3/1×1 SAME   → 23×15 × 1024
├─ MaxPool 2×2 s=2 SAME   → 12×8       ──► SSD Hook 3  ( 8×12×4 =   384 anchors)
├─ Conv8,8_2 1×1,3×3 SAME → 12×8 × 512
└─ MaxPool 2×2 s=2 SAME   →  6×4       ──► SSD Hook 4  ( 4× 6×4 =    96 anchors)

Total anchors: 7 812
Each hook: 3×3 conv → (48-class conf logits, 4-coord loc pred)
Concatenated output: (batch, 7812, 48) conf + (batch, 7812, 4) loc
```

Loss function (same as original):
- **Confidence**: sparse softmax cross-entropy with hard-negative mining
  (5 negatives per positive)
- **Localisation**: smooth-L1 on positive anchors only
- **Regularisation**: L2 weight decay (scale 1e-2)

---

## Citation

If you use the LISA dataset, please cite:

> A. Møgelmose, M. M. Trivedi, and T. B. Moeslund,
> "Vision based Traffic Sign Detection and Analysis for Intelligent Driver
> Assistance Systems: Perspectives and Survey,"
> *IEEE Transactions on Intelligent Transportation Systems*, 2012.

Original tutorial code:
> George Sung, *SSD TensorFlow Traffic Sign Detection*,
> <https://github.com/georgesung/ssd_tensorflow_traffic_sign_detection>

[orig]: https://github.com/georgesung/ssd_tensorflow_traffic_sign_detection
[lisa]: http://cvrr-nas.ucsd.edu/LISA/lisa-traffic-sign-dataset.html
