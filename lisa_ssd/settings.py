"""
settings.py — LISA All-Classes SSD Configuration
=================================================
Adapted from: https://github.com/georgesung/ssd_tensorflow_traffic_sign_detection

Key differences from original tutorial:
  * NUM_CLASSES : 3  →  48  (all 47 LISA sign types + background)
  * NUM_CHANNELS: 1  →   3  (grayscale → RGB)
  * SIGN_CLASSES: 2  →  47  (stop + ped. crossing → every US sign in LISA)
  * Feature-map sizes and anchor layout are kept identical to the original
    so the AlexNet SSD backbone remains a 1-to-1 port.
"""

# ---------------------------------------------------------------------------
# All 47 LISA traffic-sign classes.  Index 0 is always "background".
# Source: Laboratory for Intelligent & Safe Automobiles (UCSD) dataset docs.
# ---------------------------------------------------------------------------
CLASSES = [
    "background",            #  0  (no sign / ignore)
    "addedLane",             #  1
    "curveLeft",             #  2
    "curveRight",            #  3
    "dip",                   #  4
    "doNotEnter",            #  5
    "doNotPass",             #  6
    "intersection",          #  7
    "keepRight",             #  8
    "laneEnds",              #  9
    "merge",                 # 10
    "noLeftTurn",            # 11
    "noRightTurn",           # 12
    "pedestrianCrossing",    # 13
    "rampSpeedAdvisory20",   # 14
    "rampSpeedAdvisory35",   # 15
    "rampSpeedAdvisory40",   # 16
    "rampSpeedAdvisory45",   # 17
    "rampSpeedAdvisory50",   # 18
    "rampSpeedAdvisoryUrdbl",# 19
    "rightLaneMustTurn",     # 20
    "roundabout",            # 21
    "school",                # 22
    "schoolSpeedLimit25",    # 23
    "signalAhead",           # 24
    "slow",                  # 25
    "speedLimit15",          # 26
    "speedLimit25",          # 27
    "speedLimit30",          # 28
    "speedLimit35",          # 29
    "speedLimit40",          # 30
    "speedLimit45",          # 31
    "speedLimit50",          # 32
    "speedLimit55",          # 33
    "speedLimit65",          # 34
    "speedLimitUrdbl",       # 35
    "stop",                  # 36
    "stopAhead",             # 37
    "thruMergeLeft",         # 38
    "thruMergeRight",        # 39
    "thruTrafficMergeLeft",  # 40
    "truckSpeedLimit55",     # 41
    "turnLeft",              # 42
    "turnRight",             # 43
    "yield",                 # 44
    "yieldAhead",            # 45
    "zoneAhead25",           # 46
    "zoneAhead45",           # 47
]

CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
NUM_CLASSES   = len(CLASSES)   # 48

# ---------------------------------------------------------------------------
# Image dimensions — kept identical to the original tutorial for a 1-to-1
# architecture comparison.  Only NUM_CHANNELS changes: grayscale → RGB.
# ---------------------------------------------------------------------------
IMG_H        = 260
IMG_W        = 400
NUM_CHANNELS = 3   # original used 1 (grayscale)

# ---------------------------------------------------------------------------
# SSD default (anchor) boxes — same 4 configurations as the original.
# Each tuple is (Δx1, Δy1, Δx2, Δy2) relative to the cell centre.
# ---------------------------------------------------------------------------
DEFAULT_BOXES = (
    (-0.5, -0.5,  0.5,  0.5),   # square,  ~full cell
    (-0.2, -0.2,  0.2,  0.2),   # square,  ~small
    (-0.8, -0.2,  0.8,  0.2),   # wide rectangle
    (-0.2, -0.8,  0.2,  0.8),   # tall rectangle
)
NUM_DEFAULT_BOXES = len(DEFAULT_BOXES)  # 4

# ---------------------------------------------------------------------------
# Feature-map sizes produced by the AlexNet backbone for a 400×260 input.
# Derivation (VALID conv / maxpool, identical to original):
#   Input 400×260
#   Conv1 11×11 s=4 VALID → 98×63
#   MaxPool 3×3 s=2 VALID → 48×31   ← Hook 1
#   Conv2 5×5 SAME        → 48×31
#   MaxPool 3×3 s=2 VALID → 23×15   ← Hook 2
#   MaxPool 2×2 s=2 SAME  → 12×8    ← Hook 3
#   MaxPool 2×2 s=2 SAME  →  6×4    ← Hook 4
# Format: [rows (height), cols (width)]
# ---------------------------------------------------------------------------
FEATURE_MAPS = [
    [31, 48],   # Hook 1 — after 1st maxpool
    [15, 23],   # Hook 2 — after 2nd maxpool
    [ 8, 12],   # Hook 3 — after 3rd maxpool
    [ 4,  6],   # Hook 4 — after 4th maxpool
]

# Total anchor count:
#   31*48*4 + 15*23*4 + 8*12*4 + 4*6*4 = 5952+1380+384+96 = 7812
NUM_ANCHORS = sum(r * c * NUM_DEFAULT_BOXES for r, c in FEATURE_MAPS)  # 7812

# ---------------------------------------------------------------------------
# Training thresholds & hyper-parameters (same as original tutorial)
# ---------------------------------------------------------------------------
IOU_THRESH      = 0.5    # GT ↔ anchor IoU to declare a positive
NMS_IOU_THRESH  = 0.2    # suppress overlapping predictions
CONF_THRESH     = 0.5    # minimum confidence to report a detection
NEG_POS_RATIO   = 5      # hard-negative mining ratio

BATCH_SIZE      = 32
NUM_EPOCHS      = 200
VALIDATION_SIZE = 0.05
LEARNING_RATE   = 1.0    # default Adadelta lr (same as original)
REG_SCALE       = 1e-2   # L2 weight decay
LOC_LOSS_WEIGHT = 1.0    # weight for localisation loss

# ---------------------------------------------------------------------------
# File-system paths
# ---------------------------------------------------------------------------
DATA_DIR       = "./data"
LISA_DIR       = "./data/lisa"         # root of extracted LISA dataset
PICKLE_DIR     = "./data/pickles"      # preprocessed pickle files
CHECKPOINT_DIR = "./checkpoints"       # model weight checkpoints
OUTPUT_DIR     = "./output"            # annotated inference outputs

# LISA per-class annotation files
LISA_ANNOT_FILENAME  = "frameAnnotations.csv"
LISA_ANNOT_DELIMITER = ";"             # LISA CSVs use semicolons
