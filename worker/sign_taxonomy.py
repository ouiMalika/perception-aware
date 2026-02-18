"""
Traffic sign taxonomy and class-name mapping for YOLOv7-based detection.

This module serves two purposes:

1. TAXONOMY – a comprehensive registry of US traffic sign categories with
   descriptions and priority levels (used for visualisation and output).

2. CLASS MAPPING – translates raw YOLOv7 class IDs/names (which depend on
   the training dataset) into the taxonomy categories above.

Supported model families
------------------------
COCO (yolov7.pt official weights)
    80 general classes.  Only class 9 (traffic light) and 11 (stop sign)
    are relevant for traffic sign detection.

LISA (Laboratory for Intelligent & Safe Automobiles Traffic Sign Dataset)
    47 US traffic sign classes.  Class names like "stop", "yield",
    "speedLimit25", "pedestrianCrossing", etc.

MTSD (Mapillary Traffic Sign Dataset)
    400+ classes named like "regulatory--stop--g1",
    "warning--pedestrian-crossing--g1", etc.

Custom
    Any model whose class names contain keywords from the taxonomy
    (fuzzy keyword match as a last resort).

The function `map_class_to_category(class_id, class_name)` is the single
entry-point used by detector.py.
"""

import re

# ---------------------------------------------------------------------------
# Master taxonomy registry
# ---------------------------------------------------------------------------

SIGN_TAXONOMY: dict[str, dict] = {
    # ── Regulatory ──────────────────────────────────────────────────────────
    "stop": {
        "description": "Stop sign – full stop required",
        "priority": 1,
        "color": "red",
    },
    "yield": {
        "description": "Yield – give right of way",
        "priority": 1,
        "color": "red",
    },
    "speed_limit": {
        "description": "Speed limit (any numeric value)",
        "priority": 2,
        "color": "white",
    },
    "no_entry": {
        "description": "Do Not Enter / Wrong Way",
        "priority": 1,
        "color": "red",
    },
    "no_left_turn": {
        "description": "No left turn",
        "priority": 2,
        "color": "red",
    },
    "no_right_turn": {
        "description": "No right turn",
        "priority": 2,
        "color": "red",
    },
    "no_u_turn": {
        "description": "No U-turn",
        "priority": 2,
        "color": "red",
    },
    "one_way": {
        "description": "One-way traffic",
        "priority": 2,
        "color": "black",
    },
    "keep_right": {
        "description": "Keep right (or keep left)",
        "priority": 2,
        "color": "blue",
    },
    "turn_only": {
        "description": "Mandatory turn direction",
        "priority": 2,
        "color": "white",
    },
    "no_parking": {
        "description": "No parking / No standing",
        "priority": 3,
        "color": "red",
    },
    "parking": {
        "description": "Parking allowed",
        "priority": 4,
        "color": "blue",
    },
    "no_trucks": {
        "description": "No trucks / weight limit",
        "priority": 2,
        "color": "red",
    },
    "hov_lane": {
        "description": "High-Occupancy Vehicle lane",
        "priority": 3,
        "color": "white",
    },
    "divided_highway": {
        "description": "Divided highway begins / ends",
        "priority": 2,
        "color": "yellow",
    },

    # ── Warning ──────────────────────────────────────────────────────────────
    "pedestrian_crossing": {
        "description": "Pedestrian crossing ahead",
        "priority": 1,
        "color": "yellow",
    },
    "school_zone": {
        "description": "School zone – children present",
        "priority": 1,
        "color": "yellow",
    },
    "road_work": {
        "description": "Road work / construction zone",
        "priority": 2,
        "color": "orange",
    },
    "curve_warning": {
        "description": "Curve or sharp turn ahead",
        "priority": 2,
        "color": "yellow",
    },
    "intersection_warning": {
        "description": "Intersection ahead",
        "priority": 2,
        "color": "yellow",
    },
    "slippery_road": {
        "description": "Slippery when wet",
        "priority": 2,
        "color": "yellow",
    },
    "animal_crossing": {
        "description": "Animal crossing (deer, cattle, etc.)",
        "priority": 2,
        "color": "yellow",
    },
    "railroad_crossing": {
        "description": "Railroad crossing ahead",
        "priority": 1,
        "color": "yellow",
    },
    "merge_warning": {
        "description": "Merge / lane ends",
        "priority": 2,
        "color": "yellow",
    },
    "hill_grade": {
        "description": "Hill or steep grade",
        "priority": 2,
        "color": "yellow",
    },
    "narrow_bridge": {
        "description": "Narrow bridge or road narrows",
        "priority": 2,
        "color": "yellow",
    },
    "two_way_traffic": {
        "description": "Two-way traffic ahead",
        "priority": 2,
        "color": "yellow",
    },
    "low_clearance": {
        "description": "Low clearance / height limit",
        "priority": 2,
        "color": "yellow",
    },
    "bicycle_crossing": {
        "description": "Bicycle crossing",
        "priority": 2,
        "color": "yellow",
    },

    # ── Guide / Information ──────────────────────────────────────────────────
    "direction": {
        "description": "Direction sign to destination",
        "priority": 3,
        "color": "green",
    },
    "exit": {
        "description": "Highway exit",
        "priority": 3,
        "color": "green",
    },
    "highway_route": {
        "description": "Highway / interstate route marker",
        "priority": 3,
        "color": "green",
    },
    "distance_marker": {
        "description": "Distance / mileage marker",
        "priority": 4,
        "color": "green",
    },
    "rest_area": {
        "description": "Rest area / service area",
        "priority": 4,
        "color": "blue",
    },
    "information": {
        "description": "General information (hospital, gas, food)",
        "priority": 4,
        "color": "blue",
    },

    # ── Traffic Control ──────────────────────────────────────────────────────
    "traffic_light": {
        "description": "Traffic signal / traffic light",
        "priority": 1,
        "color": "yellow",
    },
    "roundabout": {
        "description": "Roundabout ahead",
        "priority": 2,
        "color": "yellow",
    },

    # ── Temporary / Construction ─────────────────────────────────────────────
    "detour": {
        "description": "Detour",
        "priority": 2,
        "color": "orange",
    },
    "lane_closure": {
        "description": "Lane closure / road closed",
        "priority": 2,
        "color": "orange",
    },
    "signal_ahead": {
        "description": "Signal / traffic light ahead (warning)",
        "priority": 1,
        "color": "yellow",
    },
    "added_lane": {
        "description": "Added lane / extra lane begins",
        "priority": 3,
        "color": "yellow",
    },
}

# Derived helpers
ALL_CATEGORIES: list[str] = list(SIGN_TAXONOMY.keys())

CATEGORY_DESCRIPTIONS: dict[str, str] = {
    k: v["description"] for k, v in SIGN_TAXONOMY.items()
}

CATEGORY_PRIORITY: dict[str, int] = {
    k: v["priority"] for k, v in SIGN_TAXONOMY.items()
}

CATEGORY_COLOR: dict[str, str] = {
    k: v["color"] for k, v in SIGN_TAXONOMY.items()
}


# ---------------------------------------------------------------------------
# COCO class mapping
# ---------------------------------------------------------------------------
# COCO only contains two traffic-sign-adjacent classes:
#   9  = traffic light
#   11 = stop sign
# Everything else is non-sign and should return None.

_COCO_MAP: dict[int, str] = {
    9: "traffic_light",
    11: "stop",
}


# ---------------------------------------------------------------------------
# LISA dataset class mapping
# ---------------------------------------------------------------------------
# 47 US traffic sign classes. Source:
#   http://cvrr.ucsd.edu/LISA/lisa-traffic-sign-dataset.html
# Names are the exact class labels used in LISA annotations.

_LISA_MAP: dict[str, str] = {
    "stop": "stop",
    "yield": "yield",
    "signalahead": "signal_ahead",
    "pedestriancrossing": "pedestrian_crossing",
    "schoolspeedlimit": "school_zone",
    "school": "school_zone",
    "speedlimit": "speed_limit",
    "speedlimit15": "speed_limit",
    "speedlimit25": "speed_limit",
    "speedlimit30": "speed_limit",
    "speedlimit35": "speed_limit",
    "speedlimit40": "speed_limit",
    "speedlimit45": "speed_limit",
    "speedlimit50": "speed_limit",
    "speedlimit55": "speed_limit",
    "speedlimit65": "speed_limit",
    "donotenter": "no_entry",
    "wrongway": "no_entry",
    "keepright": "keep_right",
    "keepleft": "keep_right",
    "noleftturn": "no_left_turn",
    "norightturn": "no_right_turn",
    "noparking": "no_parking",
    "merge": "merge_warning",
    "addedlane": "added_lane",
    "laneends": "merge_warning",
    "turnleft": "turn_only",
    "turnright": "turn_only",
    "uturns": "no_u_turn",
    "railroad": "railroad_crossing",
    "trafficlight": "traffic_light",
    "roundabout": "roundabout",
    "dip": "hill_grade",
    "bump": "hill_grade",
    "slipperywhen wet": "slippery_road",
    "slippery": "slippery_road",
    "roadwork": "road_work",
    "workzone": "road_work",
    "oneway": "one_way",
    "dividedhighway": "divided_highway",
    "dividedends": "divided_highway",
    "bicyclecrossing": "bicycle_crossing",
    "animalsoncrossing": "animal_crossing",
    "deer": "animal_crossing",
    "low clearance": "low_clearance",
    "narrowbridge": "narrow_bridge",
    "twoway traffic": "two_way_traffic",
}


# ---------------------------------------------------------------------------
# MTSD (Mapillary Traffic Sign Dataset) keyword mapping
# ---------------------------------------------------------------------------
# MTSD names follow the pattern: <type>--<name>--<version>
# e.g. "regulatory--stop--g1", "warning--pedestrian-crossing--g1"
# We strip the type prefix and version suffix, then match keywords.

_MTSD_KEYWORD_MAP: list[tuple[str, str]] = [
    # Regulatory
    ("stop", "stop"),
    ("yield", "yield"),
    ("speed-limit", "speed_limit"),
    ("maximum-speed", "speed_limit"),
    ("do-not-enter", "no_entry"),
    ("wrong-way", "no_entry"),
    ("no-left-turn", "no_left_turn"),
    ("no-right-turn", "no_right_turn"),
    ("no-u-turn", "no_u_turn"),
    ("no-parking", "no_parking"),
    ("no-stopping", "no_parking"),
    ("parking", "parking"),
    ("one-way", "one_way"),
    ("keep-right", "keep_right"),
    ("keep-left", "keep_right"),
    ("turn-left", "turn_only"),
    ("turn-right", "turn_only"),
    ("hov", "hov_lane"),
    ("no-trucks", "no_trucks"),
    ("weight-limit", "no_trucks"),
    ("height-limit", "low_clearance"),
    ("low-clearance", "low_clearance"),
    # Warning
    ("pedestrian-crossing", "pedestrian_crossing"),
    ("school", "school_zone"),
    ("road-work", "road_work"),
    ("construction", "road_work"),
    ("curve", "curve_warning"),
    ("sharp-turn", "curve_warning"),
    ("winding-road", "curve_warning"),
    ("intersection", "intersection_warning"),
    ("slippery", "slippery_road"),
    ("animal", "animal_crossing"),
    ("deer", "animal_crossing"),
    ("cattle", "animal_crossing"),
    ("railroad", "railroad_crossing"),
    ("train", "railroad_crossing"),
    ("merge", "merge_warning"),
    ("lane-ends", "merge_warning"),
    ("hill", "hill_grade"),
    ("grade", "hill_grade"),
    ("narrow", "narrow_bridge"),
    ("two-way", "two_way_traffic"),
    ("bicycle", "bicycle_crossing"),
    ("bike", "bicycle_crossing"),
    ("signal-ahead", "signal_ahead"),
    ("roundabout", "roundabout"),
    ("traffic-circle", "roundabout"),
    # Guide
    ("exit", "exit"),
    ("route", "highway_route"),
    ("interstate", "highway_route"),
    ("highway", "highway_route"),
    ("distance", "distance_marker"),
    ("milepost", "distance_marker"),
    ("direction", "direction"),
    ("rest-area", "rest_area"),
    ("service", "rest_area"),
    ("hospital", "information"),
    ("gas", "information"),
    ("food", "information"),
    # Temporary
    ("detour", "detour"),
    ("road-closed", "lane_closure"),
    ("lane-closed", "lane_closure"),
    ("added-lane", "added_lane"),
    # Traffic control
    ("traffic-light", "traffic_light"),
    ("traffic-signal", "traffic_light"),
]


# ---------------------------------------------------------------------------
# Generic keyword fallback (any model)
# ---------------------------------------------------------------------------
# Applied when COCO/LISA/MTSD pattern matching all fail.

_GENERIC_KEYWORDS: list[tuple[str, str]] = [
    ("stop", "stop"),
    ("yield", "yield"),
    ("speed", "speed_limit"),
    ("pedestrian", "pedestrian_crossing"),
    ("crosswalk", "pedestrian_crossing"),
    ("school", "school_zone"),
    ("work", "road_work"),
    ("construction", "road_work"),
    ("curve", "curve_warning"),
    ("turn", "curve_warning"),
    ("railroad", "railroad_crossing"),
    ("rail", "railroad_crossing"),
    ("merge", "merge_warning"),
    ("animal", "animal_crossing"),
    ("deer", "animal_crossing"),
    ("slippery", "slippery_road"),
    ("no entry", "no_entry"),
    ("do not enter", "no_entry"),
    ("wrong way", "no_entry"),
    ("one way", "one_way"),
    ("keep right", "keep_right"),
    ("keep left", "keep_right"),
    ("no left", "no_left_turn"),
    ("no right", "no_right_turn"),
    ("u-turn", "no_u_turn"),
    ("uturn", "no_u_turn"),
    ("parking", "parking"),
    ("no park", "no_parking"),
    ("exit", "exit"),
    ("route", "highway_route"),
    ("roundabout", "roundabout"),
    ("detour", "detour"),
    ("closed", "lane_closure"),
    ("signal", "signal_ahead"),
    ("traffic light", "traffic_light"),
    ("light", "traffic_light"),
]


# ---------------------------------------------------------------------------
# Public mapping entry-point
# ---------------------------------------------------------------------------


def map_class_to_category(class_id: int, class_name: str) -> str | None:
    """
    Map a YOLOv7 class (id + name) to a taxonomy category key.

    Returns None if the class is not a traffic sign (e.g. car, person).

    Strategy (in order):
      1. COCO numeric ID check
      2. Exact LISA name match (case-insensitive, no spaces)
      3. MTSD keyword match on the name segment
      4. Generic keyword scan of the raw class name
    """
    name_lower = class_name.lower().strip()
    name_nospace = re.sub(r"[\s\-_]", "", name_lower)

    # 1. COCO numeric mapping
    if class_id in _COCO_MAP:
        return _COCO_MAP[class_id]

    # 2. LISA exact match (normalise to no-spaces lowercase)
    lisa_key = name_nospace
    if lisa_key in _LISA_MAP:
        return _LISA_MAP[lisa_key]

    # 3. MTSD keyword scan
    #    MTSD names: "regulatory--stop--g1" → segments = ["regulatory","stop","g1"]
    #    We join the middle segment(s) for keyword matching.
    if "--" in name_lower:
        parts = name_lower.split("--")
        # Drop type prefix (first) and version suffix (last)
        middle = "--".join(parts[1:-1]) if len(parts) > 2 else parts[-1]
        for keyword, category in _MTSD_KEYWORD_MAP:
            if keyword in middle:
                return category
        # Also try full name
        full = "--".join(parts)
        for keyword, category in _MTSD_KEYWORD_MAP:
            if keyword in full:
                return category

    # 4. Generic keyword fallback
    for keyword, category in _GENERIC_KEYWORDS:
        if keyword in name_lower:
            return category

    # Not a traffic sign; caller will skip this detection
    return None


# ---------------------------------------------------------------------------
# Convenience helpers for downstream modules
# ---------------------------------------------------------------------------


def get_categories_by_priority(max_priority: int = 4) -> list[str]:
    return [c for c, p in CATEGORY_PRIORITY.items() if p <= max_priority]
