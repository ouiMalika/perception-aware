"""
Comprehensive traffic sign taxonomy for zero-shot CLIP classification.

Defines 20+ sign categories with CLIP-optimized text prompts for each.
Each category includes multiple prompt variations to improve zero-shot accuracy.
"""

# ---------------------------------------------------------------------------
# Master category registry
# ---------------------------------------------------------------------------
# Each key is a canonical sign category.
# "prompts" are fed to CLIP as candidate labels (zero-shot classification).
# "description" is a human-readable summary.
# "priority" ranks urgency for driving decisions (1 = highest).
# ---------------------------------------------------------------------------

SIGN_TAXONOMY = {
    # === Regulatory Signs ===
    "stop": {
        "description": "Stop sign - requires full stop at intersection",
        "priority": 1,
        "prompts": [
            "a red octagonal stop sign",
            "a stop sign on the road",
            "traffic stop sign",
        ],
    },
    "yield": {
        "description": "Yield sign - give right of way",
        "priority": 1,
        "prompts": [
            "a red and white triangular yield sign",
            "a yield sign on the road",
            "traffic yield sign",
            "give way sign",
        ],
    },
    "speed_limit": {
        "description": "Speed limit sign - maximum allowed speed",
        "priority": 2,
        "prompts": [
            "a speed limit sign showing a number",
            "speed limit traffic sign",
            "maximum speed sign",
            "a round white sign with a red border showing speed",
        ],
    },
    "no_entry": {
        "description": "No entry / do not enter sign",
        "priority": 1,
        "prompts": [
            "a red circular no entry sign",
            "do not enter traffic sign",
            "a round red sign with a white bar",
            "wrong way do not enter sign",
        ],
    },
    "no_turn": {
        "description": "No turn allowed (no left turn, no right turn, no U-turn)",
        "priority": 2,
        "prompts": [
            "a no turn traffic sign",
            "no left turn sign",
            "no right turn sign",
            "no U-turn sign",
            "a sign with a crossed-out turning arrow",
        ],
    },
    "one_way": {
        "description": "One way traffic sign",
        "priority": 2,
        "prompts": [
            "a one way street sign",
            "one way traffic sign with arrow",
            "black and white one way sign",
        ],
    },
    "keep_right_left": {
        "description": "Keep right or keep left sign",
        "priority": 2,
        "prompts": [
            "keep right traffic sign",
            "keep left traffic sign",
            "a blue circular sign with a white arrow",
        ],
    },

    # === Warning Signs ===
    "pedestrian_crossing": {
        "description": "Pedestrian crossing ahead",
        "priority": 1,
        "prompts": [
            "a pedestrian crossing warning sign",
            "crosswalk sign with walking person",
            "a yellow diamond sign with a person walking",
            "pedestrian crossing ahead sign",
            "school crossing sign",
        ],
    },
    "road_work": {
        "description": "Road work / construction ahead",
        "priority": 2,
        "prompts": [
            "a road work construction sign",
            "road construction ahead warning sign",
            "an orange diamond sign with a worker figure",
            "men at work traffic sign",
            "road maintenance sign",
        ],
    },
    "curve_warning": {
        "description": "Curve ahead warning (sharp turn, winding road)",
        "priority": 2,
        "prompts": [
            "a curve ahead warning sign",
            "sharp turn warning sign",
            "winding road sign",
            "a yellow diamond sign with a curved arrow",
            "dangerous curve sign",
        ],
    },
    "intersection_warning": {
        "description": "Intersection or junction ahead warning",
        "priority": 2,
        "prompts": [
            "an intersection ahead warning sign",
            "junction warning sign",
            "crossroads warning sign",
            "a yellow sign showing intersecting roads",
        ],
    },
    "slippery_road": {
        "description": "Slippery road surface warning",
        "priority": 2,
        "prompts": [
            "a slippery road warning sign",
            "slippery when wet sign",
            "a yellow sign with a skidding car",
        ],
    },
    "animal_crossing": {
        "description": "Animal crossing warning (deer, cattle, etc.)",
        "priority": 2,
        "prompts": [
            "an animal crossing warning sign",
            "deer crossing sign",
            "a yellow sign with an animal silhouette",
            "cattle crossing sign",
            "wildlife crossing sign",
        ],
    },
    "school_zone": {
        "description": "School zone - children present",
        "priority": 1,
        "prompts": [
            "a school zone warning sign",
            "school zone sign with children",
            "school crossing zone sign",
            "a yellow pentagon sign with children figures",
        ],
    },
    "railroad_crossing": {
        "description": "Railroad crossing ahead",
        "priority": 1,
        "prompts": [
            "a railroad crossing sign",
            "railway crossing warning sign",
            "a round yellow sign with an X and R R",
            "train crossing sign",
        ],
    },
    "merge_warning": {
        "description": "Merge or lane ends warning",
        "priority": 2,
        "prompts": [
            "a merge warning sign",
            "lane ends merge sign",
            "merging traffic sign",
            "a yellow diamond sign with merging arrows",
        ],
    },

    # === Guide / Information Signs ===
    "direction": {
        "description": "Direction sign (pointing to destinations)",
        "priority": 3,
        "prompts": [
            "a green highway direction sign",
            "a road direction sign pointing to a city",
            "green direction sign with white text and arrow",
            "highway guide sign",
            "road name direction sign",
        ],
    },
    "exit": {
        "description": "Highway exit sign",
        "priority": 3,
        "prompts": [
            "a highway exit sign",
            "freeway exit sign with exit number",
            "green highway exit sign",
            "a sign showing an exit ramp",
            "interstate exit sign",
        ],
    },
    "highway_route": {
        "description": "Highway / interstate route marker",
        "priority": 3,
        "prompts": [
            "a highway route number sign",
            "an interstate shield sign",
            "a US route marker sign",
            "a state highway number sign",
        ],
    },
    "distance_marker": {
        "description": "Distance or mileage marker sign",
        "priority": 4,
        "prompts": [
            "a green distance marker sign",
            "a milepost sign",
            "a sign showing distances to cities",
        ],
    },
    "parking": {
        "description": "Parking sign",
        "priority": 4,
        "prompts": [
            "a blue parking sign with letter P",
            "a parking allowed sign",
            "no parking sign",
            "parking area sign",
        ],
    },
    "information": {
        "description": "General information sign (hospital, gas, food, rest area)",
        "priority": 4,
        "prompts": [
            "a blue highway information sign",
            "a road services sign showing gas or food",
            "hospital sign on road",
            "rest area sign",
            "tourist information sign",
        ],
    },

    # === Traffic Control Signs ===
    "traffic_light": {
        "description": "Traffic signal / traffic light ahead",
        "priority": 1,
        "prompts": [
            "a traffic signal ahead sign",
            "traffic light warning sign",
            "a sign showing a traffic light",
        ],
    },
    "roundabout": {
        "description": "Roundabout / traffic circle ahead",
        "priority": 2,
        "prompts": [
            "a roundabout sign",
            "traffic circle sign",
            "a sign with circular arrows for roundabout",
            "rotary traffic sign",
        ],
    },

    # === Temporary / Construction Signs ===
    "detour": {
        "description": "Detour sign",
        "priority": 2,
        "prompts": [
            "a detour sign with arrow",
            "an orange detour traffic sign",
            "road detour sign",
        ],
    },
    "lane_closure": {
        "description": "Lane closure or road closed sign",
        "priority": 2,
        "prompts": [
            "a road closed sign",
            "lane closure sign",
            "a sign indicating road closure ahead",
            "an orange road closed barricade sign",
        ],
    },
}

# ---------------------------------------------------------------------------
# Negative prompts — these help CLIP reject non-sign objects.
# Any crop where the best category is "not_a_sign" will be discarded.
# ---------------------------------------------------------------------------
NEGATIVE_PROMPTS = [
    "a car on the road",
    "a truck on the road",
    "a bus on the road",
    "a person walking",
    "a pedestrian",
    "a bicycle on the road",
    "a motorcycle on the road",
    "a building",
    "a tree",
    "a street lamp or utility pole",
    "a traffic cone",
    "a road barrier",
    "the sky",
    "the road surface",
    "a vehicle license plate",
    "a window or door",
]

# ---------------------------------------------------------------------------
# Derived structures for pipeline use
# ---------------------------------------------------------------------------

ALL_CATEGORIES = list(SIGN_TAXONOMY.keys())

# Flat list of every CLIP prompt, mapped back to its category
PROMPT_TO_CATEGORY = {}
ALL_PROMPTS = []
for category, info in SIGN_TAXONOMY.items():
    for prompt in info["prompts"]:
        PROMPT_TO_CATEGORY[prompt] = category
        ALL_PROMPTS.append(prompt)

# Include negative prompts mapped to the "not_a_sign" pseudo-category
for prompt in NEGATIVE_PROMPTS:
    PROMPT_TO_CATEGORY[prompt] = "not_a_sign"
    ALL_PROMPTS.append(prompt)

# Priority lookup
CATEGORY_PRIORITY = {cat: info["priority"] for cat, info in SIGN_TAXONOMY.items()}

# Human-readable descriptions
CATEGORY_DESCRIPTIONS = {
    cat: info["description"] for cat, info in SIGN_TAXONOMY.items()
}


def get_categories_by_priority(max_priority: int = 4) -> list[str]:
    """Return categories filtered by maximum priority level."""
    return [
        cat
        for cat, pri in CATEGORY_PRIORITY.items()
        if pri <= max_priority
    ]


def get_prompts_for_category(category: str) -> list[str]:
    """Return the CLIP prompts for a given sign category."""
    if category not in SIGN_TAXONOMY:
        raise ValueError(f"Unknown category: {category}")
    return SIGN_TAXONOMY[category]["prompts"]
