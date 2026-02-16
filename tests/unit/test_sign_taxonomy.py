"""Unit tests for the sign taxonomy module."""

import sys
from pathlib import Path

import pytest

# Add worker directory to path so we can import the module directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "worker"))

from sign_taxonomy import (
    ALL_CATEGORIES,
    ALL_PROMPTS,
    CATEGORY_DESCRIPTIONS,
    CATEGORY_PRIORITY,
    PROMPT_TO_CATEGORY,
    SIGN_TAXONOMY,
    get_categories_by_priority,
    get_prompts_for_category,
)


class TestTaxonomyCompleteness:
    """Verify the taxonomy covers all required sign categories."""

    REQUIRED_CATEGORIES = [
        "stop", "yield", "speed_limit", "no_entry", "pedestrian_crossing",
        "road_work", "curve_warning", "direction", "exit", "school_zone",
        "traffic_light", "roundabout", "detour", "parking", "railroad_crossing",
    ]

    def test_all_required_categories_present(self):
        for cat in self.REQUIRED_CATEGORIES:
            assert cat in SIGN_TAXONOMY, f"Missing required category: {cat}"

    def test_minimum_category_count(self):
        assert len(ALL_CATEGORIES) >= 20, (
            f"Expected at least 20 categories, got {len(ALL_CATEGORIES)}"
        )

    def test_each_category_has_prompts(self):
        for cat, info in SIGN_TAXONOMY.items():
            assert len(info["prompts"]) >= 2, (
                f"Category '{cat}' should have at least 2 prompts, "
                f"got {len(info['prompts'])}"
            )

    def test_each_category_has_description(self):
        for cat, info in SIGN_TAXONOMY.items():
            assert info["description"], f"Category '{cat}' is missing a description"

    def test_each_category_has_priority(self):
        for cat, info in SIGN_TAXONOMY.items():
            assert info["priority"] in (1, 2, 3, 4), (
                f"Category '{cat}' has invalid priority: {info['priority']}"
            )

    def test_no_duplicate_prompts(self):
        seen = set()
        for prompt in ALL_PROMPTS:
            assert prompt not in seen, f"Duplicate prompt: {prompt}"
            seen.add(prompt)


class TestPromptToCategoryMapping:
    """Verify the derived PROMPT_TO_CATEGORY mapping is consistent."""

    def test_all_prompts_map_to_valid_category(self):
        for prompt, cat in PROMPT_TO_CATEGORY.items():
            assert cat in SIGN_TAXONOMY, (
                f"Prompt '{prompt}' maps to unknown category '{cat}'"
            )

    def test_prompt_count_matches(self):
        total_prompts = sum(len(info["prompts"]) for info in SIGN_TAXONOMY.values())
        assert len(ALL_PROMPTS) == total_prompts

    def test_category_descriptions_complete(self):
        assert set(CATEGORY_DESCRIPTIONS.keys()) == set(ALL_CATEGORIES)

    def test_category_priority_complete(self):
        assert set(CATEGORY_PRIORITY.keys()) == set(ALL_CATEGORIES)


class TestHelperFunctions:

    def test_get_categories_by_priority_all(self):
        result = get_categories_by_priority(4)
        assert set(result) == set(ALL_CATEGORIES)

    def test_get_categories_by_priority_critical_only(self):
        result = get_categories_by_priority(1)
        assert all(CATEGORY_PRIORITY[c] == 1 for c in result)
        assert len(result) > 0

    def test_get_categories_by_priority_excludes_low(self):
        result = get_categories_by_priority(2)
        for cat in result:
            assert CATEGORY_PRIORITY[cat] <= 2

    def test_get_prompts_for_valid_category(self):
        prompts = get_prompts_for_category("stop")
        assert len(prompts) >= 2
        assert all(isinstance(p, str) for p in prompts)

    def test_get_prompts_for_invalid_category_raises(self):
        with pytest.raises(ValueError, match="Unknown category"):
            get_prompts_for_category("nonexistent_category")
