"""Unit tests for the detector module (data classes and helpers)."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "worker"))

from detector import BoundingBox, DetectionResult, SignDetection


class TestBoundingBox:

    def test_properties(self):
        bbox = BoundingBox(x1=10, y1=20, x2=110, y2=120)
        assert bbox.width == 100
        assert bbox.height == 100
        assert bbox.area == 10000
        assert bbox.center == (60, 70)

    def test_to_dict(self):
        bbox = BoundingBox(x1=10.123, y1=20.456, x2=110.789, y2=120.012)
        d = bbox.to_dict()
        assert d == {"x1": 10.1, "y1": 20.5, "x2": 110.8, "y2": 120.0}

    def test_zero_size_box(self):
        bbox = BoundingBox(x1=50, y1=50, x2=50, y2=50)
        assert bbox.width == 0
        assert bbox.height == 0
        assert bbox.area == 0


class TestSignDetection:

    def test_to_dict(self):
        det = SignDetection(
            bbox=BoundingBox(x1=10, y1=20, x2=110, y2=120),
            category="stop",
            category_description="Stop sign",
            confidence=0.85,
            yolo_confidence=0.92,
            clip_confidence=0.924,
            clip_scores={"stop": 0.924, "yield": 0.05, "no_entry": 0.02},
        )
        d = det.to_dict()
        assert d["category"] == "stop"
        assert d["description"] == "Stop sign"
        assert d["confidence"] == 0.85
        assert "bbox" in d
        assert "top_scores" in d
        assert d["top_scores"]["stop"] == 0.924


class TestDetectionResult:

    def test_to_dict_with_detections(self):
        det = SignDetection(
            bbox=BoundingBox(x1=0, y1=0, x2=100, y2=100),
            category="yield",
            category_description="Yield sign",
            confidence=0.7,
            yolo_confidence=0.8,
            clip_confidence=0.875,
            clip_scores={"yield": 0.875},
        )
        result = DetectionResult(
            image_url="http://example.com/test.jpg",
            image_width=640,
            image_height=480,
            detections=[det],
        )
        d = result.to_dict()
        assert d["num_detections"] == 1
        assert d["image_size"]["width"] == 640
        assert d["error"] is None

    def test_to_dict_with_error(self):
        result = DetectionResult(
            image_url="http://example.com/broken.jpg",
            image_width=0,
            image_height=0,
            detections=[],
            error="Connection timeout",
        )
        d = result.to_dict()
        assert d["num_detections"] == 0
        assert d["error"] == "Connection timeout"

    def test_empty_detections(self):
        result = DetectionResult(
            image_url="http://example.com/empty.jpg",
            image_width=800,
            image_height=600,
            detections=[],
        )
        d = result.to_dict()
        assert d["num_detections"] == 0
        assert d["detections"] == []
