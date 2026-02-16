"""
Integration tests for the full detection pipeline.

These tests require the ML models (YOLOv8 + CLIP) to be available.
They are skipped in CI unless INTEGRATION_TESTS=1 is set.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "worker"))

SKIP_INTEGRATION = os.environ.get("INTEGRATION_TESTS", "0") != "1"
skip_reason = "Set INTEGRATION_TESTS=1 to run (requires GPU/models)"


@pytest.mark.skipif(SKIP_INTEGRATION, reason=skip_reason)
class TestFullPipeline:
    """End-to-end tests with real models."""

    def test_detect_signs_in_synthetic_image(self):
        """Create a simple image and verify the pipeline runs without crashing."""
        from PIL import Image

        from detector import detect_signs_in_image

        # Create a blank image (won't detect real signs, but should not crash)
        image = Image.new("RGB", (640, 480), color=(128, 128, 128))
        detections = detect_signs_in_image(image)
        assert isinstance(detections, list)

    def test_process_image_path(self, tmp_path):
        """Process a local dummy image."""
        from PIL import Image

        from detector import process_image_path

        img_path = tmp_path / "test_sign.jpg"
        Image.new("RGB", (640, 480), color=(200, 50, 50)).save(img_path)

        result = process_image_path(str(img_path))
        assert result.error is None
        assert result.image_width == 640
        assert result.image_height == 480

    def test_visualizer_draws_without_error(self):
        """Verify visualization doesn't crash on empty detections."""
        from PIL import Image

        from detector import BoundingBox, SignDetection
        from visualizer import draw_detections

        image = Image.new("RGB", (640, 480))
        detections = [
            SignDetection(
                bbox=BoundingBox(x1=100, y1=100, x2=200, y2=200),
                category="stop",
                category_description="Stop sign",
                confidence=0.9,
                yolo_confidence=0.95,
                clip_confidence=0.95,
                clip_scores={"stop": 0.95},
            )
        ]
        annotated = draw_detections(image, detections)
        assert annotated.size == (640, 480)
