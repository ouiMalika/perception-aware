"""
Visualization utilities for traffic sign detection results.

Draws bounding boxes, labels, and confidence scores on images.
Supports both interactive display and file output.
"""

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from detector import DetectionResult, SignDetection

# ---------------------------------------------------------------------------
# Colour palette - one colour per priority level
# ---------------------------------------------------------------------------
PRIORITY_COLORS = {
    1: (220, 50, 50),    # Red - critical (stop, yield, pedestrian)
    2: (240, 170, 30),   # Orange - warning (road work, curves)
    3: (50, 160, 50),    # Green - guide (direction, exit)
    4: (60, 120, 200),   # Blue - info (parking, services)
}

DEFAULT_COLOR = (180, 180, 180)


def _get_color(category: str) -> tuple[int, int, int]:
    """Get the drawing colour for a sign category based on priority."""
    from sign_taxonomy import CATEGORY_PRIORITY
    priority = CATEGORY_PRIORITY.get(category, 4)
    return PRIORITY_COLORS.get(priority, DEFAULT_COLOR)


def _get_font(size: int = 14) -> ImageFont.ImageFont:
    """Try to load a TTF font, falling back to the default bitmap font."""
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        try:
            return ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", size)
        except OSError:
            return ImageFont.load_default()


def draw_detections(
    image: Image.Image,
    detections: list[SignDetection],
    line_width: int = 3,
    font_size: int = 14,
    show_confidence: bool = True,
) -> Image.Image:
    """
    Draw bounding boxes and labels on an image.

    Returns a new image with annotations (original is not modified).
    """
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = _get_font(font_size)

    for det in detections:
        color = _get_color(det.category)
        bbox = det.bbox

        # Draw box
        draw.rectangle(
            [bbox.x1, bbox.y1, bbox.x2, bbox.y2],
            outline=color,
            width=line_width,
        )

        # Build label text
        label = det.category.replace("_", " ").title()
        if show_confidence:
            label += f" ({det.clip_confidence:.0%})"

        # Draw label background
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        label_y = max(0, bbox.y1 - text_h - 6)

        draw.rectangle(
            [bbox.x1, label_y, bbox.x1 + text_w + 8, label_y + text_h + 6],
            fill=color,
        )
        draw.text(
            (bbox.x1 + 4, label_y + 2),
            label,
            fill=(255, 255, 255),
            font=font,
        )

    return annotated


def draw_result(
    image: Image.Image,
    result: DetectionResult,
    **kwargs,
) -> Image.Image:
    """Draw all detections from a DetectionResult onto the image."""
    return draw_detections(image, result.detections, **kwargs)


def save_annotated(
    image: Image.Image,
    detections: list[SignDetection],
    output_path: str | Path,
    **kwargs,
) -> Path:
    """Draw detections and save to a file."""
    annotated = draw_detections(image, detections, **kwargs)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(output_path)
    return output_path


def annotated_to_bytes(
    image: Image.Image,
    detections: list[SignDetection],
    format: str = "JPEG",
    **kwargs,
) -> bytes:
    """Draw detections and return as bytes (for API responses)."""
    annotated = draw_detections(image, detections, **kwargs)
    buf = io.BytesIO()
    annotated.save(buf, format=format)
    return buf.getvalue()
