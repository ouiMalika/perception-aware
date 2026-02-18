"""
Visualization utilities for YOLOv7 traffic sign detection results.

Draws bounding boxes and category labels on images/frames.
Color-coded by priority level from the sign taxonomy.
"""

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from detector import DetectionResult, SignDetection

# Priority → RGB color
_PRIORITY_COLORS: dict[int, tuple[int, int, int]] = {
    1: (220, 50,  50),   # Red    – critical (stop, yield, pedestrian, railroad)
    2: (240, 170, 30),   # Amber  – warning  (road work, curves, merge)
    3: (50,  160, 50),   # Green  – guide    (direction, exit, route)
    4: (60,  120, 200),  # Blue   – info     (parking, services)
}
_DEFAULT_COLOR = (160, 160, 160)


def _priority_color(category: str) -> tuple[int, int, int]:
    from sign_taxonomy import CATEGORY_PRIORITY
    priority = CATEGORY_PRIORITY.get(category, 4)
    return _PRIORITY_COLORS.get(priority, _DEFAULT_COLOR)


def _load_font(size: int = 14) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_detections(
    image: Image.Image,
    detections: list[SignDetection],
    line_width: int = 3,
    font_size: int = 14,
    show_confidence: bool = True,
    show_raw_class: bool = False,
) -> Image.Image:
    """
    Draw bounding boxes and labels on a PIL image.

    Returns a new annotated image (original is unchanged).

    Args:
        image: Source PIL image (RGB).
        detections: List of SignDetection objects from the YOLOv7 pipeline.
        line_width: Bounding box stroke width in pixels.
        font_size: Font size for label text.
        show_confidence: Append confidence % to the label.
        show_raw_class: Also show the raw YOLOv7 class name in parentheses.
    """
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    font = _load_font(font_size)

    for det in detections:
        color = _priority_color(det.category)
        b = det.bbox

        # Bounding box
        draw.rectangle([b.x1, b.y1, b.x2, b.y2], outline=color, width=line_width)

        # Label text
        label = det.category.replace("_", " ").title()
        if show_confidence:
            label += f" {det.confidence:.0%}"
        if show_raw_class and det.raw_class_name != det.category:
            label += f" ({det.raw_class_name})"

        # Label background
        tbbox = draw.textbbox((0, 0), label, font=font)
        tw = tbbox[2] - tbbox[0]
        th = tbbox[3] - tbbox[1]
        ly = max(0, b.y1 - th - 6)
        draw.rectangle(
            [b.x1, ly, b.x1 + tw + 8, ly + th + 6],
            fill=color,
        )
        draw.text((b.x1 + 4, ly + 2), label, fill=(255, 255, 255), font=font)

    return annotated


def draw_result(image: Image.Image, result: DetectionResult, **kwargs) -> Image.Image:
    """Draw all detections from a DetectionResult onto the image."""
    return draw_detections(image, result.detections, **kwargs)


def save_annotated(
    image: Image.Image,
    detections: list[SignDetection],
    output_path: str | Path,
    **kwargs,
) -> Path:
    """Draw detections and save to file."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    draw_detections(image, detections, **kwargs).save(out)
    return out


def annotated_to_bytes(
    image: Image.Image,
    detections: list[SignDetection],
    fmt: str = "JPEG",
    **kwargs,
) -> bytes:
    """Draw detections and return as raw bytes for API responses."""
    buf = io.BytesIO()
    draw_detections(image, detections, **kwargs).save(buf, format=fmt)
    return buf.getvalue()
