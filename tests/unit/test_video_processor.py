"""Unit tests for the video processing module (data classes and deduplication)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "worker"))

from detector import BoundingBox, SignDetection
from video_processor import (
    FrameDetection,
    TrackedSign,
    VideoDetectionResult,
    _deduplicate_tracks,
    _iou,
)


# ---------------------------------------------------------------------------
# IoU tests
# ---------------------------------------------------------------------------


class TestIoU:

    def test_identical_boxes(self):
        a = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        assert _iou(a, a) == 1.0

    def test_no_overlap(self):
        a = BoundingBox(x1=0, y1=0, x2=50, y2=50)
        b = BoundingBox(x1=60, y1=60, x2=100, y2=100)
        assert _iou(a, b) == 0.0

    def test_partial_overlap(self):
        a = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        b = BoundingBox(x1=50, y1=50, x2=150, y2=150)
        # Intersection: 50x50 = 2500
        # Union: 10000 + 10000 - 2500 = 17500
        expected = 2500 / 17500
        assert abs(_iou(a, b) - expected) < 1e-6

    def test_contained_box(self):
        a = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        b = BoundingBox(x1=25, y1=25, x2=75, y2=75)
        # Intersection: 50x50 = 2500
        # Union: 10000 + 2500 - 2500 = 10000
        assert abs(_iou(a, b) - 0.25) < 1e-6

    def test_zero_area_box(self):
        a = BoundingBox(x1=50, y1=50, x2=50, y2=50)
        b = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        assert _iou(a, b) == 0.0

    def test_touching_edge(self):
        a = BoundingBox(x1=0, y1=0, x2=50, y2=50)
        b = BoundingBox(x1=50, y1=0, x2=100, y2=50)
        assert _iou(a, b) == 0.0


# ---------------------------------------------------------------------------
# FrameDetection tests
# ---------------------------------------------------------------------------


def _make_detection(category="stop", confidence=0.9, x1=10, y1=20, x2=110, y2=120):
    return SignDetection(
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        category=category,
        category_description=f"{category} sign",
        confidence=confidence,
        yolo_confidence=confidence,
        clip_confidence=confidence,
        clip_scores={category: confidence},
    )


class TestFrameDetection:

    def test_to_dict(self):
        det = _make_detection()
        fd = FrameDetection(detection=det, frame_number=30, timestamp_s=1.0)
        d = fd.to_dict()
        assert d["frame_number"] == 30
        assert d["timestamp_s"] == 1.0
        assert d["category"] == "stop"
        assert d["confidence"] == 0.9

    def test_to_dict_preserves_bbox(self):
        det = _make_detection(x1=5.5, y1=10.2, x2=200.7, y2=300.1)
        fd = FrameDetection(detection=det, frame_number=0, timestamp_s=0.0)
        d = fd.to_dict()
        assert d["bbox"]["x1"] == 5.5
        assert d["bbox"]["y2"] == 300.1


# ---------------------------------------------------------------------------
# Deduplication tests
# ---------------------------------------------------------------------------


class TestDeduplicateTracks:

    def test_single_detection(self):
        det = _make_detection()
        fds = [FrameDetection(detection=det, frame_number=0, timestamp_s=0.0)]
        tracks = _deduplicate_tracks(fds)
        assert len(tracks) == 1
        assert tracks[0].category == "stop"
        assert tracks[0].frame_count == 1

    def test_same_sign_across_frames(self):
        """Same sign at the same position across 3 frames should merge."""
        fds = []
        for i in range(3):
            det = _make_detection(category="stop", confidence=0.8 + i * 0.05)
            fds.append(FrameDetection(detection=det, frame_number=i, timestamp_s=i * 0.5))

        tracks = _deduplicate_tracks(fds, iou_threshold=0.3)
        assert len(tracks) == 1
        assert tracks[0].frame_count == 3
        assert tracks[0].best_confidence == 0.9  # highest

    def test_different_categories_not_merged(self):
        """Different categories at same position should remain separate."""
        det1 = _make_detection(category="stop")
        det2 = _make_detection(category="yield")
        fds = [
            FrameDetection(detection=det1, frame_number=0, timestamp_s=0.0),
            FrameDetection(detection=det2, frame_number=1, timestamp_s=0.5),
        ]
        tracks = _deduplicate_tracks(fds)
        assert len(tracks) == 2

    def test_non_overlapping_same_category(self):
        """Same category at different positions should remain separate."""
        det1 = _make_detection(category="stop", x1=0, y1=0, x2=50, y2=50)
        det2 = _make_detection(category="stop", x1=200, y1=200, x2=250, y2=250)
        fds = [
            FrameDetection(detection=det1, frame_number=0, timestamp_s=0.0),
            FrameDetection(detection=det2, frame_number=1, timestamp_s=0.5),
        ]
        tracks = _deduplicate_tracks(fds)
        assert len(tracks) == 2

    def test_gap_beyond_threshold_creates_new_track(self):
        """Same sign reappearing after max_gap_frames should be a new track."""
        det1 = _make_detection()
        det2 = _make_detection()
        fds = [
            FrameDetection(detection=det1, frame_number=0, timestamp_s=0.0),
            FrameDetection(detection=det2, frame_number=100, timestamp_s=50.0),
        ]
        tracks = _deduplicate_tracks(fds, max_gap_frames=15)
        assert len(tracks) == 2

    def test_track_timestamps(self):
        """Track should have correct first/last timestamps."""
        fds = [
            FrameDetection(
                detection=_make_detection(),
                frame_number=10,
                timestamp_s=0.5,
            ),
            FrameDetection(
                detection=_make_detection(),
                frame_number=11,
                timestamp_s=1.0,
            ),
            FrameDetection(
                detection=_make_detection(),
                frame_number=12,
                timestamp_s=1.5,
            ),
        ]
        tracks = _deduplicate_tracks(fds)
        assert len(tracks) == 1
        assert tracks[0].first_seen_s == 0.5
        assert tracks[0].last_seen_s == 1.5
        assert tracks[0].first_frame == 10
        assert tracks[0].last_frame == 12

    def test_empty_input(self):
        tracks = _deduplicate_tracks([])
        assert tracks == []


# ---------------------------------------------------------------------------
# TrackedSign tests
# ---------------------------------------------------------------------------


class TestTrackedSign:

    def test_to_dict(self):
        ts = TrackedSign(
            sign_id=1,
            category="stop",
            description="Stop sign",
            best_confidence=0.95,
            best_bbox=BoundingBox(x1=10, y1=20, x2=110, y2=120),
            first_seen_s=1.0,
            last_seen_s=3.5,
            first_frame=30,
            last_frame=105,
            frame_count=5,
            best_clip_scores={"stop": 0.95, "yield": 0.03},
        )
        d = ts.to_dict()
        assert d["sign_id"] == 1
        assert d["category"] == "stop"
        assert d["duration_s"] == 2.5
        assert d["frame_count"] == 5
        assert d["best_bbox"]["x1"] == 10.0


# ---------------------------------------------------------------------------
# VideoDetectionResult tests
# ---------------------------------------------------------------------------


class TestVideoDetectionResult:

    def test_to_dict(self):
        ts = TrackedSign(
            sign_id=1,
            category="stop",
            description="Stop sign",
            best_confidence=0.9,
            best_bbox=BoundingBox(x1=10, y1=20, x2=110, y2=120),
            first_seen_s=0.0,
            last_seen_s=2.0,
            first_frame=0,
            last_frame=60,
            frame_count=4,
        )
        result = VideoDetectionResult(
            video_source="/tmp/test.mp4",
            duration_s=10.0,
            fps=30.0,
            total_frames=300,
            frames_processed=20,
            frame_interval=0.5,
            tracked_signs=[ts],
            per_frame_detections=[],
        )
        d = result.to_dict()
        assert d["video_source"] == "/tmp/test.mp4"
        assert d["video_info"]["duration_s"] == 10.0
        assert d["video_info"]["fps"] == 30.0
        assert d["video_info"]["frames_processed"] == 20
        assert d["num_unique_signs"] == 1
        assert d["num_frame_detections"] == 0
        assert d["error"] is None

    def test_to_dict_with_error(self):
        result = VideoDetectionResult(
            video_source="/tmp/bad.mp4",
            duration_s=0,
            fps=0,
            total_frames=0,
            frames_processed=0,
            frame_interval=0.5,
            tracked_signs=[],
            per_frame_detections=[],
            error="Cannot open video",
        )
        d = result.to_dict()
        assert d["error"] == "Cannot open video"
        assert d["num_unique_signs"] == 0
