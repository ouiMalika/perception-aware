"""DRF serializers for sign detection API."""

from rest_framework import serializers

from .models import DetectedSign, DetectionJob, TrackedSign


# ---------------------------------------------------------------------------
# Request serializers
# ---------------------------------------------------------------------------


class DetectionRequestSerializer(serializers.Serializer):
    """Validates incoming image detection requests."""

    image_urls = serializers.ListField(
        child=serializers.URLField(max_length=2048),
        min_length=1,
        max_length=20,
        help_text="List of image URLs to analyse (max 20)",
    )
    yolo_conf = serializers.FloatField(
        required=False,
        min_value=0.05,
        max_value=0.95,
        default=0.25,
        help_text="YOLO detection confidence threshold",
    )
    clip_conf = serializers.FloatField(
        required=False,
        min_value=0.05,
        max_value=0.95,
        default=0.15,
        help_text="CLIP classification confidence threshold",
    )


# Max video file size: 200 MB
MAX_VIDEO_SIZE = 200 * 1024 * 1024

ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}


class VideoDetectionRequestSerializer(serializers.Serializer):
    """Validates incoming video detection requests."""

    video = serializers.FileField(
        help_text="Video file to analyse (mp4, avi, mov, mkv, webm; max 200 MB)",
    )
    yolo_conf = serializers.FloatField(
        required=False,
        min_value=0.05,
        max_value=0.95,
        default=0.25,
        help_text="YOLO detection confidence threshold",
    )
    clip_conf = serializers.FloatField(
        required=False,
        min_value=0.05,
        max_value=0.95,
        default=0.15,
        help_text="CLIP classification confidence threshold",
    )
    interval_s = serializers.FloatField(
        required=False,
        min_value=0.1,
        max_value=10.0,
        default=0.5,
        help_text="Seconds between sampled frames (default 0.5)",
    )
    max_frames = serializers.IntegerField(
        required=False,
        min_value=1,
        max_value=600,
        default=300,
        help_text="Maximum number of frames to process (default 300)",
    )

    def validate_video(self, value):
        # Check file extension
        name = value.name.lower()
        ext = name.rsplit(".", 1)[-1] if "." in name else ""
        if ext not in ALLOWED_VIDEO_EXTENSIONS:
            raise serializers.ValidationError(
                f"Unsupported video format '.{ext}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}"
            )
        # Check file size
        if value.size > MAX_VIDEO_SIZE:
            raise serializers.ValidationError(
                f"Video file too large ({value.size / 1024 / 1024:.1f} MB). "
                f"Maximum: {MAX_VIDEO_SIZE / 1024 / 1024:.0f} MB."
            )
        return value


# ---------------------------------------------------------------------------
# Model serializers
# ---------------------------------------------------------------------------


class DetectedSignSerializer(serializers.ModelSerializer):
    bbox = serializers.SerializerMethodField()

    class Meta:
        model = DetectedSign
        fields = [
            "id", "image_url", "category", "description",
            "confidence", "yolo_confidence", "clip_confidence",
            "bbox", "top_scores", "created_at",
        ]

    def get_bbox(self, obj):
        return {
            "x1": obj.bbox_x1,
            "y1": obj.bbox_y1,
            "x2": obj.bbox_x2,
            "y2": obj.bbox_y2,
        }


class TrackedSignSerializer(serializers.ModelSerializer):
    bbox = serializers.SerializerMethodField()
    duration_s = serializers.SerializerMethodField()

    class Meta:
        model = TrackedSign
        fields = [
            "id", "sign_id", "category", "description",
            "best_confidence", "bbox",
            "first_seen_s", "last_seen_s", "duration_s",
            "first_frame", "last_frame", "frame_count",
            "top_scores", "created_at",
        ]

    def get_bbox(self, obj):
        return {
            "x1": obj.bbox_x1,
            "y1": obj.bbox_y1,
            "x2": obj.bbox_x2,
            "y2": obj.bbox_y2,
        }

    def get_duration_s(self, obj):
        return round(obj.last_seen_s - obj.first_seen_s, 3)


class DetectionJobSerializer(serializers.ModelSerializer):
    detected_signs = DetectedSignSerializer(many=True, read_only=True)
    tracked_signs = TrackedSignSerializer(many=True, read_only=True)

    class Meta:
        model = DetectionJob
        fields = [
            "id", "job_id", "job_type", "status", "image_urls",
            "video_file", "output_video", "result", "error_message",
            "created_at", "updated_at",
            "detected_signs", "tracked_signs",
        ]
        read_only_fields = [
            "id", "job_id", "job_type", "status", "result",
            "output_video", "error_message", "created_at", "updated_at",
            "detected_signs", "tracked_signs",
        ]


class JobStatusSerializer(serializers.ModelSerializer):
    class Meta:
        model = DetectionJob
        fields = [
            "id", "job_id", "job_type", "status",
            "error_message", "created_at", "updated_at",
        ]
