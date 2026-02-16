"""DRF serializers for sign detection API."""

from rest_framework import serializers

from .models import DetectedSign, DetectionJob


class DetectionRequestSerializer(serializers.Serializer):
    """Validates incoming detection requests."""

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


class DetectionJobSerializer(serializers.ModelSerializer):
    detected_signs = DetectedSignSerializer(many=True, read_only=True)

    class Meta:
        model = DetectionJob
        fields = [
            "id", "job_id", "status", "image_urls", "result",
            "error_message", "created_at", "updated_at",
            "detected_signs",
        ]
        read_only_fields = [
            "id", "job_id", "status", "result",
            "error_message", "created_at", "updated_at",
            "detected_signs",
        ]


class JobStatusSerializer(serializers.ModelSerializer):
    class Meta:
        model = DetectionJob
        fields = ["id", "job_id", "status", "error_message", "created_at", "updated_at"]
