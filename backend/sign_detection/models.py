"""
Models for tracking detection jobs and storing results.

Follows the VisionBoard-AI pattern: an async job model tracks Celery tasks,
and result models persist the structured detection output.
"""

from django.conf import settings
from django.db import models


class DetectionJob(models.Model):
    """Tracks an async sign detection task dispatched to Celery."""

    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("processing", "Processing"),
        ("completed", "Completed"),
        ("failed", "Failed"),
    ]

    job_id = models.CharField(max_length=255, unique=True, help_text="Celery task ID")
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="detection_jobs",
    )
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")
    image_urls = models.JSONField(default=list, help_text="Input image URLs")
    result = models.JSONField(null=True, blank=True, help_text="Detection results JSON")
    error_message = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"Job {self.job_id} ({self.status})"


class DetectedSign(models.Model):
    """Persists individual sign detections extracted from completed jobs."""

    job = models.ForeignKey(
        DetectionJob,
        on_delete=models.CASCADE,
        related_name="detected_signs",
    )
    image_url = models.URLField(max_length=2048)
    category = models.CharField(max_length=100, db_index=True)
    description = models.TextField(blank=True)
    confidence = models.FloatField()
    yolo_confidence = models.FloatField()
    clip_confidence = models.FloatField()
    bbox_x1 = models.FloatField()
    bbox_y1 = models.FloatField()
    bbox_x2 = models.FloatField()
    bbox_y2 = models.FloatField()
    top_scores = models.JSONField(default=dict, help_text="Top-5 CLIP category scores")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-confidence"]
        indexes = [
            models.Index(fields=["category", "-confidence"]),
        ]

    def __str__(self):
        return f"{self.category} ({self.confidence:.2%}) in {self.image_url}"
