"""Admin site registration for sign detection models."""

from django.contrib import admin

from .models import DetectedSign, DetectionJob


@admin.register(DetectionJob)
class DetectionJobAdmin(admin.ModelAdmin):
    list_display = ["job_id", "owner", "status", "created_at"]
    list_filter = ["status", "created_at"]
    search_fields = ["job_id"]
    readonly_fields = ["result"]


@admin.register(DetectedSign)
class DetectedSignAdmin(admin.ModelAdmin):
    list_display = ["category", "confidence", "image_url", "created_at"]
    list_filter = ["category"]
    search_fields = ["category", "image_url"]
