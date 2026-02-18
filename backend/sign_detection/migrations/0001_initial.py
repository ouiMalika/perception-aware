import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name="DetectionJob",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("job_id", models.CharField(help_text="Celery task ID", max_length=255, unique=True)),
                ("job_type", models.CharField(choices=[("image", "Image"), ("video", "Video")], default="image", max_length=10)),
                ("status", models.CharField(choices=[("pending", "Pending"), ("processing", "Processing"), ("completed", "Completed"), ("failed", "Failed")], default="pending", max_length=20)),
                ("image_urls", models.JSONField(default=list, help_text="Input image URLs")),
                ("video_file", models.FileField(blank=True, help_text="Uploaded video file", null=True, upload_to="videos/")),
                ("result", models.JSONField(blank=True, help_text="Detection results JSON", null=True)),
                ("error_message", models.TextField(blank=True, default="")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("owner", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="detection_jobs", to=settings.AUTH_USER_MODEL)),
            ],
            options={
                "ordering": ["-created_at"],
            },
        ),
        migrations.CreateModel(
            name="DetectedSign",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("image_url", models.URLField(max_length=2048)),
                ("category", models.CharField(db_index=True, max_length=100)),
                ("description", models.TextField(blank=True)),
                ("confidence", models.FloatField()),
                ("yolo_confidence", models.FloatField()),
                ("clip_confidence", models.FloatField()),
                ("bbox_x1", models.FloatField()),
                ("bbox_y1", models.FloatField()),
                ("bbox_x2", models.FloatField()),
                ("bbox_y2", models.FloatField()),
                ("top_scores", models.JSONField(default=dict, help_text="Top-5 CLIP category scores")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("job", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="detected_signs", to="sign_detection.detectionjob")),
            ],
            options={
                "ordering": ["-confidence"],
                "indexes": [
                    models.Index(fields=["category", "-confidence"], name="sign_detect_categor_idx_01"),
                ],
            },
        ),
        migrations.CreateModel(
            name="TrackedSign",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("sign_id", models.IntegerField(help_text="Unique ID within the video")),
                ("category", models.CharField(db_index=True, max_length=100)),
                ("description", models.TextField(blank=True)),
                ("best_confidence", models.FloatField()),
                ("bbox_x1", models.FloatField()),
                ("bbox_y1", models.FloatField()),
                ("bbox_x2", models.FloatField()),
                ("bbox_y2", models.FloatField()),
                ("first_seen_s", models.FloatField(help_text="Timestamp of first appearance")),
                ("last_seen_s", models.FloatField(help_text="Timestamp of last appearance")),
                ("first_frame", models.IntegerField()),
                ("last_frame", models.IntegerField()),
                ("frame_count", models.IntegerField(help_text="Number of frames this sign appeared in")),
                ("top_scores", models.JSONField(default=dict)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("job", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="tracked_signs", to="sign_detection.detectionjob")),
            ],
            options={
                "ordering": ["first_seen_s"],
                "indexes": [
                    models.Index(fields=["category", "-best_confidence"], name="sign_detect_categor_idx_02"),
                ],
            },
        ),
    ]
