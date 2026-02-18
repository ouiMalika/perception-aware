from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("sign_detection", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="detectionjob",
            name="output_video",
            field=models.FileField(blank=True, help_text="Annotated output video with bounding boxes", null=True, upload_to="videos/"),
        ),
    ]
