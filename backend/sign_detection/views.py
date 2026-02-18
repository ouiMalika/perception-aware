"""
API views for the traffic sign detection service.

Endpoints:
  POST /api/detect/         - Submit images for detection (returns job ID)
  POST /api/detect/video/   - Upload a video for detection (returns job ID)
  GET  /api/jobs/<job_id>/  - Poll job status / retrieve results
  GET  /api/jobs/           - List all jobs for the authenticated user
  GET  /api/signs/          - List all detected signs (filterable by category)
  GET  /api/taxonomy/       - Return the full sign taxonomy (no auth required)
  POST /api/auth/register/  - Register a new user
  POST /api/auth/login/     - Log in and get an auth token
"""

import json
import uuid

from celery.result import AsyncResult
from django.conf import settings
from django.contrib.auth import authenticate
from django.contrib.auth.models import User
from rest_framework import generics, permissions, status
from rest_framework.authtoken.models import Token
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import DetectedSign, DetectionJob, TrackedSign
from .serializers import (
    DetectedSignSerializer,
    DetectionJobSerializer,
    DetectionRequestSerializer,
    JobStatusSerializer,
    TrackedSignSerializer,
    VideoDetectionRequestSerializer,
)

# ---------------------------------------------------------------------------
# Sign taxonomy (served from worker module)
# ---------------------------------------------------------------------------

# Inline copy so the backend doesn't depend on the worker package at import
# time.  The canonical source is worker/sign_taxonomy.py.
_TAXONOMY_CACHE = None


def _get_taxonomy():
    global _TAXONOMY_CACHE
    if _TAXONOMY_CACHE is None:
        try:
            import importlib
            mod = importlib.import_module("sign_taxonomy")
            _TAXONOMY_CACHE = {
                cat: {
                    "description": info["description"],
                    "priority": info["priority"],
                    "num_prompts": len(info["prompts"]),
                }
                for cat, info in mod.SIGN_TAXONOMY.items()
            }
        except ImportError:
            # Fallback: return a summary without needing the worker module
            _TAXONOMY_CACHE = _BUILTIN_TAXONOMY_SUMMARY
    return _TAXONOMY_CACHE


_BUILTIN_TAXONOMY_SUMMARY = {
    "stop": {"description": "Stop sign", "priority": 1, "num_prompts": 3},
    "yield": {"description": "Yield sign", "priority": 1, "num_prompts": 4},
    "speed_limit": {"description": "Speed limit sign", "priority": 2, "num_prompts": 4},
    "no_entry": {"description": "No entry / do not enter", "priority": 1, "num_prompts": 4},
    "no_turn": {"description": "No turn allowed", "priority": 2, "num_prompts": 5},
    "one_way": {"description": "One way street", "priority": 2, "num_prompts": 3},
    "keep_right_left": {"description": "Keep right or left", "priority": 2, "num_prompts": 3},
    "pedestrian_crossing": {"description": "Pedestrian crossing", "priority": 1, "num_prompts": 5},
    "road_work": {"description": "Road work / construction", "priority": 2, "num_prompts": 5},
    "curve_warning": {"description": "Curve ahead warning", "priority": 2, "num_prompts": 5},
    "intersection_warning": {"description": "Intersection ahead", "priority": 2, "num_prompts": 4},
    "slippery_road": {"description": "Slippery road", "priority": 2, "num_prompts": 3},
    "animal_crossing": {"description": "Animal crossing", "priority": 2, "num_prompts": 5},
    "school_zone": {"description": "School zone", "priority": 1, "num_prompts": 4},
    "railroad_crossing": {"description": "Railroad crossing", "priority": 1, "num_prompts": 4},
    "merge_warning": {"description": "Merge / lane ends", "priority": 2, "num_prompts": 4},
    "direction": {"description": "Direction sign", "priority": 3, "num_prompts": 5},
    "exit": {"description": "Highway exit sign", "priority": 3, "num_prompts": 5},
    "highway_route": {"description": "Route marker", "priority": 3, "num_prompts": 4},
    "distance_marker": {"description": "Distance marker", "priority": 4, "num_prompts": 3},
    "parking": {"description": "Parking sign", "priority": 4, "num_prompts": 4},
    "information": {"description": "Info sign (hospital, gas, etc.)", "priority": 4, "num_prompts": 5},
    "traffic_light": {"description": "Traffic signal ahead", "priority": 1, "num_prompts": 3},
    "roundabout": {"description": "Roundabout ahead", "priority": 2, "num_prompts": 4},
    "detour": {"description": "Detour", "priority": 2, "num_prompts": 3},
    "lane_closure": {"description": "Lane closure / road closed", "priority": 2, "num_prompts": 4},
}


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------


class TaxonomyView(APIView):
    """Return the full sign taxonomy. No authentication required."""

    permission_classes = [permissions.AllowAny]

    def get(self, request):
        return Response({
            "categories": _get_taxonomy(),
            "total_categories": len(_get_taxonomy()),
        })


class DetectView(APIView):
    """Submit image URLs for asynchronous sign detection."""

    def post(self, request):
        serializer = DetectionRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        image_urls = serializer.validated_data["image_urls"]
        options = {
            "yolo_conf": serializer.validated_data.get("yolo_conf", 0.25),
            "clip_conf": serializer.validated_data.get("clip_conf", 0.15),
        }

        # Dispatch to Celery
        from celery import current_app
        task = current_app.send_task(
            "detect_signs_batch",
            args=[image_urls, options],
        )

        # Create tracking record
        job = DetectionJob.objects.create(
            job_id=task.id,
            owner=request.user,
            job_type="image",
            image_urls=image_urls,
            status="pending",
        )

        return Response(
            {
                "job_id": task.id,
                "status": "pending",
                "job_type": "image",
                "num_images": len(image_urls),
                "message": "Detection job submitted. Poll /api/jobs/<job_id>/ for results.",
            },
            status=status.HTTP_202_ACCEPTED,
        )


class VideoDetectView(APIView):
    """Upload a video file for asynchronous sign detection."""

    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        serializer = VideoDetectionRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        video_file = serializer.validated_data["video"]
        options = {
            "yolo_conf": serializer.validated_data.get("yolo_conf", 0.25),
            "clip_conf": serializer.validated_data.get("clip_conf", 0.15),
            "interval_s": serializer.validated_data.get("interval_s", 0.5),
            "max_frames": serializer.validated_data.get("max_frames", 300),
        }

        # Create job first to get the file saved via Django's FileField.
        # Use a temporary UUID as job_id, then replace with Celery task ID.
        temp_id = f"video-{uuid.uuid4().hex[:16]}"
        job = DetectionJob(
            job_id=temp_id,
            owner=request.user,
            job_type="video",
            status="pending",
        )
        job.video_file = video_file
        job.save()

        # Dispatch to Celery with the saved file path
        from celery import current_app
        task = current_app.send_task(
            "detect_signs_video",
            args=[job.video_file.path, options],
        )

        # Update job with real Celery task ID
        job.job_id = task.id
        job.save(update_fields=["job_id"])

        return Response(
            {
                "job_id": task.id,
                "status": "pending",
                "job_type": "video",
                "video_filename": video_file.name,
                "options": {
                    "interval_s": options["interval_s"],
                    "max_frames": options["max_frames"],
                },
                "message": "Video detection job submitted. Poll /api/jobs/<job_id>/ for results.",
            },
            status=status.HTTP_202_ACCEPTED,
        )


class JobStatusView(APIView):
    """Poll a detection job's status and retrieve results when complete."""

    def get(self, request, job_id):
        try:
            job = DetectionJob.objects.get(job_id=job_id, owner=request.user)
        except DetectionJob.DoesNotExist:
            return Response(
                {"error": "Job not found"},
                status=status.HTTP_404_NOT_FOUND,
            )

        # If still pending/processing, check Celery
        if job.status in ("pending", "processing"):
            result = AsyncResult(job.job_id)

            if result.state == "PROGRESS":
                job.status = "processing"
                job.save(update_fields=["status"])
            elif result.state == "SUCCESS":
                job.status = "completed"
                job.result = result.result
                save_fields = ["status", "result"]
                # For video jobs, link the annotated output video
                if job.job_type == "video":
                    annotated_path = (result.result or {}).get("result", {}).get("annotated_video_path")
                    if annotated_path:
                        # Convert absolute worker path to a relative media path
                        # Worker saves to /app/media/videos/..., we need videos/...
                        import os
                        media_root = str(settings.MEDIA_ROOT)
                        if annotated_path.startswith(media_root):
                            rel_path = annotated_path[len(media_root):].lstrip("/")
                        else:
                            rel_path = os.path.basename(annotated_path)
                        job.output_video = rel_path
                        save_fields.append("output_video")
                job.save(update_fields=save_fields)
                if job.job_type == "video":
                    _persist_tracked_signs(job)
                else:
                    _persist_detections(job)
            elif result.state == "FAILURE":
                job.status = "failed"
                job.error_message = str(result.result)
                job.save(update_fields=["status", "error_message"])

        serializer = DetectionJobSerializer(job)
        return Response(serializer.data)


class JobListView(generics.ListAPIView):
    """List all detection jobs for the authenticated user."""

    serializer_class = JobStatusSerializer

    def get_queryset(self):
        return DetectionJob.objects.filter(owner=self.request.user)


class DetectedSignListView(generics.ListAPIView):
    """
    List all detected signs for the authenticated user.

    Query params:
        - category: filter by sign category (e.g. ?category=yield)
        - min_confidence: minimum combined confidence (e.g. ?min_confidence=0.5)
    """

    serializer_class = DetectedSignSerializer

    def get_queryset(self):
        qs = DetectedSign.objects.filter(job__owner=self.request.user)

        category = self.request.query_params.get("category")
        if category:
            qs = qs.filter(category=category)

        min_conf = self.request.query_params.get("min_confidence")
        if min_conf:
            qs = qs.filter(confidence__gte=float(min_conf))

        return qs


class TrackedSignListView(generics.ListAPIView):
    """
    List tracked signs from video detection jobs.

    Query params:
        - category: filter by sign category
        - min_confidence: minimum best confidence
        - job_id: filter by specific job
    """

    serializer_class = TrackedSignSerializer

    def get_queryset(self):
        qs = TrackedSign.objects.filter(job__owner=self.request.user)

        category = self.request.query_params.get("category")
        if category:
            qs = qs.filter(category=category)

        min_conf = self.request.query_params.get("min_confidence")
        if min_conf:
            qs = qs.filter(best_confidence__gte=float(min_conf))

        job_id = self.request.query_params.get("job_id")
        if job_id:
            qs = qs.filter(job__job_id=job_id)

        return qs


# ---------------------------------------------------------------------------
# Auth views
# ---------------------------------------------------------------------------


class RegisterView(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request):
        username = request.data.get("username")
        password = request.data.get("password")
        email = request.data.get("email", "")

        if not username or not password:
            return Response(
                {"error": "username and password are required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if User.objects.filter(username=username).exists():
            return Response(
                {"error": "Username already taken"},
                status=status.HTTP_409_CONFLICT,
            )

        user = User.objects.create_user(username=username, password=password, email=email)
        token, _ = Token.objects.get_or_create(user=user)

        return Response(
            {"token": token.key, "user_id": user.id, "username": user.username},
            status=status.HTTP_201_CREATED,
        )


class LoginView(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request):
        username = request.data.get("username")
        password = request.data.get("password")

        user = authenticate(username=username, password=password)
        if user is None:
            return Response(
                {"error": "Invalid credentials"},
                status=status.HTTP_401_UNAUTHORIZED,
            )

        token, _ = Token.objects.get_or_create(user=user)
        return Response({"token": token.key, "user_id": user.id, "username": user.username})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _persist_detections(job: DetectionJob):
    """Extract individual sign detections from image job results and save to DB."""
    if not job.result or "results" not in job.result:
        return

    signs = []
    for img_result in job.result["results"]:
        for det in img_result.get("detections", []):
            bbox = det.get("bbox", {})
            signs.append(DetectedSign(
                job=job,
                image_url=img_result.get("image_url", ""),
                category=det.get("category", "unknown"),
                description=det.get("description", ""),
                confidence=det.get("confidence", 0),
                yolo_confidence=det.get("yolo_confidence", 0),
                clip_confidence=det.get("clip_confidence", 0),
                bbox_x1=bbox.get("x1", 0),
                bbox_y1=bbox.get("y1", 0),
                bbox_x2=bbox.get("x2", 0),
                bbox_y2=bbox.get("y2", 0),
                top_scores=det.get("top_scores", {}),
            ))

    if signs:
        DetectedSign.objects.bulk_create(signs)


def _persist_tracked_signs(job: DetectionJob):
    """Extract tracked signs from video job results and save to DB."""
    if not job.result:
        return

    result = job.result.get("result", job.result)
    tracked_list = result.get("tracked_signs", [])

    signs = []
    for ts in tracked_list:
        bbox = ts.get("best_bbox", {})
        signs.append(TrackedSign(
            job=job,
            sign_id=ts.get("sign_id", 0),
            category=ts.get("category", "unknown"),
            description=ts.get("description", ""),
            best_confidence=ts.get("best_confidence", 0),
            bbox_x1=bbox.get("x1", 0),
            bbox_y1=bbox.get("y1", 0),
            bbox_x2=bbox.get("x2", 0),
            bbox_y2=bbox.get("y2", 0),
            first_seen_s=ts.get("first_seen_s", 0),
            last_seen_s=ts.get("last_seen_s", 0),
            first_frame=ts.get("first_frame", 0),
            last_frame=ts.get("last_frame", 0),
            frame_count=ts.get("frame_count", 0),
            top_scores=ts.get("top_scores", {}),
        ))

    if signs:
        TrackedSign.objects.bulk_create(signs)
