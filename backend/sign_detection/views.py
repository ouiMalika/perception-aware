"""
API views for the traffic sign detection service.

Endpoints:
  POST /api/detect/         - Submit images for detection (returns job ID)
  GET  /api/jobs/<job_id>/  - Poll job status / retrieve results
  GET  /api/jobs/           - List all jobs for the authenticated user
  GET  /api/signs/          - List all detected signs (filterable by category)
  GET  /api/taxonomy/       - Return the full sign taxonomy (no auth required)
  POST /api/auth/register/  - Register a new user
  POST /api/auth/login/     - Log in and get an auth token
"""

import json

from celery.result import AsyncResult
from django.contrib.auth import authenticate
from django.contrib.auth.models import User
from rest_framework import generics, permissions, status
from rest_framework.authtoken.models import Token
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import DetectedSign, DetectionJob
from .serializers import (
    DetectedSignSerializer,
    DetectionJobSerializer,
    DetectionRequestSerializer,
    JobStatusSerializer,
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
            image_urls=image_urls,
            status="pending",
        )

        return Response(
            {
                "job_id": task.id,
                "status": "pending",
                "num_images": len(image_urls),
                "message": "Detection job submitted. Poll /api/jobs/<job_id>/ for results.",
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
                job.save(update_fields=["status", "result"])
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
    """Extract individual sign detections from job results and save to DB."""
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
