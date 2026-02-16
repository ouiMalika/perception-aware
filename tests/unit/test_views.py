"""Unit tests for Django API views."""

import io
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import django

# Configure Django settings before importing anything else
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "perception_backend.settings")
os.environ.setdefault("POSTGRES_HOST", "localhost")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "backend"))

# Use SQLite for tests
from django.conf import settings
settings.DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": ":memory:",
    }
}

django.setup()

import pytest
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import RequestFactory
from rest_framework.authtoken.models import Token
from rest_framework.test import APIRequestFactory, force_authenticate

from sign_detection.views import (
    DetectView,
    LoginView,
    RegisterView,
    TaxonomyView,
    VideoDetectView,
)


@pytest.fixture
def api_factory():
    return APIRequestFactory()


@pytest.fixture
def user(db):
    user = User.objects.create_user(username="testuser", password="testpass123")
    Token.objects.create(user=user)
    return user


@pytest.mark.django_db
class TestTaxonomyView:

    def test_returns_categories(self, api_factory):
        request = api_factory.get("/api/taxonomy/")
        view = TaxonomyView.as_view()
        response = view(request)
        assert response.status_code == 200
        assert "categories" in response.data
        assert response.data["total_categories"] >= 20

    def test_no_auth_required(self, api_factory):
        request = api_factory.get("/api/taxonomy/")
        view = TaxonomyView.as_view()
        response = view(request)
        assert response.status_code == 200


@pytest.mark.django_db
class TestRegisterView:

    def test_register_success(self, api_factory):
        request = api_factory.post("/api/auth/register/", {
            "username": "newuser",
            "password": "securepass123",
            "email": "new@example.com",
        })
        view = RegisterView.as_view()
        response = view(request)
        assert response.status_code == 201
        assert "token" in response.data
        assert response.data["username"] == "newuser"

    def test_register_missing_fields(self, api_factory):
        request = api_factory.post("/api/auth/register/", {"username": "nopass"})
        view = RegisterView.as_view()
        response = view(request)
        assert response.status_code == 400

    def test_register_duplicate_username(self, api_factory, user):
        request = api_factory.post("/api/auth/register/", {
            "username": "testuser",
            "password": "anotherpass",
        })
        view = RegisterView.as_view()
        response = view(request)
        assert response.status_code == 409


@pytest.mark.django_db
class TestLoginView:

    def test_login_success(self, api_factory, user):
        request = api_factory.post("/api/auth/login/", {
            "username": "testuser",
            "password": "testpass123",
        })
        view = LoginView.as_view()
        response = view(request)
        assert response.status_code == 200
        assert "token" in response.data

    def test_login_bad_credentials(self, api_factory):
        request = api_factory.post("/api/auth/login/", {
            "username": "nobody",
            "password": "wrong",
        })
        view = LoginView.as_view()
        response = view(request)
        assert response.status_code == 401


@pytest.mark.django_db
class TestDetectView:

    @patch("sign_detection.views.current_app")
    def test_submit_detection_job(self, mock_celery, api_factory, user):
        mock_task = MagicMock()
        mock_task.id = "fake-task-id-123"
        mock_celery.send_task.return_value = mock_task

        request = api_factory.post("/api/detect/", {
            "image_urls": ["http://example.com/sign1.jpg"],
        }, format="json")
        force_authenticate(request, user=user)

        view = DetectView.as_view()
        response = view(request)
        assert response.status_code == 202
        assert response.data["job_id"] == "fake-task-id-123"
        assert response.data["job_type"] == "image"

    def test_detect_requires_auth(self, api_factory):
        request = api_factory.post("/api/detect/", {
            "image_urls": ["http://example.com/sign1.jpg"],
        }, format="json")
        view = DetectView.as_view()
        response = view(request)
        assert response.status_code in (401, 403)

    @patch("sign_detection.views.current_app")
    def test_detect_validates_urls(self, mock_celery, api_factory, user):
        request = api_factory.post("/api/detect/", {
            "image_urls": [],
        }, format="json")
        force_authenticate(request, user=user)
        view = DetectView.as_view()
        response = view(request)
        assert response.status_code == 400


@pytest.mark.django_db
class TestVideoDetectView:

    @patch("sign_detection.views.current_app")
    def test_submit_video_job(self, mock_celery, api_factory, user, tmp_path):
        mock_task = MagicMock()
        mock_task.id = "fake-video-task-456"
        mock_celery.send_task.return_value = mock_task

        video_content = b"\x00" * 1024  # Dummy video bytes
        video_file = SimpleUploadedFile(
            "test_video.mp4",
            video_content,
            content_type="video/mp4",
        )

        request = api_factory.post(
            "/api/detect/video/",
            {"video": video_file, "interval_s": 1.0},
            format="multipart",
        )
        force_authenticate(request, user=user)

        view = VideoDetectView.as_view()
        response = view(request)
        assert response.status_code == 202
        assert response.data["job_id"] == "fake-video-task-456"
        assert response.data["job_type"] == "video"

    def test_video_detect_requires_auth(self, api_factory):
        video_file = SimpleUploadedFile(
            "test.mp4", b"\x00" * 100, content_type="video/mp4",
        )
        request = api_factory.post(
            "/api/detect/video/",
            {"video": video_file},
            format="multipart",
        )
        view = VideoDetectView.as_view()
        response = view(request)
        assert response.status_code in (401, 403)

    def test_rejects_invalid_extension(self, api_factory, user):
        bad_file = SimpleUploadedFile(
            "test.txt", b"not a video", content_type="text/plain",
        )
        request = api_factory.post(
            "/api/detect/video/",
            {"video": bad_file},
            format="multipart",
        )
        force_authenticate(request, user=user)
        view = VideoDetectView.as_view()
        response = view(request)
        assert response.status_code == 400

    def test_rejects_missing_video(self, api_factory, user):
        request = api_factory.post(
            "/api/detect/video/",
            {},
            format="multipart",
        )
        force_authenticate(request, user=user)
        view = VideoDetectView.as_view()
        response = view(request)
        assert response.status_code == 400
