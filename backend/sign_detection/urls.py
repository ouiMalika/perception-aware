"""URL routing for the sign detection API."""

from django.urls import path

from . import views

urlpatterns = [
    # Detection
    path("detect/", views.DetectView.as_view(), name="detect"),
    path("detect/video/", views.VideoDetectView.as_view(), name="detect-video"),
    path("jobs/", views.JobListView.as_view(), name="job-list"),
    path("jobs/<str:job_id>/", views.JobStatusView.as_view(), name="job-status"),
    path("signs/", views.DetectedSignListView.as_view(), name="sign-list"),
    path("signs/tracked/", views.TrackedSignListView.as_view(), name="tracked-sign-list"),

    # Reference
    path("taxonomy/", views.TaxonomyView.as_view(), name="taxonomy"),

    # Auth
    path("auth/register/", views.RegisterView.as_view(), name="register"),
    path("auth/login/", views.LoginView.as_view(), name="login"),
]
