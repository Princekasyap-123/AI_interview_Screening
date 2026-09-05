# apps/interviews/urls.py

from django.urls import path

from apps.interviews.views import (
    InterviewRoomView,
    InterviewSessionCreateView,
    InterviewSessionDetailView,
    InterviewSessionJoinInfoView,
    InterviewSessionListView,
    recruiter_dashboard,
)

app_name = "interviews"

urlpatterns = [
    path("", InterviewSessionListView.as_view(), name="list"),
    path("create/", InterviewSessionCreateView.as_view(), name="create"),
    path("<uuid:session_id>/", InterviewSessionDetailView.as_view(), name="detail"),
    path("<uuid:session_id>/join/", InterviewSessionJoinInfoView.as_view(), name="join-info"),
    path(
        "<uuid:session_id>/room/<str:access_token>/",
        InterviewRoomView.as_view(),
        name="room",
    ),
    path("dashboard/", recruiter_dashboard, name="dashboard"),
]