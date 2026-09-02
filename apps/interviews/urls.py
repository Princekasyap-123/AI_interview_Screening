# apps/interviews/urls.py

from django.urls import path

from apps.interviews.views import (
    InterviewRoomView,               # add this import
    InterviewSessionCreateView,
    InterviewSessionDetailView,
    InterviewSessionJoinInfoView,
    InterviewSessionListView,
)

app_name = "interviews"

urlpatterns = [
    path("", InterviewSessionListView.as_view(), name="list"),
    path("create/", InterviewSessionCreateView.as_view(), name="create"),
    path("<uuid:session_id>/", InterviewSessionDetailView.as_view(), name="detail"),
    path("<uuid:session_id>/join/", InterviewSessionJoinInfoView.as_view(), name="join-info"),
    path("<uuid:session_id>/room/", InterviewRoomView.as_view(), name="room"),   # add this line
]