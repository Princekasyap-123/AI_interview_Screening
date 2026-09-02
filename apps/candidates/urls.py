# apps/candidates/urls.py

from django.urls import path

from apps.candidates.views import (
    CandidateCreateView,
    CandidateDetailView,
    CandidateListView,
)

app_name = "candidates"

urlpatterns = [
    path("", CandidateListView.as_view(), name="list"),
    path("create/", CandidateCreateView.as_view(), name="create"),
    path("<uuid:candidate_id>/", CandidateDetailView.as_view(), name="detail"),
]