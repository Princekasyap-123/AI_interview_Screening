# apps/avatar/urls.py

from django.urls import path

from apps.avatar import views

urlpatterns = [
    path("tts/", views.tts_view, name="avatar-tts"),
    path("tts/warmup/", views.tts_warmup_view, name="avatar-tts-warmup"),
    path("lipsync/", views.lipsync_view, name="avatar-lipsync"),
    path("lipsync/status/", views.lipsync_status_view, name="avatar-lipsync-status"),
    path("lipsync/clip/<str:clip_id>.mp4", views.lipsync_clip_view, name="avatar-lipsync-clip"),
]