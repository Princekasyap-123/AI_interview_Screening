# apps/proctoring/routing.py

from django.urls import re_path

from apps.proctoring.consumers import ProctoringConsumer

websocket_urlpatterns = [
    re_path(
        r"^ws/proctoring/(?P<session_id>[0-9a-f-]{36})/$",
        ProctoringConsumer.as_asgi(),
    ),
]