# config/asgi.py

import os

from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.security.websocket import AllowedHostsOriginValidator
from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

# get_asgi_application() must be called before importing anything that
# touches Django models/apps (like our routing modules below) — this
# populates the app registry first. Importing routing.py too early is a
# common source of "Apps aren't loaded yet" errors.
django_asgi_app = get_asgi_application()

from apps.interviews.routing import websocket_urlpatterns as interview_ws_patterns  # noqa: E402
from apps.proctoring.routing import websocket_urlpatterns as proctoring_ws_patterns  # noqa: E402

websocket_urlpatterns = interview_ws_patterns + proctoring_ws_patterns

application = ProtocolTypeRouter({
    "http": django_asgi_app,
    "websocket": AllowedHostsOriginValidator(
        AuthMiddlewareStack(
            URLRouter(websocket_urlpatterns)
        )
    ),
})