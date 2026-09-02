# config/wsgi.py

"""
WSGI config for the ai_interview_screening project.

Exposes the WSGI callable as a module-level variable named `application`.

Note: this project's live interview flow (WebSocket-based) runs through
config/asgi.py, not this file — WSGI here only serves plain HTTP
requests: Django admin, the REST endpoints in candidates/views.py and
interviews/views.py, and the recruiter dashboard pages. If you deploy
with an ASGI server (Daphne/uvicorn) handling everything, as recommended
given you need WebSocket support, this file may end up unused in
production — but Django's tooling (and some hosting platforms) still
expect it to exist, so it's kept for compatibility.
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

application = get_wsgi_application()