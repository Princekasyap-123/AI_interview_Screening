# config/urls.py

from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),

    path("api/candidates/", include("apps.candidates.urls")),
    path("api/interviews/", include("apps.interviews.urls")),

    # recruiter_dashboard serves HTML templates (not a REST API), so it's
    # mounted at a plain path rather than under /api/ — not yet written,
    # this include will fail until recruiter_dashboard/urls.py exists.
    path("dashboard/", include("apps.recruiter_dashboard.urls")),

    # frontend/templates (interview_room.html, dashboard.html) are served
    # via Django's TEMPLATES DIRS config already, not routed here directly
    # — views that render them live in whichever app owns that page
    # (interviews for the room, recruiter_dashboard for the dashboard).
]

if settings.DEBUG:
    # Serve uploaded resumes/media locally in dev only — in production
    # this should be served by nginx/Hostinger directly, not Django.
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)