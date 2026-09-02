# apps/recruiter_dashboard/views.py

from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, redirect, render
from django.utils.decorators import method_decorator
from django.views import View

from apps.candidates.models import Candidate
from apps.interviews.models import InterviewSession


@method_decorator(login_required, name="dispatch")
class DashboardView(View):
    """
    Recruiter-facing session list + detail page. Requires Django login
    (login_required) — this is internal tooling, never candidate-facing,
    consistent with IsAuthenticated on the equivalent REST endpoints in
    interviews/views.py.

    Renders both the list (left panel) and, if session_id is provided
    via query param, the detail panel (right side) — single-page rather
    than a separate detail route, so a recruiter can click through
    several candidates without a full page reload each time (detail
    panel could later be swapped for an AJAX/HTMX partial without
    changing this view's data shape).

    Also handles POST for creating a new InterviewSession directly from
    the dashboard (candidate + JD text + max_questions form) — plain
    Django form POST with CSRF, not routed through the DRF API, since
    this view is already session-authenticated via login_required.
    """

    def get(self, request):
        sessions = (
            InterviewSession.objects.select_related("candidate", "final_rating")
            .prefetch_related("questions__answer", "proctoring_session")
            .order_by("-created_at")
        )

        status_filter = request.GET.get("status")
        if status_filter:
            sessions = sessions.filter(status=status_filter)

        selected_session = None
        selected_id = request.GET.get("session_id")
        if selected_id:
            selected_session = get_object_or_404(
                InterviewSession.objects.select_related(
                    "candidate", "final_rating"
                ).prefetch_related(
                    "questions__answer",
                    "proctoring_session__flags",
                ),
                id=selected_id,
            )

        return render(
            request,
            "dashboard.html",
            {
                "sessions": sessions,
                "candidates": Candidate.objects.order_by("-created_at"),
                "selected_session": selected_session,
                "status_filter": status_filter or "",
                "status_choices": InterviewSession.Status.choices,
            },
        )

    def post(self, request):
        candidate = get_object_or_404(Candidate, id=request.POST.get("candidate_id"))
        session = InterviewSession.objects.create(
            candidate=candidate,
            job_description_text=request.POST.get("job_description_text", ""),
            max_questions=int(request.POST.get("max_questions") or 5),
        )
        return redirect(f"/dashboard/?session_id={session.id}")