# apps/recruiter_dashboard/views.py — merged version, keep this

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.exceptions import ValidationError
from django.db.models import Avg, Count, Q
from django.http import Http404
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils.decorators import method_decorator
from django.views import View

from apps.candidates.models import Candidate
from apps.interviews.models import InterviewSession
from apps.interviews.services.language import initial_language


@method_decorator(login_required, name="dispatch")
class DashboardView(View):
    """
    Recruiter-facing session list + detail page. Requires Django login
    (login_required) — internal tooling, never candidate-facing.

    Also handles POST for creating a new InterviewSession directly from
    the dashboard (candidate + JD text + max_questions form).
    """

    def get(self, request):
        sessions = (
            InterviewSession.objects.select_related("candidate", "final_rating")
            .prefetch_related("questions__answer", "proctoring_session")
            .order_by("-created_at")
        )

        # Headline numbers always describe ALL sessions, not the filtered list.
        stats = InterviewSession.objects.aggregate(
            total=Count("id"),
            completed=Count("id", filter=Q(status=InterviewSession.Status.COMPLETED)),
            in_progress=Count("id", filter=Q(status=InterviewSession.Status.IN_PROGRESS)),
            pending=Count("id", filter=Q(status=InterviewSession.Status.PENDING)),
            terminated=Count("id", filter=Q(status=InterviewSession.Status.TERMINATED)),
            avg_score=Avg("final_rating__overall_score"),
            flagged=Count(
                "id", filter=Q(proctoring_session__countable_violation_count__gt=0)
            ),
        )

        status_filter = request.GET.get("status")
        if status_filter not in InterviewSession.Status.values:
            status_filter = ""
        if status_filter:
            sessions = sessions.filter(status=status_filter)

        selected_session = None
        interview_link = None
        selected_id = request.GET.get("session_id")
        if selected_id:
            try:
                selected_session = get_object_or_404(
                    InterviewSession.objects.select_related(
                        "candidate", "final_rating"
                    ).prefetch_related(
                        "questions__answer",
                        "conversation",
                        "proctoring_session__flags",
                    ),
                    id=selected_id,
                )
            except (ValidationError, ValueError):
                # a malformed ?session_id= used to be a 500
                raise Http404("Unknown interview session.")
            # Build the full candidate-facing interview URL via reverse().
            # This is the top-level route defined directly in
            # config/urls.py (NOT the one under apps/interviews/urls.py,
            # which is a separate /api/interviews/<id>/room/<token>/
            # path also pointing at InterviewRoomView but is not the
            # link recruiters should hand to candidates):
            #   path("interview/<uuid:session_id>/<str:access_token>/",
            #        InterviewRoomView.as_view(), name="interview-room")
            # It has no app namespace since it's declared directly in
            # urlpatterns, not inside an include().
            self._ensure_rating(selected_session)
            interview_link = request.build_absolute_uri(
                reverse(
                    "interview-room",
                    kwargs={
                        "session_id": selected_session.id,
                        "access_token": selected_session.access_token,
                    },
                )
            )

        # Stable serial number per session (oldest = 1), so a card keeps its
        # number whatever filter is applied.
        serials = {
            sid: n for n, sid in enumerate(
                InterviewSession.objects.order_by("created_at").values_list("id", flat=True), start=1
            )
        }
        sessions = list(sessions)
        for s in sessions:
            s.serial = serials.get(s.id, 0)
        if selected_session is not None:
            selected_session.serial = serials.get(selected_session.id, 0)

        return render(
            request,
            "dashboard.html",
            {
                "sessions": sessions,
                "stats": stats,
                "language_choices": InterviewSession.LanguageMode.choices,
                "candidates": Candidate.objects.order_by("-created_at"),
                "selected_session": selected_session,
                "interview_link": interview_link,
                "status_filter": status_filter or "",
                "status_choices": InterviewSession.Status.choices,
            },
        )

    @staticmethod
    def _ensure_rating(session):
        """An ended interview with no rating (no Celery worker was running, or the job failed) gets one now."""
        from django.utils import timezone

        if session.status not in (InterviewSession.Status.COMPLETED, InterviewSession.Status.TERMINATED):
            return
        if hasattr(session, "final_rating") and getattr(session, "final_rating", None) is not None:
            return
        if session.termination_reason == InterviewSession.TerminationReason.RESCHEDULED and not session.questions_asked_count:
            return                                   # rescheduled before answering anything: nothing to rate
        if session.ended_at and (timezone.now() - session.ended_at).total_seconds() < 60:
            return                                   # the normal path is still working on it
        from apps.interviews.services.final_rating import ensure_rating_in_background
        ensure_rating_in_background(session.id, delay_seconds=0)

    def post(self, request):
        candidate_id = request.POST.get("candidate_id")
        job_description_text = request.POST.get("job_description_text", "").strip()
        max_questions_raw = request.POST.get("max_questions", "5")

        if not candidate_id:
            messages.error(request, "Please select a candidate.")
            return redirect("recruiter_dashboard:index")

        if not job_description_text:
            messages.error(request, "Job description cannot be empty.")
            return redirect("recruiter_dashboard:index")

        candidate = Candidate.objects.filter(id=candidate_id).first()
        if candidate is None:
            messages.error(request, "Selected candidate no longer exists.")
            return redirect("recruiter_dashboard:index")

        try:
            max_questions = int(max_questions_raw)
            if max_questions < 1:
                raise ValueError
        except (TypeError, ValueError):
            max_questions = 5

        job_role_type = request.POST.get("job_role_type", "").strip()
        if job_role_type not in InterviewSession.JobRoleType.values:
            job_role_type = InterviewSession.JobRoleType.WHITE_COLLAR

        language_mode = request.POST.get("language_mode", "").strip()
        if language_mode not in InterviewSession.LanguageMode.values:
            language_mode = InterviewSession.LanguageMode.ENGLISH

        session = InterviewSession.objects.create(
            language_mode=language_mode,
            current_language=initial_language(language_mode),
            candidate=candidate,
            job_description_text=job_description_text,
            max_questions=max_questions,
            job_role_type=job_role_type,
        )
        messages.success(request, f"Interview session created for {candidate.name}.")
        return redirect(f"/dashboard/?session_id={session.id}")