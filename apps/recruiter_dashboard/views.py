# apps/recruiter_dashboard/views.py — merged version, keep this

from django.contrib import messages
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

        session = InterviewSession.objects.create(
            candidate=candidate,
            job_description_text=job_description_text,
            max_questions=max_questions,
        )

        messages.success(request, f"Interview session created for {candidate.name}.")
        return redirect(f"/dashboard/?session_id={session.id}")