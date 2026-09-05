# apps/interviews/views.py

import logging

from django.conf import settings
from rest_framework import status
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, render
from django.views import View

from apps.candidates.models import Candidate
from apps.interviews.models import Answer, FinalRating, InterviewSession, Question
from apps.interviews.serializers import (
    InterviewSessionCreateSerializer,
    InterviewSessionDetailSerializer,
    InterviewSessionSummarySerializer,
)

logger = logging.getLogger(__name__)


class InterviewRoomView(View):
    """
    GET /api/interviews/<uuid:session_id>/room/<str:access_token>/ —
    candidate-facing HTML page. Renders interview_room.html.

    access_token is required and validated against the session (not
    just session_id alone) so a candidate link can't be guessed from a
    sequential/leaked UUID — matches the trust model already used by
    InterviewSessionJoinInfoView, just extended with the extra token.
    """

    def get(self, request, session_id, access_token):
        session = get_object_or_404(
            InterviewSession, id=session_id, access_token=access_token
        )
        return render(request, "interview_room.html", {"session_id": str(session.id)})


class InterviewSessionCreateView(APIView):
    """
    POST /api/interviews/  — recruiter creates a new interview session
    for an already-shortlisted candidate, providing the JD text and
    (optionally) max_questions. This does NOT start the interview itself
    — it just creates the PENDING session row. The actual question
    generation + intro turn happens when the candidate connects to
    InterviewConsumer and sends {"type": "join"}.

    Recruiter-only (IsAuthenticated, per REST_FRAMEWORK default) — a
    candidate should never be able to create their own session with an
    arbitrary max_questions or JD.

    Response now also includes `interview_link` — the full URL the
    recruiter can copy/send to the candidate, built from access_token
    (auto-generated on the model's save()).
    """

    permission_classes = [IsAuthenticated]

    def post(self, request):
        serializer = InterviewSessionCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        candidate_id = serializer.validated_data["candidate_id"]
        candidate = Candidate.objects.filter(id=candidate_id).first()
        if candidate is None:
            return Response(
                {"detail": "Candidate not found."}, status=status.HTTP_404_NOT_FOUND
            )

        session = InterviewSession.objects.create(
            candidate=candidate,
            job_description_text=serializer.validated_data.get("job_description_text", ""),
            max_questions=serializer.validated_data.get(
                "max_questions", settings.INTERVIEW_DEFAULT_MAX_QUESTIONS
            ),
        )

        logger.info(
            "Created interview session %s for candidate %s", session.id, candidate.id
        )

        data = InterviewSessionDetailSerializer(session).data
        base_url = getattr(settings, "FRONTEND_BASE_URL", request.build_absolute_uri("/").rstrip("/"))
        data["interview_link"] = f"{base_url}/interview/{session.id}/{session.access_token}/"

        return Response(data, status=status.HTTP_201_CREATED)


class InterviewSessionJoinInfoView(APIView):
    """
    GET /api/interviews/<uuid:session_id>/join/  — candidate-facing.
    Returns just enough info for the frontend to render the interview
    room and open the two WebSocket connections (interview + proctoring)
    — NOT the questions themselves (those only come via the WebSocket
    "join" flow) and definitely not any analysis/scoring data.

    AllowAny deliberately: the candidate reaches this via a link (e.g.
    emailed to them), not a logged-in Django session. Session UUIDs are
    unguessable, which is the access control here — same trust model
    already used for the WebSocket routes.
    """

    permission_classes = [AllowAny]

    def get(self, request, session_id):
        session = InterviewSession.objects.filter(id=session_id).first()
        if session is None:
            return Response(
                {"detail": "Interview session not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        if session.status == InterviewSession.Status.COMPLETED:
            return Response(
                {"detail": "This interview has already been completed."},
                status=status.HTTP_410_GONE,
            )

        if session.status == InterviewSession.Status.TERMINATED:
            return Response(
                {"detail": "This interview session has ended."},
                status=status.HTTP_410_GONE,
            )

        return Response({
            "session_id": str(session.id),
            "candidate_name": session.candidate.name,
            "status": session.status,
            # Explicitly NOT included: max_questions, questions_asked_count,
            # job_description_text, any Question/Answer/analysis data —
            # candidate-facing endpoint stays minimal per steps 6-7.
        })


class InterviewSessionDetailView(APIView):
    """
    GET /api/interviews/<uuid:session_id>/  — recruiter-facing full view:
    session metadata, all questions asked, transcripts, and FinalRating
    if available. This is the raw data source the recruiter_dashboard
    app will build views on top of.
    """

    permission_classes = [IsAuthenticated]

    def get(self, request, session_id):
        session = (
            InterviewSession.objects.select_related("candidate", "final_rating")
            .prefetch_related("questions__answer")
            .filter(id=session_id)
            .first()
        )
        if session is None:
            return Response(
                {"detail": "Interview session not found."},
                status=status.HTTP_404_NOT_FOUND,
            )

        return Response(InterviewSessionDetailSerializer(session).data)


class InterviewSessionListView(APIView):
    """
    GET /api/interviews/?candidate_id=<uuid>  — recruiter-facing list,
    optionally filtered by candidate. Summary shape only (no full
    transcript) — use InterviewSessionDetailView for the full record.
    """

    permission_classes = [IsAuthenticated]

    def get(self, request):
        from rest_framework.pagination import PageNumberPagination

        sessions = InterviewSession.objects.select_related(
            "candidate", "final_rating"
        ).order_by("-created_at")

        candidate_id = request.query_params.get("candidate_id")
        if candidate_id:
            sessions = sessions.filter(candidate_id=candidate_id)

        status_filter = request.query_params.get("status")
        if status_filter:
            sessions = sessions.filter(status=status_filter)

        paginator = PageNumberPagination()
        page = paginator.paginate_queryset(sessions, request)
        serializer = InterviewSessionSummarySerializer(page, many=True)
        return paginator.get_paginated_response(serializer.data)


@login_required
def recruiter_dashboard(request):
    """
    GET /dashboard/  — plain HTML recruiter dashboard (the template you
    already have, interviews/dashboard.html). This is a separate,
    session-authenticated (Django login) surface from the /api/
    endpoints above, which are DRF/token or Django-session-authenticated
    APIViews. It renders the same InterviewSession data as
    InterviewSessionListView/DetailView but as server-rendered HTML for
    a human recruiter clicking around, not a JSON API consumer.

    Note: session CREATION on this page happens via the dashboard's
    <script> fetch()'ing InterviewSessionCreateView directly (see
    dashboard.html) rather than duplicating creation logic here — one
    source of truth for how a session gets created.
    """
    status_filter = request.GET.get("status", "")
    sessions = InterviewSession.objects.select_related(
        "candidate", "final_rating", "proctoring_session"
    ).order_by("-created_at")
    if status_filter:
        sessions = sessions.filter(status=status_filter)

    selected_session = None
    session_id = request.GET.get("session_id")
    if session_id:
        selected_session = (
            InterviewSession.objects.select_related(
                "candidate", "final_rating", "proctoring_session"
            )
            .prefetch_related("questions__answer", "proctoring_session__flags")
            .filter(id=session_id)
            .first()
        )

    interview_link = None
    if selected_session:
        base_url = getattr(
            settings, "FRONTEND_BASE_URL", request.build_absolute_uri("/").rstrip("/")
        )
        interview_link = (
            f"{base_url}/interview/{selected_session.id}/{selected_session.access_token}/"
        )

    return render(request, "interviews/dashboard.html", {
        "sessions": sessions,
        "selected_session": selected_session,
        "status_filter": status_filter,
        "status_choices": InterviewSession.Status.choices,
        "candidates": Candidate.objects.all(),
        "interview_link": interview_link,
    })