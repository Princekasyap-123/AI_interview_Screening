# apps/interviews/views.py

import logging

from django.conf import settings
from rest_framework import status
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
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
    GET /api/interviews/<uuid:session_id>/room/  — candidate-facing HTML
    page. Renders interview_room.html, which contains the frontend JS
    that calls InterviewSessionJoinInfoView and opens the WebSocket
    connections. AllowAny-equivalent: plain View has no DRF permission
    classes, session UUID is the access control (same trust model as
    InterviewSessionJoinInfoView).
    """

    def get(self, request, session_id):
        session = get_object_or_404(InterviewSession, id=session_id)
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

        return Response(
            InterviewSessionDetailSerializer(session).data,
            status=status.HTTP_201_CREATED,
        )


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