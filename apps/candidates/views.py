# apps/candidates/views.py

import logging

from rest_framework import status
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.candidates.models import Candidate
from apps.candidates.serializers import (
    CandidateCreateSerializer,
    CandidateSerializer,
)

logger = logging.getLogger(__name__)


class CandidateCreateView(APIView):
    """
    POST /api/candidates/  — create a candidate + upload resume.

    Permission note: this is set to AllowAny deliberately, since this
    endpoint is likely called by your recruiter-side tooling (or a
    candidate-facing application form) rather than requiring a logged-in
    Django user. If this should actually be recruiter-only (staff auth
    required), change permission_classes to [IsAuthenticated] — flagging
    this explicitly since REST_FRAMEWORK in settings.py defaults every
    OTHER view to IsAuthenticated, and this one deliberately opts out.

    Resume parsing trigger note: this view does NOT call
    submit_and_parse_resume itself. apps/candidates/signals.py's
    post_save handler on Candidate already fires for every creation
    path — admin, this endpoint, shell, future imports — in one place.
    Calling it again here would submit the same resume to the
    bulkresume service twice per upload.
    """

    permission_classes = [AllowAny]
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        serializer = CandidateCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        candidate = serializer.save()

        logger.info("Created candidate %s (%s)", candidate.id, candidate.email)

        return Response(
            CandidateSerializer(candidate).data,
            status=status.HTTP_201_CREATED,
        )


class CandidateDetailView(APIView):
    """
    GET /api/candidates/<uuid:candidate_id>/  — recruiter-facing view of
    a candidate including parsed profile, if available.

    Recruiter-only by default (inherits IsAuthenticated from
    REST_FRAMEWORK settings) — this exposes profile data that shouldn't
    be candidate-accessible.
    """

    permission_classes = [IsAuthenticated]

    def get(self, request, candidate_id):
        candidate = Candidate.objects.select_related("resume_profile").filter(
            id=candidate_id
        ).first()

        if candidate is None:
            return Response(
                {"detail": "Candidate not found."}, status=status.HTTP_404_NOT_FOUND
            )

        return Response(CandidateSerializer(candidate).data)


class CandidateListView(APIView):
    """
    GET /api/candidates/  — recruiter-facing list, paginated per
    REST_FRAMEWORK settings (PAGE_SIZE=20).
    """

    permission_classes = [IsAuthenticated]

    def get(self, request):
        from rest_framework.pagination import PageNumberPagination

        candidates = Candidate.objects.select_related("resume_profile").order_by(
            "-created_at"
        )

        paginator = PageNumberPagination()
        page = paginator.paginate_queryset(candidates, request)
        serializer = CandidateSerializer(page, many=True)
        return paginator.get_paginated_response(serializer.data)