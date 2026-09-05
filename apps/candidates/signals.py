# apps/candidates/signals.py

import logging

from django.db.models.signals import post_save
from django.dispatch import receiver

from apps.candidates.models import Candidate

logger = logging.getLogger(__name__)


@receiver(post_save, sender=Candidate)
def trigger_resume_parsing_on_upload(sender, instance, created, **kwargs):
    """
    Automatically triggers resume parsing whenever a Candidate is saved
    WITH a resume_file attached — covers every creation path (admin,
    REST API via CandidateCreateView, shell, future imports) in one
    place, rather than relying on each individual call site to remember
    to trigger it manually. This is the ONLY place that should call
    submit_and_parse_resume.delay() — do not also call it from views,
    or a resume gets submitted to the bulkresume service twice per
    upload.

    Only fires when resume_file is actually present, and only on the
    save that has it (so editing other fields later doesn't re-trigger
    parsing every time — see the has_profile check below).
    """
    if not instance.resume_file:
        return

    # Avoid re-parsing every time the candidate is saved for unrelated
    # edits (e.g. admin changing "applied_role") once a profile already
    # exists — only parse if there's no profile yet.
    if hasattr(instance, "resume_profile"):
        return

    # Mark as pending via a signal-safe queryset .update() — this
    # bypasses Candidate.save(), so it does NOT re-trigger this same
    # post_save signal (which would otherwise re-submit the resume for
    # parsing in an infinite loop).
    Candidate.objects.filter(id=instance.id).update(
        parsing_status=Candidate.ParsingStatus.PENDING
    )

    from apps.candidates.tasks import submit_and_parse_resume
    logger.info("Triggering resume parsing for candidate %s via signal", instance.id)
    submit_and_parse_resume.delay(str(instance.id))