# apps/candidates/services.py

import logging

logger = logging.getLogger(__name__)


def resolve_candidate_profile(candidate) -> dict:
    """
    Normalizes Candidate/ResumeProfile into the dict shape the AI engine
    prompts expect. experience_years and role_level are NOT reliably
    available from the bulkresume parser and are not used to calibrate
    question difficulty — interviews are JD + resume-content driven only,
    not seniority-calibrated. Both keys are kept in the returned dict
    purely so existing prompt-building functions don't need signature
    changes, but their values will almost always be "not specified".
    """
    profile = getattr(candidate, "resume_profile", None)

    if profile is None:
        logger.warning(
            "Candidate %s has no linked ResumeProfile — using minimal "
            "fallback profile.",
            candidate.id,
        )
        return {
            "name": getattr(candidate, "name", "the candidate"),
            "skills": [],
            "experience_years": "not specified",
            "recent_projects": [],
            "role_level": "not specified",
        }

    return {
        "name": getattr(candidate, "name", "the candidate"),
        "skills": getattr(profile, "skills", []) or [],
        "experience_years": getattr(profile, "experience_years", None) or "not specified",
        "recent_projects": getattr(profile, "recent_projects", []) or [],
        "role_level": getattr(profile, "role_level", "") or "not specified",
    }