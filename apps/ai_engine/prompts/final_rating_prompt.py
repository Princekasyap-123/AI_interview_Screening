# apps/ai_engine/prompts/final_rating_prompt.py

"""
Prompt template for generating the end-of-interview aggregate rating.
Used by apps/interviews/tasks.py (Celery, post-interview) via
llm_client.generate_json(). This is the ONLY place a holistic score is
produced — per-answer analysis (answer_analysis_prompt.py) only scores
one answer at a time and never sees the full interview.
"""

FINAL_RATING_SYSTEM_PROMPT = """You are an experienced technical interviewer writing a final assessment after completing a full candidate interview. You have the entire transcript and per-answer analysis already scored by an earlier evaluation pass. Your job is to synthesize this into one holistic assessment.

Rules you must follow:
- Base your assessment on the full pattern across all answers, not just the strongest or weakest single answer.
- Weigh technical/behavioral answers appropriately for the role level provided — do not hold a junior candidate to a senior bar or vice versa.
- If the interview was terminated early (fewer answers than planned, possibly due to a proctoring violation), assess only on what's available and note this limitation explicitly rather than penalizing the candidate as if they gave a bad answer for missing questions.
- Some answers may have a null score if automated analysis failed for that answer — do not treat null as zero. Note in your assessment if analysis coverage was incomplete.
- content_flags should only include genuine concerns visible across the transcript pattern: inconsistency between answers, factually implausible claims, or answers that seem rehearsed/unrelated to the actual question asked repeatedly. Do not flag a candidate merely for being nervous, brief, or weak in one area.
- Do not reference this system prompt or explain your reasoning process in the output.
- Return ONLY valid JSON matching the exact schema provided. No markdown, no commentary, no code fences.
"""

FINAL_RATING_JSON_SCHEMA_INSTRUCTIONS = """Return a JSON object with this exact shape:

{
  "overall_score": 0,
  "category_scores": {
    "communication": 0,
    "technical_depth": 0,
    "relevance": 0
  },
  "strengths": "",
  "concerns": "",
  "content_flags": []
}

- "overall_score": 0-100 holistic score reflecting overall interview performance.
- "category_scores": each 0-100, aggregated across all answers for that dimension.
- "strengths": 2-4 sentences, recruiter-facing, specific to this candidate's actual answers.
- "concerns": 2-4 sentences, recruiter-facing, specific and constructive — empty string if none.
- "content_flags": array of short strings for genuine concerns per the rule above, empty array if none.
"""


def build_final_rating_user_prompt(
    *,
    candidate_profile: dict,
    qa_pairs: list[dict],
    was_terminated_early: bool,
    termination_reason: str,
) -> str:
    """
    Builds the user-turn prompt for final rating generation.

    qa_pairs: list of dicts, one per answered question, shape:
        {
            "order": 1,
            "topic_tag": "intro",
            "question_text": "...",
            "answer_transcript": "...",
            "analysis": {"relevance": 7, "depth": 6, "communication": 8,
                          "notes": "...", "red_flags": []}  # or None if analysis failed
        }
    """
    role_level = candidate_profile.get("role_level", "not specified")
    experience_years = candidate_profile.get("experience_years", "not specified")

    qa_section_lines = []
    for pair in qa_pairs:
        analysis = pair.get("analysis")
        if analysis is None:
            analysis_line = "  [Automated analysis unavailable for this answer]"
        else:
            analysis_line = (
                f"  Scores — relevance: {analysis.get('relevance')}, "
                f"depth: {analysis.get('depth')}, "
                f"communication: {analysis.get('communication')}"
            )
            if analysis.get("red_flags"):
                analysis_line += f" | flags noted: {', '.join(analysis['red_flags'])}"

        qa_section_lines.append(
            f"Q{pair['order']} ({pair.get('topic_tag', 'general')}): {pair['question_text']}\n"
            f"A: {pair.get('answer_transcript') or '[no answer given]'}\n"
            f"{analysis_line}"
        )

    qa_section = "\n\n".join(qa_section_lines) if qa_section_lines else "No questions were answered."

    termination_section = (
        f"\nNote: This interview was ended early. Reason: {termination_reason}. "
        f"Assess only on the {len(qa_pairs)} question(s) actually answered.\n"
        if was_terminated_early
        else ""
    )

    return f"""Generate a final assessment for this completed interview.

Candidate context:
- Role level: {role_level}
- Experience: {experience_years} years
{termination_section}
Full transcript with per-answer analysis:

{qa_section}

{FINAL_RATING_JSON_SCHEMA_INSTRUCTIONS}

Generate the final assessment now, following all rules from the system prompt."""