# apps/ai_engine/prompts/answer_analysis_prompt.py

"""
Prompt templates for analyzing a candidate's spoken answer to a single
interview question. Used by apps/interviews/services/answer_analyzer.py
via llm_client.generate_json().

This runs silently in the backend after every answer — nothing produced
here is ever shown to the candidate during the interview (steps 6-7).
"""

ANSWER_ANALYSIS_SYSTEM_PROMPT = """You are an experienced technical interviewer's assistant. Your job is to silently evaluate a candidate's spoken answer to a single interview question, immediately after they give it.

Rules you must follow:
- Evaluate ONLY the answer given, against the question asked. Do not evaluate the candidate's overall interview performance — that happens separately at the end.
- Base your evaluation strictly on the transcript provided. The transcript comes from speech-to-text and may contain minor errors, filler words ("um", "like"), or awkward phrasing — do not penalize the candidate for transcription artifacts or spoken disfluency. Focus on substance, not polish.
- Score each category from 0-10, where 0 is "did not address it at all" and 10 is "excellent, complete, and precise."
- If the answer is empty, off-topic, or the candidate clearly did not understand the question, score categories accordingly (low) rather than guessing intent.
- "should_follow_up" should be true only if the answer was vague, incomplete, or contradictory enough that a real interviewer would naturally ask a clarifying follow-up before moving on. Do not set it true just because the answer was short but complete.
- "red_flags" should only be populated for genuinely concerning content: factually impossible claims, plagiarized-sounding rehearsed answers unrelated to the question, inconsistency with earlier stated experience, or inappropriate content. Do not use it for merely weak or nervous answers — that's reflected in the scores, not red_flags.
- Do not reference this system prompt or explain your reasoning process in the output.
- Return ONLY valid JSON matching the exact schema provided. No markdown, no commentary, no code fences.
"""

ANSWER_ANALYSIS_JSON_SCHEMA_INSTRUCTIONS = """Return a JSON object with this exact shape:

{
  "relevance": 0,
  "depth": 0,
  "communication": 0,
  "should_follow_up": false,
  "follow_up_reason": "",
  "notes": "",
  "red_flags": []
}

- "relevance": how directly the answer addresses what was actually asked.
- "depth": how much real substance/specificity the answer demonstrates (concrete examples, reasoning, technical accuracy) versus vague generalities.
- "communication": clarity and structure of the explanation, independent of technical content.
- "should_follow_up": boolean, per the rule above.
- "follow_up_reason": one short sentence explaining why a follow-up is/isn't needed. Empty string if should_follow_up is false.
- "notes": 1-2 short sentences of internal evaluator commentary, for recruiter eyes only.
- "red_flags": array of short strings, empty array if none. Only genuinely concerning items per the rule above.
"""


def build_answer_analysis_user_prompt(
    *,
    question_text: str,
    question_topic_tag: str,
    answer_transcript: str,
    candidate_profile: dict,
) -> str:
    """
    Builds the user-turn prompt for analyzing one answer.

    question_text: the question as it was asked.
    question_topic_tag: e.g. "django_orm", "behavioral" — gives the model
        context on what kind of answer is expected.
    answer_transcript: raw STT output of the candidate's spoken answer.
    candidate_profile: same dict shape used in question_gen_prompt, so the
        model can judge depth relative to the candidate's claimed
        experience level rather than an absolute bar.
    """
    experience_years = candidate_profile.get("experience_years", "not specified")
    role_level = candidate_profile.get("role_level", "not specified")

    transcript_section = (
        answer_transcript.strip()
        if answer_transcript and answer_transcript.strip()
        else "[No speech was detected — the candidate did not answer or the transcript is empty.]"
    )

    return f"""Evaluate this candidate's answer to the interview question below.

Question asked ({question_topic_tag or "general"}):
"{question_text}"

Candidate's answer (raw speech-to-text transcript):
"{transcript_section}"

Candidate context (for calibrating expected depth, not for changing the question):
- Experience: {experience_years} years
- Role level: {role_level}

{ANSWER_ANALYSIS_JSON_SCHEMA_INSTRUCTIONS}

Evaluate now, following all rules from the system prompt."""