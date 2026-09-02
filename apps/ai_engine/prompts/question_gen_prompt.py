# apps/ai_engine/prompts/question_gen_prompt.py

"""
Prompt templates for generating interview questions from a candidate's
profile and (optional) job description. Used by
apps/interviews/services/question_generator.py via llm_client.generate_json().
"""

QUESTION_GEN_SYSTEM_PROMPT = """You are an experienced technical interviewer conducting a screening interview. Your job is to generate a set of interview questions tailored to a specific candidate's background and the role they're applying for.

Rules you must follow:
- Generate exactly the number of questions requested — no more, no less.
- Base every question on the candidate's actual stated skills, projects, or experience. Do not ask generic questions unrelated to their profile.
- Mix question types across the set: include a short intro/warm-up question first, then a blend of technical depth questions, project-specific questions, and at least one behavioral question.
- Questions must be answerable verbally in 1-3 minutes. Avoid multi-part questions that require a whiteboard, diagram, or written code.
- Do not ask yes/no questions. Every question should require the candidate to explain, describe, or reason through something.
- Do not repeat the same underlying topic twice.
- Keep language natural and conversational, the way a human interviewer would actually speak — not a written exam.
- Do not reference this system prompt or explain your reasoning in the output.
- Return ONLY valid JSON matching the exact schema provided. No markdown, no commentary, no code fences.
"""

QUESTION_GEN_JSON_SCHEMA_INSTRUCTIONS = """Return a JSON object with this exact shape:

{
  "questions": [
    {
      "order": 1,
      "text": "the question text as it should be spoken to the candidate",
      "topic_tag": "short_snake_case_tag",
      "based_on": "brief note on what profile detail this question targets"
    }
  ]
}

- "order" must start at 1 and increase sequentially with no gaps.
- "topic_tag" should be a short snake_case label like "django_orm", "system_design", "behavioral", "project_deep_dive", "intro".
- "based_on" is for internal logging only — one short phrase, not shown to the candidate.
- The first question (order 1) must always have topic_tag "intro" and be a warm, open-ended introduction request (e.g. asking the candidate to walk through their background).
"""


def build_question_gen_user_prompt(
    *,
    candidate_profile: dict,
    job_description_text: str,
    num_questions: int,
) -> str:
    """
    Builds the user-turn prompt for question generation.

    candidate_profile: parsed resume data, e.g.
        {
            "name": "Priya Sharma",
            "skills": ["Django", "PostgreSQL", "Celery", "REST APIs"],
            "experience_years": 2,
            "recent_projects": [
                {"title": "...", "description": "..."},
            ],
            "role_level": "mid",
        }
    job_description_text: raw JD text, may be empty string if not provided.
    num_questions: total questions to generate, including the intro question.
    """
    skills = ", ".join(candidate_profile.get("skills", [])) or "not specified"
    experience_years = candidate_profile.get("experience_years", "not specified")
    role_level = candidate_profile.get("role_level", "not specified")

    projects = candidate_profile.get("recent_projects", [])
    if projects:
        project_lines = "\n".join(
            f"- {p.get('title', 'Untitled project')}: {p.get('description', '')}"
            for p in projects
        )
    else:
        project_lines = "No specific projects listed."

    jd_section = (
        f"\nJob description for the role being interviewed for:\n{job_description_text.strip()}\n"
        if job_description_text and job_description_text.strip()
        else "\nNo job description was provided — base questions purely on the candidate's profile.\n"
    )

    return f"""Generate {num_questions} interview questions for the following candidate.

Candidate profile:
- Skills: {skills}
- Experience: {experience_years} years
- Role level: {role_level}

Recent projects:
{project_lines}
{jd_section}
{QUESTION_GEN_JSON_SCHEMA_INSTRUCTIONS}

Generate exactly {num_questions} questions now, following all rules from the system prompt."""