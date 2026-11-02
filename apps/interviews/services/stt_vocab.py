# apps/interviews/services/stt_vocab.py

"""
Domain vocabulary for speech-to-text.

Whisper hears "B.Tech CSE" as "BTEC CIC", "White Force" as "Whiteboards", "Ollama" as
"olama" - and the answer is then scored on the wrong words. Two cheap fixes:

  build_prompt()   a short priming text (company, candidate name, skills, tech terms from the
                   job description) passed to Whisper as its `prompt`, so those words are
                   expected and spelled correctly.
  correct_terms()  a conservative clean-up of the transcript for mishearings that still slip
                   through: a small table of known ones, plus near-miss spellings of the
                   vocabulary words themselves.
"""

import re
from difflib import SequenceMatcher

COMPANY_TERMS = ["White Force", "B.Tech", "CSE", "M.Tech", "MCA", "BCA", "MBA"]

TECH_TERMS = [
    "Python", "Django", "Django REST Framework", "FastAPI", "Flask", "Redis", "Celery", "RabbitMQ", "Kafka",
    "PostgreSQL", "MySQL", "MongoDB", "SQLite", "SQL", "NoSQL", "Docker", "Kubernetes", "AWS", "Azure", "GCP",
    "Git", "GitHub", "REST API", "GraphQL", "JavaScript", "TypeScript", "React", "Angular", "Vue", "Node.js",
    "Next.js", "Java", "Spring Boot", "Kotlin", "Swift", "Golang", "C++", "C#", ".NET", "PHP", "Laravel",
    "Ollama", "LLM", "Groq", "Whisper", "OpenAI", "LangChain", "TensorFlow", "PyTorch", "Pandas", "NumPy",
    "WebSocket", "Channels", "ORM", "CI/CD", "Jenkins", "Linux", "Nginx", "Gunicorn", "Elasticsearch",
    "Selenium", "Pytest", "Agile", "Scrum", "Jira", "HTML", "CSS", "Tailwind", "Bootstrap", "Excel", "Tally",
]

# Mishearings seen in real interviews -> what was meant. Whole phrases, case-insensitive.
KNOWN_MISHEARINGS = {
    "btec cic": "B.Tech CSE",
    "btec cse": "B.Tech CSE",
    "b tech cic": "B.Tech CSE",
    "btec": "B.Tech",
    "white boards": "White Force",
    "whiteboards": "White Force",
    "whiteboard": "White Force",
    "white course": "White Force",
    "olama": "Ollama",
    "alama": "Ollama",
    "allama": "Ollama",
    "o llama": "Ollama",
    "riddies": "Redis",
    "reddies": "Redis",
    "rediz": "Redis",
    "ready's": "Redis",
    "jango": "Django",
    "gango": "Django",
    "reddis": "Redis",
    "redish": "Redis",
    "cellery": "Celery",
    "fast api": "FastAPI",
    "post gres": "PostgreSQL",
}

# Patterns for spellings that vary too much for the table above ("B, Tech", "TechCSC", "B tech C S E").
REGEX_FIXES = [
    (re.compile(r"\bb[,.]?\s*tech[,.]?\s*c[\s.]?[sc][\s.]?[ce]\b", re.IGNORECASE), "B.Tech CSE"),
    (re.compile(r"\bb[,.]?\s*tech\b", re.IGNORECASE), "B.Tech"),
    (re.compile(r"\bm[,.]?\s*tech\b", re.IGNORECASE), "M.Tech"),
]

_MAX_PROMPT_CHARS = 480            # Whisper reads only the last ~224 tokens of a prompt
_FUZZY_MIN_LEN = 6                 # only longer vocabulary words are matched approximately
_FUZZY_RATIO = 0.88


def _norm(term: str) -> str:
    return re.sub(r"\s+", " ", term.strip())


def build_terms(job_description: str = "", profile: dict | None = None) -> list[str]:
    """Company words, the candidate's own skills, then tech terms found in the job description."""
    profile = profile or {}
    jd = job_description or ""
    jd_low = jd.lower()
    found: list[str] = []

    def add(term):
        term = _norm(str(term))
        if term and term.lower() not in {t.lower() for t in found} and len(term) <= 40:
            found.append(term)

    for term in COMPANY_TERMS:
        add(term)
    skills = profile.get("skills") or []
    if isinstance(skills, str):
        skills = [x for x in re.split(r"[,;/\n]", skills)]
    for skill in list(skills)[:25]:
        add(skill)
    for term in TECH_TERMS:
        if re.search(r"(?<![A-Za-z0-9])" + re.escape(term.lower()) + r"(?![A-Za-z0-9])", jd_low):
            add(term)
    if len(found) < 12:                      # little context: still help with the most common tech words
        for term in TECH_TERMS[:14]:
            add(term)
    return found


def build_prompt(job_description: str = "", profile: dict | None = None, candidate_name: str = "",
                 language: str = "en") -> str:
    """Whisper priming text. Only for English: on short or quiet Hindi audio Whisper repeats a hint back
    as if the candidate had said it (and an English hint pulls Hindi towards English spellings), and in
    tests it did not improve Hindi anyway - so Hindi / auto-detected speech gets no hint."""
    if language in ("hi", ""):
        return ""
    terms = build_terms(job_description, profile)
    head = "Interview with a candidate at White Force"
    if candidate_name:
        head += f", {candidate_name}"
    text = head + ". Terms: " + ", ".join(terms) + "."
    return text[:_MAX_PROMPT_CHARS]


def prompt_echo(text: str, prompt: str) -> bool:
    """True when the transcript is just the hint's opening sentence read back (Whisper does this on
    near-silent audio). The vocabulary list itself is not checked: a candidate may well say "Django, Redis"."""
    head = (prompt or "").split("Terms:")[0]
    t = re.findall(r"\w+", (text or "").lower())
    h = re.findall(r"\w+", head.lower())
    if len(t) < 3 or len(h) < 3:
        return False
    match = SequenceMatcher(None, t, h, autojunk=False).find_longest_match(0, len(t), 0, len(h))
    return match.size >= 3 and match.size / len(t) >= 0.7


def correct_terms(text: str, terms: list[str] | None = None) -> str:
    """Fixes known mishearings and near-miss spellings of vocabulary words. Never raises."""
    if not text:
        return text
    out = text
    for wrong in sorted(KNOWN_MISHEARINGS, key=len, reverse=True):
        out = re.sub(r"(?<![A-Za-z0-9])" + re.escape(wrong) + r"(?![A-Za-z0-9])", KNOWN_MISHEARINGS[wrong], out, flags=re.IGNORECASE)

    for pattern, proper in REGEX_FIXES:
        out = pattern.sub(proper, out)

    singles = {t.lower(): t for t in (terms or []) if " " not in t and len(t) >= _FUZZY_MIN_LEN and t.isascii()}
    if singles:
        def fix(m):
            word = m.group(0)
            low = word.lower()
            if low in singles or len(low) < _FUZZY_MIN_LEN:
                return singles.get(low, word)
            for key, proper in singles.items():
                if abs(len(key) - len(low)) <= 2 and SequenceMatcher(None, low, key).ratio() >= _FUZZY_RATIO:
                    return proper
            return word
        out = re.sub(r"[A-Za-z][A-Za-z.+#]{5,}", fix, out)
    return out
