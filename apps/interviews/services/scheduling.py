# apps/interviews/services/scheduling.py

"""
Rescheduling an interview that the candidate cannot take now.

The candidate says a date and time; this module turns the spoken words into a
datetime (one small LLM call) and checks it against the rules:

  * a working day (Monday-Friday), not a holiday
  * inside working hours (INTERVIEW_WORK_START .. INTERVIEW_WORK_END)
  * not in the past (at least INTERVIEW_MIN_NOTICE_HOURS from now)
  * within INTERVIEW_RESCHEDULE_MAX_DAYS (one month) from today

validate_slot() is pure (no I/O) so the rules are easy to test.
"""

import logging
from datetime import datetime, time, timedelta

from django.conf import settings
from django.utils import timezone

from apps.ai_engine.llm_client import llm_client

logger = logging.getLogger(__name__)

WORK_DAYS = (0, 1, 2, 3, 4)          # Monday..Friday


def _cfg(name, default):
    return getattr(settings, name, default)


def work_start() -> time:
    return time(int(_cfg("INTERVIEW_WORK_START_HOUR", 10)), 0)


def work_end() -> time:
    return time(int(_cfg("INTERVIEW_WORK_END_HOUR", 18)), 0)


def max_days() -> int:
    return int(_cfg("INTERVIEW_RESCHEDULE_MAX_DAYS", 30))


def min_notice_hours() -> int:
    return int(_cfg("INTERVIEW_MIN_NOTICE_HOURS", 2))


def holidays() -> dict:
    """{date: name} from settings.INTERVIEW_HOLIDAYS."""
    out = {}
    for item in _cfg("INTERVIEW_HOLIDAYS", []):
        day, _, name = str(item).partition(":")
        try:
            out[datetime.strptime(day.strip(), "%Y-%m-%d").date()] = name.strip() or "a public holiday"
        except ValueError:
            logger.warning("Ignoring bad INTERVIEW_HOLIDAYS entry %r", item)
    return out


def _hour_label(t: time) -> str:
    return t.strftime("%I %p").lstrip("0")


def hours_phrase() -> str:
    return f"{_hour_label(work_start())} and {_hour_label(work_end())}"


def format_slot(dt: datetime) -> str:
    dt = timezone.localtime(dt) if timezone.is_aware(dt) else dt
    return f"{dt.strftime('%A')}, {dt.day} {dt.strftime('%B')} at {dt.strftime('%I:%M %p').lstrip('0')}"


def validate_slot(dt: datetime, now: datetime | None = None) -> tuple[bool, str, str]:
    """
    Returns (ok, code, reason). `reason` finishes the sentence "Sorry, ..." and is
    safe to speak. Codes: ok, past, too_far, weekend, holiday, hours.
    """
    now = timezone.localtime(now or timezone.now())
    if timezone.is_naive(dt):
        dt = timezone.make_aware(dt)
    dt = timezone.localtime(dt)

    if dt < now + timedelta(hours=min_notice_hours()):
        return False, "past", "that time has already passed, or is too soon for us to arrange"
    last_day = (now + timedelta(days=max_days())).date()
    if dt.date() > last_day:
        return False, "too_far", (
            f"the interview has to be within the next {max_days()} days, that is by "
            f"{last_day.day} {last_day.strftime('%B')}"
        )
    if dt.weekday() not in WORK_DAYS:
        return False, "weekend", "interviews are held only on working days, Monday to Friday"
    name = holidays().get(dt.date())
    if name:
        return False, "holiday", f"{dt.day} {dt.strftime('%B')} is a holiday, {name}"
    start, end = work_start(), work_end()
    if not (start <= dt.time() <= (datetime.combine(dt.date(), end) - timedelta(minutes=30)).time()):
        return False, "hours", f"interviews are held only between {hours_phrase()}"
    return True, "ok", ""


# ----------------------------------------------------------------------
# Understanding what the candidate said
# ----------------------------------------------------------------------

_PARSE_PROMPT = """You extract a date and a time from a job candidate's spoken reply (speech-to-text; it may be Hindi or Hinglish).
The candidate was asked when they would like to take an interview. Today is {today} ({weekday}). Working hours are {start} to {end}.
Calendar (use it for every weekday name, never compute weekdays yourself):
{calendar}

Return ONLY JSON: {{"date": "YYYY-MM-DD" or null, "time": "HH:MM" (24 hour) or null, "refuses": true|false}}
Rules:
- Resolve relative words with the calendar: "tomorrow", "day after tomorrow", "Monday" / "next Monday" = the first such weekday after today in the calendar, "this Friday", "in two days". "next week" or "sometime" alone -> null (too vague).
- A bare hour such as "at 3" or "3 o'clock" means the working-hours reading (3 PM). "morning" alone, "afternoon" alone -> time null.
- Say null for anything the candidate did not state. Never invent a date or time.
- "refuses": true only if the candidate says they do not want to give the interview at all."""


def _calendar(now: datetime) -> str:
    """The next 15 days with their weekday names, so the model never has to work them out."""
    lines = []
    for i in range(15):
        d = now + timedelta(days=i)
        tag = " (today)" if i == 0 else " (tomorrow)" if i == 1 else ""
        lines.append(f"{d.strftime('%Y-%m-%d')} = {d.strftime('%A')}{tag}")
    return "\n".join(lines)


def parse_slot(text: str, now: datetime | None = None) -> dict:
    """{'date': date|None, 'time': time|None, 'refuses': bool}. Never raises."""
    now = timezone.localtime(now or timezone.now())
    empty = {"date": None, "time": None, "refuses": False}
    if not (text or "").strip():
        return empty
    try:
        data = llm_client.generate_json(
            system_prompt=_PARSE_PROMPT.format(
                today=now.strftime("%Y-%m-%d"), weekday=now.strftime("%A"),
                start=work_start().strftime("%H:%M"), end=work_end().strftime("%H:%M"),
                calendar=_calendar(now),
            ),
            user_prompt=f"Candidate said: {text.strip()}",
            model=llm_client.FAST_MODEL,
            temperature=0.0,
            max_tokens=512,
            reasoning_effort="low",
        )
    except Exception as exc:
        logger.warning("Could not read the reschedule date/time: %s", exc)
        return empty

    out = dict(empty)
    try:
        if data.get("date"):
            out["date"] = datetime.strptime(str(data["date"]), "%Y-%m-%d").date()
    except ValueError:
        pass
    try:
        if data.get("time"):
            out["time"] = datetime.strptime(str(data["time"])[:5], "%H:%M").time()
    except ValueError:
        pass
    out["refuses"] = bool(data.get("refuses"))
    return out


def next_working_slot(now: datetime | None = None) -> datetime:
    """The soonest valid slot (11 AM on the next working day that is not a holiday)."""
    now = timezone.localtime(now or timezone.now())
    day = now.date() + timedelta(days=1)
    hol = holidays()
    while day.weekday() not in WORK_DAYS or day in hol:
        day += timedelta(days=1)
    return timezone.make_aware(datetime.combine(day, time(11, 0)))
