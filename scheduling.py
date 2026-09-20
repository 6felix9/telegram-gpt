"""Scheduled-prompt cadence logic and the agent's scheduling tools.

Kept separate from tools.py (web search / fetch) and image_store.py so
schedule concerns stay in one focused module. All clock times are
Asia/Singapore; next_run_at values are tz-aware UTC.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from croniter import croniter

logger = logging.getLogger(__name__)

SGT = ZoneInfo("Asia/Singapore")

# Guardrails. Deliberately constants rather than env vars: none of these is
# something an operator would tune per environment.
MAX_SCHEDULES_PER_CHAT = 10
MAX_PROMPT_CHARS = 500
MIN_INTERVAL_MINUTES = 60
MAX_ONESHOT_DAYS = 365
MAX_CONSECUTIVE_FAILURES = 5
POLL_INTERVAL_SECONDS = 30


class ScheduleError(ValueError):
    """Invalid schedule request. The message is safe to hand to the model."""


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def next_run_from_cron(cron: str, after: datetime | None = None) -> datetime:
    """Next firing of `cron` strictly after `after`, as tz-aware UTC.

    The expression is evaluated in Asia/Singapore, so "0 8 * * *" means
    08:00 SGT regardless of where the process runs.
    """
    base_utc = after or _now_utc()
    base_sgt = base_utc.astimezone(SGT)
    try:
        nxt = croniter(cron, base_sgt).get_next(datetime)
    except Exception as e:
        raise ScheduleError(f"'{cron}' is not a valid cron expression: {e}") from e
    return nxt.astimezone(timezone.utc)


def validate_cron_interval(cron: str) -> None:
    """Reject a cron that fires more often than MIN_INTERVAL_MINUTES.

    Checked by asking for the next two occurrences rather than by pattern
    matching, so every runaway shape is caught without enumerating them.
    """
    first = next_run_from_cron(cron)
    second = next_run_from_cron(cron, after=first)
    if (second - first) < timedelta(minutes=MIN_INTERVAL_MINUTES):
        raise ScheduleError(
            f"That schedule fires more often than once an hour. "
            f"The minimum interval is {MIN_INTERVAL_MINUTES} minutes."
        )


def next_run_from_at(at: str, now: datetime | None = None) -> datetime:
    """Parse a one-shot ISO-8601 local datetime (SGT) into tz-aware UTC."""
    now_utc = now or _now_utc()
    try:
        naive = datetime.fromisoformat(at)
    except ValueError as e:
        raise ScheduleError(
            f"'{at}' is not a valid date and time. "
            "Use an ISO format like 2026-09-22T15:00."
        ) from e
    if naive.tzinfo is not None:
        raise ScheduleError("Give the time without a timezone offset; it is read as SGT.")
    run_utc = naive.replace(tzinfo=SGT).astimezone(timezone.utc)
    if run_utc <= now_utc:
        raise ScheduleError("That time is in the past.")
    if run_utc > now_utc + timedelta(days=MAX_ONESHOT_DAYS):
        raise ScheduleError(
            f"That is more than {MAX_ONESHOT_DAYS} days away. Check the year."
        )
    return run_utc


def format_sgt(dt: datetime) -> str:
    """Render a UTC instant as 'Mon 22 Sep, 8:00am SGT'."""
    local = dt.astimezone(SGT)
    hour = local.hour % 12 or 12
    meridiem = "am" if local.hour < 12 else "pm"
    return f"{local:%a %-d %b}, {hour}:{local:%M}{meridiem} SGT"
