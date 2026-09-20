"""Scheduled-prompt cadence logic and the agent's scheduling tools.

Kept separate from tools.py (web search / fetch) and image_store.py so
schedule concerns stay in one focused module. All clock times are
Asia/Singapore; next_run_at values are tz-aware UTC.
"""
from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from croniter import croniter
from langchain.tools import ToolRuntime, tool

logger = logging.getLogger(__name__)

SGT = ZoneInfo("Asia/Singapore")

# Guardrails. Deliberately constants rather than env vars: none of these is
# something an operator would tune per environment.
MAX_SCHEDULES_PER_CHAT = 10
MAX_PROMPT_CHARS = 500
MIN_INTERVAL_MINUTES = 60
MAX_ONESHOT_DAYS = 365
MAX_CONSECUTIVE_FAILURES = 5
# How many future firings the interval guardrail inspects. Two was not
# enough: a cron like "0,30 0 * * *" hides its 30-minute pair behind a
# 23.5-hour first gap depending on when it was created.
INTERVAL_CHECK_OCCURRENCES = 32
POLL_INTERVAL_SECONDS = 30


class ScheduleError(ValueError):
    """Invalid schedule request. The message is safe to hand to the model."""


def _now_utc() -> datetime:
    return datetime.now(UTC)


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
    return nxt.astimezone(UTC)


def validate_cron_interval(cron: str, after: datetime | None = None) -> None:
    """Reject a cron that fires more often than MIN_INTERVAL_MINUTES.

    Checked by measuring the smallest gap across the next
    INTERVAL_CHECK_OCCURRENCES firings rather than by pattern matching, so
    every runaway shape is caught without enumerating them. Looking at more
    than one gap matters: a multi-value field can place its tight pair
    anywhere in the cycle, so the first gap alone would make the guardrail
    depend on what time the schedule happened to be created.
    """
    base_sgt = (after or _now_utc()).astimezone(SGT)
    try:
        cycle = croniter(cron, base_sgt)
        occurrences = [
            cycle.get_next(datetime) for _ in range(INTERVAL_CHECK_OCCURRENCES)
        ]
    except Exception as e:
        raise ScheduleError(f"'{cron}' is not a valid cron expression: {e}") from e
    smallest = min(b - a for a, b in zip(occurrences, occurrences[1:]))
    if smallest < timedelta(minutes=MIN_INTERVAL_MINUTES):
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
    run_utc = naive.replace(tzinfo=SGT).astimezone(UTC)
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


PROMPT_PREVIEW_CHARS = 80


def create_schedule(db, chat_id, prompt: str, label: str, cron: str | None,
                    at: str | None, created_by: int | None) -> str:
    """Validate and store one schedule. Returns the model-facing confirmation,
    or an explanatory string. Never raises: tool errors go to the model."""
    if chat_id is None:
        return "Scheduling is not available here."
    if bool(cron) == bool(at):
        return "Provide exactly one of cron (recurring) or at (one-shot)."

    prompt = (prompt or "").strip()
    if not prompt:
        return "The prompt to run is empty."
    if len(prompt) > MAX_PROMPT_CHARS:
        return (
            f"That prompt is too long ({len(prompt)} characters). "
            f"The limit is {MAX_PROMPT_CHARS}."
        )

    try:
        if cron:
            validate_cron_interval(cron)
            next_run_at = next_run_from_cron(cron)
        elif at:
            next_run_at = next_run_from_at(at)
        else:  # unreachable: the exactly-one check above guarantees one is set
            return "Provide exactly one of cron (recurring) or at (one-shot)."
    except ScheduleError as e:
        return str(e)

    # Every database call shares one failure boundary: this function promises
    # never to raise, so a transient outage on the reads must surface as a
    # model-facing string exactly like one on the insert.
    try:
        existing = db.find_duplicate_schedule(
            str(chat_id), cron, prompt, next_run_at
        )
        if existing is not None:
            return (
                f"That is already scheduled as #{existing}. "
                "Nothing new was created."
            )

        if db.count_schedules(str(chat_id)) >= MAX_SCHEDULES_PER_CHAT:
            return (
                f"This chat already has {MAX_SCHEDULES_PER_CHAT} schedules, "
                "which is the limit. Cancel one first."
            )

        new_id = db.add_schedule(
            chat_id=str(chat_id), prompt=prompt, label=label or "as scheduled",
            cron=cron, next_run_at=next_run_at, created_by=created_by,
        )
    except Exception:
        logger.exception("Failed to store schedule for chat %s", chat_id)
        return "Could not save that schedule. Try again."

    return (
        f"Scheduled #{new_id} — {label}. "
        f"First run {format_sgt(next_run_at)}.\n"
        f'Will run: "{prompt}"'
    )


def render_schedule_list(records) -> str:
    """Format one chat's schedules for the model to relay, ids intact."""
    if not records:
        return "No schedules in this chat."
    blocks = []
    for r in records:
        preview = r.prompt
        if len(preview) > PROMPT_PREVIEW_CHARS:
            preview = preview[:PROMPT_PREVIEW_CHARS].rstrip() + "..."
        disabled = "" if r.enabled else (
            f"  (disabled — {r.consecutive_failures} failed runs)"
        )
        blocks.append(
            f"#{r.id}  {r.label}{disabled}\n"
            f"     Next run: {format_sgt(r.next_run_at)}\n"
            f'     "{preview}"'
        )
    count = len(records)
    header = f"{count} schedule{'s' if count != 1 else ''} in this chat:"
    return header + "\n\n" + "\n\n".join(blocks)


def build_schedule_tools(db) -> list:
    """Return the three scheduling tools bound to db, scoped to the calling
    chat via ToolRuntime context (same pattern as get_image).

    ToolRuntime and tool are imported at module level, not here: this module
    uses `from __future__ import annotations`, so pydantic resolves the tools'
    annotations against module globals when building their schemas."""

    def _chat_id(runtime):
        context = getattr(runtime, "context", None)
        return getattr(context, "thread_id", None)

    def _user_id(runtime):
        context = getattr(runtime, "context", None)
        return getattr(context, "user_id", None)

    @tool
    def schedule_prompt(prompt: str, label: str, runtime: ToolRuntime,
                        cron: str | None = None, at: str | None = None) -> str:
        """Schedule a prompt to be run automatically in this chat later.

        Only use this when someone explicitly asks for something recurring or
        for a reminder at a specific time. Always relay the returned
        confirmation to them, including the id and the next run time.

        Write `prompt` so it stands alone. It is replayed into this chat later,
        when nobody is asking and the surrounding conversation is gone, so
        resolve every "me", "you", "this", "tomorrow" and "what we discussed"
        into absolutes first. Write "Post a good morning message for Felix",
        not "give me a good morning message".

        Args:
            prompt: The self-contained prompt to run at each firing.
            label: Your plain-English phrasing of the cadence, e.g.
                "every weekday at 9:00am". Shown back to the user in listings.
            cron: A 5-field cron expression for a recurring schedule, read in
                Asia/Singapore time, e.g. "0 8 * * *" for 8am daily. Must not
                fire more than once an hour.
            at: An ISO-8601 local date and time for a one-shot, e.g.
                "2026-09-22T15:00". Also Asia/Singapore.
        """
        return create_schedule(
            db, _chat_id(runtime), prompt=prompt, label=label,
            cron=cron, at=at, created_by=_user_id(runtime),
        )

    @tool
    def list_schedules(runtime: ToolRuntime) -> str:
        """List the prompts scheduled to run in this chat.

        Relay the ids exactly as given — the user cancels by id.
        """
        chat_id = _chat_id(runtime)
        if chat_id is None:
            return "Scheduling is not available here."
        try:
            return render_schedule_list(db.list_schedules(str(chat_id)))
        except Exception:
            logger.exception("Failed to list schedules for chat %s", chat_id)
            return "Could not read this chat's schedules."

    @tool
    def cancel_schedule(schedule_id: int, runtime: ToolRuntime) -> str:
        """Cancel a scheduled prompt in this chat.

        Args:
            schedule_id: The numeric id shown by list_schedules.
        """
        chat_id = _chat_id(runtime)
        if chat_id is None:
            return "Scheduling is not available here."
        try:
            record = db.delete_schedule(str(chat_id), schedule_id)
        except Exception:
            logger.exception("Failed to cancel schedule %s", schedule_id)
            return "Could not cancel that schedule."
        if record is None:
            return f"Schedule #{schedule_id} not found."
        return f"Cancelled #{schedule_id} — {record.label}."

    return [schedule_prompt, list_schedules, cancel_schedule]
