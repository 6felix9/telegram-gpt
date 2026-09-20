"""Cadence core: cron/one-shot parsing, SGT->UTC next-run math, guardrails."""
from datetime import datetime, timezone

import pytest

import scheduling


def _utc(y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm, tzinfo=timezone.utc)


def test_next_run_from_cron_interprets_clock_time_as_sgt():
    """0 8 * * * means 08:00 SGT, which is 00:00 UTC."""
    after = _utc(2026, 9, 21, 3, 0)  # 11:00 SGT on Mon 21 Sep
    result = scheduling.next_run_from_cron("0 8 * * *", after=after)
    assert result == _utc(2026, 9, 22, 0, 0)


def test_next_run_from_cron_crosses_sgt_midnight_correctly():
    """23:00 SGT on the 21st is 15:00 UTC on the 21st, not the 22nd."""
    after = _utc(2026, 9, 21, 3, 0)
    result = scheduling.next_run_from_cron("0 23 * * *", after=after)
    assert result == _utc(2026, 9, 21, 15, 0)


def test_next_run_from_cron_handles_weekday_ranges():
    """0 9 * * 1-5 from a Saturday lands on the following Monday."""
    after = _utc(2026, 9, 26, 3, 0)  # Sat 26 Sep, 11:00 SGT
    result = scheduling.next_run_from_cron("0 9 * * 1-5", after=after)
    assert result == _utc(2026, 9, 28, 1, 0)  # Mon 28 Sep, 09:00 SGT


def test_next_run_from_cron_rejects_malformed_expression():
    with pytest.raises(scheduling.ScheduleError):
        scheduling.next_run_from_cron("not a cron")


def test_validate_cron_interval_accepts_hourly():
    scheduling.validate_cron_interval("0 * * * *")  # no raise


def test_validate_cron_interval_rejects_every_minute():
    with pytest.raises(scheduling.ScheduleError) as exc:
        scheduling.validate_cron_interval("* * * * *")
    assert "hour" in str(exc.value).lower()


def test_validate_cron_interval_rejects_every_thirty_minutes():
    with pytest.raises(scheduling.ScheduleError):
        scheduling.validate_cron_interval("*/30 * * * *")


def test_next_run_from_at_parses_iso_local_time_as_sgt():
    now = _utc(2026, 9, 21, 3, 0)
    result = scheduling.next_run_from_at("2026-09-22T15:00", now=now)
    assert result == _utc(2026, 9, 22, 7, 0)  # 15:00 SGT


def test_next_run_from_at_rejects_a_past_time():
    now = _utc(2026, 9, 21, 3, 0)
    with pytest.raises(scheduling.ScheduleError) as exc:
        scheduling.next_run_from_at("2026-09-20T15:00", now=now)
    assert "past" in str(exc.value).lower()


def test_next_run_from_at_rejects_beyond_the_one_year_horizon():
    now = _utc(2026, 9, 21, 3, 0)
    with pytest.raises(scheduling.ScheduleError):
        scheduling.next_run_from_at("2029-09-22T15:00", now=now)


def test_next_run_from_at_rejects_malformed_datetime():
    with pytest.raises(scheduling.ScheduleError):
        scheduling.next_run_from_at("next tuesday")


def test_format_sgt_renders_utc_instant_in_singapore_time():
    assert scheduling.format_sgt(_utc(2026, 9, 22, 0, 0)) == "Tue 22 Sep, 8:00am SGT"


def test_format_sgt_renders_afternoon_without_leading_zero():
    assert scheduling.format_sgt(_utc(2026, 9, 22, 7, 0)) == "Tue 22 Sep, 3:00pm SGT"


# --- tools ---------------------------------------------------------------

from dataclasses import dataclass
from types import SimpleNamespace


@dataclass
class _Rec:
    id: int
    chat_id: str
    prompt: str
    label: str
    cron: str | None
    next_run_at: datetime
    enabled: bool = True
    created_by: int | None = 55
    last_run_at: datetime | None = None
    consecutive_failures: int = 0


def _db(**overrides):
    base = dict(
        count_schedules=lambda chat_id: 0,
        find_duplicate_schedule=lambda chat_id, cron, prompt, next_run_at: None,
        add_schedule=lambda **kw: 7,
        list_schedules=lambda chat_id: [],
        delete_schedule=lambda chat_id, schedule_id: None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _create(db=None, **overrides):
    kwargs = dict(
        chat_id="123", prompt="Post a good morning message for Felix.",
        label="every day at 8:00am", cron="0 8 * * *", at=None, created_by=55,
    )
    kwargs.update(overrides)
    return scheduling.create_schedule(db or _db(), **kwargs)


def test_create_schedule_confirmation_names_id_next_run_and_prompt():
    out = _create()
    assert "#7" in out
    assert "every day at 8:00am" in out
    assert "SGT" in out
    assert "Post a good morning message for Felix." in out


def test_create_schedule_rejects_prompt_over_the_length_cap():
    out = _create(prompt="x" * (scheduling.MAX_PROMPT_CHARS + 1))
    assert "too long" in out.lower()


def test_create_schedule_rejects_when_chat_is_at_the_cap():
    db = _db(count_schedules=lambda chat_id: scheduling.MAX_SCHEDULES_PER_CHAT)
    out = _create(db=db)
    assert str(scheduling.MAX_SCHEDULES_PER_CHAT) in out
    assert "cancel" in out.lower()


def test_create_schedule_rejects_a_sub_hourly_cron():
    out = _create(cron="* * * * *")
    assert "hour" in out.lower()


def test_create_schedule_rejects_both_cron_and_at():
    out = _create(at="2026-12-01T09:00")
    assert "exactly one" in out.lower()


def test_create_schedule_rejects_neither_cron_nor_at():
    out = _create(cron=None)
    assert "exactly one" in out.lower()


def test_create_schedule_returns_existing_id_for_a_duplicate():
    db = _db(find_duplicate_schedule=lambda chat_id, cron, prompt, next_run_at: 4)
    out = _create(db=db)
    assert "#4" in out
    assert "already" in out.lower()


def test_create_schedule_refuses_without_a_chat_id():
    out = _create(chat_id=None)
    assert "not available" in out.lower()


def test_render_schedule_list_shows_id_label_next_run_and_preview():
    records = [_Rec(7, "123", "Post a good morning message.",
                    "every day at 8:00am", "0 8 * * *", _utc(2026, 9, 22, 0, 0))]
    out = scheduling.render_schedule_list(records)
    assert "#7" in out
    assert "every day at 8:00am" in out
    assert "Tue 22 Sep, 8:00am SGT" in out
    assert "Post a good morning message." in out


def test_render_schedule_list_truncates_a_long_prompt():
    records = [_Rec(7, "123", "y" * 200, "daily", "0 8 * * *",
                    _utc(2026, 9, 22, 0, 0))]
    out = scheduling.render_schedule_list(records)
    assert "..." in out
    assert "y" * 200 not in out


def test_render_schedule_list_marks_a_disabled_schedule():
    records = [_Rec(7, "123", "p", "daily", "0 8 * * *",
                    _utc(2026, 9, 22, 0, 0), enabled=False,
                    consecutive_failures=scheduling.MAX_CONSECUTIVE_FAILURES)]
    assert "disabled" in scheduling.render_schedule_list(records).lower()


def test_render_schedule_list_handles_an_empty_chat():
    assert scheduling.render_schedule_list([]) == "No schedules in this chat."


def test_build_schedule_tools_exposes_three_named_tools():
    names = {t.name for t in scheduling.build_schedule_tools(_db())}
    assert names == {"schedule_prompt", "list_schedules", "cancel_schedule"}


# --- review fixes --------------------------------------------------------

def test_validate_cron_interval_rejects_a_pair_hidden_past_the_first_gap():
    """Checking only the next two occurrences made the hour floor depend on
    creation time: '0,30 0 * * *' at 00:15 shows a 23.5h gap first, then 30m."""
    base = _utc(2026, 9, 20, 16, 15)  # 00:15 SGT on 21 Sep
    with pytest.raises(scheduling.ScheduleError):
        scheduling.validate_cron_interval("0,30 0 * * *", after=base)


def test_validate_cron_interval_still_accepts_a_genuinely_hourly_cron():
    base = _utc(2026, 9, 20, 16, 15)
    scheduling.validate_cron_interval("0 */2 * * *", after=base)  # no raise


def test_create_schedule_allows_the_same_one_shot_prompt_at_another_time():
    """Two one-shots differ by time: cron is NULL for both, so the duplicate
    key must include next_run_at or the second is wrongly rejected."""
    seen = []

    def _find(chat_id, cron, prompt, next_run_at):
        seen.append(next_run_at)
        return None

    db = _db(find_duplicate_schedule=_find)
    first = _create(db=db, cron=None, at="2027-01-04T09:00",
                    prompt="Take medicine.", label="once at 9am")
    second = _create(db=db, cron=None, at="2027-01-04T17:00",
                     prompt="Take medicine.", label="once at 5pm")

    assert "Scheduled #" in first
    assert "Scheduled #" in second
    assert seen[0] != seen[1]  # the time is part of the identity


def test_create_schedule_still_dedupes_an_identical_recurring_schedule():
    db = _db(find_duplicate_schedule=lambda chat_id, cron, prompt, next_run_at: 4)
    assert "#4" in _create(db=db)


def test_create_schedule_survives_a_failing_duplicate_lookup():
    """create_schedule promises never to raise; the reads must be inside the
    same failure boundary as the insert."""
    def _boom(*args, **kwargs):
        raise RuntimeError("db down")

    out = _create(db=_db(find_duplicate_schedule=_boom))
    assert "could not save" in out.lower()


def test_create_schedule_survives_a_failing_count_query():
    def _boom(*args, **kwargs):
        raise RuntimeError("db down")

    out = _create(db=_db(count_schedules=_boom))
    assert "could not save" in out.lower()
