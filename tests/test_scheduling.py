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
