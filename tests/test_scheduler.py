"""ScheduledRunner: due selection, advancement, one-shot deletion, and the
failure cutout. No database, no Telegram, no sleeping."""
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import scheduling
from scheduler import ScheduledRunner


@dataclass
class _Rec:
    id: int = 7
    chat_id: str = "123"
    prompt: str = "Post a good morning message."
    label: str = "every day at 8:00am"
    cron: str | None = "0 8 * * *"
    next_run_at: datetime = datetime(2026, 9, 22, 0, 0, tzinfo=timezone.utc)
    enabled: bool = True
    created_by: int | None = 55
    last_run_at: datetime | None = None
    consecutive_failures: int = 0


def _db(records):
    return SimpleNamespace(
        due_schedules=Mock(return_value=records),
        mark_schedule_fired=Mock(),
        record_schedule_failure=Mock(),
        delete_schedule_by_id=Mock(),
    )


def _runner(db, turn=None):
    # process_agent_turn returns True on success; a mock must say so explicitly.
    processor = SimpleNamespace(
        process_agent_turn=turn or AsyncMock(return_value=True)
    )
    prompt_builder = SimpleNamespace(
        to_lc_human_message=Mock(return_value="human-message")
    )
    bot = SimpleNamespace(send_message=AsyncMock())
    return ScheduledRunner(db, processor, bot, prompt_builder), processor


def test_fire_due_runs_the_turn_and_advances_a_recurring_schedule():
    db = _db([_Rec()])
    runner, processor = _runner(db)
    now = _Rec().next_run_at  # the schedule is due exactly now

    fired = asyncio.run(runner.fire_due(now=now))

    assert fired == 1
    processor.process_agent_turn.assert_awaited_once()
    db.mark_schedule_fired.assert_called_once()
    db.delete_schedule_by_id.assert_not_called()
    # Advanced to the next 08:00 SGT after now: 24h later, same wall-clock time.
    assert db.mark_schedule_fired.call_args.kwargs["next_run_at"] == datetime(
        2026, 9, 23, 0, 0, tzinfo=timezone.utc
    )
    assert db.mark_schedule_fired.call_args.kwargs["last_run_at"] == now


def test_fire_due_advances_from_now_not_from_the_missed_time():
    """Downtime skips occurrences rather than backfilling them."""
    db = _db([_Rec(next_run_at=datetime(2026, 9, 1, 0, 0, tzinfo=timezone.utc))])
    runner, _ = _runner(db)
    now = datetime(2026, 9, 22, 0, 0, tzinfo=timezone.utc)  # three weeks late

    asyncio.run(runner.fire_due(now=now))

    assert db.mark_schedule_fired.call_args.kwargs["next_run_at"] == datetime(
        2026, 9, 23, 0, 0, tzinfo=timezone.utc
    )


def test_fire_due_deletes_a_one_shot_after_it_runs():
    db = _db([_Rec(cron=None)])
    runner, _ = _runner(db)

    asyncio.run(runner.fire_due())

    db.delete_schedule_by_id.assert_called_once_with(7)
    db.mark_schedule_fired.assert_not_called()


def test_fire_due_records_a_failure_reported_by_the_return_value():
    """process_agent_turn swallows its own errors, so False is the signal."""
    db = _db([_Rec()])
    runner, _ = _runner(db, turn=AsyncMock(return_value=False))

    fired = asyncio.run(runner.fire_due())

    assert fired == 0
    db.mark_schedule_fired.assert_not_called()
    db.record_schedule_failure.assert_called_once()
    assert db.record_schedule_failure.call_args.kwargs["disable"] is False


def test_fire_due_records_a_failure_when_the_turn_raises():
    db = _db([_Rec()])
    runner, _ = _runner(db, turn=AsyncMock(side_effect=RuntimeError("boom")))

    fired = asyncio.run(runner.fire_due())

    assert fired == 0
    db.record_schedule_failure.assert_called_once()


def test_fire_due_disables_a_schedule_at_the_failure_cutout():
    rec = _Rec(consecutive_failures=scheduling.MAX_CONSECUTIVE_FAILURES - 1)
    db = _db([rec])
    runner, _ = _runner(db, turn=AsyncMock(side_effect=RuntimeError("boom")))

    asyncio.run(runner.fire_due())

    assert db.record_schedule_failure.call_args.kwargs["disable"] is True


def test_fire_due_deletes_a_one_shot_even_when_it_failed():
    db = _db([_Rec(cron=None)])
    runner, _ = _runner(db, turn=AsyncMock(side_effect=RuntimeError("boom")))

    asyncio.run(runner.fire_due())

    db.delete_schedule_by_id.assert_called_once_with(7)


def test_fire_due_keeps_going_after_one_schedule_fails():
    db = _db([_Rec(id=1), _Rec(id=2)])
    turn = AsyncMock(side_effect=[RuntimeError("boom"), True])
    runner, _ = _runner(db, turn=turn)

    fired = asyncio.run(runner.fire_due())

    assert fired == 1
    assert turn.await_count == 2


def test_fire_due_treats_a_negative_chat_id_as_a_group():
    """Telegram group ids are negative; DMs are not."""
    db = _db([_Rec(chat_id="-100123")])
    runner, processor = _runner(db)

    asyncio.run(runner.fire_due())

    assert processor.process_agent_turn.call_args.kwargs["is_group"] is True


def test_fire_due_treats_a_positive_chat_id_as_a_dm():
    db = _db([_Rec(chat_id="123")])
    runner, processor = _runner(db)

    asyncio.run(runner.fire_due())

    assert processor.process_agent_turn.call_args.kwargs["is_group"] is False


def test_fire_due_survives_a_failing_due_query():
    db = _db([])
    db.due_schedules = Mock(side_effect=RuntimeError("db down"))
    runner, _ = _runner(db)

    assert asyncio.run(runner.fire_due()) == 0
