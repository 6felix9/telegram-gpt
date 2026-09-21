# Agent-Scheduled Prompts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the agent three tools so a user can say "chatgpt give me a good morning message at 8am every day" and get a durable, recurring, agent-run prompt — with no new Telegram command.

**Architecture:** A new `scheduling.py` holds pure cadence logic and the three chat-scoped tools (bound to a `Database`, scoped via `ToolRuntime.context.thread_id`, exactly like `image_store.build_image_tool`). A new `scheduled_prompts` table behind a new `database/schedule_repository.py` is the single source of truth. A new `scheduler.py` runs a 30s asyncio poll loop started from `bot.py`'s `post_init`, firing due rows through a `process_agent_turn` extracted out of `RequestProcessor.process` so scheduled runs share the real request path.

**Tech Stack:** Python 3.12+, LangChain `@tool` + `ToolRuntime`, psycopg2 + Alembic, `croniter`, `zoneinfo`, pytest.

**Spec:** `docs/superpowers/specs/2026-09-21-agent-scheduled-prompts-design.md`

## Global Constraints

- Timezone is **Asia/Singapore** everywhere. Cron and `at=` are read in SGT; `next_run_at` is stored UTC; all user-facing times are rendered SGT.
- Guardrails, as module constants in `scheduling.py`: `MAX_SCHEDULES_PER_CHAT = 10`, `MAX_PROMPT_CHARS = 500`, `MIN_INTERVAL_MINUTES = 60`, `MAX_ONESHOT_DAYS = 365`, `MAX_CONSECUTIVE_FAILURES = 5`, `POLL_INTERVAL_SECONDS = 30`.
- **No new environment variables.** `config.py` stays focused on credentials and model settings.
- Tools never raise. Every failure path returns an explanatory string for the model to relay, matching `fetch_url` and `get_image`.
- Scheduled firings never post an error into the chat — nobody is waiting on them.
- Tests are pure logic: no database, no network, no Telegram. Use the `_FakeConn` / `_fake_manager` doubles already in `tests/test_repositories.py`.
- Run tests with `venv/bin/python3.12 -m pytest` — a bare `pytest` on this machine can hit a stray 3.14 interpreter.
- Target Python 3.12+, 4-space indent, `snake_case`, type hints, short docstrings on public methods.
- Work on `dev`. Promote to `main` by PR, never a local merge.
- End every commit message with the attribution footer:
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`, and every PR
  description with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.

---

### Task 1: Cadence core — validation, next-run computation, guardrails

Pure functions with no database and no LangChain. Everything later tasks need to turn a cron string or an ISO datetime into a UTC `next_run_at`, and to reject bad input.

**Files:**
- Create: `scheduling.py`
- Modify: `requirements.txt`
- Test: `tests/test_scheduling.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `SGT: ZoneInfo` — `ZoneInfo("Asia/Singapore")`
  - `MAX_SCHEDULES_PER_CHAT: int`, `MAX_PROMPT_CHARS: int`, `MIN_INTERVAL_MINUTES: int`, `MAX_ONESHOT_DAYS: int`, `MAX_CONSECUTIVE_FAILURES: int`
  - `class ScheduleError(ValueError)` — message is model-safe
  - `next_run_from_cron(cron: str, after: datetime | None = None) -> datetime` — returns tz-aware UTC
  - `next_run_from_at(at: str, now: datetime | None = None) -> datetime` — returns tz-aware UTC
  - `validate_cron_interval(cron: str) -> None` — raises `ScheduleError` if fires more often than `MIN_INTERVAL_MINUTES`
  - `format_sgt(dt: datetime) -> str` — `"Mon 22 Sep, 8:00am SGT"`

- [ ] **Step 1: Add the dependency**

Append to the `# Tools` block at the end of `requirements.txt`:

```
croniter>=3.0
```

Then install it:

```bash
venv/bin/python3.12 -m pip install 'croniter>=3.0'
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_scheduling.py`:

```python
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
    assert scheduling.format_sgt(_utc(2026, 9, 22, 0, 0)) == "Mon 22 Sep, 8:00am SGT"


def test_format_sgt_renders_afternoon_without_leading_zero():
    assert scheduling.format_sgt(_utc(2026, 9, 22, 7, 0)) == "Tue 22 Sep, 3:00pm SGT"
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `venv/bin/python3.12 -m pytest tests/test_scheduling.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scheduling'`

- [ ] **Step 4: Write the implementation**

Create `scheduling.py`:

```python
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
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `venv/bin/python3.12 -m pytest tests/test_scheduling.py -v`
Expected: PASS, 13 tests.

- [ ] **Step 6: Commit**

```bash
git add scheduling.py tests/test_scheduling.py requirements.txt
git commit -m "Add cadence core for scheduled prompts"
```

---

### Task 2: Storage — migration, repository, facade

The durable half. A table, a repository following the five existing ones, and the `Database` facade methods the tools and the runner call.

**Files:**
- Create: `alembic/versions/0005_scheduled_prompts.py`
- Create: `database/schedule_repository.py`
- Modify: `database/__init__.py`
- Test: `tests/test_repositories.py` (append)

**Interfaces:**
- Consumes: `database.db_connection.ConnectionManager` (existing).
- Produces, on `Database` and on `ScheduleRepository`:
  - `@dataclass ScheduleRecord` with fields `id: int`, `chat_id: str`, `prompt: str`, `label: str`, `cron: str | None`, `next_run_at: datetime`, `enabled: bool`, `created_by: int | None`, `last_run_at: datetime | None`, `consecutive_failures: int`
  - `add_schedule(chat_id, prompt, label, cron, next_run_at, created_by) -> int`
  - `list_schedules(chat_id) -> list[ScheduleRecord]`
  - `find_duplicate_schedule(chat_id, cron, prompt) -> int | None`
  - `count_schedules(chat_id) -> int`
  - `delete_schedule(chat_id, schedule_id) -> ScheduleRecord | None`
  - `due_schedules(now) -> list[ScheduleRecord]`
  - `mark_schedule_fired(schedule_id, next_run_at, last_run_at) -> None`
  - `record_schedule_failure(schedule_id, next_run_at, disable) -> None`

- [ ] **Step 1: Write the migration**

Create `alembic/versions/0005_scheduled_prompts.py`:

```python
"""Add scheduled_prompts table for agent-created recurring prompts

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-21

"""
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE scheduled_prompts (
            id SERIAL PRIMARY KEY,
            chat_id TEXT NOT NULL,
            prompt TEXT NOT NULL,
            label TEXT NOT NULL,
            cron TEXT,
            next_run_at TIMESTAMPTZ NOT NULL,
            enabled BOOLEAN NOT NULL DEFAULT TRUE,
            created_by BIGINT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            last_run_at TIMESTAMPTZ,
            consecutive_failures INTEGER NOT NULL DEFAULT 0
        )
    """)
    op.execute(
        "CREATE INDEX idx_scheduled_prompts_due "
        "ON scheduled_prompts (enabled, next_run_at)"
    )
    op.execute(
        "CREATE INDEX idx_scheduled_prompts_chat ON scheduled_prompts (chat_id)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS scheduled_prompts")
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_repositories.py`. Add the import at the top of the file next to the other repository imports:

```python
from database.schedule_repository import ScheduleRecord, ScheduleRepository
```

Then append these tests to the end of the file:

```python
# --- ScheduleRepository --------------------------------------------------

def _schedule_row(**overrides):
    row = {
        "id": 7, "chat_id": "123", "prompt": "Post a good morning message.",
        "label": "every day at 8:00am", "cron": "0 8 * * *",
        "next_run_at": datetime(2026, 9, 22, 0, 0), "enabled": True,
        "created_by": 55, "last_run_at": None, "consecutive_failures": 0,
    }
    row.update(overrides)
    return row


def test_add_schedule_inserts_and_returns_id():
    manager, conn = _fake_manager(results=[(7,)])
    repo = ScheduleRepository(manager)
    new_id = repo.add_schedule(
        chat_id=123, prompt="Post a good morning message.",
        label="every day at 8:00am", cron="0 8 * * *",
        next_run_at=datetime(2026, 9, 22, 0, 0), created_by=55,
    )
    assert new_id == 7
    sql, params = conn.executed[0]
    assert "INSERT INTO scheduled_prompts" in sql
    assert params[0] == "123"  # chat_id coerced to str


def test_list_schedules_is_scoped_to_the_chat():
    manager, conn = _fake_manager(results=[[_schedule_row()]])
    repo = ScheduleRepository(manager)
    records = repo.list_schedules("123")
    assert [r.id for r in records] == [7]
    assert records[0].label == "every day at 8:00am"
    sql, params = conn.executed[0]
    assert "WHERE chat_id = %s" in sql
    assert params == ("123",)


def test_find_duplicate_schedule_returns_existing_id():
    manager, conn = _fake_manager(results=[(7,)])
    repo = ScheduleRepository(manager)
    assert repo.find_duplicate_schedule("123", "0 8 * * *", "Post it.") == 7


def test_find_duplicate_schedule_returns_none_when_absent():
    manager, _ = _fake_manager(results=[])
    repo = ScheduleRepository(manager)
    assert repo.find_duplicate_schedule("123", "0 8 * * *", "Post it.") is None


def test_delete_schedule_returns_the_deleted_record_scoped_to_chat():
    manager, conn = _fake_manager(results=[_schedule_row()])
    repo = ScheduleRepository(manager)
    record = repo.delete_schedule("123", 7)
    assert record is not None and record.id == 7
    sql, params = conn.executed[0]
    assert "DELETE FROM scheduled_prompts" in sql
    assert params == ("123", 7)


def test_delete_schedule_returns_none_for_another_chats_id():
    manager, _ = _fake_manager(results=[])
    repo = ScheduleRepository(manager)
    assert repo.delete_schedule("999", 7) is None


def test_due_schedules_selects_enabled_rows_at_or_before_now():
    manager, conn = _fake_manager(results=[[_schedule_row()]])
    repo = ScheduleRepository(manager)
    now = datetime(2026, 9, 22, 0, 0)
    assert len(repo.due_schedules(now)) == 1
    sql, params = conn.executed[0]
    assert "enabled = TRUE" in sql
    assert "next_run_at <= %s" in sql
    assert params == (now,)


def test_record_schedule_failure_disables_when_asked():
    manager, conn = _fake_manager()
    repo = ScheduleRepository(manager)
    repo.record_schedule_failure(7, datetime(2026, 9, 23, 0, 0), disable=True)
    sql, _ = conn.executed[0]
    assert "enabled = FALSE" in sql
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `venv/bin/python3.12 -m pytest tests/test_repositories.py -v -k schedule`
Expected: FAIL — `ModuleNotFoundError: No module named 'database.schedule_repository'`

- [ ] **Step 4: Write the repository**

Create `database/schedule_repository.py`:

```python
"""CRUD for `scheduled_prompts`, the durable store behind the agent's
scheduling tools and the background runner. Not cached: the poll loop needs
current rows and the row count is trivial."""
import logging
from dataclasses import dataclass
from datetime import datetime

from psycopg2.extras import RealDictCursor

from .db_connection import ConnectionManager

logger = logging.getLogger(__name__)

_COLUMNS = (
    "id, chat_id, prompt, label, cron, next_run_at, enabled, "
    "created_by, last_run_at, consecutive_failures"
)


@dataclass
class ScheduleRecord:
    """One row of `scheduled_prompts`. cron is None for a one-shot."""

    id: int
    chat_id: str
    prompt: str
    label: str
    cron: str | None
    next_run_at: datetime
    enabled: bool
    created_by: int | None
    last_run_at: datetime | None
    consecutive_failures: int


def _record(row) -> ScheduleRecord:
    return ScheduleRecord(
        id=row["id"], chat_id=row["chat_id"], prompt=row["prompt"],
        label=row["label"], cron=row["cron"], next_run_at=row["next_run_at"],
        enabled=row["enabled"], created_by=row["created_by"],
        last_run_at=row["last_run_at"],
        consecutive_failures=row["consecutive_failures"],
    )


class ScheduleRepository:
    """CRUD for `scheduled_prompts`."""

    def __init__(self, conn: ConnectionManager):
        self._conn = conn

    def add_schedule(self, chat_id, prompt: str, label: str, cron: str | None,
                     next_run_at: datetime, created_by: int | None) -> int:
        """Insert a schedule and return its id."""
        with self._conn.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO scheduled_prompts
                    (chat_id, prompt, label, cron, next_run_at, created_by)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (str(chat_id), prompt, label, cron, next_run_at, created_by),
                )
                return cur.fetchone()[0]

    def list_schedules(self, chat_id) -> list[ScheduleRecord]:
        """Every schedule for one chat, soonest first."""
        with self._conn.connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    f"SELECT {_COLUMNS} FROM scheduled_prompts "
                    "WHERE chat_id = %s ORDER BY next_run_at",
                    (str(chat_id),),
                )
                return [_record(r) for r in cur.fetchall()]

    def count_schedules(self, chat_id) -> int:
        """How many schedules this chat already has, for the per-chat cap."""
        with self._conn.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM scheduled_prompts WHERE chat_id = %s",
                    (str(chat_id),),
                )
                return cur.fetchone()[0]

    def find_duplicate_schedule(self, chat_id, cron: str | None,
                                prompt: str) -> int | None:
        """Id of an identical existing schedule, so a retried tool call is
        idempotent rather than creating a twin."""
        with self._conn.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id FROM scheduled_prompts "
                    "WHERE chat_id = %s AND cron IS NOT DISTINCT FROM %s "
                    "AND prompt = %s LIMIT 1",
                    (str(chat_id), cron, prompt),
                )
                row = cur.fetchone()
                return row[0] if row else None

    def delete_schedule(self, chat_id, schedule_id: int) -> ScheduleRecord | None:
        """Delete one schedule, scoped to the chat. Returns what was deleted."""
        with self._conn.connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    f"DELETE FROM scheduled_prompts "
                    f"WHERE chat_id = %s AND id = %s RETURNING {_COLUMNS}",
                    (str(chat_id), schedule_id),
                )
                row = cur.fetchone()
                return _record(row) if row else None

    def due_schedules(self, now: datetime) -> list[ScheduleRecord]:
        """Enabled schedules whose next_run_at has arrived, across all chats."""
        with self._conn.connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    f"SELECT {_COLUMNS} FROM scheduled_prompts "
                    "WHERE enabled = TRUE AND next_run_at <= %s "
                    "ORDER BY next_run_at",
                    (now,),
                )
                return [_record(r) for r in cur.fetchall()]

    def mark_schedule_fired(self, schedule_id: int, next_run_at: datetime,
                            last_run_at: datetime) -> None:
        """Advance a schedule after a successful firing."""
        with self._conn.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE scheduled_prompts SET next_run_at = %s, "
                    "last_run_at = %s, consecutive_failures = 0 WHERE id = %s",
                    (next_run_at, last_run_at, schedule_id),
                )

    def record_schedule_failure(self, schedule_id: int, next_run_at: datetime,
                                disable: bool) -> None:
        """Advance past a failed firing, optionally tripping the cutout."""
        enabled_clause = ", enabled = FALSE" if disable else ""
        with self._conn.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE scheduled_prompts SET next_run_at = %s, "
                    "consecutive_failures = consecutive_failures + 1"
                    f"{enabled_clause} WHERE id = %s",
                    (next_run_at, schedule_id),
                )

    def delete_by_id(self, schedule_id: int) -> None:
        """Remove a one-shot once it has fired. Not chat-scoped: the caller is
        the runner, which already holds the row it read."""
        with self._conn.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM scheduled_prompts WHERE id = %s", (schedule_id,)
                )
```

- [ ] **Step 5: Wire the facade**

In `database/__init__.py`, add the import next to the other repository imports:

```python
from .schedule_repository import ScheduleRecord, ScheduleRepository
```

In `Database.__init__`, after `self._images = ImageRepository(self._conn)`:

```python
        self._schedules = ScheduleRepository(self._conn)
```

At the end of the class, after the image delegations, add:

```python
    # --- schedules -----------------------------------------------------
    def add_schedule(self, *args, **kwargs) -> int:
        return self._schedules.add_schedule(*args, **kwargs)

    def list_schedules(self, *args, **kwargs) -> list[ScheduleRecord]:
        return self._schedules.list_schedules(*args, **kwargs)

    def count_schedules(self, *args, **kwargs) -> int:
        return self._schedules.count_schedules(*args, **kwargs)

    def find_duplicate_schedule(self, *args, **kwargs) -> int | None:
        return self._schedules.find_duplicate_schedule(*args, **kwargs)

    def delete_schedule(self, *args, **kwargs) -> ScheduleRecord | None:
        return self._schedules.delete_schedule(*args, **kwargs)

    def due_schedules(self, *args, **kwargs) -> list[ScheduleRecord]:
        return self._schedules.due_schedules(*args, **kwargs)

    def mark_schedule_fired(self, *args, **kwargs) -> None:
        return self._schedules.mark_schedule_fired(*args, **kwargs)

    def record_schedule_failure(self, *args, **kwargs) -> None:
        return self._schedules.record_schedule_failure(*args, **kwargs)

    def delete_schedule_by_id(self, *args, **kwargs) -> None:
        return self._schedules.delete_by_id(*args, **kwargs)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `venv/bin/python3.12 -m pytest tests/test_repositories.py -v`
Expected: PASS, including the 8 new schedule tests.

- [ ] **Step 7: Verify the migration applies**

Run: `venv/bin/python3.12 -m alembic upgrade head`
Expected: `Running upgrade 0004 -> 0005`. If no database is reachable locally, skip this step and note it — the Railway `preDeployCommand` applies it on deploy.

- [ ] **Step 8: Commit**

```bash
git add alembic/versions/0005_scheduled_prompts.py database/schedule_repository.py database/__init__.py tests/test_repositories.py
git commit -m "Add scheduled_prompts table and repository"
```

---

### Task 3: Carry the caller's user id into tool runtime

`AgentContext` currently exposes only `is_group`, `reply_context` and
`thread_id`, so a tool cannot tell who is calling it. The spec relies on
`created_by` for traceability — without this, every schedule stores NULL and
that mitigation is void.

**Files:**
- Modify: `agent.py:222-226` (the `AgentContext` dataclass), `agent.py:435-446` (`Agent.run`)
- Modify: `handlers/request_processor.py`
- Test: `tests/test_agent.py` (append)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `AgentContext.user_id: int | None = None`
  - `Agent.run(chat_id, human_message, is_group, reply_context=None, user_id=None) -> str`
    — the new parameter is keyword-optional, so no existing caller breaks.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_agent.py`:

```python
def test_agent_context_carries_the_caller_user_id():
    """Tools read the caller off runtime.context; scheduling needs created_by."""
    from agent import AgentContext

    ctx = AgentContext(is_group=True, thread_id="123", user_id=55)
    assert ctx.user_id == 55


def test_agent_context_user_id_defaults_to_none():
    from agent import AgentContext

    assert AgentContext().user_id is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `venv/bin/python3.12 -m pytest tests/test_agent.py -v -k user_id`
Expected: FAIL — `TypeError: AgentContext.__init__() got an unexpected keyword argument 'user_id'`

- [ ] **Step 3: Add the field**

In `agent.py`, extend the dataclass:

```python
@dataclass
class AgentContext:
    """Per-invocation context read by middleware and tools (not persisted)."""
    is_group: bool = False
    reply_context: tuple[str, str] | None = None
    thread_id: str = "unknown"
    user_id: int | None = None
```

- [ ] **Step 4: Thread it through `Agent.run`**

In `agent.py`, change the signature and the context construction:

```python
    async def run(self, chat_id, human_message, is_group, reply_context=None,
                  user_id=None) -> str:
```

and, a few lines below:

```python
        context = AgentContext(
            is_group=is_group,
            reply_context=reply_context,
            thread_id=str(chat_id),
            user_id=user_id,
        )
```

Leave the rest of the method alone.

- [ ] **Step 5: Pass it from the request processor**

In `handlers/request_processor.py`, in the `agent.run(...)` call inside
`process`, add the argument:

```python
                response = await agent.run(
                    chat_id, human_message, is_group,
                    reply_context=reply_context, user_id=user_id,
                )
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `venv/bin/python3.12 -m pytest tests/test_agent.py tests/test_request_processor.py tests/test_message_handlers.py -v`
Expected: PASS. The new parameter is optional, so no existing caller or test changes.

- [ ] **Step 7: Commit**

```bash
git add agent.py handlers/request_processor.py tests/test_agent.py
git commit -m "Carry the caller's user id in AgentContext"
```

---

### Task 4: The three agent tools

The user-facing surface. Formatting and validation are pure functions so they can be tested without LangChain; the `@tool` wrappers are thin.

**Files:**
- Modify: `scheduling.py`
- Modify: `tools.py:104-114`
- Test: `tests/test_scheduling.py` (append), `tests/test_tools.py` (append)

**Interfaces:**
- Consumes: Task 1's `ScheduleError`, `next_run_from_cron`, `next_run_from_at`, `validate_cron_interval`, `format_sgt`, and the constants; Task 2's `Database` schedule methods and `ScheduleRecord`; Task 3's `AgentContext.user_id`.
- Produces:
  - `create_schedule(db, chat_id, prompt, label, cron, at, created_by) -> str` — pure core, returns the confirmation or an error string
  - `render_schedule_list(records) -> str`
  - `build_schedule_tools(db) -> list` — `[schedule_prompt, list_schedules, cancel_schedule]`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_scheduling.py`:

```python
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
        find_duplicate_schedule=lambda chat_id, cron, prompt: None,
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
    db = _db(find_duplicate_schedule=lambda chat_id, cron, prompt: 4)
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
    assert "Mon 22 Sep, 8:00am SGT" in out
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
```

Append to `tests/test_tools.py`:

```python
def test_build_tools_includes_schedule_tools_when_db_present():
    db = object()
    names = {t.name for t in tools.build_tools(_CfgNoKey, db=db)}
    assert {"schedule_prompt", "list_schedules", "cancel_schedule"} <= names


def test_build_tools_omits_schedule_tools_without_db():
    names = {t.name for t in tools.build_tools(_CfgNoKey, db=None)}
    assert "schedule_prompt" not in names
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/bin/python3.12 -m pytest tests/test_scheduling.py tests/test_tools.py -v`
Expected: FAIL — `AttributeError: module 'scheduling' has no attribute 'create_schedule'`

- [ ] **Step 3: Implement the pure core**

Append to `scheduling.py`:

```python
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
        else:
            next_run_at = next_run_from_at(at)
    except ScheduleError as e:
        return str(e)

    existing = db.find_duplicate_schedule(str(chat_id), cron, prompt)
    if existing is not None:
        return f"That is already scheduled as #{existing}. Nothing new was created."

    if db.count_schedules(str(chat_id)) >= MAX_SCHEDULES_PER_CHAT:
        return (
            f"This chat already has {MAX_SCHEDULES_PER_CHAT} schedules, "
            "which is the limit. Cancel one first."
        )

    try:
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
```

- [ ] **Step 4: Implement the tool wrappers**

Append to `scheduling.py`:

```python
def build_schedule_tools(db) -> list:
    """Return the three scheduling tools bound to db, scoped to the calling
    chat via ToolRuntime context (same pattern as get_image)."""
    from langchain.tools import ToolRuntime, tool

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
```

- [ ] **Step 5: Wire into build_tools**

In `tools.py`, replace the body of `build_tools` after `built = [search, fetch_url]`:

```python
    if db is not None:
        from image_store import build_image_tool
        from scheduling import build_schedule_tools
        built.append(build_image_tool(db))
        built.extend(build_schedule_tools(db))
    return built
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `venv/bin/python3.12 -m pytest tests/test_scheduling.py tests/test_tools.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add scheduling.py tools.py tests/test_scheduling.py tests/test_tools.py
git commit -m "Add schedule_prompt, list_schedules and cancel_schedule tools"
```

---

### Task 5: Extract `process_agent_turn` from `RequestProcessor`

A pure refactor with no behaviour change, so the runner can reuse the real request path instead of reimplementing it. The existing `tests/test_request_processor.py` must pass untouched — that is the proof.

**Files:**
- Modify: `handlers/request_processor.py:36-92`
- Test: `tests/test_request_processor.py` (append; existing tests unchanged)

**Interfaces:**
- Consumes: `HandlerDependencies` (existing).
- Produces: `RequestProcessor.process_agent_turn(bot, chat_id, *, user_id, sender_name, sender_username, is_group, build_payload, reply_context, send, telegram_message_id=None, success_log, error_log_prefix, on_error=None, generic_error_text="", post_success=None) -> bool`
  - Returns **True only if the agent replied and the reply was sent.** It catches
    its own exceptions, so a caller cannot detect failure any other way — the
    scheduled runner depends on this return value to decide whether to advance
    or to count a failure.
  - `send` is an async callable taking the reply text.
  - `on_error` is an optional async callable taking a user-facing string; when None, failures are logged only. `process()` passes `message.reply_text`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_request_processor.py`:

```python
def test_process_agent_turn_sends_through_the_send_callable():
    db = SimpleNamespace(add_message=Mock())
    agent = SimpleNamespace(run=AsyncMock(return_value="reply text"))
    processor = RequestProcessor(_deps(db=db, agent=agent))
    send = AsyncMock()

    asyncio.run(processor.process_agent_turn(
        _bot(), 123, user_id=55, sender_name=None, sender_username=None,
        is_group=True, build_payload=_payload, reply_context=None, send=send,
        success_log="ok", error_log_prefix="err",
    ))

    assert db.add_message.call_count == 2
    send.assert_awaited_once_with("reply text")


def test_process_agent_turn_without_on_error_stays_silent_on_failure():
    """A scheduled run has no one waiting, so a failure posts nothing."""
    db = SimpleNamespace(add_message=Mock())
    agent = SimpleNamespace(run=AsyncMock(side_effect=CompletionError("nope")))
    processor = RequestProcessor(_deps(db=db, agent=agent))
    send = AsyncMock()

    asyncio.run(processor.process_agent_turn(
        _bot(), 123, user_id=55, sender_name=None, sender_username=None,
        is_group=True, build_payload=_payload, reply_context=None, send=send,
        success_log="ok", error_log_prefix="err",
    ))

    send.assert_not_awaited()


def test_process_agent_turn_returns_true_on_success_false_on_failure():
    """The return value is the scheduled runner's only failure signal."""
    db = SimpleNamespace(add_message=Mock())
    ok_agent = SimpleNamespace(run=AsyncMock(return_value="reply text"))
    bad_agent = SimpleNamespace(run=AsyncMock(side_effect=CompletionError("nope")))

    def _call(agent):
        processor = RequestProcessor(_deps(db=db, agent=agent))
        return asyncio.run(processor.process_agent_turn(
            _bot(), 123, user_id=55, sender_name=None, sender_username=None,
            is_group=True, build_payload=_payload, reply_context=None,
            send=AsyncMock(), success_log="ok", error_log_prefix="err",
        ))

    assert _call(ok_agent) is True
    assert _call(bad_agent) is False


def test_process_agent_turn_reports_through_on_error_when_given():
    db = SimpleNamespace(add_message=Mock())
    agent = SimpleNamespace(run=AsyncMock(side_effect=CompletionError("nope")))
    processor = RequestProcessor(_deps(db=db, agent=agent))
    send, on_error = AsyncMock(), AsyncMock()

    asyncio.run(processor.process_agent_turn(
        _bot(), 123, user_id=55, sender_name=None, sender_username=None,
        is_group=True, build_payload=_payload, reply_context=None, send=send,
        success_log="ok", error_log_prefix="err", on_error=on_error,
    ))

    on_error.assert_awaited_once_with("nope")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/bin/python3.12 -m pytest tests/test_request_processor.py -v`
Expected: the three new tests FAIL with `AttributeError: 'RequestProcessor' object has no attribute 'process_agent_turn'`; the existing tests PASS.

- [ ] **Step 3: Extract the method**

In `handlers/request_processor.py`, replace the whole `process` method with these two:

```python
    async def process(
        self,
        bot,
        message,
        *,
        user_id: int,
        sender_name: str,
        sender_username: str,
        is_group: bool,
        build_payload,
        reply_context: tuple[str, str] | None,
        generic_error_text: str,
        success_log: str,
        error_log_prefix: str,
        post_success=None,
    ) -> None:
        """Message-shaped entry point used by the text and photo handlers."""

        async def _on_error(text: str) -> None:
            await message.reply_text(text)

        await self.process_agent_turn(
            bot, message.chat_id,
            user_id=user_id, sender_name=sender_name,
            sender_username=sender_username, is_group=is_group,
            build_payload=build_payload, reply_context=reply_context,
            send=message.reply_text, telegram_message_id=message.message_id,
            success_log=success_log, error_log_prefix=error_log_prefix,
            on_error=_on_error, generic_error_text=generic_error_text,
            post_success=post_success,
        )

    async def process_agent_turn(
        self,
        bot,
        chat_id,
        *,
        user_id: int | None,
        sender_name: str | None,
        sender_username: str | None,
        is_group: bool,
        build_payload,
        reply_context: tuple[str, str] | None,
        send,
        success_log: str,
        error_log_prefix: str,
        telegram_message_id: int | None = None,
        on_error=None,
        generic_error_text: str = "",
        post_success=None,
    ) -> bool:
        """Audit-log -> agent.run -> audit-log -> send, shared by real messages
        and scheduled firings. `send` receives the reply text; `on_error`, when
        given, receives a user-facing failure string (a scheduled run passes
        none, because nobody is waiting on it).

        Returns True only if the reply was generated and sent. Exceptions are
        handled here, so the return value is the caller's only failure signal."""
        chat_id = str(chat_id)
        db = self._deps.db
        agent = self._deps.agent
        try:
            async with typing_action(bot, chat_id):
                content, token_count, human_message = await build_payload()
                db.add_message(
                    chat_id=chat_id, role="user", content=content,
                    user_id=user_id, message_id=telegram_message_id,
                    token_count=token_count,
                    sender_name=sender_name, sender_username=sender_username,
                    is_group_chat=is_group,
                )
                response = await agent.run(
                    chat_id, human_message, is_group, reply_context=reply_context
                )
                db.add_message(
                    chat_id=chat_id, role="assistant", content=response,
                    token_count=count_tokens(response), is_group_chat=is_group,
                )
            await send(response)
            logger.info(success_log)
            if post_success is not None:
                try:
                    await post_success()
                except Exception:
                    logger.exception(
                        "post_success hook failed for chat %s", chat_id
                    )
            return True
        except CompletionError as e:
            logger.warning("%s: %s", error_log_prefix, e.user_message)
            if on_error is not None:
                await on_error(e.user_message)
            return False
        except Exception as e:
            logger.error(f"{error_log_prefix}: {e}", exc_info=True)
            if on_error is not None:
                await on_error(generic_error_text)
            return False
```

- [ ] **Step 4: Run the whole handler suite to verify nothing regressed**

Run: `venv/bin/python3.12 -m pytest tests/test_request_processor.py tests/test_message_handlers.py tests/test_handlers_characterization.py -v`
Expected: PASS, all of them, including the three new tests. If a pre-existing test fails, the extraction changed behaviour — fix the extraction, not the test.

- [ ] **Step 5: Commit**

```bash
git add handlers/request_processor.py tests/test_request_processor.py
git commit -m "Extract process_agent_turn from RequestProcessor.process"
```

---

### Task 6: The runner

The background loop and its wiring. `ScheduledRunner.fire_due()` is separated from the sleep loop so it can be tested without waiting 30 seconds.

**Files:**
- Create: `scheduler.py`
- Modify: `bot.py:23-46`, `bot.py:88-90`
- Test: `tests/test_scheduler.py`

**Interfaces:**
- Consumes: Task 1's `next_run_from_cron`, `MAX_CONSECUTIVE_FAILURES`, `POLL_INTERVAL_SECONDS`; Task 2's `Database` schedule methods; Task 5's `process_agent_turn`.
- Produces:
  - `class ScheduledRunner(db, processor, bot, prompt_builder)`
  - `async fire_due(now: datetime | None = None) -> int` — returns how many fired
  - `async run_forever() -> None`
  - `start(app, db, processor, prompt_builder) -> asyncio.Task`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_scheduler.py`:

```python
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

    fired = asyncio.run(runner.fire_due())

    assert fired == 1
    processor.process_agent_turn.assert_awaited_once()
    db.mark_schedule_fired.assert_called_once()
    db.delete_schedule_by_id.assert_not_called()
    # Advanced to the next 08:00 SGT, i.e. a later instant than it fired at.
    assert db.mark_schedule_fired.call_args.kwargs["next_run_at"] > _Rec().next_run_at


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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/bin/python3.12 -m pytest tests/test_scheduler.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scheduler'`

- [ ] **Step 3: Write the runner**

Create `scheduler.py`:

```python
"""Background runner for scheduled prompts.

A 30s poll over `scheduled_prompts.next_run_at`, rather than an in-memory job
registry, so the database stays the single source of truth and a restart or
redeploy needs no re-registration. Everything fails quietly: no user is
waiting on a scheduled firing.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from agent import count_tokens
from scheduling import (
    MAX_CONSECUTIVE_FAILURES,
    POLL_INTERVAL_SECONDS,
    next_run_from_cron,
)

logger = logging.getLogger(__name__)


class ScheduledRunner:
    """Fires due schedules through the normal request path."""

    def __init__(self, db, processor, bot, prompt_builder):
        self._db = db
        self._processor = processor
        self._bot = bot
        self._prompt_builder = prompt_builder

    async def fire_due(self, now: datetime | None = None) -> int:
        """Run every schedule whose time has come. Returns how many succeeded."""
        now = now or datetime.now(timezone.utc)
        try:
            due = self._db.due_schedules(now)
        except Exception:
            logger.exception("Failed to read due schedules")
            return 0

        fired = 0
        for record in due:
            if await self._fire_one(record, now):
                fired += 1
        return fired

    async def _fire_one(self, record, now: datetime) -> bool:
        """Run one schedule and advance it. Never raises."""
        # Telegram group and supergroup chat ids are negative; DMs are not.
        is_group = str(record.chat_id).startswith("-")

        async def _build_payload():
            human = self._prompt_builder.to_lc_human_message(
                text=record.prompt, is_group=is_group, sender_name=None,
            )
            return record.prompt, count_tokens(record.prompt), human

        async def _send(text: str) -> None:
            await self._bot.send_message(chat_id=record.chat_id, text=text)

        ok = False
        try:
            ok = await self._processor.process_agent_turn(
                self._bot, record.chat_id,
                user_id=record.created_by, sender_name=None,
                sender_username=None, is_group=is_group,
                build_payload=_build_payload, reply_context=None,
                send=_send,
                success_log=f"Scheduled prompt #{record.id} ran for chat {record.chat_id}",
                error_log_prefix=f"Scheduled prompt #{record.id} failed",
            )
        except Exception:
            # process_agent_turn handles its own errors and reports them via its
            # return value; this guards only against it raising unexpectedly.
            logger.exception("Scheduled prompt #%s raised", record.id)
            ok = False

        self._advance(record, now, ok)
        return ok

    def _advance(self, record, now: datetime, ok: bool) -> None:
        """Persist the outcome. A one-shot is deleted either way — never retried."""
        try:
            if record.cron is None:
                self._db.delete_schedule_by_id(record.id)
                return
            next_run_at = next_run_from_cron(record.cron, after=now)
            if ok:
                self._db.mark_schedule_fired(
                    record.id, next_run_at=next_run_at, last_run_at=now
                )
            else:
                disable = (
                    record.consecutive_failures + 1 >= MAX_CONSECUTIVE_FAILURES
                )
                if disable:
                    logger.warning(
                        "Disabling schedule #%s after %s consecutive failures",
                        record.id, record.consecutive_failures + 1,
                    )
                self._db.record_schedule_failure(
                    record.id, next_run_at=next_run_at, disable=disable
                )
        except Exception:
            logger.exception("Failed to advance schedule #%s", record.id)

    async def run_forever(self) -> None:
        """Poll until cancelled. A failure in one pass never ends the loop."""
        logger.info(
            "Scheduled-prompt runner started (%ss poll)", POLL_INTERVAL_SECONDS
        )
        while True:
            try:
                await self.fire_due()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Scheduler pass failed")
            await asyncio.sleep(POLL_INTERVAL_SECONDS)


def start(app, db, processor, prompt_builder) -> asyncio.Task:
    """Launch the runner as a background task bound to the running app."""
    runner = ScheduledRunner(db, processor, app.bot, prompt_builder)
    return asyncio.create_task(runner.run_forever())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `venv/bin/python3.12 -m pytest tests/test_scheduler.py -v`
Expected: PASS, 10 tests.

- [ ] **Step 5: Expose the RequestProcessor from the handlers facade**

`init_handlers` builds a `RequestProcessor` as a local today, so the runner
cannot reach it. In `handlers/__init__.py`, add a global next to the two
existing ones:

```python
_processor: RequestProcessor | None = None
```

In `init_handlers`, add `_processor` to the `global` statement and assign it
instead of using a bare local:

```python
def init_handlers(cfg, database, bot_agent, prompt_bldr, username=None):
    """Initialize handler dependencies."""
    global _deps, _message_handlers, _command_handlers, _processor
    _deps = HandlerDependencies(
        config=cfg, db=database, agent=bot_agent,
        prompt_builder=prompt_bldr, bot_username=username,
    )
    _processor = RequestProcessor(_deps)
    _message_handlers = MessageHandlers(_deps, _processor)
    _command_handlers = CommandHandlers(_deps)
```

Then add the accessor below `is_main_authorized_user`:

```python
def get_request_processor() -> RequestProcessor:
    """The RequestProcessor built by init_handlers, for the scheduled runner."""
    assert _processor is not None, "init_handlers() must run before get_request_processor()"
    return _processor
```

- [ ] **Step 6: Wire the runner into bot.py**

In `bot.py`, add to the imports after `import handlers`:

```python
import scheduler
```

`prompt_builder` is a local inside `main()` today, but `post_init` needs it.
Extend the globals block:

```python
# Global instances
db = None
application = None
bot_agent = None
checkpointer_pool = None
prompt_builder = None
scheduler_task = None
```

In `main()`, add both names to the `global` statement:

```python
    global db, application, bot_agent, checkpointer_pool, prompt_builder, scheduler_task
```

The existing `prompt_builder = stack.prompt_builder` line then assigns the
global rather than a local — no other change needed there.

Start the runner at the end of `post_init`, after the closing log line:

```python
    logger.info("=" * 50)
    scheduler_task = scheduler.start(
        app, db, handlers.get_request_processor(), prompt_builder
    )
```

Add `global scheduler_task` as the first line of `post_init`, since it assigns
to it.

Cancel it in `post_shutdown`, before the db close:

```python
async def post_shutdown(app: Application):
    """Called before bot stops."""
    logger.info("Bot shutting down gracefully...")
    if scheduler_task:
        scheduler_task.cancel()
    # Close database connection pool
    if db:
        db.close()
    if checkpointer_pool:
        checkpointer_pool.close()
```

- [ ] **Step 7: Verify the bot still boots and the module compiles**

Run: `venv/bin/python3.12 -m py_compile *.py && venv/bin/python3.12 -m pytest tests/ -v`
Expected: compile clean, full suite PASS.

- [ ] **Step 8: Commit**

```bash
git add scheduler.py bot.py handlers/__init__.py tests/test_scheduler.py
git commit -m "Add background runner for scheduled prompts"
```

---

### Task 7: Documentation

`CLAUDE.md` currently asserts in two places that there is no scheduler. Both become false with this change and must be corrected in the same commit.

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`

- [ ] **Step 1: Find the stale claims**

Run: `grep -n "no scheduler" CLAUDE.md`
Expected: two hits — one in the Open Access Behavior section explaining lazy expiry, one elsewhere. Read both in context before editing.

- [ ] **Step 2: Correct the Open Access note**

The lazy-expiry *behaviour* does not change; only its stated justification does. Rewrite the sentence so it reads as a deliberate choice rather than a consequence of having no scheduler — e.g. "Expiry is evaluated lazily on read (`SettingsRepository.get_open_access()`) rather than by the scheduled-prompt runner, which only fires `scheduled_prompts` rows." Keep the rest of that bullet, including the guarantee that nothing ever authorizes past expiry.

- [ ] **Step 3: Add a Scheduled Prompts section**

Add to `CLAUDE.md` after the Voice Handling section:

```markdown
### Scheduled Prompts

- The agent has three tools — `schedule_prompt`, `list_schedules`,
  `cancel_schedule` — so scheduling is conversational. There is no `/schedule`
  command and no callback handler.
- Any user already authorized to chat can create a schedule. Abuse is bounded by
  guardrails, not by an admin gate: 10 schedules per chat, 500-character
  prompts, and a one-hour minimum interval. `created_by` records who made each.
- `schedule_prompt` takes a 5-field `cron` (recurring) or an ISO `at` (one-shot),
  both read in **Asia/Singapore**. `next_run_at` is stored UTC; every
  user-facing time is rendered SGT.
- The model writes the stored `prompt`, and it must be self-contained — it is
  replayed cold at fire time, so "give me..." and "what we just discussed" have
  to be resolved into absolutes at creation. The confirmation echoes the stored
  prompt so a bad rewrite is visible immediately.
- `scheduler.py` polls `scheduled_prompts` every 30 seconds from a task started
  in `bot.py`'s `post_init`. The database is the only source of truth, so
  restarts and redeploys need no re-registration.
- A firing runs through `RequestProcessor.process_agent_turn` — the same path as
  a real message — so compaction, trimming, context and persistence are
  identical. It uses whatever `active_model` and `active_personality` are live
  at fire time, and its `messages` rows obey `MESSAGE_RETENTION_DAYS`.
- Missed firings during downtime are skipped, never backfilled. A failed firing
  posts nothing to the chat, and a schedule is auto-disabled after 5 consecutive
  failures. A one-shot is deleted after it fires, successfully or not.
```

- [ ] **Step 4: Update the schema and testing lists**

In `CLAUDE.md`'s "Database Schema" expected-tables list, add `scheduled_prompts`. In the same section's "Important details", add: "`scheduled_prompts` is the durable store for agent-created schedules; `cron IS NULL` marks a one-shot".

In the "Unit Tests" list, add:

```markdown
- `scheduling.create_schedule()` / `next_run_from_cron()` / `render_schedule_list()` — cadence math, guardrails, and tool output (`tests/test_scheduling.py`)
- `scheduler.ScheduledRunner.fire_due()` — due selection, advancement, and the failure cutout (`tests/test_scheduler.py`)
```

- [ ] **Step 5: Add a README usage example**

In `README.md`'s usage section, add an example showing conversational scheduling:

```markdown
### Scheduling

Ask in plain English — there is no command:

> chatgpt post a summary of the day's messages every weekday at 6pm

The bot confirms with an id, the next run time in SGT, and the exact prompt it
stored. "chatgpt what's scheduled here" lists them; "chatgpt cancel 7" removes
one. Times are Asia/Singapore, and a schedule fires at most once an hour.
```

- [ ] **Step 6: Verify nothing else went stale**

Run: `grep -rn "no scheduler\|there is no scheduler" CLAUDE.md README.md AGENTS.md`
Expected: no hits.

- [ ] **Step 7: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "Document scheduled prompts"
```

---

### Task 8: Full verification and PR

- [ ] **Step 1: Run the whole suite**

Run: `venv/bin/python3.12 -m py_compile *.py && venv/bin/python3.12 -m pytest tests/ -v`
Expected: compile clean, every test PASS. Record the count.

- [ ] **Step 2: Exercise it in the CLI simulator**

Run: `venv/bin/python3.12 scripts/chat_cli.py --chat-id test`

Then type: `chatgpt remind me to check the deploy at 9am tomorrow`
Expected: a confirmation naming an id, a next run in SGT, and the stored prompt — and the stored prompt should be self-contained ("Check the deploy"), not "remind me to check the deploy".

Then type: `chatgpt what's scheduled here`
Expected: the schedule listed with the same id.

Then type: `chatgpt cancel <that id>`
Expected: a cancellation naming the label.

- [ ] **Step 3: Push and open the PR**

```bash
git push -u origin dev
```

The push auto-deploys to the Railway `dev` environment, whose `preDeployCommand` applies migration `0005`. Watch the deploy, then verify against the real dev bot: schedule something a few minutes out and confirm it fires.

- [ ] **Step 4: Promote by PR**

```bash
gh pr create --base main --head dev \
  --title "Add agent-scheduled prompts" \
  --body "$(cat <<'BODY'
## What changed

Implements #75 with an agent tool surface instead of the proposed `/schedule`
command. The agent gets `schedule_prompt`, `list_schedules` and
`cancel_schedule`, so scheduling is conversational — the model translates
"every morning at 8" into cron itself, which removes the hand-rolled time
grammar and the inline keyboard the issue proposed as its fallback.

Firings run through `RequestProcessor.process_agent_turn`, the same path a real
message takes, so compaction, trimming, context and persistence are identical.
A 30s poll loop in `scheduler.py` drives it, with `scheduled_prompts` as the
single source of truth so restarts need no re-registration.

## Env / config changes

- New dependency: `croniter>=3.0`.
- New migration `0005_scheduled_prompts` — applied by the existing Railway
  `preDeployCommand`.
- **No new environment variables.** Guardrails are module constants.

## Notable behaviour

- Any user authorized to chat can schedule, bounded by 10 schedules per chat,
  a 500-character prompt cap, and a one-hour minimum interval.
- All times are Asia/Singapore; `next_run_at` is stored UTC.
- Missed firings are skipped, never backfilled. A schedule auto-disables after
  5 consecutive failures.

## Validation

- `python3 -m py_compile *.py` clean.
- Full `pytest tests/ -v` suite passing, including the new
  `tests/test_scheduling.py` and `tests/test_scheduler.py`.
- Exercised create / list / cancel in `scripts/chat_cli.py`.
- Verified a real firing against the dev bot on the Railway `dev` environment.

Closes #75.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
BODY
)"
```

Use a plain merge commit, not squash, for the `dev` → `main` promotion, so
`main`'s history stays a readable sequence of releases.
