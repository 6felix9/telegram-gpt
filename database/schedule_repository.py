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
