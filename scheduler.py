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
