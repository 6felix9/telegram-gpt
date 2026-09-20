"""Shared agent-turn workflow for triggering text and image requests."""
import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager

from telegram.constants import ChatAction

from agent import CompletionError, count_tokens

from .handler_deps import HandlerDependencies

logger = logging.getLogger(__name__)


@asynccontextmanager
async def typing_action(bot, chat_id: str):
    """Keep the Telegram typing indicator active for the duration of the block."""
    async def _loop():
        while True:
            try:
                await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            except Exception as e:
                logger.debug(f"Failed to send typing action: {e}")
            await asyncio.sleep(4)
    task = asyncio.create_task(_loop())
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


class RequestProcessor:
    """Runs the audit-log -> agent.run -> audit-log -> reply workflow shared
    by text and image requests, and maps failures to a user-facing reply."""

    def __init__(self, deps: HandlerDependencies):
        self._deps = deps

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
                    chat_id, human_message, is_group,
                    reply_context=reply_context, user_id=user_id,
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
