"""Shared agent-turn workflow for triggering text and image requests."""
import asyncio
import logging
from contextlib import asynccontextmanager

from telegram.constants import ChatAction

from agent import CompletionError, count_tokens
from handler_deps import HandlerDependencies

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
    ) -> None:
        chat_id = str(message.chat_id)
        db = self._deps.db
        agent = self._deps.agent
        try:
            async with typing_action(bot, chat_id):
                content, token_count, human_message = await build_payload()
                db.add_message(
                    chat_id=chat_id, role="user", content=content,
                    user_id=user_id, message_id=message.message_id,
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
            await message.reply_text(response)
            logger.info(success_log)
        except CompletionError as e:
            await message.reply_text(e.user_message)
        except Exception as e:
            logger.error(f"{error_log_prefix}: {e}", exc_info=True)
            await message.reply_text(generic_error_text)
