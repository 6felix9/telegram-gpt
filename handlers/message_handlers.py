"""Telegram-facing text/photo intake: activation parsing, auth gate, and
handing off to the shared request processor."""
import base64
import logging
import re
import uuid

from agent import count_tokens

from .authorization import is_authorized
from .handler_deps import HandlerDependencies
from .request_processor import RequestProcessor

logger = logging.getLogger(__name__)


def extract_keyword(text: str, bot_username: str | None = None) -> tuple[bool, str]:
    """
    Check for activation keyword or @mention and extract prompt.

    Args:
        text: Message text
        bot_username: Bot's username (without @) for mention detection

    Returns:
        Tuple of (has_keyword, prompt_without_keyword)
    """
    if not text:
        return False, ""

    text_lower = text.lower()
    has_activation = False
    cleaned = text

    if "chatgpt" in text_lower:
        has_activation = True
        cleaned = re.sub(r'\bchatgpt\b', '', cleaned, flags=re.IGNORECASE)

    if bot_username:
        mention = f"@{bot_username}"
        if mention.lower() in text_lower:
            has_activation = True
            cleaned = re.sub(rf'@{re.escape(bot_username)}', '', cleaned, flags=re.IGNORECASE)

    prompt = cleaned.strip()
    return has_activation, prompt


def extract_reply_data(message) -> tuple[str, str] | None:
    """
    Extracts raw data from the message being replied to.

    Args:
        message: Telegram message object

    Returns:
        Tuple of (sender_name, content) or None if no valid reply
    """
    if not message.reply_to_message:
        return None

    reply = message.reply_to_message
    content = reply.text or reply.caption or ""
    if not content:
        return None

    sender = reply.from_user.first_name if reply.from_user else "Unknown"
    return (sender, content)


class MessageHandlers:
    """Text and photo Telegram handlers, bound to an explicit dependency set."""

    def __init__(self, deps: HandlerDependencies, processor: RequestProcessor):
        self._deps = deps
        self._processor = processor

    async def message_handler(self, update, context):
        message = update.message
        if not message or not message.text:
            return

        user_id = message.from_user.id
        chat_id = str(message.chat_id)
        is_group = message.chat.type in ["group", "supergroup"]
        sender_name = message.from_user.first_name or "Unknown"
        sender_username = message.from_user.username

        has_keyword, prompt = extract_keyword(message.text, self._deps.bot_username)

        if not has_keyword:
            try:
                self._deps.db.add_message(
                    chat_id=chat_id, role="user", content=message.text,
                    user_id=user_id, message_id=message.message_id,
                    token_count=count_tokens(message.text),
                    sender_name=sender_name, sender_username=sender_username,
                    is_group_chat=is_group,
                )
                self._deps.agent.append_context_message(
                    chat_id,
                    self._deps.prompt_builder.to_lc_human_message(
                        text=message.text, is_group=is_group, sender_name=sender_name),
                )
            except Exception as e:
                logger.error(f"Failed to store context message: {e}")
            return

        if not is_authorized(user_id, self._deps.config, self._deps.db):
            await message.reply_text("Sorry, you have no access to me.")
            return

        reply_data = extract_reply_data(message)
        reply = message.reply_to_message
        if reply is not None and getattr(reply, "photo", None):
            image_ref = await self._resolve_reply_image_ref(chat_id, reply, is_group)
            if image_ref is not None:
                reply_data = image_ref

        if not prompt:
            await message.reply_text("Yes, what's your request?")
            return

        async def _build_payload():
            human = self._deps.prompt_builder.to_lc_human_message(
                text=prompt, is_group=is_group, sender_name=sender_name)
            return message.text, count_tokens(prompt), human

        await self._processor.process(
            context.bot, message,
            user_id=user_id, sender_name=sender_name, sender_username=sender_username,
            is_group=is_group, build_payload=_build_payload, reply_context=reply_data,
            generic_error_text=(
                "Sorry, I encountered an error processing your request. Please try again."
            ),
            success_log=f"Response sent for chat {chat_id}",
            error_log_prefix="Error processing request",
        )

    async def photo_handler(self, update, context):
        message = update.message
        if not message or not message.photo:
            return

        user_id = message.from_user.id
        chat_id = str(message.chat_id)
        is_group = message.chat.type in ["group", "supergroup"]
        sender_name = message.from_user.first_name or "Unknown"
        sender_username = message.from_user.username

        caption = message.caption or ""
        has_keyword, prompt = (
            extract_keyword(caption, self._deps.bot_username) if caption else (False, "")
        )

        if not has_keyword:
            await self._passive_persist(
                message, chat_id, is_group, sender_name, sender_username, user_id
            )
            return

        if not is_authorized(user_id, self._deps.config, self._deps.db):
            await message.reply_text("Sorry, you have no access to me.")
            return

        reply_data = extract_reply_data(message)
        image_message_id = str(uuid.uuid4())
        captured: dict[str, str] = {}

        async def _build_payload():
            image_data_url = await self._photo_data_url(message.photo[-1])
            caption_marker = f"[image] {message.caption}" if message.caption else "[image]"
            human = self._deps.prompt_builder.to_lc_human_message(
                text=prompt, is_group=is_group, sender_name=sender_name,
                image_data_url=image_data_url, message_id=image_message_id)
            captured["image_data_url"] = image_data_url
            return caption_marker, count_tokens(caption_marker), human

        async def _post_success():
            image_data_url = captured.get("image_data_url")
            if not image_data_url:
                return
            await self._deps.agent.persist_image(
                chat_id=chat_id,
                image_message_id=image_message_id,
                image_data_url=image_data_url,
                mime_type="image/jpeg",
                caption=message.caption,
                telegram_message_id=message.message_id,
                is_group=is_group,
                sender_name=sender_name,
            )

        await self._processor.process(
            context.bot, message,
            user_id=user_id, sender_name=sender_name, sender_username=sender_username,
            is_group=is_group, build_payload=_build_payload, reply_context=reply_data,
            generic_error_text=(
                "Sorry, I encountered an error processing your image. Please try again."
            ),
            success_log=f"Image processed for chat {chat_id}",
            error_log_prefix="Error processing image",
            post_success=_post_success,
        )

    @staticmethod
    async def _photo_data_url(photo) -> str:
        """Download a Telegram PhotoSize and return it as a JPEG data URL."""
        photo_file = await photo.get_file()
        photo_bytes = await photo_file.download_as_bytearray()
        base64_image = base64.b64encode(photo_bytes).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_image}"

    async def _passive_persist(
        self, message, chat_id, is_group, sender_name, sender_username, user_id
    ) -> None:
        """Store and summarize a photo that did not trigger a reply, so it is
        retrievable later. Mirrors non-triggering text: an audit `messages` row
        plus a checkpoint [image #N] marker (a fresh id appends it). Fully
        fail-open — never surfaced to the user and no reply is sent."""
        try:
            image_data_url = await self._photo_data_url(message.photo[-1])
            caption_marker = (
                f"[image] {message.caption}" if message.caption else "[image]"
            )
            self._deps.db.add_message(
                chat_id=chat_id, role="user", content=caption_marker,
                user_id=user_id, message_id=message.message_id,
                token_count=count_tokens(caption_marker),
                sender_name=sender_name, sender_username=sender_username,
                is_group_chat=is_group,
            )
            await self._deps.agent.persist_image(
                chat_id=chat_id,
                image_message_id=str(uuid.uuid4()),
                image_data_url=image_data_url,
                mime_type="image/jpeg",
                caption=message.caption,
                telegram_message_id=message.message_id,
                is_group=is_group,
                sender_name=sender_name,
            )
        except Exception:
            logger.exception("Failed to passively persist photo for chat %s", chat_id)

    async def _resolve_reply_image_ref(
        self, chat_id, reply, is_group: bool
    ) -> tuple[str, str] | None:
        """Resolve the persisted [image #N] a triggering message is replying to,
        so the agent can pull it back with get_image(N). Falls back to persisting
        the replied photo on the fly (legacy photos predating passive ingest).
        Returns (poster_name, "[image #N] <summary>") or None on any failure."""
        poster = reply.from_user.first_name if reply.from_user else "Unknown"
        try:
            record = self._deps.db.get_image_by_message_id(chat_id, reply.message_id)
            if record is None:
                image_data_url = await self._photo_data_url(reply.photo[-1])
                image_id = await self._deps.agent.persist_image(
                    chat_id=chat_id,
                    image_message_id=str(uuid.uuid4()),
                    image_data_url=image_data_url,
                    mime_type="image/jpeg",
                    caption=reply.caption,
                    telegram_message_id=reply.message_id,
                    is_group=is_group,
                    sender_name=poster,
                )
                if image_id is None:
                    return None
                record = self._deps.db.get_image_by_message_id(
                    chat_id, reply.message_id
                )
            if record is None:
                return None
            return (poster, f"[image #{record.id}] {record.summary}")
        except Exception:
            logger.exception("Failed to resolve replied image for chat %s", chat_id)
            return None
