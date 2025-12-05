"""Telegram message handlers and bot logic."""
import logging
import re
import random
from telegram import Update
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)

# Global instances (will be set by bot.py)
config = None
db = None
token_manager = None
openai_client = None


def init_handlers(cfg, database, token_mgr, openai_cl):
    """Initialize handler dependencies."""
    global config, db, token_manager, openai_client
    config = cfg
    db = database
    token_manager = token_mgr
    openai_client = openai_cl


def is_authorized(user_id: int) -> bool:
    """Check if user is authorized to use the bot."""
    # Check if user is the main authorized user
    if str(user_id) == config.AUTHORIZED_USER_ID:
        return True

    # Check if user has been granted access
    return db.is_user_granted(user_id)


def is_main_authorized_user(user_id: int) -> bool:
    """Check if user is the main authorized user (for admin commands)."""
    return str(user_id) == config.AUTHORIZED_USER_ID


def extract_keyword(text: str) -> tuple[bool, str]:
    """
    Check for activation keyword and extract prompt.

    Args:
        text: Message text

    Returns:
        Tuple of (has_keyword, prompt_without_keyword)
    """
    text_lower = text.lower()

    # Check if "chatgpt" keyword is present (case-insensitive)
    if "chatgpt" not in text_lower:
        return False, ""

    # Remove keyword from message (preserve case of rest)
    # Use word boundary to avoid matching "chatgpt123" etc.
    cleaned = re.sub(r'\bchatgpt\b', '', text, flags=re.IGNORECASE)
    prompt = cleaned.strip()

    return True, prompt


async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Main message handler for all text messages."""

    # 1. Extract message details
    message = update.message
    if not message or not message.text:
        return  # Ignore non-text messages

    user_id = message.from_user.id
    chat_id = str(message.chat_id)
    is_group = message.chat.type in ["group", "supergroup"]

    # Get sender information
    sender_name = message.from_user.first_name or "Unknown"
    sender_username = message.from_user.username

    # 2. Check for activation keyword
    has_keyword, prompt = extract_keyword(message.text)

    # In group chats, store ALL messages for context (even without keyword)
    if is_group and not has_keyword:
        # Store this message for context
        try:
            message_tokens = token_manager.count_message_tokens("user", message.text)
            db.add_message(
                chat_id=chat_id,
                role="user",
                content=message.text,
                user_id=user_id,
                message_id=message.message_id,
                token_count=message_tokens,
                sender_name=sender_name,
                sender_username=sender_username,
                is_group_chat=True,
            )
            logger.debug(f"Stored group message from {sender_name} for context")

            # Periodically cleanup old messages (10% chance)
            if random.random() < 0.1:
                db.cleanup_old_group_messages(chat_id, config.MAX_GROUP_CONTEXT_MESSAGES)

        except Exception as e:
            logger.error(f"Failed to store group message: {e}")
        return  # Don't process, just store for context

    # If no keyword at all (private chat or group), ignore
    if not has_keyword:
        return

    # 3. Authorization check
    if not is_authorized(user_id):
        await message.reply_text("Sorry, you have no access to me.")
        return

    # 4. Handle empty prompt
    if not prompt:
        await message.reply_text("Yes, what's your request?")
        return

    # 5. Process request
    await process_request(message, prompt, user_id, sender_name, sender_username, is_group)


async def process_request(message, prompt: str, user_id: int, sender_name: str, sender_username: str, is_group: bool):
    """Process GPT request with context management."""

    chat_id = str(message.chat_id)
    message_id = message.message_id

    try:
        # 1. Count tokens in user's message
        user_tokens = token_manager.count_message_tokens("user", prompt)

        # 2. Store user message
        db.add_message(
            chat_id=chat_id,
            role="user",
            content=prompt,
            user_id=user_id,
            message_id=message_id,
            token_count=user_tokens,
            sender_name=sender_name,
            sender_username=sender_username,
            is_group_chat=is_group,
        )

        # 3. Get conversation history within token budget
        max_tokens = config.MAX_CONTEXT_TOKENS
        messages = db.get_messages_by_tokens(chat_id, max_tokens)

        # 4. Final trim to ensure we fit (accounting for response)
        messages = token_manager.trim_to_fit(messages, reserve_tokens=1000)

        logger.info(
            f"Processing request for chat {chat_id}: "
            f"{len(messages)} messages, {user_tokens} tokens"
        )

        # 5. Get completion from OpenAI
        response = await openai_client.get_completion(messages, is_group)

        # 6. Count and store assistant's response
        assistant_tokens = token_manager.count_message_tokens("assistant", response)
        db.add_message(
            chat_id=chat_id,
            role="assistant",
            content=response,
            token_count=assistant_tokens,
            is_group_chat=is_group,
        )

        # 7. Send response to user
        await message.reply_text(response)

        logger.info(
            f"Response sent for chat {chat_id}: {assistant_tokens} tokens"
        )

    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        await message.reply_text(
            "Sorry, I encountered an error processing your request. "
            "Please try again."
        )


async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle photo messages with OpenAI vision."""

    # 1. Extract message details
    message = update.message
    if not message or not message.photo:
        return

    user_id = message.from_user.id
    chat_id = str(message.chat_id)
    is_group = message.chat.type in ["group", "supergroup"]

    # Get sender information
    sender_name = message.from_user.first_name or "Unknown"
    sender_username = message.from_user.username

    # 2. Check for activation keyword in caption
    caption = message.caption or ""
    has_keyword, prompt = extract_keyword(caption) if caption else (False, "")

    # In group chats without keyword, ignore
    if is_group and not has_keyword:
        return

    # If no keyword at all, ignore
    if not has_keyword:
        return

    # 3. Authorization check
    if not is_authorized(user_id):
        await message.reply_text("Sorry, you have no access to me.")
        return

    # 4. Process image request
    await process_image_request(
        message, prompt, user_id, sender_name, sender_username, is_group
    )


async def process_image_request(
    message,
    prompt: str,
    user_id: int,
    sender_name: str,
    sender_username: str,
    is_group: bool
):
    """Process image request with vision model - split storage approach."""

    chat_id = str(message.chat_id)
    message_id = message.message_id

    try:
        # 1. Download image as bytes (in-memory only)
        import base64
        photo = message.photo[-1]  # Get highest resolution
        photo_file = await photo.get_file()
        photo_bytes = await photo_file.download_as_bytearray()

        # 2. Convert to base64 data URL
        base64_image = base64.b64encode(photo_bytes).decode('utf-8')
        image_data_url = f"data:image/jpeg;base64,{base64_image}"

        # 3. Prepare content text
        content_text = prompt if prompt else ""

        # 4. Store caption as separate text message (preserved in future context)
        # Prefix with [image] to indicate image was part of this turn
        if content_text:
            caption_with_marker = f"[image] {content_text}"
        else:
            caption_with_marker = "[image]"

        caption_tokens = token_manager.count_message_tokens("user", caption_with_marker)
        db.add_message(
            chat_id=chat_id,
            role="user",
            content=caption_with_marker,
            user_id=user_id,
            message_id=message_id,
            token_count=caption_tokens,
            sender_name=sender_name,
            sender_username=sender_username,
            is_group_chat=is_group,
            has_image=False,  # This is the text part, will be included in context
        )

        # 5. Get conversation history (now includes caption from step 4)
        max_tokens = config.MAX_CONTEXT_TOKENS
        messages = db.get_messages_by_tokens(chat_id, max_tokens)

        # 7. Trim text history conservatively (reserve more for image)
        messages = token_manager.trim_to_fit(messages, reserve_tokens=3000)

        # 8. Build multimodal message array for OpenAI
        # Use the original content_text (without [image] prefix) for API call
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": content_text if content_text else "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data_url,
                        "detail": "auto"
                    }
                }
            ],
            "sender_name": sender_name,
            "sender_username": sender_username,
            "is_group_chat": is_group,
        })

        logger.info(
            f"Processing image request for chat {chat_id}: "
            f"{len(messages)} messages in context, caption={caption_tokens} tokens"
        )

        # 9. Call OpenAI with vision support
        response_text = await openai_client.get_completion(messages, is_group)

        # 10. Store assistant response with tiktoken-counted tokens
        response_tokens = token_manager.count_message_tokens("assistant", response_text)
        db.add_message(
            chat_id=chat_id,
            role="assistant",
            content=response_text,
            token_count=response_tokens,
            is_group_chat=is_group,
        )

        # 11. Send response to user
        await message.reply_text(response_text)

        logger.info(
            f"Image processed for chat {chat_id}: caption={caption_tokens} tokens"
        )

    except Exception as e:
        logger.error(f"Error processing image: {e}", exc_info=True)
        await message.reply_text(
            "Sorry, I encountered an error processing your image. "
            "Please try again."
        )


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Clear conversation history for current chat."""

    user_id = update.message.from_user.id
    if not is_authorized(user_id):
        await update.message.reply_text("Sorry, you have no access to me.")
        return

    chat_id = str(update.message.chat_id)

    try:
        db.clear_history(chat_id)
        await update.message.reply_text(
            "✅ Conversation history cleared for this chat."
        )
        logger.info(f"History cleared for chat {chat_id}")

    except Exception as e:
        logger.error(f"Error clearing history: {e}", exc_info=True)
        await update.message.reply_text(
            "❌ Failed to clear history. Please try again."
        )


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show statistics for current chat."""

    user_id = update.message.from_user.id
    if not is_authorized(user_id):
        await update.message.reply_text("Sorry, you have no access to me.")
        return

    chat_id = str(update.message.chat_id)

    try:
        stats = db.get_stats(chat_id)

        # Format timestamp for display
        first_msg = stats["first_message"]
        if first_msg != "N/A":
            first_msg = first_msg.split("T")[0]  # Just the date

        await update.message.reply_text(
            f"📊 Chat Statistics:\n"
            f"Messages: {stats['total_messages']}\n"
            f"Total tokens: {stats['total_tokens']:,}\n"
            f"Since: {first_msg}"
        )

        logger.info(f"Stats shown for chat {chat_id}")

    except Exception as e:
        logger.error(f"Error getting stats: {e}", exc_info=True)
        await update.message.reply_text(
            "❌ Failed to retrieve statistics. Please try again."
        )


async def grant_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Grant access to a user (main authorized user only)."""

    user_id = update.message.from_user.id
    if not is_main_authorized_user(user_id):
        await update.message.reply_text("Sorry, only the main authorized user can grant access.")
        return

    # Check if user_id argument is provided
    if not context.args or len(context.args) == 0:
        await update.message.reply_text(
            "❌ Usage: /grant <user_id>\n"
            "Example: /grant 123456789"
        )
        return

    try:
        # Parse user_id from argument
        target_user_id = int(context.args[0])
        if target_user_id <= 0:
            await update.message.reply_text(
                "❌ Invalid user ID. User IDs must be positive integers."
            )
            return

        # Check if trying to grant access to self
        if str(target_user_id) == config.AUTHORIZED_USER_ID:
            await update.message.reply_text(
                "ℹ️ You are already the main authorized user."
            )
            return

        # Grant access
        was_granted = db.grant_access(target_user_id)

        if was_granted:
            await update.message.reply_text(
                f"✅ Access granted to user {target_user_id}.\n"
                f"They can now use the bot with 'chatgpt' keyword."
            )
            logger.info(f"User {user_id} granted access to {target_user_id}")
        else:
            await update.message.reply_text(
                f"ℹ️ User {target_user_id} already has access."
            )

    except ValueError:
        await update.message.reply_text(
            "❌ Invalid user ID. Please provide a numeric user ID.\n"
            "Example: /grant 123456789"
        )
    except Exception as e:
        logger.error(f"Error granting access: {e}", exc_info=True)
        await update.message.reply_text(
            "❌ Failed to grant access. Please try again."
        )


async def revoke_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Revoke access from a user (main authorized user only)."""

    user_id = update.message.from_user.id
    if not is_main_authorized_user(user_id):
        await update.message.reply_text("Sorry, only the main authorized user can revoke access.")
        return

    # Check if user_id argument is provided
    if not context.args or len(context.args) == 0:
        await update.message.reply_text(
            "❌ Usage: /revoke <user_id>\n"
            "Example: /revoke 123456789"
        )
        return

    try:
        # Parse user_id from argument
        target_user_id = int(context.args[0])

        # Validate that the user ID is positive
        if target_user_id <= 0:
            await update.message.reply_text(
                "❌ Invalid user ID. User IDs must be positive integers."
            )
            return
        # Check if trying to revoke access from self
        if str(target_user_id) == config.AUTHORIZED_USER_ID:
            await update.message.reply_text(
                "❌ Cannot revoke access from the main authorized user."
            )
            return

        # Revoke access
        was_revoked = db.revoke_access(target_user_id)

        if was_revoked:
            await update.message.reply_text(
                f"✅ Access revoked from user {target_user_id}."
            )
            logger.info(f"User {user_id} revoked access from {target_user_id}")
        else:
            await update.message.reply_text(
                f"ℹ️ User {target_user_id} didn't have access."
            )

    except ValueError:
        await update.message.reply_text(
            "❌ Invalid user ID. Please provide a numeric user ID.\n"
            "Example: /revoke 123456789"
        )
    except Exception as e:
        logger.error(f"Error revoking access: {e}", exc_info=True)
        await update.message.reply_text(
            "❌ Failed to revoke access. Please try again."
        )


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Global error handler for unhandled exceptions."""

    logger.error("Exception while handling update:", exc_info=context.error)

    # Try to notify user if possible
    if update and update.message:
        try:
            await update.message.reply_text(
                "An error occurred while processing your request. "
                "The error has been logged."
            )
        except Exception:
            # If we can't send message, just log
            pass
