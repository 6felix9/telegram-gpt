"""Telegram message handlers and bot logic."""
import logging
import re
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

    # 2. Check for activation keyword
    has_keyword, prompt = extract_keyword(message.text)
    if not has_keyword:
        return  # Silently ignore messages without keyword

    # 3. Authorization check
    user_id = message.from_user.id
    if not is_authorized(user_id):
        await message.reply_text("Sorry, you have no access to me.")
        return

    # 4. Handle empty prompt
    if not prompt:
        await message.reply_text("Yes, what's your request?")
        return

    # 5. Process request
    await process_request(message, prompt, user_id)


async def process_request(message, prompt: str, user_id: int):
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
        response = await openai_client.get_completion(messages)

        # 6. Count and store assistant's response
        assistant_tokens = token_manager.count_message_tokens("assistant", response)
        db.add_message(
            chat_id=chat_id,
            role="assistant",
            content=response,
            token_count=assistant_tokens,
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
