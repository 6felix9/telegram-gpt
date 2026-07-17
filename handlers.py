"""Telegram message handlers and bot logic."""
import asyncio
import logging
import re
import base64
from contextlib import asynccontextmanager
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ContextTypes
from telegram.helpers import escape_markdown
from agent import MODEL_PROVIDERS, CompletionError, count_tokens

logger = logging.getLogger(__name__)

# Global instances (will be set by bot.py)
config = None
db = None
agent = None
prompt_builder = None
bot_username = None


def init_handlers(cfg, database, bot_agent, prompt_bldr, username=None):
    """Initialize handler dependencies."""
    global config, db, agent, prompt_builder, bot_username
    config = cfg
    db = database
    agent = bot_agent
    prompt_builder = prompt_bldr
    bot_username = username


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


def extract_keyword(text: str, bot_username: str = None) -> tuple[bool, str]:
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

    # Check for "chatgpt" keyword
    if "chatgpt" in text_lower:
        has_activation = True
        # Remove keyword from message (preserve case of rest)
        # Use word boundary to avoid matching "chatgpt123" etc.
        cleaned = re.sub(r'\bchatgpt\b', '', cleaned, flags=re.IGNORECASE)

    # Check for @mention if bot_username provided
    if bot_username:
        mention = f"@{bot_username}"
        if mention.lower() in text_lower:
            has_activation = True
            # Remove @mention from message (case-insensitive)
            cleaned = re.sub(rf'@{re.escape(bot_username)}', '', cleaned, flags=re.IGNORECASE)

    # Clean up extra whitespace
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
    # Check if this is a reply
    if not message.reply_to_message:
        return None

    reply = message.reply_to_message

    # Extract content (prioritize text, then caption for media)
    content = reply.text or reply.caption or ""

    # If no text content, return None
    if not content:
        return None

    # Get sender name
    sender = reply.from_user.first_name if reply.from_user else "Unknown"

    return (sender, content)


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
    has_keyword, prompt = extract_keyword(message.text, bot_username)

    # Store non-triggering text for context (private and group); do not reply.
    if not has_keyword:
        try:
            db.add_message(
                chat_id=chat_id, role="user", content=message.text,
                user_id=user_id, message_id=message.message_id,
                token_count=count_tokens(message.text),
                sender_name=sender_name, sender_username=sender_username,
                is_group_chat=is_group,
            )
            await agent.append_context_message(
                chat_id,
                prompt_builder.to_lc_human_message(
                    text=message.text, is_group=is_group, sender_name=sender_name),
            )
            # Messages are currently stored without a database retention limit.
            # TODO: add a coordinated cleanup policy for stored messages
            # and checkpoint state.
            # if is_group and random.random() < 0.1:
            #     db.cleanup_old_group_messages(
            #         chat_id, config.MAX_GROUP_CONTEXT_MESSAGES
            #     )
        except Exception as e:
            logger.error(f"Failed to store context message: {e}")
        return

    # 3. Authorization check
    if not is_authorized(user_id):
        await message.reply_text("Sorry, you have no access to me.")
        return

    # 4. Extract reply context if it exists
    reply_data = extract_reply_data(message)

    # 5. Handle empty prompt
    if not prompt:
        await message.reply_text("Yes, what's your request?")
        return

    # 6. Process request
    await process_request(
        context.bot, message, prompt, user_id, sender_name, sender_username, is_group, reply_data
    )


async def process_request(
    bot,
    message,
    prompt: str,
    user_id: int,
    sender_name: str,
    sender_username: str,
    is_group: bool,
    reply_context: tuple[str, str] | None = None,
):
    """Process a triggering text request through the agent."""
    chat_id = str(message.chat_id)
    try:
        async with typing_action(bot, chat_id):
            # Audit-log the user message (context lives in the checkpoint).
            db.add_message(
                chat_id=chat_id, role="user", content=message.text,
                user_id=user_id, message_id=message.message_id,
                token_count=count_tokens(prompt),
                sender_name=sender_name, sender_username=sender_username,
                is_group_chat=is_group,
            )

            human = prompt_builder.to_lc_human_message(
                text=prompt, is_group=is_group, sender_name=sender_name)
            response = await agent.run(chat_id, human, is_group, reply_context=reply_context)

            db.add_message(
                chat_id=chat_id, role="assistant", content=response,
                token_count=count_tokens(response), is_group_chat=is_group,
            )
        await message.reply_text(response)
        logger.info(f"Response sent for chat {chat_id}")

    except CompletionError as e:
        await message.reply_text(e.user_message)
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        await message.reply_text(
            "Sorry, I encountered an error processing your request. Please try again."
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
    has_keyword, prompt = extract_keyword(caption, bot_username) if caption else (False, "")

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

    # 4. Extract reply context if it exists
    reply_data = extract_reply_data(message)

    # 5. Process image request
    await process_image_request(
        context.bot, message, prompt, user_id, sender_name, sender_username, is_group, reply_data
    )


async def process_image_request(
    bot,
    message,
    prompt: str,
    user_id: int,
    sender_name: str,
    sender_username: str,
    is_group: bool,
    reply_context: tuple[str, str] | None = None,
):
    """Process a triggering image request through the agent."""
    chat_id = str(message.chat_id)
    try:
        async with typing_action(bot, chat_id):
            photo = message.photo[-1]
            photo_file = await photo.get_file()
            photo_bytes = await photo_file.download_as_bytearray()
            base64_image = base64.b64encode(photo_bytes).decode("utf-8")
            image_data_url = f"data:image/jpeg;base64,{base64_image}"

            # Audit-log a text marker only (never the base64 payload).
            caption_marker = f"[image] {message.caption}" if message.caption else "[image]"
            db.add_message(
                chat_id=chat_id, role="user", content=caption_marker,
                user_id=user_id, message_id=message.message_id,
                token_count=count_tokens(caption_marker),
                sender_name=sender_name, sender_username=sender_username,
                is_group_chat=is_group,
            )

            human = prompt_builder.to_lc_human_message(
                text=prompt, is_group=is_group, sender_name=sender_name,
                image_data_url=image_data_url)
            response = await agent.run(chat_id, human, is_group, reply_context=reply_context)

            db.add_message(
                chat_id=chat_id, role="assistant", content=response,
                token_count=count_tokens(response), is_group_chat=is_group,
            )
        await message.reply_text(response)
        logger.info(f"Image processed for chat {chat_id}")

    except CompletionError as e:
        await message.reply_text(e.user_message)
    except Exception as e:
        logger.error(f"Error processing image: {e}", exc_info=True)
        await message.reply_text(
            "Sorry, I encountered an error processing your image. Please try again."
        )


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Clear conversation history for current chat (main authorized user only)."""

    user_id = update.message.from_user.id
    if not is_main_authorized_user(user_id):
        await update.message.reply_text("Sorry, only the main authorized user can clear history.")
        return

    chat_id = str(update.message.chat_id)
    try:
        agent.clear_thread(chat_id)
        await update.message.reply_text("✅ Conversation history cleared for this chat.")
        logger.info(f"History cleared for chat {chat_id}")
    except Exception as e:
        logger.error(f"Error clearing history: {e}", exc_info=True)
        await update.message.reply_text("❌ Failed to clear history. Please try again.")


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show statistics for current chat (main authorized user only)."""

    user_id = update.message.from_user.id
    if not is_main_authorized_user(user_id):
        await update.message.reply_text("Sorry, only the main authorized user can view stats.")
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

        # Try to fetch user info from Telegram
        first_name = None
        username = None
        try:
            chat = await context.bot.get_chat(target_user_id)
            first_name = chat.first_name
            username = chat.username
        except Exception as e:
            logger.warning(f"Could not fetch user info for {target_user_id}: {e}")

        # Grant access
        was_granted = db.grant_access(target_user_id, first_name=first_name, username=username)

        if was_granted:
            name_display = first_name or str(target_user_id)
            if username:
                name_display += f" (@{username})"
            await update.message.reply_text(
                f"✅ Access granted to {name_display}.\n"
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


async def version_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show the version of the bot (main authorized user only)."""

    user_id = update.message.from_user.id
    if not is_main_authorized_user(user_id):
        await update.message.reply_text("Sorry, only the main authorized user can view the bot version.")
        return

    await update.message.reply_text(f"Bot version: {config.BOT_VERSION}")
    logger.info(f"Version shown for chat {update.message.chat_id}")


async def allowlist_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show all users who are validated to use the bot (main authorized user only)."""

    user_id = update.message.from_user.id
    if not is_main_authorized_user(user_id):
        await update.message.reply_text("Sorry, only the main authorized user can see the allowlist.")
        return

    try:
        # Get list of granted users
        granted_users = db.get_granted_users()

        # Format message
        message = "📋 **Bot Allowlist**\n\n"
        message += f"👑 **Main Admin:**\n- `{config.AUTHORIZED_USER_ID}`\n\n"

        if granted_users:
            message += "👥 **Granted Users:**\n"
            for target_user_id, granted_at, first_name, username in granted_users:
                parts = [f"`{target_user_id}`"]
                name_parts = []
                if first_name:
                    name_parts.append(escape_markdown(first_name))
                if username:
                    name_parts.append(f"@{escape_markdown(username)}")
                if name_parts:
                    parts.append(f"({' / '.join(name_parts)})")
                parts.append(f"(granted: {granted_at.split('T')[0]})")
                message += f"- {' '.join(parts)}\n"
        else:
            message += "👥 No other users have been granted access."

        await update.message.reply_text(message, parse_mode="Markdown")
        logger.info(f"Allowlist shown to admin user {user_id}")

    except Exception as e:
        logger.error(f"Error showing allowlist: {e}", exc_info=True)
        await update.message.reply_text(
            "❌ Failed to retrieve allowlist. Please try again."
        )


async def personality_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Change or view the active personality (main authorized user only)."""

    user_id = update.message.from_user.id
    if not is_main_authorized_user(user_id):
        await update.message.reply_text("Sorry, only the main authorized user can change personality.")
        return

    # If no args, show current active personality
    if not context.args or len(context.args) == 0:
        try:
            active_personality = db.get_active_personality()
            await update.message.reply_text(
                f"Current personality: **{active_personality}**\n\n"
                f"Usage: /personality <name>\n"
                f"Example: /personality villain",
                parse_mode="Markdown"
            )
        except Exception as e:
            logger.error(f"Error getting active personality: {e}", exc_info=True)
            await update.message.reply_text(
                "❌ Failed to retrieve active personality. Please try again."
            )
        return

    # Parse personality name from argument
    personality_name = context.args[0].strip()

    try:
        if not db.personality_exists(personality_name):
            await update.message.reply_text(
                f"❌ No personality '{personality_name}' found."
            )
            return

        # Set active personality
        db.set_active_personality(personality_name)
        await update.message.reply_text(
            f"✅ Personality changed to '{personality_name}'"
        )
        logger.info(f"User {user_id} changed personality to {personality_name}")

    except Exception as e:
        logger.error(f"Error changing personality: {e}", exc_info=True)
        await update.message.reply_text(
            "❌ Failed to change personality. Please try again."
        )


async def list_personality_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """List all available personalities."""
    
    user_id = update.message.from_user.id
    if not is_main_authorized_user(user_id):
        await update.message.reply_text("Sorry, only the main authorized user can view personalities.")
        return
    
    try:
        personalities = db.list_personalities()
        active = db.get_active_personality()
        
        if not personalities:
            await update.message.reply_text(
                "No personalities found in database."
            )
            return

        # Build message
        message = f"**Available Personalities:**\n"
        message += f"Currently active: **{active}**\n\n"

        for name, prompt_preview in personalities:
            marker = "✓" if name == active else "-"
            message += f"{marker} `{name}`\n"
            message += f"  _{prompt_preview}_\n\n"
        
        await update.message.reply_text(message, parse_mode="Markdown")
        logger.info(f"Listed personalities for user {user_id}")
        
    except Exception as e:
        logger.error(f"Error listing personalities: {e}", exc_info=True)
        await update.message.reply_text(
            "❌ Failed to list personalities. Please try again."
        )


async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """View or change the active model (main authorized user only)."""
    user_id = update.message.from_user.id
    if not is_main_authorized_user(user_id):
        await update.message.reply_text("Sorry, only the main authorized user can change the model.")
        return

    available = "\n".join(f"  `{m}`" for m in MODEL_PROVIDERS)

    if not context.args:
        current = db.get_active_model()
        await update.message.reply_text(
            f"Current model: `{current}`\n\nAvailable models:\n{available}\n\nUsage: `/model <name>`",
            parse_mode="Markdown",
        )
        return

    new_model = context.args[0].strip()
    if new_model not in MODEL_PROVIDERS:
        await update.message.reply_text(
            f"❌ Unknown model `{new_model}`.\n\nAvailable models:\n{available}",
            parse_mode="Markdown",
        )
        return

    try:
        db.set_active_model(new_model)
        agent.set_model(new_model)
        await update.message.reply_text(f"✅ Model switched to `{new_model}`", parse_mode="Markdown")
        logger.info(f"User {user_id} switched model to {new_model}")
    except Exception as e:
        logger.error(f"Error switching model: {e}", exc_info=True)
        await update.message.reply_text("❌ Failed to switch model. Please try again.")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show all available commands (main authorized user only)."""

    user_id = update.message.from_user.id
    if not is_main_authorized_user(user_id):
        await update.message.reply_text("Sorry, only the main authorized user can use this command.")
        return

    help_text = (
        "📖 **Available Commands:**\n\n"
        "/help - Show this help message\n"
        "/clear - Clear conversation history for current chat\n"
        "/stats - Show message count, token usage, and first message date\n"
        "/grant <user\\_id> - Grant bot access to a user\n"
        "/revoke <user\\_id> - Revoke bot access from a user\n"
        "/allowlist - Show all authorized users\n"
        "/model - View or change the active AI model\n"
        "/personality <name> - View or change active personality\n"
        "/list\\_personality - List all available personalities\n"
        "/version - Show current bot version"
    )

    await update.message.reply_text(help_text, parse_mode="Markdown")
    logger.info(f"Help shown to admin user {user_id}")


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
