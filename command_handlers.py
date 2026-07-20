"""Admin-only Telegram commands, bound to an explicit dependency set."""
import logging

from telegram.helpers import escape_markdown

from authorization import is_main_authorized_user
from handler_deps import HandlerDependencies
from model_registry import MODEL_PROVIDERS

logger = logging.getLogger(__name__)


class CommandHandlers:
    """All /commands gated by is_main_authorized_user, bound to deps."""

    def __init__(self, deps: HandlerDependencies):
        self._deps = deps

    async def clear_command(self, update, context):
        user_id = update.message.from_user.id
        if not is_main_authorized_user(user_id, self._deps.config):
            await update.message.reply_text(
                "Sorry, only the main authorized user can clear history."
            )
            return

        chat_id = str(update.message.chat_id)
        try:
            self._deps.agent.clear_thread(chat_id)
            await update.message.reply_text("✅ Conversation history cleared for this chat.")
            logger.info(f"History cleared for chat {chat_id}")
        except Exception as e:
            logger.error(f"Error clearing history: {e}", exc_info=True)
            await update.message.reply_text("❌ Failed to clear history. Please try again.")

    async def stats_command(self, update, context):
        user_id = update.message.from_user.id
        if not is_main_authorized_user(user_id, self._deps.config):
            await update.message.reply_text("Sorry, only the main authorized user can view stats.")
            return

        chat_id = str(update.message.chat_id)
        try:
            stats = self._deps.db.get_stats(chat_id)
            first_msg = stats["first_message"]
            if first_msg != "N/A":
                first_msg = first_msg.split("T")[0]

            await update.message.reply_text(
                f"📊 Chat Statistics:\n"
                f"Messages: {stats['total_messages']}\n"
                f"Total tokens: {stats['total_tokens']:,}\n"
                f"Since: {first_msg}"
            )
            logger.info(f"Stats shown for chat {chat_id}")
        except Exception as e:
            logger.error(f"Error getting stats: {e}", exc_info=True)
            await update.message.reply_text("❌ Failed to retrieve statistics. Please try again.")

    async def grant_command(self, update, context):
        user_id = update.message.from_user.id
        if not is_main_authorized_user(user_id, self._deps.config):
            await update.message.reply_text(
                "Sorry, only the main authorized user can grant access."
            )
            return

        if not context.args or len(context.args) == 0:
            await update.message.reply_text(
                "❌ Usage: /grant <user_id>\n"
                "Example: /grant 123456789"
            )
            return

        try:
            target_user_id = int(context.args[0])
            if target_user_id <= 0:
                await update.message.reply_text(
                    "❌ Invalid user ID. User IDs must be positive integers."
                )
                return

            if str(target_user_id) == self._deps.config.AUTHORIZED_USER_ID:
                await update.message.reply_text("ℹ️ You are already the main authorized user.")
                return

            first_name = None
            username = None
            try:
                chat = await context.bot.get_chat(target_user_id)
                first_name = chat.first_name
                username = chat.username
            except Exception as e:
                logger.warning(f"Could not fetch user info for {target_user_id}: {e}")

            was_granted = self._deps.db.grant_access(
                target_user_id, first_name=first_name, username=username
            )

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
                await update.message.reply_text(f"ℹ️ User {target_user_id} already has access.")

        except ValueError:
            await update.message.reply_text(
                "❌ Invalid user ID. Please provide a numeric user ID.\n"
                "Example: /grant 123456789"
            )
        except Exception as e:
            logger.error(f"Error granting access: {e}", exc_info=True)
            await update.message.reply_text("❌ Failed to grant access. Please try again.")

    async def revoke_command(self, update, context):
        user_id = update.message.from_user.id
        if not is_main_authorized_user(user_id, self._deps.config):
            await update.message.reply_text(
                "Sorry, only the main authorized user can revoke access."
            )
            return

        if not context.args or len(context.args) == 0:
            await update.message.reply_text(
                "❌ Usage: /revoke <user_id>\n"
                "Example: /revoke 123456789"
            )
            return

        try:
            target_user_id = int(context.args[0])
            if target_user_id <= 0:
                await update.message.reply_text(
                    "❌ Invalid user ID. User IDs must be positive integers."
                )
                return
            if str(target_user_id) == self._deps.config.AUTHORIZED_USER_ID:
                await update.message.reply_text(
                    "❌ Cannot revoke access from the main authorized user."
                )
                return

            was_revoked = self._deps.db.revoke_access(target_user_id)

            if was_revoked:
                await update.message.reply_text(f"✅ Access revoked from user {target_user_id}.")
                logger.info(f"User {user_id} revoked access from {target_user_id}")
            else:
                await update.message.reply_text(f"ℹ️ User {target_user_id} didn't have access.")

        except ValueError:
            await update.message.reply_text(
                "❌ Invalid user ID. Please provide a numeric user ID.\n"
                "Example: /revoke 123456789"
            )
        except Exception as e:
            logger.error(f"Error revoking access: {e}", exc_info=True)
            await update.message.reply_text("❌ Failed to revoke access. Please try again.")

    async def version_command(self, update, context):
        user_id = update.message.from_user.id
        if not is_main_authorized_user(user_id, self._deps.config):
            await update.message.reply_text(
                "Sorry, only the main authorized user can view the bot version."
            )
            return

        await update.message.reply_text(f"Bot version: {self._deps.config.BOT_VERSION}")
        logger.info(f"Version shown for chat {update.message.chat_id}")

    async def allowlist_command(self, update, context):
        user_id = update.message.from_user.id
        if not is_main_authorized_user(user_id, self._deps.config):
            await update.message.reply_text(
                "Sorry, only the main authorized user can see the allowlist."
            )
            return

        try:
            granted_users = self._deps.db.get_granted_users()

            message = "📋 **Bot Allowlist**\n\n"
            message += f"👑 **Main Admin:**\n- `{self._deps.config.AUTHORIZED_USER_ID}`\n\n"

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
            await update.message.reply_text("❌ Failed to retrieve allowlist. Please try again.")

    async def personality_command(self, update, context):
        user_id = update.message.from_user.id
        if not is_main_authorized_user(user_id, self._deps.config):
            await update.message.reply_text(
                "Sorry, only the main authorized user can change personality."
            )
            return

        if not context.args or len(context.args) == 0:
            try:
                active_personality = self._deps.db.get_active_personality()
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

        personality_name = context.args[0].strip()

        try:
            if not self._deps.db.personality_exists(personality_name):
                await update.message.reply_text(f"❌ No personality '{personality_name}' found.")
                return

            self._deps.db.set_active_personality(personality_name)
            await update.message.reply_text(f"✅ Personality changed to '{personality_name}'")
            logger.info(f"User {user_id} changed personality to {personality_name}")

        except Exception as e:
            logger.error(f"Error changing personality: {e}", exc_info=True)
            await update.message.reply_text("❌ Failed to change personality. Please try again.")

    async def list_personality_command(self, update, context):
        user_id = update.message.from_user.id
        if not is_main_authorized_user(user_id, self._deps.config):
            await update.message.reply_text(
                "Sorry, only the main authorized user can view personalities."
            )
            return

        try:
            personalities = self._deps.db.list_personalities()
            active = self._deps.db.get_active_personality()

            if not personalities:
                await update.message.reply_text("No personalities found in database.")
                return

            message = "**Available Personalities:**\n"
            message += f"Currently active: **{active}**\n\n"

            for name, prompt_preview in personalities:
                marker = "✓" if name == active else "-"
                message += f"{marker} `{name}`\n"
                message += f"  _{prompt_preview}_\n\n"

            await update.message.reply_text(message, parse_mode="Markdown")
            logger.info(f"Listed personalities for user {user_id}")

        except Exception as e:
            logger.error(f"Error listing personalities: {e}", exc_info=True)
            await update.message.reply_text("❌ Failed to list personalities. Please try again.")

    async def model_command(self, update, context):
        user_id = update.message.from_user.id
        if not is_main_authorized_user(user_id, self._deps.config):
            await update.message.reply_text(
                "Sorry, only the main authorized user can change the model."
            )
            return

        available = "\n".join(f"  `{m}`" for m in MODEL_PROVIDERS)

        if not context.args:
            current = self._deps.db.get_active_model()
            await update.message.reply_text(
                f"Current model: `{current}`\n\nAvailable models:\n{available}\n\n"
                "Usage: `/model <name>`",
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
            self._deps.db.set_active_model(new_model)
            self._deps.agent.set_model(new_model)
            await update.message.reply_text(
                f"✅ Model switched to `{new_model}`", parse_mode="Markdown"
            )
            logger.info(f"User {user_id} switched model to {new_model}")
        except Exception as e:
            logger.error(f"Error switching model: {e}", exc_info=True)
            await update.message.reply_text("❌ Failed to switch model. Please try again.")

    async def help_command(self, update, context):
        user_id = update.message.from_user.id
        if not is_main_authorized_user(user_id, self._deps.config):
            await update.message.reply_text(
                "Sorry, only the main authorized user can use this command."
            )
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


async def error_handler(update, context):
    """Global error handler for unhandled exceptions."""
    logger.error("Exception while handling update:", exc_info=context.error)

    if update and update.message:
        try:
            await update.message.reply_text(
                "An error occurred while processing your request. "
                "The error has been logged."
            )
        except Exception:
            pass
