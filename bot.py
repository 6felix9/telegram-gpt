"""Main bot entry point."""
import logging
import signal
import sys
import asyncio
from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, filters

from config import config
from database import Database
from token_manager import TokenManager
from openai_client import OpenAIClient
import handlers

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=getattr(logging, config.LOG_LEVEL)
)
logger = logging.getLogger(__name__)

# Global instances
db = None
token_manager = None
openai_client = None
application = None


async def post_init(app: Application):
    """Called after bot starts."""
    logger.info("=" * 50)
    logger.info("Bot started successfully!")
    logger.info(f"Model: {config.OPENAI_MODEL}")
    logger.info(f"Max context tokens: {config.MAX_CONTEXT_TOKENS}")
    logger.info(f"Authorized user: {config.AUTHORIZED_USER_ID}")
    # Hide sensitive connection string details in logs
    db_display = config.DATABASE_URL[:50] + "..." if len(config.DATABASE_URL) > 50 else config.DATABASE_URL
    logger.info(f"Database: {db_display}")
    logger.info("=" * 50)


async def post_shutdown(app: Application):
    """Called before bot stops."""
    logger.info("Bot shutting down gracefully...")
    # Close database connection pool
    if db:
        db.close()


def signal_handler(signum, frame):
    """Handle shutdown signals (SIGINT, SIGTERM)."""
    logger.info(f"Received signal {signum}, initiating shutdown...")
    sys.exit(0)


def main():
    """Initialize and run the bot."""

    global db, token_manager, openai_client, application

    try:
        # 1. Validate configuration
        logger.info("Validating configuration...")
        config.validate()

        # 2. Initialize database
        logger.info("Initializing database...")
        db = Database(config.DATABASE_URL)

        # 3. Initialize token manager
        logger.info("Initializing token manager...")
        model_limit = config.get_model_context_limit(config.OPENAI_MODEL)
        max_tokens = min(config.MAX_CONTEXT_TOKENS, model_limit - 2000)
        token_manager = TokenManager(config.OPENAI_MODEL, max_tokens)

        # 4. Initialize OpenAI client
        logger.info("Initializing OpenAI client...")
        client_kwargs = {
            "api_key": config.OPENAI_API_KEY,
            "model": config.OPENAI_MODEL,
            "timeout": config.OPENAI_TIMEOUT,
        }
        # Add base_url if configured (for xAI or other OpenAI-compatible APIs)
        if config.OPENAI_BASE_URL:
            client_kwargs["base_url"] = config.OPENAI_BASE_URL
        openai_client = OpenAIClient(**client_kwargs)

        # 5. Build Telegram application
        logger.info("Building Telegram application...")
        application = (
            Application.builder()
            .token(config.TELEGRAM_BOT_TOKEN)
            .post_init(post_init)
            .post_shutdown(post_shutdown)
            .build()
        )

        # 6. Initialize handlers with dependencies
        bot_username = config.BOT_USERNAME.lstrip("@")
        handlers.init_handlers(config, db, token_manager, openai_client, bot_username)

        # 7. Register handlers
        # Message handler (non-command text messages)
        application.add_handler(
            MessageHandler(
                filters.TEXT & ~filters.COMMAND,
                handlers.message_handler
            )
        )

        # Photo handler (images with optional captions)
        application.add_handler(
            MessageHandler(
                filters.PHOTO,
                handlers.photo_handler
            )
        )

        # Command handlers
        application.add_handler(CommandHandler("clear", handlers.clear_command))
        application.add_handler(CommandHandler("stats", handlers.stats_command))
        application.add_handler(CommandHandler("grant", handlers.grant_command))
        application.add_handler(CommandHandler("revoke", handlers.revoke_command))
        application.add_handler(CommandHandler("allowlist", handlers.allowlist_command))
        application.add_handler(CommandHandler("version", handlers.version_command))
        application.add_handler(CommandHandler("personality", handlers.personality_command))
        application.add_handler(CommandHandler("list_personality", handlers.list_personality_command))

        # Error handler
        application.add_error_handler(handlers.error_handler)

        # 8. Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # 9. Start bot polling
        logger.info("Starting bot polling...")
        application.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True,  # Ignore messages sent while bot was offline
        )

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Fatal error during initialization: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
