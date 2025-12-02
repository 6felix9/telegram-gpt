"""Main bot entry point."""
import logging
import signal
import sys
import asyncio
from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, filters

from config import config
from database import Database
import llm_factory
import handlers

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=getattr(logging, config.LOG_LEVEL)
)
logger = logging.getLogger(__name__)

# Global instances
db = None
llm_provider = None
application = None


async def post_init(app: Application):
    """Called after bot starts."""
    logger.info("=" * 50)
    logger.info("Bot started successfully!")
    logger.info(f"Provider: {config.PROVIDER}")
    logger.info(f"Model: {config.MODEL}")
    logger.info(f"Max context tokens: {config.MAX_CONTEXT_TOKENS}")
    logger.info(f"Authorized user: {config.AUTHORIZED_USER_ID}")
    logger.info(f"Database: {config.DATABASE_PATH}")
    logger.info("=" * 50)


async def post_shutdown(app: Application):
    """Called before bot stops."""
    logger.info("Bot shutting down gracefully...")


def signal_handler(signum, frame):
    """Handle shutdown signals (SIGINT, SIGTERM)."""
    logger.info(f"Received signal {signum}, initiating shutdown...")
    sys.exit(0)


def main():
    """Initialize and run the bot."""

    global db, llm_provider, application

    try:
        # 1. Validate configuration
        logger.info("Validating configuration...")
        config.validate()

        # 2. Initialize database
        logger.info("Initializing database...")
        db = Database(config.DATABASE_PATH)

        # 3. Initialize LLM provider
        logger.info("Initializing LLM provider...")
        llm_provider = llm_factory.create_provider(config)

        # 4. Initialize handlers with dependencies
        handlers.init_handlers(config, db, llm_provider)

        # 6. Build Telegram application
        logger.info("Building Telegram application...")
        application = (
            Application.builder()
            .token(config.TELEGRAM_BOT_TOKEN)
            .post_init(post_init)
            .post_shutdown(post_shutdown)
            .build()
        )

        # 7. Register handlers
        # Message handler (non-command text messages)
        application.add_handler(
            MessageHandler(
                filters.TEXT & ~filters.COMMAND,
                handlers.message_handler
            )
        )

        # Command handlers
        application.add_handler(CommandHandler("clear", handlers.clear_command))
        application.add_handler(CommandHandler("stats", handlers.stats_command))
        application.add_handler(CommandHandler("grant", handlers.grant_command))
        application.add_handler(CommandHandler("revoke", handlers.revoke_command))

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
