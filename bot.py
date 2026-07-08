"""Main bot entry point."""
import logging
import signal
import sys
import asyncio
from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, filters

from config import config
from database import Database
from prompt_builder import PromptBuilder
from agent import Agent
import agent as agent_module
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres import PostgresSaver
import handlers

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=getattr(logging, config.LOG_LEVEL)
)
logger = logging.getLogger(__name__)

# Global instances
db = None
application = None
bot_agent = None
checkpointer_pool = None


async def post_init(app: Application):
    """Called after bot starts."""
    logger.info("=" * 50)
    logger.info("Bot started successfully!")
    logger.info(f"Default model: {config.DEFAULT_MODEL}")
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
    if checkpointer_pool:
        checkpointer_pool.close()


def signal_handler(signum, frame):
    """Handle shutdown signals (SIGINT, SIGTERM)."""
    logger.info(f"Received signal {signum}, initiating shutdown...")
    sys.exit(0)


def main():
    """Initialize and run the bot."""

    global db, application, bot_agent, checkpointer_pool

    try:
        # 1. Validate configuration
        logger.info("Validating configuration...")
        config.validate()

        # 2. Initialize database
        logger.info("Initializing database...")
        db = Database(config.DATABASE_URL)

        # Seed active_model on first run with env var default, then load persisted value
        db.init_active_model(config.DEFAULT_MODEL)
        effective_model = db.get_active_model()
        logger.info(f"Active model: {effective_model}")

        # 3. Build the Postgres checkpointer over a dedicated psycopg3 pool.
        #    Tables are created out-of-band by scripts/setup_checkpointer.py
        #    (deploy preDeployCommand); we do NOT call .setup() here.
        logger.info("Initializing checkpointer pool...")
        checkpointer_pool = ConnectionPool(
            conninfo=config.DATABASE_URL,
            max_size=10,
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
            check=ConnectionPool.check_connection,
        )
        checkpointer = PostgresSaver(checkpointer_pool)

        # 4. Prompt builder (shared with the dynamic-prompt middleware).
        logger.info("Initializing prompt builder...")
        prompt_builder = PromptBuilder(
            default_private_prompt=agent_module.SYSTEM_PROMPT,
            default_group_prompt=agent_module.SYSTEM_PROMPT_GROUP,
            get_active_personality=db.get_active_personality,
            get_personality_prompt=db.get_personality_prompt,
        )

        # 5. Build the agent for the active model.
        logger.info("Building agent...")
        bot_agent = Agent(
            config=config,
            prompt_builder=prompt_builder,
            checkpointer=checkpointer,
            model_name=effective_model,
        )

        # 6. Build Telegram application
        logger.info("Building Telegram application...")
        application = (
            Application.builder()
            .token(config.TELEGRAM_BOT_TOKEN)
            .post_init(post_init)
            .post_shutdown(post_shutdown)
            .build()
        )

        # 7. Initialize handlers with dependencies
        bot_username = config.BOT_USERNAME.lstrip("@")
        handlers.init_handlers(config, db, bot_agent, prompt_builder, bot_username)

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
        application.add_handler(CommandHandler("model", handlers.model_command))
        application.add_handler(CommandHandler("personality", handlers.personality_command))
        application.add_handler(CommandHandler("list_personality", handlers.list_personality_command))
        application.add_handler(CommandHandler("help", handlers.help_command))

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
