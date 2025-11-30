"""Configuration management with environment variable validation."""
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class Config:
    """Centralized configuration with validation."""

    # Telegram Configuration
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", "60"))
    MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "16000"))

    # Authorization
    AUTHORIZED_USER_ID = os.getenv("AUTHORIZED_USER_ID", "")

    # Database
    DATABASE_PATH = os.getenv("DATABASE_PATH", "data/messages.db")

    # Group chat settings
    MAX_GROUP_CONTEXT_MESSAGES = int(os.getenv("MAX_GROUP_CONTEXT_MESSAGES", "100"))

    # Feature Flags
    ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "true").lower() == "true"

    # Web Search Configuration
    MAX_WEB_SOURCES = int(os.getenv("MAX_WEB_SOURCES", "1"))  # Max number of source links to show
    SHOW_WEB_SOURCES = os.getenv("SHOW_WEB_SOURCES", "true").lower() == "true"  # Whether to show sources at all
    WEB_SUMMARY_SENTENCES = int(os.getenv("WEB_SUMMARY_SENTENCES", "3"))  # Target number of sentences in summary

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    @classmethod
    def validate(cls):
        """Validate all required environment variables are present and correct format."""
        errors = []

        # Check required variables
        if not cls.TELEGRAM_BOT_TOKEN:
            errors.append("TELEGRAM_BOT_TOKEN is required")
        elif not cls.TELEGRAM_BOT_TOKEN.strip():
            errors.append("TELEGRAM_BOT_TOKEN cannot be empty")

        if not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required")
        elif not cls.OPENAI_API_KEY.strip():
            errors.append("OPENAI_API_KEY cannot be empty")

        if not cls.AUTHORIZED_USER_ID:
            errors.append("AUTHORIZED_USER_ID is required")
        elif not cls.AUTHORIZED_USER_ID.isdigit():
            errors.append("AUTHORIZED_USER_ID must be numeric")

        # Validate numeric ranges
        if cls.OPENAI_TIMEOUT <= 0:
            errors.append("OPENAI_TIMEOUT must be positive")

        if cls.MAX_CONTEXT_TOKENS <= 0:
            errors.append("MAX_CONTEXT_TOKENS must be positive")

        # Validate model is known
        known_models = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "gpt-4-turbo", "gpt-4"]
        if cls.OPENAI_MODEL not in known_models:
            logger.warning(
                f"Unknown model '{cls.OPENAI_MODEL}'. "
                f"Known models: {', '.join(known_models)}"
            )

        # Ensure database directory exists
        db_dir = Path(cls.DATABASE_PATH).parent
        if not db_dir.exists():
            try:
                db_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created database directory: {db_dir}")
            except Exception as e:
                errors.append(f"Cannot create database directory {db_dir}: {e}")

        # Report all errors
        if errors:
            logger.error("Configuration validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            sys.exit(1)

        logger.info("Configuration validated successfully")

    @classmethod
    def get_model_context_limit(cls, model: str) -> int:
        """Return maximum tokens for given model."""
        LIMITS = {
            "gpt-4o-mini": 128000,
            "gpt-4o": 128000,
            "gpt-3.5-turbo": 16385,
            "gpt-4-turbo": 128000,
            "gpt-4": 8192,
        }
        return LIMITS.get(model, 8000)  # Conservative default


# Create singleton instance
config = Config()
