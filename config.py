"""Configuration management with environment variable validation."""
import os
import sys
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class Config:
    """Centralized configuration with validation."""

    # Telegram Configuration
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    BOT_USERNAME = os.getenv("BOT_USERNAME", "")

    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", "60"))
    MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "16000"))
    
    # Token Reserve Configuration
    RESERVE_TOKENS_TEXT = int(os.getenv("RESERVE_TOKENS_TEXT", "1000"))
    RESERVE_TOKENS_IMAGE = int(os.getenv("RESERVE_TOKENS_IMAGE", "3000"))

    # Authorization
    AUTHORIZED_USER_ID = os.getenv("AUTHORIZED_USER_ID", "")

    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "")

    # Group chat settings
    MAX_GROUP_CONTEXT_MESSAGES = int(os.getenv("MAX_GROUP_CONTEXT_MESSAGES", "100"))

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # Bot version
    BOT_VERSION = "1.1.1"

    @classmethod
    def validate(cls):
        """Validate all required environment variables are present and correct format."""
        errors = []

        # Check required variables
        if not cls.TELEGRAM_BOT_TOKEN:
            errors.append("TELEGRAM_BOT_TOKEN is required")
        elif not cls.TELEGRAM_BOT_TOKEN.strip():
            errors.append("TELEGRAM_BOT_TOKEN cannot be empty")

        if not cls.BOT_USERNAME:
            errors.append("BOT_USERNAME is required")
        elif not cls.BOT_USERNAME.strip():
            errors.append("BOT_USERNAME cannot be empty")

        if not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required")
        elif not cls.OPENAI_API_KEY.strip():
            errors.append("OPENAI_API_KEY cannot be empty")

        if not cls.AUTHORIZED_USER_ID:
            errors.append("AUTHORIZED_USER_ID is required")
        elif not cls.AUTHORIZED_USER_ID.isdigit():
            errors.append("AUTHORIZED_USER_ID must be numeric")

        if not cls.DATABASE_URL:
            errors.append("DATABASE_URL is required")
        elif not cls.DATABASE_URL.strip():
            errors.append("DATABASE_URL cannot be empty")

        # Validate numeric ranges
        if cls.OPENAI_TIMEOUT <= 0:
            errors.append("OPENAI_TIMEOUT must be positive")

        if cls.MAX_CONTEXT_TOKENS <= 0:
            errors.append("MAX_CONTEXT_TOKENS must be positive")

        if cls.RESERVE_TOKENS_TEXT <= 0:
            errors.append("RESERVE_TOKENS_TEXT must be positive")

        if cls.RESERVE_TOKENS_IMAGE <= 0:
            errors.append("RESERVE_TOKENS_IMAGE must be positive")

        # Optional warning for very large context windows (but don't fail)
        if cls.MAX_CONTEXT_TOKENS > 100000:
            logger.warning(
                f"MAX_CONTEXT_TOKENS is very large ({cls.MAX_CONTEXT_TOKENS}). "
                "Make sure this matches your model's actual context window limit."
            )

        # Report all errors
        if errors:
            logger.error("Configuration validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            sys.exit(1)

        logger.info("Configuration validated successfully")


# Create singleton instance
config = Config()
