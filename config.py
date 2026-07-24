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

    # AI Provider API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")    # OpenAI models (gpt-*)
    XAI_API_KEY = os.getenv("XAI_API_KEY", "")          # xAI Grok models (grok-*)
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")    # Google Gemini models (gemini-*)

    # Default model to use on first startup (persisted in DB after first run)
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-5.4-mini")

    MODEL_TIMEOUT = int(os.getenv("MODEL_TIMEOUT", "60"))
    MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "16000"))

    # Max tokens the model may generate per reply; also used as the trimming
    # middleware's reserve (history budget = MAX_CONTEXT_TOKENS - this).
    MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "2048"))

    # Rolling checkpoint summary. Summarization runs only on triggered requests.
    SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "gpt-4.1-mini")
    # Dedicated vision model that describes images on ingest so later turns
    # keep a text description. Fixed, independent of /model and SUMMARY_MODEL.
    # A missing provider key does not block startup (image persist fails open).
    VISION_SUMMARY_MODEL = os.getenv("VISION_SUMMARY_MODEL", "gpt-5.4-nano")
    SUMMARY_TRIGGER_TOKENS = int(os.getenv("SUMMARY_TRIGGER_TOKENS", "10000"))
    SUMMARY_KEEP_TOKENS = int(os.getenv("SUMMARY_KEEP_TOKENS", "4000"))
    # Input budget for the summary model call, independent of MAX_CONTEXT_TOKENS
    # (which bounds the reply model instead).
    SUMMARY_CONTEXT_TOKENS = int(os.getenv("SUMMARY_CONTEXT_TOKENS", "14000"))

    # Web search tool (Tavily); blank falls back to DuckDuckGo at runtime
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

    # Authorization
    AUTHORIZED_USER_ID = os.getenv("AUTHORIZED_USER_ID", "")

    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "")

    # Group chat settings
    MAX_GROUP_CONTEXT_MESSAGES = int(os.getenv("MAX_GROUP_CONTEXT_MESSAGES", "500"))

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # Bot version
    BOT_VERSION = "2.0.0"

    @classmethod
    def validate(cls):
        """Validate the small required set; optional vars fall back to defaults."""
        errors = []

        if not cls.TELEGRAM_BOT_TOKEN.strip():
            errors.append("TELEGRAM_BOT_TOKEN is required")

        if not cls.OPENAI_API_KEY.strip():
            errors.append("OPENAI_API_KEY is required")

        if not cls.AUTHORIZED_USER_ID:
            errors.append("AUTHORIZED_USER_ID is required")
        elif not cls.AUTHORIZED_USER_ID.isdigit():
            errors.append("AUTHORIZED_USER_ID must be numeric")

        if not cls.DATABASE_URL.strip():
            errors.append("DATABASE_URL is required")

        for name in (
            "MODEL_TIMEOUT",
            "MAX_CONTEXT_TOKENS",
            "MAX_OUTPUT_TOKENS",
            "SUMMARY_TRIGGER_TOKENS",
            "SUMMARY_KEEP_TOKENS",
            "SUMMARY_CONTEXT_TOKENS",
        ):
            if getattr(cls, name) <= 0:
                errors.append(f"{name} must be positive")

        if cls.SUMMARY_KEEP_TOKENS >= cls.SUMMARY_TRIGGER_TOKENS:
            errors.append(
                "SUMMARY_KEEP_TOKENS must be less than SUMMARY_TRIGGER_TOKENS"
            )

        older_partition_tokens = cls.SUMMARY_TRIGGER_TOKENS - cls.SUMMARY_KEEP_TOKENS
        if cls.SUMMARY_CONTEXT_TOKENS < older_partition_tokens:
            errors.append(
                "SUMMARY_CONTEXT_TOKENS must be at least "
                "SUMMARY_TRIGGER_TOKENS - SUMMARY_KEEP_TOKENS "
                f"({older_partition_tokens}), or an ordinary (non-backlog) trigger "
                "would silently drop history before it reaches the summary model"
            )

        if cls.MAX_CONTEXT_TOKENS > 100000:
            logger.warning(
                f"MAX_CONTEXT_TOKENS is very large ({cls.MAX_CONTEXT_TOKENS}). "
                "Make sure this matches your model's actual context window limit."
            )

        if errors:
            logger.error("Configuration validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            sys.exit(1)

        logger.info("Configuration validated successfully")


# Create singleton instance
config = Config()
