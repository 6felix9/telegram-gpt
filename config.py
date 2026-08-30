"""Configuration management with environment variable validation."""
import os
import sys
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


def _int_env(name: str, default: int) -> int:
    """Read an int setting, treating an empty value as unset.

    os.getenv's default only applies when the variable is absent, but
    .env.example ships these keys blank ("MODEL_TIMEOUT=") to mean "use the
    default" — and Railway/Docker pass a set-but-empty var through the same
    way. Without this, int("") crashes at import.
    """
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer; using default %s", name, raw, default)
        return default


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

    MODEL_TIMEOUT = _int_env("MODEL_TIMEOUT", 60)
    MAX_CONTEXT_TOKENS = _int_env("MAX_CONTEXT_TOKENS", 16000)

    # Max tokens the model may generate per reply; also used as the trimming
    # middleware's reserve (history budget = MAX_CONTEXT_TOKENS - this).
    MAX_OUTPUT_TOKENS = _int_env("MAX_OUTPUT_TOKENS", 2048)

    # Rolling checkpoint summary. Compaction runs before every checkpoint
    # update, triggered or passive, and is independent of /model.
    SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "gpt-5.6-luna")
    # Dedicated vision model that describes images on ingest so later turns
    # keep a text description. Fixed, independent of /model and SUMMARY_MODEL.
    # A missing provider key does not block startup (image persist fails open).
    VISION_SUMMARY_MODEL = os.getenv("VISION_SUMMARY_MODEL", "gpt-5.4-nano")
    # Compact the checkpoint when active message state reaches this many
    # approximate tokens. The sole threshold governing checkpoint size.
    SUMMARIZATION_TRIGGER = _int_env("SUMMARIZATION_TRIGGER", 8000)
    # Hard output cap for one generated summary.
    MAX_SUMMARY_OUTPUT = _int_env("MAX_SUMMARY_OUTPUT", 1000)

    # Dedicated speech-to-text model for voice notes. Uses OpenAI's audio
    # transcription endpoint, not init_chat_model — deliberately absent from
    # MODEL_PROVIDERS. Fixed and independent of /model and SUMMARY_MODEL.
    TRANSCRIPTION_MODEL = os.getenv("TRANSCRIPTION_MODEL", "gpt-transcribe")
    # Voice notes longer than this are skipped before download, so an
    # over-limit note costs nothing. At gpt-transcribe's $0.0045/min this caps
    # a single note at roughly $0.045.
    MAX_VOICE_DURATION_SECONDS = _int_env("MAX_VOICE_DURATION_SECONDS", 600)

    # Web search tool (Tavily); blank falls back to DuckDuckGo at runtime
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

    # Authorization
    AUTHORIZED_USER_ID = os.getenv("AUTHORIZED_USER_ID", "")

    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "")

    # Retention: age-based deletion of `messages` audit rows, run out-of-band by
    # scripts/cleanup_retention.py (wired into Railway preDeployCommand). 0 means
    # "disabled" (skip the delete).
    MESSAGE_RETENTION_DAYS = _int_env("MESSAGE_RETENTION_DAYS", 30)
    # Same shape, for `images` rows. Kept a separate knob despite the shared
    # default because image rows carry blobs, so this is the threshold worth
    # tightening if storage gets tight. 0 means "disabled".
    IMAGE_RETENTION_DAYS = _int_env("IMAGE_RETENTION_DAYS", 30)

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
            "SUMMARIZATION_TRIGGER",
            "MAX_SUMMARY_OUTPUT",
            "MAX_VOICE_DURATION_SECONDS",
        ):
            if getattr(cls, name) <= 0:
                errors.append(f"{name} must be positive")

        if cls.MESSAGE_RETENTION_DAYS < 0:
            errors.append("MESSAGE_RETENTION_DAYS must be >= 0 (0 disables retention cleanup)")

        if cls.IMAGE_RETENTION_DAYS < 0:
            errors.append("IMAGE_RETENTION_DAYS must be >= 0 (0 disables retention cleanup)")

        if cls.MAX_SUMMARY_OUTPUT >= cls.SUMMARIZATION_TRIGGER:
            errors.append(
                "MAX_SUMMARY_OUTPUT must be less than SUMMARIZATION_TRIGGER, or a "
                "compaction could not bring checkpoint state below the trigger and "
                "every subsequent message would re-trigger a summary call"
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
