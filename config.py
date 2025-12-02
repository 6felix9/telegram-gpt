"""Configuration management with environment variable validation."""
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# Provider-specific default models
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash-preview-09-2025",
    "groq": "llama-3.3-70b-versatile"
}

# Model context limits
MODEL_CONTEXT_LIMITS = {
    # OpenAI
    "gpt-4o-mini": 128000,
    "gpt-4o": 128000,
    "gpt-3.5-turbo": 16385,
    "gpt-4-turbo": 128000,
    "gpt-4": 8192,
    # Gemini
    "gemini-2.5-flash-preview-09-2025": 200000,
    "gemini-2.5-flash-lite-preview-09-2025": 200000,
    # Groq
    "openai/gpt-oss-120b": 128000,
    "llama-3.3-70b-versatile": 128000,
    "llama-3.1-8b-instant": 128000,
    "mixtral-8x7b-32768": 32768,
}


class Config:
    """Centralized configuration with validation."""

    # Telegram Configuration
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

    # Provider Configuration
    _provider_env = os.getenv("PROVIDER", "openai")
    PROVIDER = _provider_env.strip().lower() if _provider_env else "openai"

    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

    # Model Configuration (with backward compatibility)
    # New: MODEL, TIMEOUT
    # Old: OPENAI_MODEL, OPENAI_TIMEOUT
    _model_env = os.getenv("MODEL")
    _legacy_model = os.getenv("OPENAI_MODEL")
    MODEL = (_model_env or _legacy_model or "").strip()
    TIMEOUT = int(os.getenv("TIMEOUT") or os.getenv("OPENAI_TIMEOUT", "60"))
    MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "16000"))

    # Authorization
    AUTHORIZED_USER_ID = os.getenv("AUTHORIZED_USER_ID", "")

    # Database
    DATABASE_PATH = os.getenv("DATABASE_PATH", "data/messages.db")

    # Group chat settings
    MAX_GROUP_CONTEXT_MESSAGES = int(os.getenv("MAX_GROUP_CONTEXT_MESSAGES", "100"))

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

        # Validate provider
        provider = cls.PROVIDER.lower()
        if provider not in ["openai", "gemini", "groq"]:
            errors.append(f"PROVIDER must be one of: openai, gemini, groq (got: {cls.PROVIDER})")

        # Check provider-specific API key
        if provider == "openai":
            if not cls.OPENAI_API_KEY:
                errors.append("OPENAI_API_KEY is required for OpenAI provider")
            elif not cls.OPENAI_API_KEY.strip():
                errors.append("OPENAI_API_KEY cannot be empty")
        elif provider == "gemini":
            if not cls.GEMINI_API_KEY:
                errors.append("GEMINI_API_KEY is required for Gemini provider")
            elif not cls.GEMINI_API_KEY.strip():
                errors.append("GEMINI_API_KEY cannot be empty")
        elif provider == "groq":
            if not cls.GROQ_API_KEY:
                errors.append("GROQ_API_KEY is required for Groq provider")
            elif not cls.GROQ_API_KEY.strip():
                errors.append("GROQ_API_KEY cannot be empty")

        # Set default model if not specified
        if not cls.MODEL:
            cls.MODEL = DEFAULT_MODELS.get(provider, "gpt-4o-mini")
            logger.info(f"Using default model for {provider}: {cls.MODEL}")

        if not cls.AUTHORIZED_USER_ID:
            errors.append("AUTHORIZED_USER_ID is required")
        elif not cls.AUTHORIZED_USER_ID.isdigit():
            errors.append("AUTHORIZED_USER_ID must be numeric")

        # Validate numeric ranges
        if cls.TIMEOUT <= 0:
            errors.append("TIMEOUT must be positive")

        if cls.MAX_CONTEXT_TOKENS <= 0:
            errors.append("MAX_CONTEXT_TOKENS must be positive")

        # Validate model is known (warn if unknown, don't fail)
        if cls.MODEL not in MODEL_CONTEXT_LIMITS:
            logger.warning(
                f"Unknown model '{cls.MODEL}'. "
                f"Known models: {', '.join(MODEL_CONTEXT_LIMITS.keys())}"
            )

        # Backward compatibility warnings
        if os.getenv("OPENAI_MODEL") and not os.getenv("MODEL"):
            logger.warning(
                "OPENAI_MODEL is deprecated, please use MODEL instead. "
                "Set PROVIDER=openai and MODEL=<model-name>"
            )
        if os.getenv("OPENAI_TIMEOUT") and not os.getenv("TIMEOUT"):
            logger.warning(
                "OPENAI_TIMEOUT is deprecated, please use TIMEOUT instead."
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
        return MODEL_CONTEXT_LIMITS.get(model, 8000)  # Conservative default


# Create singleton instance
config = Config()
