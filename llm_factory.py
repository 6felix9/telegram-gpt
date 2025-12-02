"""Factory for creating LLM provider instances."""
import logging
from llm_provider import LLMProvider

logger = logging.getLogger(__name__)


def create_provider(config) -> LLMProvider:
    """
    Factory to create LLM provider based on config.

    Args:
        config: Configuration object with provider settings

    Returns:
        LLMProvider instance

    Raises:
        ValueError: If provider type is unknown or configuration is invalid
    """
    provider_type = config.PROVIDER.lower()

    logger.info(f"Creating {provider_type} provider...")

    if provider_type == "openai":
        from providers.openai_provider import OpenAIProvider

        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required for OpenAI provider")

        return OpenAIProvider(
            api_key=config.OPENAI_API_KEY,
            model=config.MODEL,
            timeout=config.TIMEOUT
        )

    elif provider_type == "gemini":
        from providers.gemini_provider import GeminiProvider

        if not config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is required for Gemini provider")

        return GeminiProvider(
            api_key=config.GEMINI_API_KEY,
            model=config.MODEL,
            timeout=config.TIMEOUT
        )

    elif provider_type == "groq":
        from providers.groq_provider import GroqProvider

        if not config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is required for Groq provider")

        return GroqProvider(
            api_key=config.GROQ_API_KEY,
            model=config.MODEL,
            timeout=config.TIMEOUT
        )

    else:
        raise ValueError(
            f"Unknown provider: {provider_type}. "
            f"Supported providers: openai, gemini, groq"
        )
