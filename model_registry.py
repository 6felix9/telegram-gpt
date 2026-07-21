"""Model -> provider mapping, independent of agent.py's LangChain wiring."""

MODEL_PROVIDERS: dict[str, str] = {
    "gpt-4.1-mini": "openai",
    "gpt-5.4-mini": "openai",
    "gpt-5.4": "openai",
    "gpt-5.6-luna": "openai",
    "gpt-5.6-terra": "openai",
    "grok-4.20-0309-reasoning": "xai",
    "grok-4.20-0309-non-reasoning": "xai",
    "grok-4-1-fast-reasoning": "xai",
    "gemini-3.1-flash-lite-preview": "google_genai",
    "gemini-3.5-flash": "google_genai",
}

PROVIDER_LABEL: dict[str, str] = {
    "openai": "OpenAI", "xai": "xAI", "google_genai": "Gemini"
}

# OpenAI reasoning-capable models: a low reasoning effort leaves more of
# MAX_OUTPUT_TOKENS free for visible text, reducing empty (reasoning-only) replies.
REASONING_EFFORT_LOW: set[str] = {"gpt-5.6-luna", "gpt-5.6-terra"}


def resolve_model(name: str) -> tuple[str, str]:
    """Map a bare model name to (provider, provider-prefixed id)."""
    provider = MODEL_PROVIDERS[name]  # KeyError for unknown models (caught by /model)
    return provider, f"{provider}:{name}"


def provider_api_key(provider: str, config) -> str:
    """Return the configured API key for a provider (may be empty)."""
    return {
        "openai": config.OPENAI_API_KEY,
        "xai": config.XAI_API_KEY,
        "google_genai": config.GEMINI_API_KEY,
    }[provider]
