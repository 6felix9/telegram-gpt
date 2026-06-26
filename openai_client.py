"""OpenAI API client wrapper with error handling."""
import logging
import asyncio
from dataclasses import dataclass
import openai
import httpx
from prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


class CompletionError(Exception):
    """API completion failed; user_message is safe to show in Telegram."""

    def __init__(self, user_message: str):
        self.user_message = user_message
        super().__init__(user_message)


@dataclass
class ModelConfig:
    api: str          # "responses" | "chat_completions"
    provider: str     # "openai" | "xai" | "gemini"
    reasoning: bool = False  # Responses API only: use reasoning/verbosity params


# Base URLs for each provider (None = use the openai SDK default)
PROVIDER_BASE_URLS: dict[str, str | None] = {
    "openai": None,
    "xai":    "https://api.x.ai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
}

MODEL_REGISTRY: dict[str, ModelConfig] = {
    # OpenAI — Responses API
    "gpt-4o-mini":                   ModelConfig(api="responses", provider="openai"),
    "gpt-4.1-mini":                  ModelConfig(api="responses", provider="openai"),
    "gpt-5.4-mini":                  ModelConfig(api="responses", provider="openai", reasoning=True),
    "gpt-5":                         ModelConfig(api="responses", provider="openai", reasoning=True),
    # xAI Grok — Responses API
    "grok-4.20-0309-reasoning":      ModelConfig(api="responses", provider="xai"),
    "grok-4.20-0309-non-reasoning":  ModelConfig(api="responses", provider="xai"),
    "grok-4-1-fast-reasoning":       ModelConfig(api="responses", provider="xai"),
    # Google Gemini — Chat Completions API
    "gemini-3.1-flash-lite-preview": ModelConfig(api="chat_completions", provider="gemini", reasoning=True),
    "gemini-3-flash-preview":        ModelConfig(api="chat_completions", provider="gemini", reasoning=True),
}


class OpenAIClient:
    """Wrapper for OpenAI/xAI/Gemini APIs with comprehensive error handling."""

    SYSTEM_PROMPT = """You are Tze Foong's Assistant, an AI helper in Telegram.

Key behaviors:
- Be direct and concise - no unnecessary preambles
- Provide clear, helpful responses
- Never claim to be OpenAI or reference being a language model
- Respond naturally as a personal assistant"""

    SYSTEM_PROMPT_GROUP = """You are Tze Foong's Assistant, an AI helper in Telegram group chats.

Key behaviors:
- Be direct and concise - no unnecessary preambles
- Provide clear, helpful responses
- Never claim to be OpenAI or reference being a language model
- Track conversation context from multiple participants
- Messages are formatted as [Name]: content - reply naturally without mimicking this format"""

    def __init__(
        self,
        openai_api_key: str,
        xai_api_key: str,
        gemini_api_key: str,
        model: str,
        timeout: int,
        prompt_builder: PromptBuilder | None = None,
    ):
        """
        Initialize the AI client.

        Args:
            openai_api_key: API key for OpenAI models (gpt-*, o3, o4-mini)
            xai_api_key: API key for xAI Grok models (grok-*)
            gemini_api_key: API key for Google Gemini models (gemini-*)
            model: Initial model name
            timeout: Request timeout in seconds
            prompt_builder: Shared prompt builder for system/messages assembly
        """
        self._openai_api_key = openai_api_key
        self._xai_api_key = xai_api_key
        self._gemini_api_key = gemini_api_key
        self.timeout = timeout
        self.prompt_builder = prompt_builder or PromptBuilder(
            default_private_prompt=self.SYSTEM_PROMPT,
            default_group_prompt=self.SYSTEM_PROMPT_GROUP,
        )
        self.model = model
        self.client = self._make_client(model)

    def _make_client(self, model: str) -> openai.OpenAI:
        """Create an openai.OpenAI client configured for the given model's provider."""
        model_cfg = MODEL_REGISTRY.get(model, ModelConfig(api="responses", provider="openai"))
        api_key = {
            "openai": self._openai_api_key,
            "xai":    self._xai_api_key,
            "gemini": self._gemini_api_key,
        }[model_cfg.provider]
        base_url = PROVIDER_BASE_URLS[model_cfg.provider]

        kwargs: dict = {"api_key": api_key, "timeout": httpx.Timeout(float(self.timeout))}
        if base_url:
            kwargs["base_url"] = base_url

        logger.info(
            f"Configured {model_cfg.provider} client for model {model}"
            + (f" (base_url: {base_url})" if base_url else "")
        )
        return openai.OpenAI(**kwargs)

    def set_model(self, model: str) -> None:
        """Switch to a different model, re-initializing the underlying client if the provider changes."""
        self.model = model
        self.client = self._make_client(model)

    async def get_completion(
        self,
        messages: list[dict],
        is_group: bool = False,
        custom_system_prompt: str | None = None,
        reply_context: tuple[str, str] | None = None,
    ) -> str:
        """
        Get completion from the active model's API.

        Args:
            messages: List of message dicts with 'role', 'content', and optionally sender info
            is_group: Whether this is a group chat (affects formatting and system prompt)
            custom_system_prompt: Optional custom system prompt to use instead of default
            reply_context: Optional tuple of (sender_name, content) being replied to

        Returns:
            Assistant's response text on success.

        Raises:
            CompletionError: On API or transport failure, with a user-safe message.
        """
        try:
            logger.debug(f"Requesting completion with {len(messages)} messages (group={is_group})")

            model_cfg = MODEL_REGISTRY.get(self.model, ModelConfig(api="responses", provider="openai"))

            formatted_messages = self.prompt_builder.format_messages(
                messages, is_group, api_format=model_cfg.api
            )
            system_prompt = self.prompt_builder.build_system_prompt(
                is_group=is_group,
                custom_system_prompt=custom_system_prompt,
                reply_context=reply_context,
            )

            # Log system prompt metadata only to avoid leaking sensitive content
            logger.debug("System prompt generated (length=%d chars)", len(system_prompt))

            if model_cfg.api == "chat_completions":
                all_messages = [{"role": "system", "content": system_prompt}, *formatted_messages]
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=self.model,
                    messages=all_messages,
                    reasoning_effort="low",
                )
                content = response.choices[0].message.content
                logger.debug(
                    f"Received Chat Completions response: {len(content)} chars, "
                    f"usage: {response.usage.total_tokens} tokens"
                )
            else:
                api_kwargs = (
                    {"text": {"verbosity": "low"}, "reasoning": {"effort": "low"}}
                    if model_cfg.reasoning
                    else {}
                )
                response = await asyncio.to_thread(
                    self.client.responses.create,
                    model=self.model,
                    instructions=system_prompt,
                    input=formatted_messages,
                    **api_kwargs,
                )
                content = response.output_text
                logger.debug(
                    f"Received Responses API completion: {len(content)} chars, "
                    f"usage: {response.usage.total_tokens} tokens"
                )

            return content

        except openai.AuthenticationError as e:
            logger.error(f"Authentication failed: {e}")
            raise CompletionError(
                "❌ API key is invalid or missing for this model's provider. "
                "Please check your configuration."
            ) from e

        except openai.RateLimitError as e:
            logger.warning(f"Rate limit exceeded: {e}")
            raise CompletionError(
                "⏱️ Rate limit exceeded. "
                "Please wait a moment and try again."
            ) from e

        except openai.APITimeoutError as e:
            logger.warning(f"Request timed out: {e}")
            raise CompletionError(
                f"⏱️ Request timed out after {self.timeout}s. "
                "Please try again."
            ) from e

        except openai.BadRequestError as e:
            error_msg = str(e)
            logger.error(f"Bad request: {error_msg}")

            if "context_length_exceeded" in error_msg:
                raise CompletionError(
                    "❌ Message history is too long for the model. "
                    "Use /clear to clear history and try again."
                ) from e

            raise CompletionError(f"❌ Invalid request: {error_msg}") from e

        except openai.APIConnectionError as e:
            logger.error(f"Connection error: {e}")
            raise CompletionError(
                "❌ Network error connecting to the API. "
                "Please check your internet connection."
            ) from e

        except openai.InternalServerError as e:
            logger.error(f"Server error: {e}")
            raise CompletionError(
                "❌ API service is experiencing issues. "
                "Please try again in a moment."
            ) from e

        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            raise CompletionError(
                "❌ An unexpected error occurred. "
                "Please try again or contact support."
            ) from e

    def test_connection(self) -> bool:
        """
        Test the API connection for the current model.

        Returns:
            True if connection successful, False otherwise
        """
        model_cfg = MODEL_REGISTRY.get(self.model, ModelConfig(api="responses", provider="openai"))
        try:
            if model_cfg.api == "chat_completions":
                self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": "Hi"}],
                )
            else:
                self.client.responses.create(
                    model=self.model,
                    input="Hi",
                )
            logger.info("API connection test successful")
            return True

        except Exception as e:
            logger.error(f"API connection test failed: {e}")
            return False
