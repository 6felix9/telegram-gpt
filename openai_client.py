"""OpenAI API client wrapper with error handling."""
import logging
import asyncio
import openai
import httpx
from prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


class OpenAIClient:
    """Wrapper for OpenAI API with comprehensive error handling."""

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
        api_key: str,
        model: str,
        timeout: int,
        base_url: str | None = None,
        prompt_builder: PromptBuilder | None = None,
    ):
        """
        Initialize OpenAI client.

        Args:
            api_key: OpenAI API key (or xAI API key)
            model: Model name (e.g., "gpt-4o-mini" or "grok-4")
            timeout: Request timeout in seconds
            base_url: Optional base URL for API (e.g., "https://api.x.ai/v1" for xAI)
            prompt_builder: Shared prompt builder for system/messages assembly
        """
        # Use httpx.Timeout for better timeout handling, especially for reasoning models
        timeout_obj = httpx.Timeout(float(timeout))
        
        # Initialize client with optional base_url
        client_kwargs = {
            "api_key": api_key,
            "timeout": timeout_obj,
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        
        self.client = openai.OpenAI(**client_kwargs)
        self.model = model
        self.timeout = timeout
        self.base_url = base_url
        self.prompt_builder = prompt_builder or PromptBuilder(
            default_private_prompt=self.SYSTEM_PROMPT,
            default_group_prompt=self.SYSTEM_PROMPT_GROUP,
        )

        api_provider = "xAI" if base_url else "OpenAI"
        logger.info(f"Initialized {api_provider} client with model {model}" + (f" (base_url: {base_url})" if base_url else ""))

    async def get_completion(self, messages: list[dict], is_group: bool = False, custom_system_prompt: str | None = None) -> str:
        """
        Get completion from OpenAI API.

        Args:
            messages: List of message dicts with 'role', 'content', and optionally sender info
            is_group: Whether this is a group chat (affects formatting and system prompt)
            custom_system_prompt: Optional custom system prompt to use instead of default

        Returns:
            Assistant's response text or error message
        """
        try:
            logger.debug(f"Requesting completion with {len(messages)} messages (group={is_group})")

            formatted_messages = self.prompt_builder.format_messages(messages, is_group)
            system_prompt = self.prompt_builder.build_system_prompt(
                is_group=is_group,
                custom_system_prompt=custom_system_prompt,
            )

            # Run sync OpenAI call in thread pool using Responses API
            # GPT-5 models don't support temperature parameter
            if self.model.startswith("gpt-5"):
                response = await asyncio.to_thread(
                    self.client.responses.create,
                    model=self.model,
                    instructions=system_prompt,
                    input=formatted_messages,
                    text={ "verbosity": "low" },
                    reasoning={ "effort": "low" },
                )
            else:
                response = await asyncio.to_thread(
                    self.client.responses.create,
                    model=self.model,
                    instructions=system_prompt,
                    input=formatted_messages,
                    temperature=0.7,  # Balanced creativity
                )

            content = response.output_text

            logger.debug(
                f"Received Responses API completion: {len(content)} chars, "
                f"usage: {response.usage.total_tokens} tokens"
            )

            return content

        except openai.AuthenticationError as e:
            logger.error(f"Authentication failed: {e}")
            return (
                "❌ OpenAI API key is invalid. "
                "Please check your configuration."
            )

        except openai.RateLimitError as e:
            logger.warning(f"Rate limit exceeded: {e}")
            return (
                "⏱️ Rate limit exceeded. "
                "Please wait a moment and try again."
            )

        except openai.APITimeoutError as e:
            logger.warning(f"Request timed out: {e}")
            return (
                f"⏱️ Request timed out after {self.timeout}s. "
                "Please try again."
            )

        except openai.BadRequestError as e:
            error_msg = str(e)
            logger.error(f"Bad request: {error_msg}")

            if "context_length_exceeded" in error_msg:
                return (
                    "❌ Message history is too long for the model. "
                    "Use /clear to clear history and try again."
                )

            return f"❌ Invalid request: {error_msg}"

        except openai.APIConnectionError as e:
            logger.error(f"Connection error: {e}")
            return (
                "❌ Network error connecting to OpenAI. "
                "Please check your internet connection."
            )

        except openai.InternalServerError as e:
            logger.error(f"OpenAI server error: {e}")
            return (
                "❌ OpenAI service is experiencing issues. "
                "Please try again in a moment."
            )

        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            return (
                "❌ An unexpected error occurred. "
                "Please try again or contact support."
            )

    def test_connection(self) -> bool:
        """
        Test the OpenAI API connection.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Simple test with minimal tokens using Responses API
            response = self.client.responses.create(
                model=self.model,
                input="Hi",
            )

            logger.info("OpenAI API connection test successful")
            return True

        except Exception as e:
            logger.error(f"OpenAI API connection test failed: {e}")
            return False
