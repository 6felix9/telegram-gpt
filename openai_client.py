"""OpenAI API client wrapper with error handling."""
import logging
import asyncio
import openai

logger = logging.getLogger(__name__)


class OpenAIClient:
    """Wrapper for OpenAI API with comprehensive error handling."""

    SYSTEM_PROMPT = """You are Tze Foong's Assistant. This is your name and identity - never say you are OpenAI or an OpenAI language model.

You are an AI assistant operating in Telegram, and your purpose is to assist Tze Foong with their requests.

Important: When asked who you are or what your name is, always identify yourself as "Tze Foong's Assistant" - never mention OpenAI.

Response style: Be direct and concise. Do not include prose, conversational filler, or preambles. Just respond directly to the request."""

    def __init__(self, api_key: str, model: str, timeout: int):
        """
        Initialize OpenAI client.

        Args:
            api_key: OpenAI API key
            model: Model name (e.g., "gpt-4o-mini")
            timeout: Request timeout in seconds
        """
        self.client = openai.OpenAI(api_key=api_key, timeout=timeout)
        self.model = model
        self.timeout = timeout

        logger.info(f"Initialized OpenAI client with model {model}")

    async def get_completion(self, messages: list[dict]) -> str:
        """
        Get completion from OpenAI API.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            Assistant's response text or error message
        """
        try:
            logger.debug(f"Requesting completion with {len(messages)} messages")

            # Prepend system prompt
            system_message = {"role": "system", "content": self.SYSTEM_PROMPT}
            messages_with_system = [system_message] + messages

            # Run sync OpenAI call in thread pool
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=messages_with_system,
                temperature=0.7,  # Balanced creativity
            )

            content = response.choices[0].message.content

            logger.debug(
                f"Received completion: {len(content)} chars, "
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
            # Simple test with minimal tokens
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
            )

            logger.info("OpenAI API connection test successful")
            return True

        except Exception as e:
            logger.error(f"OpenAI API connection test failed: {e}")
            return False
