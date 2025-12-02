"""OpenAI provider implementation."""
import logging
import asyncio
import openai
import tiktoken
from llm_provider import LLMProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI API provider implementation."""

    SYSTEM_PROMPT = """You are Tze Foong's Assistant. This is your name and identity - never say you are OpenAI or an OpenAI language model.

You are an AI assistant operating in Telegram, and your purpose is to assist Tze Foong with their requests.

Important: When asked who you are or what your name is, always identify yourself as "Tze Foong's Assistant" - never mention OpenAI.

Response style: Be direct and concise. Do not include prose, conversational filler, or preambles. Just respond directly to the request."""

    SYSTEM_PROMPT_GROUP = """You are Tze Foong's Assistant, an AI assistant operating in a Telegram group chat.

Your purpose is to assist Tze Foong and provide helpful responses based on the conversation context.

Important:
- Messages are formatted as [Name]: message content
- Pay attention to who is speaking and reference previous messages when relevant
- When someone says "answer her question" or similar, look at the previous messages to understand the context
- Be conversational and context-aware of the group discussion"""

    # Model context limits
    MODEL_LIMITS = {
        "gpt-4o-mini": 128000,
        "gpt-4o": 128000,
        "gpt-3.5-turbo": 16385,
        "gpt-4-turbo": 128000,
        "gpt-4": 8192,
    }

    def __init__(self, api_key: str, model: str, timeout: int):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key
            model: Model name (e.g., "gpt-4o-mini")
            timeout: Request timeout in seconds
        """
        self.client = openai.OpenAI(api_key=api_key, timeout=timeout)
        self.model = model
        self.timeout = timeout
        self.encoding = None

        # Initialize tiktoken encoding
        try:
            self.encoding = tiktoken.encoding_for_model(model)
            logger.info(f"Initialized OpenAI provider with model {model}")
        except KeyError:
            logger.warning(
                f"Model {model} not found in tiktoken, "
                "using cl100k_base encoding as fallback"
            )
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.error(f"Failed to initialize tiktoken: {e}", exc_info=True)
            self.encoding = None

    async def get_completion(self, messages: list[dict], **kwargs) -> str:
        """
        Get completion from OpenAI API.

        Args:
            messages: List of message dicts with 'role', 'content', and optionally sender info
            **kwargs: Additional options (is_group: bool)

        Returns:
            Assistant's response text or error message
        """
        is_group = kwargs.get("is_group", False)

        try:
            logger.debug(f"Requesting completion with {len(messages)} messages (group={is_group})")

            # Format messages
            formatted_messages = self.format_messages(messages, is_group)

            # Choose system prompt based on chat type
            system_prompt = self.SYSTEM_PROMPT_GROUP if is_group else self.SYSTEM_PROMPT
            system_message = {"role": "system", "content": system_prompt}
            messages_with_system = [system_message] + formatted_messages

            # Run sync OpenAI call in thread pool
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=messages_with_system,
                temperature=0.7,
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

    async def test_connection(self) -> bool:
        """
        Test the OpenAI API connection.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Simple test with minimal tokens
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
            )

            logger.info("OpenAI API connection test successful")
            return True

        except Exception as e:
            logger.error(f"OpenAI API connection test failed: {e}")
            return False

    def get_model_name(self) -> str:
        """Return current model identifier."""
        return self.model

    def get_max_context_tokens(self) -> int:
        """Return max context window for current model."""
        return self.MODEL_LIMITS.get(self.model, 8000)

    def count_tokens(self, messages: list[dict]) -> int:
        """
        Count tokens in message list.

        OpenAI uses approximately 4 tokens per message for formatting:
        - <im_start>{role/name}\n{content}<im_end>\n
        """
        if not self.encoding:
            # Fallback: estimate 4 chars ≈ 1 token
            return self._estimate_tokens(messages)

        try:
            num_tokens = 0

            for message in messages:
                # Every message follows <im_start>{role/name}\n{content}<im_end>\n
                num_tokens += 4

                for key, value in message.items():
                    num_tokens += len(self.encoding.encode(str(value)))

                    # If there's a name, add special tokens
                    if key == "name":
                        num_tokens += -1  # Role is omitted if name present

            num_tokens += 2  # Every reply is primed with <im_start>assistant

            return num_tokens

        except Exception as e:
            logger.error(f"Token counting failed: {e}", exc_info=True)
            return self._estimate_tokens(messages)

    def format_messages(self, messages: list[dict], is_group: bool) -> list[dict]:
        """
        Format messages for OpenAI API.

        Args:
            messages: Raw message list
            is_group: Whether this is a group chat

        Returns:
            Formatted message list
        """
        formatted_messages = []

        for msg in messages:
            formatted_content = msg["content"]

            # For group chats, prepend sender name to user messages
            if is_group and msg["role"] == "user":
                sender_name = msg.get("sender_name", "Unknown")
                # Only add name prefix if not already there
                if not formatted_content.startswith("["):
                    formatted_content = f"[{sender_name}]: {formatted_content}"

            formatted_messages.append({
                "role": msg["role"],
                "content": formatted_content
            })

        return formatted_messages

    def _estimate_tokens(self, messages: list[dict]) -> int:
        """Fallback token estimation when tiktoken is unavailable."""
        total_chars = 0

        for message in messages:
            for value in message.values():
                total_chars += len(str(value))

        # Rough estimate: 4 characters ≈ 1 token
        # Add overhead for message formatting
        estimated = (total_chars // 4) + (len(messages) * 4)

        logger.debug(f"Estimated {estimated} tokens (fallback method)")
        return estimated
