"""Groq provider implementation."""
import logging
import asyncio
import tiktoken
from llm_provider import LLMProvider

logger = logging.getLogger(__name__)


class GroqProvider(LLMProvider):
    """Groq API provider implementation (OpenAI-compatible)."""

    SYSTEM_PROMPT = """You are Tze Foong's Assistant. This is your name and identity - never say you are Groq or mention Groq.

You are an AI assistant operating in Telegram, and your purpose is to assist Tze Foong with their requests.

Important: When asked who you are or what your name is, always identify yourself as "Tze Foong's Assistant" - never mention Groq or the underlying model.

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
        "openai/gpt-oss-120b": 128000,
        "llama-3.3-70b-versatile": 128000,
        "llama-3.1-8b-instant": 128000,
        "mixtral-8x7b-32768": 32768,
        "llama-3.1-70b-versatile": 128000,
    }

    def __init__(self, api_key: str, model: str, timeout: int):
        """
        Initialize Groq provider.

        Args:
            api_key: Groq API key
            model: Model name (e.g., "llama-3.3-70b-versatile")
            timeout: Request timeout in seconds
        """
        try:
            from groq import Groq
        except ImportError:
            logger.error("groq package not installed. Run: pip install groq")
            raise

        self.client = Groq(api_key=api_key, timeout=timeout)
        self.model = model
        self.timeout = timeout
        self.encoding = None

        # Initialize tiktoken encoding for token counting
        try:
            # Use cl100k_base for Llama models
            self.encoding = tiktoken.get_encoding("cl100k_base")
            logger.info(f"Initialized Groq provider with model {model}")
        except Exception as e:
            logger.warning(f"Failed to initialize tiktoken: {e}")
            self.encoding = None

    async def get_completion(self, messages: list[dict], **kwargs) -> str:
        """
        Get completion from Groq API.

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

            # Run sync Groq call in thread pool
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

        except Exception as e:
            error_msg = str(e).lower()
            logger.error(f"Groq API error: {e}", exc_info=True)

            if "authentication" in error_msg or "api key" in error_msg:
                return (
                    "❌ Groq API key is invalid. "
                    "Please check your configuration."
                )
            elif "rate limit" in error_msg:
                return (
                    "⏱️ Rate limit exceeded. "
                    "Please wait a moment and try again."
                )
            elif "timeout" in error_msg:
                return (
                    f"⏱️ Request timed out after {self.timeout}s. "
                    "Please try again."
                )
            elif "context" in error_msg or "too long" in error_msg:
                return (
                    "❌ Message history is too long for the model. "
                    "Use /clear to clear history and try again."
                )
            elif "connection" in error_msg or "network" in error_msg:
                return (
                    "❌ Network error connecting to Groq. "
                    "Please check your internet connection."
                )
            else:
                return (
                    f"❌ Groq API error: {str(e)}"
                )

    async def test_connection(self) -> bool:
        """
        Test the Groq API connection.

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

            logger.info("Groq API connection test successful")
            return True

        except Exception as e:
            logger.error(f"Groq API connection test failed: {e}")
            return False

    def get_model_name(self) -> str:
        """Return current model identifier."""
        return self.model

    def get_max_context_tokens(self) -> int:
        """Return max context window for current model."""
        return self.MODEL_LIMITS.get(self.model, 32768)

    def count_tokens(self, messages: list[dict]) -> int:
        """
        Count tokens in message list using tiktoken.

        Args:
            messages: List of message dicts

        Returns:
            Total token count
        """
        if not self.encoding:
            # Fallback: estimate tokens
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
        Format messages for Groq API (OpenAI-compatible).

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
