"""Gemini provider implementation."""
import asyncio
import logging
from llm_provider import LLMProvider

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    """Google Gemini API provider implementation."""

    SYSTEM_PROMPT = """You are Tze Foong's Assistant. This is your name and identity - never say you are Google or a Google AI model.

You are an AI assistant operating in Telegram, and your purpose is to assist Tze Foong with their requests.

Important: When asked who you are or what your name is, always identify yourself as "Tze Foong's Assistant" - never mention Google or Gemini.

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
        "gemini-2.5-flash-preview-09-2025": 200000,
        "gemini-2.5-flash-lite-preview-09-2025": 200000,
        "gemini-1.5-pro": 2097152,
        "gemini-1.5-flash": 1048576,
    }

    def __init__(self, api_key: str, model: str, timeout: int):
        """
        Initialize Gemini provider.

        Args:
            api_key: Gemini API key
            model: Model name (e.g., "gemini-2.5-flash-preview-09-2025")
            timeout: Request timeout in seconds
        """
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            logger.error("google-genai package not installed. Run: pip install google-genai")
            raise

        self.client = genai.Client(api_key=api_key, http_options={'timeout': timeout})
        self.model_name = model
        self.timeout = timeout
        self.genai = genai
        self.types = types

        logger.info(f"Initialized Gemini provider with model {model}")

    async def get_completion(self, messages: list[dict], **kwargs) -> str:
        """
        Get completion from Gemini API.

        Args:
            messages: List of message dicts with 'role', 'content', and optionally sender info
            **kwargs: Additional options (is_group: bool)

        Returns:
            Assistant's response text or error message
        """
        is_group = kwargs.get("is_group", False)

        try:
            logger.debug(f"Requesting completion with {len(messages)} messages (group={is_group})")

            # Format messages for Gemini
            formatted_messages = self.format_messages(messages, is_group)

            # Choose system prompt based on chat type
            system_prompt = self.SYSTEM_PROMPT_GROUP if is_group else self.SYSTEM_PROMPT

            # Convert messages to Gemini format
            contents = []
            for msg in formatted_messages:
                role = "model" if msg["role"] == "assistant" else "user"
                contents.append(self.types.Content(
                    role=role,
                    parts=[self.types.Part(text=msg["content"])]
                ))

            # Generate response (run blocking call in thread pool)
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=contents,
                config=self.types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.7,
                )
            )

            content = response.text

            logger.debug(f"Received completion: {len(content)} chars")

            return content

        except Exception as e:
            error_msg = str(e).lower()
            logger.error(f"Gemini API error: {e}", exc_info=True)

            if "api key" in error_msg or "authentication" in error_msg:
                return (
                    "❌ Gemini API key is invalid. "
                    "Please check your configuration."
                )
            elif "quota" in error_msg or "rate limit" in error_msg:
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
            else:
                return (
                    f"❌ Gemini API error: {str(e)}"
                )

    async def test_connection(self) -> bool:
        """
        Test the Gemini API connection.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Simple test with minimal tokens (run blocking call in thread pool)
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents="Hi",
                config=self.types.GenerateContentConfig(
                    max_output_tokens=5,
                )
            )

            logger.info("Gemini API connection test successful")
            return True

        except Exception as e:
            logger.error(f"Gemini API connection test failed: {e}")
            return False

    def get_model_name(self) -> str:
        """Return current model identifier."""
        return self.model_name

    def get_max_context_tokens(self) -> int:
        """Return max context window for current model."""
        return self.MODEL_LIMITS.get(self.model_name, 1000000)

    def count_tokens(self, messages: list[dict]) -> int:
        """
        Count tokens in message list using estimation.

        Note: Uses fallback estimation to avoid blocking the event loop.
        Gemini's count_tokens API is a blocking network call, so we use
        a local estimation (4 chars ≈ 1 token) for better performance.

        Args:
            messages: List of message dicts

        Returns:
            Total token count (estimated)
        """
        # Use fallback estimation to avoid blocking
        # The Gemini token counting API would block the event loop
        return self._estimate_tokens(messages)

    def format_messages(self, messages: list[dict], is_group: bool) -> list[dict]:
        """
        Format messages for Gemini API.

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
        """Fallback token estimation."""
        total_chars = 0

        for message in messages:
            for value in message.values():
                total_chars += len(str(value))

        # Rough estimate: 4 characters ≈ 1 token
        estimated = (total_chars // 4) + (len(messages) * 4)

        logger.debug(f"Estimated {estimated} tokens (fallback method)")
        return estimated
