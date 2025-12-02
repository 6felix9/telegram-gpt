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

    SYSTEM_PROMPT_GROUP = """You are Tze Foong's Assistant, an AI assistant operating in a Telegram group chat.

Your purpose is to assist Tze Foong and provide helpful responses based on the conversation context.

Important:
- Messages are formatted as [Name]: message content
- Pay attention to who is speaking and reference previous messages when relevant
- When someone says "answer her question" or similar, look at the previous messages to understand the context
- Be conversational and context-aware of the group discussion"""

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

    async def get_completion(self, messages: list[dict], is_group: bool = False) -> str:
        """
        Get completion from OpenAI API.

        Args:
            messages: List of message dicts with 'role', 'content', and optionally sender info
            is_group: Whether this is a group chat (affects formatting and system prompt)

        Returns:
            Assistant's response text or error message
        """
        try:
            logger.debug(f"Requesting completion with {len(messages)} messages (group={is_group})")

            # Format messages for group chats with sender names
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

            # Choose system prompt based on chat type
            system_prompt = self.SYSTEM_PROMPT_GROUP if is_group else self.SYSTEM_PROMPT
            system_message = {"role": "system", "content": system_prompt}
            messages_with_system = [system_message] + formatted_messages
            
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

    async def get_completion_with_usage(
        self,
        messages: list[dict],
        is_group: bool = False
    ) -> tuple[str, dict]:
        """
        Get completion from OpenAI API with usage statistics.

        Args:
            messages: List of message dicts. Can include:
                - Text: {"role": "user", "content": "text"}
                - Multimodal: {"role": "user", "content": [{"type": "text", ...}, {"type": "image_url", ...}]}
            is_group: Whether this is a group chat

        Returns:
            Tuple of (response_text, usage_dict) where usage_dict contains:
                - prompt_tokens: int
                - completion_tokens: int
                - total_tokens: int
        """
        try:
            logger.debug(f"Requesting completion with {len(messages)} messages (group={is_group})")

            # Format messages for group chats with sender names
            formatted_messages = []
            for msg in messages:
                content = msg["content"]

                # Handle text-only messages
                if isinstance(content, str):
                    formatted_content = content

                    if is_group and msg["role"] == "user":
                        sender_name = msg.get("sender_name", "Unknown")
                        if not formatted_content.startswith("["):
                            formatted_content = f"[{sender_name}]: {formatted_content}"

                    formatted_messages.append({
                        "role": msg["role"],
                        "content": formatted_content
                    })

                # Handle multimodal messages (image + text)
                elif isinstance(content, list):
                    if is_group and msg["role"] == "user":
                        sender_name = msg.get("sender_name", "Unknown")
                        updated_content = []
                        for part in content:
                            if part.get("type") == "text":
                                text = part["text"]
                                if not text.startswith("["):
                                    text = f"[{sender_name}]: {text}"
                                updated_content.append({"type": "text", "text": text})
                            else:
                                updated_content.append(part)
                        formatted_messages.append({
                            "role": msg["role"],
                            "content": updated_content
                        })
                    else:
                        formatted_messages.append({
                            "role": msg["role"],
                            "content": content
                        })

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
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

            logger.debug(
                f"Received completion: {len(content)} chars, "
                f"usage: {usage['total_tokens']} tokens "
                f"({usage['prompt_tokens']} prompt + {usage['completion_tokens']} completion)"
            )

            return content, usage

        except openai.AuthenticationError as e:
            logger.error(f"Authentication failed: {e}")
            return (
                "❌ OpenAI API key is invalid. Please check your configuration.",
                {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            )

        except openai.RateLimitError as e:
            logger.warning(f"Rate limit exceeded: {e}")
            return (
                "⏱️ Rate limit exceeded. Please wait a moment and try again.",
                {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            )

        except openai.APITimeoutError as e:
            logger.warning(f"Request timed out: {e}")
            return (
                f"⏱️ Request timed out after {self.timeout}s. Please try again.",
                {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            )

        except openai.BadRequestError as e:
            error_msg = str(e)
            logger.error(f"Bad request: {error_msg}")

            if "context_length_exceeded" in error_msg:
                return (
                    "❌ Message history is too long for the model. Use /clear to clear history and try again.",
                    {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                )

            return (
                f"❌ Invalid request: {error_msg}",
                {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            )

        except openai.APIConnectionError as e:
            logger.error(f"Connection error: {e}")
            return (
                "❌ Network error connecting to OpenAI. Please check your internet connection.",
                {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            )

        except openai.InternalServerError as e:
            logger.error(f"OpenAI server error: {e}")
            return (
                "❌ OpenAI service is experiencing issues. Please try again in a moment.",
                {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            )

        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            return (
                "❌ An unexpected error occurred. Please try again or contact support.",
                {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
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
