"""OpenAI API client wrapper with error handling."""
import logging
import asyncio
import openai

logger = logging.getLogger(__name__)


class OpenAIClient:
    """Wrapper for OpenAI API with comprehensive error handling."""

    SYSTEM_PROMPT = """You are Tze Foong's Assistant. This is your name and identity - never say you are OpenAI or an OpenAI language model.

You are an AI assistant operating in Telegram, and your purpose is to assist Tze Foong with their requests.

Response style: Be direct and concise. Do not include prose, conversational filler, or preambles. Just respond directly to the request."""

    SYSTEM_PROMPT_GROUP = """You are Tze Foong's Assistant, an AI assistant operating in a Telegram group chat.

Your purpose is to assist Tze Foong and provide helpful responses based on the conversation context.

Important:
- Messages are formatted as [Name]: message content
- Pay attention to who is speaking and reference previous messages when relevant
- When someone says "answer her question" or similar, look at the previous messages to understand the context
- Be conversational and context-aware of the group discussion
- Just respond directly, no need to prefix your response with [Tze Foong's Assistant]"""

    def __init__(self, api_key: str, model: str, timeout: int, base_url: str | None = None):
        """
        Initialize OpenAI client (supports both OpenAI and Gemini via OpenAI SDK).

        Args:
            api_key: API key (OpenAI or Gemini)
            model: Model name (e.g., "gpt-4o-mini" or "gemini-2.5-flash")
            timeout: Request timeout in seconds
            base_url: Base URL for API (None for OpenAI, Gemini URL for Gemini)
        """
        if base_url:
            self.client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
            logger.info(f"Initialized Gemini client with model {model}")
        else:
            self.client = openai.OpenAI(api_key=api_key, timeout=timeout)
            logger.info(f"Initialized OpenAI client with model {model}")
        self.model = model
        self.timeout = timeout
        self.base_url = base_url

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
                content = msg["content"]

                # Handle text-only messages
                if isinstance(content, str):
                    formatted_content = content

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

                # Handle multimodal messages (image + text)
                elif isinstance(content, list):
                    updated_content = []
                    for part in content:
                        # Handle text parts
                        if part.get("type") == "text":
                            text = part["text"]
                            # For group chats, prepend sender name to user messages
                            if is_group and msg["role"] == "user":
                                sender_name = msg.get("sender_name", "Unknown")
                                if not text.startswith("["):
                                    text = f"[{sender_name}]: {text}"
                            updated_content.append({
                                "type": "text",
                                "text": text
                            })
                        # Handle image parts (standard OpenAI format)
                        elif part.get("type") == "image_url":
                            image_url_obj = part.get("image_url", {})
                            if isinstance(image_url_obj, dict):
                                image_url_str = image_url_obj.get("url", "")
                            else:
                                image_url_str = str(image_url_obj)
                            updated_content.append({
                                "type": "image_url",
                                "image_url": {"url": image_url_str}
                            })
                        else:
                            # Pass through other formats
                            updated_content.append(part)
                    
                    formatted_messages.append({
                        "role": msg["role"],
                        "content": updated_content
                    })

            # Choose system prompt based on chat type
            system_prompt = self.SYSTEM_PROMPT_GROUP if is_group else self.SYSTEM_PROMPT
            
            # Add system message to the beginning of messages
            formatted_messages.insert(0, {
                "role": "system",
                "content": system_prompt
            })
            
            # Run sync OpenAI call in thread pool using chat.completions API
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=formatted_messages,
                temperature=0.7,  # Balanced creativity
            )

            content = response.choices[0].message.content

            # Normalize and guard against None / list payloads (e.g., Gemini-style parts)
            if content is None:
                logger.error(f"Provider returned empty content: {response}")
                return "❌ Model returned empty content. Please try again."

            if isinstance(content, list):
                content = " ".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )

            logger.debug(
                f"Received chat completion: {len(content)} chars, "
                f"usage: {response.usage.total_tokens} tokens"
            )

            return content

        except openai.AuthenticationError as e:
            logger.error(f"Authentication failed: {e}")
            provider = "Gemini" if self.base_url else "OpenAI"
            return (
                f"❌ {provider} API key is invalid. "
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
            provider = "Gemini" if self.base_url else "OpenAI"
            return (
                f"❌ Network error connecting to {provider}. "
                "Please check your internet connection."
            )

        except openai.InternalServerError as e:
            logger.error(f"API server error: {e}")
            provider = "Gemini" if self.base_url else "OpenAI"
            return (
                f"❌ {provider} service is experiencing issues. "
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
        Test the API connection.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Simple test with minimal tokens using chat.completions
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hi"}],
            )

            provider = "Gemini" if self.base_url else "OpenAI"
            logger.info(f"{provider} API connection test successful")
            return True

        except Exception as e:
            provider = "Gemini" if self.base_url else "OpenAI"
            logger.error(f"{provider} API connection test failed: {e}")
            return False
