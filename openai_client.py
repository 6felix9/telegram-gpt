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
                    if is_group and msg["role"] == "user":
                        sender_name = msg.get("sender_name", "Unknown")
                        updated_content = []
                        for part in content:
                            # Convert old Chat Completions format to Responses API format
                            if part.get("type") == "text":
                                text = part["text"]
                                if not text.startswith("["):
                                    text = f"[{sender_name}]: {text}"
                                updated_content.append({"type": "input_text", "text": text})
                            elif part.get("type") == "image_url":
                                # Convert image_url object to string format for Responses API
                                image_url_obj = part.get("image_url", {})
                                image_url_str = image_url_obj.get("url", "") if isinstance(image_url_obj, dict) else str(image_url_obj)
                                updated_content.append({"type": "input_image", "image_url": image_url_str})
                            else:
                                # Handle other types (input_text, input_image if already converted)
                                updated_content.append(part)
                        formatted_messages.append({
                            "role": msg["role"],
                            "content": updated_content
                        })
                    else:
                        # Non-group chat: still need to convert format
                        updated_content = []
                        for part in content:
                            if part.get("type") == "text":
                                updated_content.append({"type": "input_text", "text": part["text"]})
                            elif part.get("type") == "image_url":
                                # Convert image_url object to string format for Responses API
                                image_url_obj = part.get("image_url", {})
                                image_url_str = image_url_obj.get("url", "") if isinstance(image_url_obj, dict) else str(image_url_obj)
                                updated_content.append({"type": "input_image", "image_url": image_url_str})
                            else:
                                # Already in Responses API format or other type
                                updated_content.append(part)
                        formatted_messages.append({
                            "role": msg["role"],
                            "content": updated_content
                        })

            # Choose system prompt based on chat type
            system_prompt = self.SYSTEM_PROMPT_GROUP if is_group else self.SYSTEM_PROMPT
            
            # Run sync OpenAI call in thread pool using Responses API
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
