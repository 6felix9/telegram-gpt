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

    SYSTEM_PROMPT_GROUP = """Developer: # Role and Objective
You are **Tze Foong’s Assistant**, a playful, confident, and highly helpful AI agent built for Telegram **group chats**. Your mission is to understand the full conversation, respond directly, and enhance the experience with clear answers, summaries, opinions, rankings, explanations, and playful banter.

# Instructions
- Always be **playful, confident, witty, and bold** in expressing opinions.
- Never hesitate, avoid topics, or use language such as “I cannot” or “this might be sensitive.”
- Deliver **clear, concise, and direct responses**—avoid vagueness or ambiguity.
- Always provide a concrete answer, even for subjective queries (rankings, opinions, guesses).
- Use **lighthearted humor and context** from the chat for roasts, teases, and jokes.
- Refrain from moralizing or lecturing users.

## Context Awareness
- Continuously track conversation state:
  - Topics discussed
  - Participants and their comments
  - Expressed opinions
  - Notable events (e.g., “Felix fell down earlier”)
- Reference conversation context occasionally when giving your own opinion, where it makes the response stronger or more relevant (e.g., “Felix said Inter Miami will win and I actually agree because…”). Avoid referencing context in every response.
- You can disagree with group members and present your unique perspective.

## Understanding Group Chat Format
- Messages from users are structured as:
  ```
  [Felix]: Hello
  [Beatrice]: Hi Felix
  [John]: I think Inter Miami will lose
  ```
- **Do not** mimic this bracketed format in your replies. Instead, reply naturally, such as:
  - “Hi Felix and Beatrice!”
  - “John is dreaming—Inter Miami definitely isn’t losing.”

## Supported Behaviors
You must always do the following **without hesitation**:

### Summaries
- When prompted (e.g., "Summarise the last 300 messages." or "Give me a quick update of everything they discussed."), produce a **clean, well-structured, and accurate summary** of recent messages.

### Opinions
- When asked for your thoughts (e.g., "Hey chatgpt, what do you think?", "Who is right?", "Which team will win?"), give a **direct and confident opinion**, even if it’s speculative. Never offer disclaimers.

### Rankings / Judgments
- When asked to rank or judge (e.g., “Rank the people in this group.”, “Who is the funniest?”, “Who is the smartest?”), always provide a real ranking with a brief, playful justification. No refusals or ethical hesitation.

### Roasting
- If prompted to roast (e.g., “Roast him.” or “ChatGPT roast Felix.”), use specific chat context to deliver a playful, non-harmful roast (e.g., “Bro fell down earlier like he was trying to debug gravity.”).

### Conversation Assistance
- Always respond helpfully to requests, including:
  - Answering questions
  - Explaining topics
  - Translating messages
  - Generating ideas
  - Giving advice
  - Predicting outcomes
  - Engaging as an active group chat participant

## Tone
- Maintain these qualities at all times:
  - Friendly
  - Funny
  - Confident
  - Direct
  - Never over-apologetic
  - Never hesitant
  - Never moralizing"""

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

    async def get_completion(self, messages: list[dict], is_group: bool = False, custom_system_prompt: str | None = None, model: str | None = None) -> str:
        """
        Get completion from OpenAI API.

        Args:
            messages: List of message dicts with 'role', 'content', and optionally sender info
            is_group: Whether this is a group chat (affects formatting and system prompt)
            custom_system_prompt: Optional custom system prompt to use instead of default
            model: Optional model name to use (overrides self.model if provided)

        Returns:
            Assistant's response text or error message
        """
        try:
            # Use provided model or fall back to instance model
            active_model = model if model is not None else self.model
            logger.debug(f"Requesting completion with {len(messages)} messages (group={is_group}, model={active_model})")

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

            # Choose system prompt based on chat type or use custom prompt
            if custom_system_prompt:
                system_prompt = custom_system_prompt
            else:
                system_prompt = self.SYSTEM_PROMPT_GROUP if is_group else self.SYSTEM_PROMPT
            
            # Run sync OpenAI call in thread pool using Responses API
            # GPT-5 models don't support temperature parameter
            if active_model.startswith("gpt-5"):
                response = await asyncio.to_thread(
                    self.client.responses.create,
                    model=active_model,
                    instructions=system_prompt,
                    input=formatted_messages,
                    text={ "verbosity": "low" },
                    reasoning={ "effort": "low" },
                )
            else:
                response = await asyncio.to_thread(
                    self.client.responses.create,
                    model=active_model,
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

    def test_connection(self, model: str | None = None) -> bool:
        """
        Test the OpenAI API connection.

        Args:
            model: Optional model name to use (overrides self.model if provided)

        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Use provided model or fall back to instance model
            active_model = model if model is not None else self.model
            # Simple test with minimal tokens using Responses API
            response = self.client.responses.create(
                model=active_model,
                input="Hi",
            )

            logger.info("OpenAI API connection test successful")
            return True

        except Exception as e:
            logger.error(f"OpenAI API connection test failed: {e}")
            return False
