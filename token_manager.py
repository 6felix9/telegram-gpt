"""Token counting and context window management using tiktoken."""
import logging
import tiktoken

logger = logging.getLogger(__name__)


class TokenManager:
    """Manage token counting and context window trimming."""

    def __init__(self, model: str, max_tokens: int):
        """
        Initialize token manager.

        Args:
            model: OpenAI model name
            max_tokens: Maximum tokens allowed in context
        """
        self.model = model
        self.max_tokens = max_tokens
        self.encoding = None

        try:
            self.encoding = tiktoken.encoding_for_model(model)
            logger.info(f"Initialized token manager for model {model}")
        except KeyError:
            logger.warning(
                f"Model {model} not found in tiktoken, "
                "using cl100k_base encoding as fallback"
            )
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.error(f"Failed to initialize tiktoken: {e}", exc_info=True)
            self.encoding = None

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

    def count_message_tokens(self, role: str, content: str) -> int:
        """Count tokens for a single message."""
        message = {"role": role, "content": content}
        return self.count_tokens([message]) - 2  # Remove reply priming

    def trim_to_fit(
        self,
        messages: list[dict],
        reserve_tokens: int = 1000
    ) -> list[dict]:
        """
        Trim messages with internal safety checks and a smarter fallback.
        """
        if not messages:
            return []

        try:
            available_tokens = self.max_tokens - reserve_tokens
            
            # 1. Basic Validation: Ensure the last message (the prompt) always exists
            # If the last message alone is too big, we still have to send it and 
            # let the API handle the error, otherwise we send an empty prompt.
            last_msg = messages[-1]
            
            # 2. Start counting tokens for the last message
            current_tokens = self.count_tokens([last_msg])
            kept_messages = [last_msg]

            # 3. Iterate backwards through the rest of the history
            # We use a loop with a try-except to handle individual message corruption
            for message in reversed(messages[:-1]):
                try:
                    msg_tokens = self.count_tokens([message])
                    
                    if current_tokens + msg_tokens <= available_tokens:
                        kept_messages.insert(0, message)
                        current_tokens += msg_tokens
                    else:
                        break # Reached the limit
                except Exception as msg_err:
                    logger.error(f"Failed to process a message during trimming: {msg_err}")
                    continue # Skip the corrupted message and try the next one

            return kept_messages

        except Exception as e:
            logger.error(f"Critical failure in trim_to_fit: {e}", exc_info=True)
            # 4. Ultimate Fallback: Return last 20 messages as conservative fallback.
            # This preserves conversation context while being unlikely to exceed limits.
            # Better to fail with context than succeed with no history.
            return messages[-20:] if messages else []

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

    def get_max_tokens(self) -> int:
        """Return the maximum tokens allowed."""
        return self.max_tokens

    def get_model(self) -> str:
        """Return the model name."""
        return self.model
