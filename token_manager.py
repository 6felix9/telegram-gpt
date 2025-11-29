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
        Trim messages to fit in context window.

        Args:
            messages: List of messages in chronological order
            reserve_tokens: Tokens to reserve for response

        Returns:
            Trimmed list of messages that fit in context window
        """
        if not messages:
            return []

        # Calculate available tokens
        available_tokens = self.max_tokens - reserve_tokens

        # Always keep the last message (user's current prompt)
        if len(messages) == 1:
            return messages

        try:
            # Start from the end and work backwards
            current_tokens = 0
            kept_messages = []

            for message in reversed(messages):
                msg_tokens = self.count_tokens([message])

                if current_tokens + msg_tokens <= available_tokens:
                    kept_messages.insert(0, message)
                    current_tokens += msg_tokens
                elif not kept_messages:
                    # If even the last message doesn't fit, keep it anyway
                    kept_messages = [message]
                    logger.warning(
                        f"Last message exceeds token budget "
                        f"({msg_tokens} > {available_tokens})"
                    )
                    break
                else:
                    # Stop adding messages
                    break

            trimmed_count = len(messages) - len(kept_messages)
            if trimmed_count > 0:
                logger.info(
                    f"Trimmed {trimmed_count} messages to fit context window "
                    f"({current_tokens}/{available_tokens} tokens)"
                )

            return kept_messages

        except Exception as e:
            logger.error(f"Trimming failed: {e}", exc_info=True)
            # Fallback: keep last N messages
            return messages[-20:]  # Conservative fallback

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
