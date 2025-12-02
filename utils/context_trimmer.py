"""Context trimming utility for managing token budgets."""
import logging

logger = logging.getLogger(__name__)


def trim_messages_to_fit(
    messages: list[dict],
    max_tokens: int,
    provider,
    reserve_tokens: int = 1000
) -> list[dict]:
    """
    Trim messages to fit within token budget.

    Args:
        messages: List of messages in chronological order
        max_tokens: Maximum tokens allowed in context
        provider: LLM provider instance for token counting
        reserve_tokens: Tokens to reserve for response

    Returns:
        Trimmed list of messages that fit in context window
    """
    if not messages:
        return []

    # Calculate available tokens
    available_tokens = max_tokens - reserve_tokens

    # Always keep the last message (user's current prompt)
    if len(messages) == 1:
        return messages

    try:
        # Start from the end and work backwards
        current_tokens = 0
        kept_messages = []

        for message in reversed(messages):
            msg_tokens = provider.count_tokens([message])

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
