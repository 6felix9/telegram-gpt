"""LangChain agent: model resolution, middleware, tools wiring, and the
Telegram-facing entry point. Replaces openai_client.py and token_manager.py."""
from __future__ import annotations

import logging

import tiktoken
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage

logger = logging.getLogger(__name__)

# tiktoken encoding is model-independent for our budgeting purposes.
_ENCODING = tiktoken.get_encoding("cl100k_base")


class CompletionError(Exception):
    """Agent run failed; user_message is safe to show in Telegram."""

    def __init__(self, user_message: str):
        self.user_message = user_message
        super().__init__(user_message)


def count_tokens(text: str) -> int:
    """Token count of a plain string."""
    if not text:
        return 0
    try:
        return len(_ENCODING.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def _message_text(message: BaseMessage) -> str:
    """Flatten a message's content (str or content blocks) to countable text."""
    content = message.content
    if isinstance(content, str):
        return content
    parts: list[str] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                if block.get("type") in ("text", "input_text"):
                    parts.append(str(block.get("text", "")))
                elif block.get("type") in ("image_url", "image", "input_image"):
                    # Do not count base64 payloads; charge a flat image cost instead.
                    parts.append("[image]")
            else:
                parts.append(str(block))
    return " ".join(parts)


def _has_image(message: BaseMessage) -> bool:
    content = message.content
    return isinstance(content, list) and any(
        isinstance(b, dict) and b.get("type") in ("image_url", "image", "input_image")
        for b in content
    )


def count_message_tokens(message: BaseMessage) -> int:
    """Approximate token count of one message, including per-message overhead."""
    return count_tokens(_message_text(message)) + 4


def trim_messages(
    messages: list[BaseMessage],
    max_context_tokens: int,
    reserve_text: int,
    reserve_image: int,
) -> list[BaseMessage]:
    """Keep as much recent history as fits the budget, newest-first.

    Non-destructive: returns a new list. Always keeps the last message.
    Never returns a list beginning with a ToolMessage orphaned from its
    AIMessage tool call.
    """
    if not messages:
        return []

    reserve = reserve_image if any(_has_image(m) for m in messages) else reserve_text
    available = max(0, max_context_tokens - reserve)

    kept: list[BaseMessage] = [messages[-1]]
    total = count_message_tokens(messages[-1])
    for message in reversed(messages[:-1]):
        cost = count_message_tokens(message)
        if total + cost > available:
            break
        kept.insert(0, message)
        total += cost

    # Drop a leading orphaned ToolMessage (its AIMessage tool_call was trimmed).
    # Guard with len(kept) > 1 so the most-recent message is never removed.
    while len(kept) > 1 and isinstance(kept[0], ToolMessage):
        kept.pop(0)

    return kept


def make_trim_middleware(max_context_tokens: int, reserve_text: int, reserve_image: int):
    """Build a wrap_model_call middleware that trims request.messages non-destructively."""

    @wrap_model_call
    def trim(request: ModelRequest, handler) -> ModelResponse:
        trimmed = trim_messages(
            list(request.messages), max_context_tokens, reserve_text, reserve_image
        )
        return handler(request.override(messages=trimmed))

    return trim
