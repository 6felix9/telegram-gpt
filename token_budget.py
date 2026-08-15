"""Token counting and context-window trimming, independent of provider wiring."""
from __future__ import annotations

from collections.abc import Iterable

import tiktoken
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, ToolMessage

# tiktoken encoding is model-independent for our budgeting purposes.
_ENCODING = tiktoken.get_encoding("cl100k_base")

# Heading the compacted summary message carries in checkpoint state.
# Single source of truth: conversation_summary.py re-exports this rather than
# duplicating the literal, since it imports _is_summary_message from here.
SUMMARY_HEADING = "## Conversation summary"


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


def count_message_tokens(message: BaseMessage) -> int:
    """Approximate token count of one message, including per-message overhead."""
    return count_tokens(_message_text(message)) + 4


def count_messages_tokens(messages: Iterable[BaseMessage]) -> int:
    """Approximate total tokens for LangChain summary trigger/keep policies."""
    return sum(count_message_tokens(message) for message in messages)


def _is_summary_message(message: BaseMessage) -> bool:
    """True for the rolling-summary HumanMessage inserted by checkpoint compaction."""
    if message.additional_kwargs.get("lc_source") == "summarization":
        return True
    text = _message_text(message)
    return text.startswith(SUMMARY_HEADING)


def _drop_orphaned_tool_messages(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Remove tool results whose retained tool call no longer precedes them."""
    retained_tool_call_ids: set[str] = set()
    valid_messages: list[AnyMessage] = []

    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in message.tool_calls:
                tool_call_id = tool_call.get("id")
                if tool_call_id:
                    retained_tool_call_ids.add(tool_call_id)
            valid_messages.append(message)
        elif isinstance(message, ToolMessage):
            if message.tool_call_id in retained_tool_call_ids:
                valid_messages.append(message)
        else:
            valid_messages.append(message)

    return valid_messages


def trim_messages(
    messages: list[AnyMessage],
    max_context_tokens: int,
    reserve: int,
) -> list[AnyMessage]:
    """Keep as much recent history as fits the budget, newest-first.

    Non-destructive: returns a new list. Always keeps the last message unless
    it is an orphaned tool result, and keeps any summarization-tagged rolling
    summary messages (even if their combined cost exceeds the budget). Never
    returns a ToolMessage without its preceding AIMessage tool call.
    """
    if not messages:
        return []

    available = max(0, max_context_tokens - reserve)

    always_keep = {len(messages) - 1}
    for i, message in enumerate(messages):
        if _is_summary_message(message):
            always_keep.add(i)

    total = sum(count_message_tokens(messages[i]) for i in always_keep)
    selected = set(always_keep)

    for i in range(len(messages) - 2, -1, -1):
        if i in selected:
            continue
        cost = count_message_tokens(messages[i])
        if total + cost > available:
            break
        selected.add(i)
        total += cost

    kept = [messages[i] for i in range(len(messages)) if i in selected]

    return _drop_orphaned_tool_messages(kept)


def make_trim_middleware(max_context_tokens: int, reserve: int):
    """Build a wrap_model_call middleware that trims request.messages non-destructively."""

    @wrap_model_call
    def trim(request: ModelRequest, handler) -> ModelResponse:
        trimmed = trim_messages(list(request.messages), max_context_tokens, reserve)
        return handler(request.override(messages=trimmed))

    return trim
