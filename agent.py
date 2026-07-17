"""LangChain agent: model resolution, middleware, tools wiring, and the
Telegram-facing entry point. Replaces openai_client.py and token_manager.py."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import tiktoken
from langchain.agents import create_agent
from langchain.agents.middleware import (
    wrap_model_call,
    dynamic_prompt,
    ModelRequest,
    ModelResponse,
)
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, RemoveMessage, ToolMessage

from prompt_builder import PromptBuilder
from tools import build_tools

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are Tze Foong's Assistant, an AI helper in Telegram.

Key behaviors:
- Be direct and concise - no unnecessary preambles
- Provide clear, helpful responses
- Never claim to be OpenAI or reference being a language model
- Respond naturally as a personal assistant
- Do not use Markdown formatting (no **bold**, *italics*, headers, or bullet asterisks)"""

SYSTEM_PROMPT_GROUP = """You are Tze Foong's Assistant, an AI helper in Telegram group chats.

Key behaviors:
- Be direct and concise - no unnecessary preambles
- Provide clear, helpful responses
- Never claim to be OpenAI or reference being a language model
- Track conversation context from multiple participants
- Messages are formatted as [Name]: content - reply naturally without mimicking this format"""

# tiktoken encoding is model-independent for our budgeting purposes.
_ENCODING = tiktoken.get_encoding("cl100k_base")

# Temporary latest-checkpoint guard. Replace with summarization and durable
# long-term memory as part of the future context-engineering architecture.
MAX_CHECKPOINT_MESSAGES = 500
CHECKPOINT_PRUNE_TARGET_MESSAGES = 400


class CompletionError(Exception):
    """Agent run failed; user_message is safe to show in Telegram."""

    def __init__(self, user_message: str):
        self.user_message = user_message
        super().__init__(user_message)


def _to_completion_error(exc: Exception) -> CompletionError:
    """Map a provider/agent exception to a user-safe CompletionError.

    LangChain surfaces provider SDK exceptions; classify by type name and
    message so the Telegram-facing messages stay equivalent to the old client.
    """
    name = type(exc).__name__
    text = str(exc).lower()

    if "authentication" in name.lower() or "unauthorized" in text or "api key" in text:
        return CompletionError(
            "❌ API key is invalid or missing for this model's provider. "
            "Please check your configuration."
        )
    if "ratelimit" in name.lower() or "rate limit" in text or "429" in text:
        return CompletionError("⏱️ Rate limit exceeded. Please wait a moment and try again.")
    if "timeout" in name.lower() or "timed out" in text:
        return CompletionError("⏱️ Request timed out. Please try again.")
    if "context_length_exceeded" in text or "context length" in text:
        return CompletionError(
            "❌ Message history is too long for the model. "
            "Use /clear to clear history and try again."
        )
    if "connection" in name.lower() or "connection" in text:
        logger.error("Connection error in agent run: %s", exc, exc_info=True)
        return CompletionError(
            "❌ Network error connecting to the API. "
            "Please check your internet connection."
        )
    logger.error("Unhandled agent error: %s", exc, exc_info=True)
    return CompletionError(
        "❌ An unexpected error occurred. Please try again or contact support."
    )


MODEL_PROVIDERS: dict[str, str] = {
    "gpt-4.1-mini": "openai",
    "gpt-5.4-mini": "openai",
    "gpt-5.4": "openai",
    "gpt-5.6-luna": "openai",
    "gpt-5.6-terra": "openai",
    "grok-4.20-0309-reasoning": "xai",
    "grok-4.20-0309-non-reasoning": "xai",
    "grok-4-1-fast-reasoning": "xai",
    "gemini-3.1-flash-lite-preview": "google_genai",
    "gemini-3.5-flash": "google_genai",
}

PROVIDER_LABEL: dict[str, str] = {
    "openai": "OpenAI", "xai": "xAI", "google_genai": "Gemini"
}


def resolve_model(name: str) -> tuple[str, str]:
    """Map a bare model name to (provider, provider-prefixed id)."""
    provider = MODEL_PROVIDERS[name]  # KeyError for unknown models (caught by /model)
    return provider, f"{provider}:{name}"


def provider_api_key(provider: str, config) -> str:
    """Return the configured API key for a provider (may be empty)."""
    return {
        "openai": config.OPENAI_API_KEY,
        "xai": config.XAI_API_KEY,
        "google_genai": config.GEMINI_API_KEY,
    }[provider]


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


def trim_messages(
    messages: list[BaseMessage],
    max_context_tokens: int,
    reserve: int,
) -> list[BaseMessage]:
    """Keep as much recent history as fits the budget, newest-first.

    Non-destructive: returns a new list. Always keeps the last message.
    Never returns a list beginning with a ToolMessage orphaned from its
    AIMessage tool call.
    """
    if not messages:
        return []

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


def checkpoint_messages_to_remove(
    messages: list[BaseMessage],
    max_messages: int | None = None,
    target_messages: int | None = None,
) -> list[RemoveMessage]:
    """Return removals that reduce an oversized checkpoint to its target."""
    if max_messages is None:
        max_messages = MAX_CHECKPOINT_MESSAGES
    if target_messages is None:
        target_messages = CHECKPOINT_PRUNE_TARGET_MESSAGES

    if len(messages) <= max_messages:
        return []

    remove_count = len(messages) - target_messages

    # Do not leave a ToolMessage without its preceding AI tool call.
    while remove_count < len(messages) and isinstance(messages[remove_count], ToolMessage):
        remove_count += 1

    return [
        RemoveMessage(id=message.id)
        for message in messages[:remove_count]
        if message.id is not None
    ]


def make_trim_middleware(max_context_tokens: int, reserve: int):
    """Build a wrap_model_call middleware that trims request.messages non-destructively."""

    @wrap_model_call
    def trim(request: ModelRequest, handler) -> ModelResponse:
        trimmed = trim_messages(list(request.messages), max_context_tokens, reserve)
        return handler(request.override(messages=trimmed))

    return trim


@dataclass
class AgentContext:
    """Per-invocation context read by middleware (not persisted)."""
    is_group: bool = False
    reply_context: tuple[str, str] | None = None


def _make_dynamic_prompt(prompt_builder):
    """Build a @dynamic_prompt middleware that resolves the system prompt per call."""

    @dynamic_prompt
    def system_prompt(request) -> str:
        ctx = getattr(request.runtime, "context", None) or AgentContext()
        return prompt_builder.build_system_prompt(
            is_group=ctx.is_group,
            reply_context=ctx.reply_context,
        )

    return system_prompt


class Agent:
    """Compiled LangChain agent with DB-driven model + personality."""

    def __init__(self, config, prompt_builder, checkpointer, model_name: str):
        self._config = config
        self._prompt_builder = prompt_builder
        self._checkpointer = checkpointer
        self._tools = build_tools(config)  # from tools.py
        self._middleware = [
            _make_dynamic_prompt(prompt_builder),
            make_trim_middleware(config.MAX_CONTEXT_TOKENS, config.MAX_OUTPUT_TOKENS),
        ]
        self.model_name = model_name
        self._provider = None
        self._graph = None
        self.set_model(model_name)

    # --- compilation -----------------------------------------------------
    def _compile(self, model) -> None:
        self._graph = create_agent(
            model=model,
            tools=self._tools,
            middleware=self._middleware,
            checkpointer=self._checkpointer,
            context_schema=AgentContext,
        )

    def set_model(self, model_name: str) -> None:
        self.model_name = model_name
        provider, prefixed_id = resolve_model(model_name)
        self._provider = provider
        key = provider_api_key(provider, self._config)
        if not key.strip():
            logger.warning("%s API key not set; model %s will error on use",
                           PROVIDER_LABEL[provider], model_name)
            self._graph = None
            return
        model = init_chat_model(
            prefixed_id,
            api_key=key,
            timeout=self._config.OPENAI_TIMEOUT,
            max_retries=2,
            max_tokens=self._config.MAX_OUTPUT_TOKENS,
            **({"use_responses_api": True} if provider == "openai" else {}),
        )
        self._compile(model)
        logger.info("Agent compiled for %s (%s)", model_name, provider)

    # --- runtime ---------------------------------------------------------
    def _config_for(self, chat_id: str) -> dict:
        return {"configurable": {"thread_id": str(chat_id)}}

    def _prune_checkpoint(self, chat_id: str, messages: list[BaseMessage]) -> None:
        """Best-effort temporary cap for the latest checkpoint message state."""
        removals = checkpoint_messages_to_remove(
            messages,
            MAX_CHECKPOINT_MESSAGES,
            CHECKPOINT_PRUNE_TARGET_MESSAGES,
        )
        if not removals:
            return

        try:
            self._graph.update_state(
                self._config_for(chat_id), {"messages": removals}
            )
            logger.info(
                "Pruned %s checkpoint messages for chat %s",
                len(removals),
                chat_id,
            )
        except Exception as e:
            logger.error(
                "Failed to prune checkpoint messages for chat %s: %s",
                chat_id,
                e,
                exc_info=True,
            )

    async def run(self, chat_id, human_message, is_group, reply_context=None) -> str:
        if self._graph is None:
            raise CompletionError(
                f"❌ {PROVIDER_LABEL[self._provider]} API key is not set. "
                "Set it or switch models with /model."
            )
        try:
            result = await asyncio.to_thread(
                self._graph.invoke,
                {"messages": [human_message]},
                config=self._config_for(chat_id),
                context=AgentContext(is_group=is_group, reply_context=reply_context),
            )
            response = _message_text(result["messages"][-1])
            self._prune_checkpoint(chat_id, list(result["messages"]))
            return response
        except CompletionError:
            raise
        except Exception as e:
            raise _to_completion_error(e) from e

    async def append_context_message(self, chat_id, human_message) -> None:
        """Append a non-triggering message to the thread (no model call)."""
        if self._graph is None:
            return
        await asyncio.to_thread(
            self._append_context_message_sync, chat_id, human_message
        )

    def _append_context_message_sync(self, chat_id, human_message) -> None:
        try:
            updated_config = self._graph.update_state(
                self._config_for(chat_id), {"messages": [human_message]}
            )
            state = self._graph.get_state(updated_config)
            self._prune_checkpoint(
                chat_id, list(state.values.get("messages", []))
            )
        except Exception as e:
            logger.error("Failed to append context message: %s", e, exc_info=True)

    def clear_thread(self, chat_id) -> None:
        self._checkpointer.delete_thread(str(chat_id))
