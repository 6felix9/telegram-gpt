"""LangChain agent: model resolution, middleware, tools wiring, and the
Telegram-facing entry point. Replaces openai_client.py and token_manager.py."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, RemoveMessage

from conversation_summary import (
    PendingSummaryAuditRecord,
    ResilientSummarizationMiddleware,
    SUMMARY_PROMPT,
    SummaryAuditRecord,
)
from prompt_builder import PromptBuilder
from tools import build_tools
from model_registry import (
    MODEL_PROVIDERS,
    PROVIDER_LABEL,
    REASONING_EFFORT_LOW,
    resolve_model,
    provider_api_key,
)
from token_budget import (
    _message_text,
    count_tokens,
    count_message_tokens,
    count_messages_tokens,
    trim_messages,
    make_trim_middleware,
)

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

SUMMARY_MAX_OUTPUT_TOKENS = 1024

# An empty (e.g. reasoning-only, no visible text) model reply is treated as a
# retryable failure, mirroring the transport-level max_retries=2 already used
# for the underlying provider client.
EMPTY_RESPONSE_MAX_RETRIES = 2


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


def make_summary_model(config):
    """Build and validate the fixed model used for checkpoint summaries."""
    try:
        provider, prefixed_id = resolve_model(config.SUMMARY_MODEL)
    except KeyError as exc:
        raise ValueError(
            f"Unsupported SUMMARY_MODEL: {config.SUMMARY_MODEL}"
        ) from exc

    key = provider_api_key(provider, config)
    if not key.strip():
        env_name = {
            "openai": "OPENAI_API_KEY",
            "xai": "XAI_API_KEY",
            "google_genai": "GEMINI_API_KEY",
        }[provider]
        raise ValueError(
            f"{env_name} is required for SUMMARY_MODEL={config.SUMMARY_MODEL}"
        )

    return init_chat_model(
        prefixed_id,
        api_key=key,
        timeout=config.MODEL_TIMEOUT,
        max_retries=2,
        max_tokens=SUMMARY_MAX_OUTPUT_TOKENS,
        **({"use_responses_api": True} if provider == "openai" else {}),
        **({"reasoning": {"effort": "low"}} if config.SUMMARY_MODEL in REASONING_EFFORT_LOW else {}),
    )


def make_vision_summary_model(config):
    """Build and validate the fixed model used to describe images on ingest."""
    try:
        provider, prefixed_id = resolve_model(config.VISION_SUMMARY_MODEL)
    except KeyError as exc:
        raise ValueError(
            f"Unsupported VISION_SUMMARY_MODEL: {config.VISION_SUMMARY_MODEL}"
        ) from exc

    key = provider_api_key(provider, config)
    if not key.strip():
        env_name = {
            "openai": "OPENAI_API_KEY",
            "xai": "XAI_API_KEY",
            "google_genai": "GEMINI_API_KEY",
        }[provider]
        raise ValueError(
            f"{env_name} is required for VISION_SUMMARY_MODEL={config.VISION_SUMMARY_MODEL}"
        )

    return init_chat_model(
        prefixed_id,
        api_key=key,
        timeout=config.MODEL_TIMEOUT,
        max_retries=2,
        **({"use_responses_api": True} if provider == "openai" else {}),
    )


def _build_vision_model(config):
    """Fail-open wrapper: return the vision model or None if it can't be built."""
    try:
        return make_vision_summary_model(config)
    except ValueError as exc:
        logger.warning("Vision summary model unavailable: %s", exc)
        return None


@dataclass
class AgentContext:
    """Per-invocation context read by middleware (not persisted)."""
    is_group: bool = False
    reply_context: tuple[str, str] | None = None
    thread_id: str = "unknown"
    pending_summary_records: list[PendingSummaryAuditRecord] = field(default_factory=list)
    summary_compacted: bool = False


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

    def __init__(
        self,
        config,
        prompt_builder,
        checkpointer,
        model_name: str,
        *,
        summary_model=None,
        db=None,
    ):
        self._config = config
        self._prompt_builder = prompt_builder
        self._checkpointer = checkpointer
        self._tools = build_tools(config)  # from tools.py
        self._db = db
        self._summary_model = summary_model or make_summary_model(config)
        self._summary_middleware = ResilientSummarizationMiddleware(
            model=self._summary_model,
            summary_model_name=config.SUMMARY_MODEL,
            trigger=("tokens", config.SUMMARY_TRIGGER_TOKENS),
            keep=("tokens", config.SUMMARY_KEEP_TOKENS),
            token_counter=count_messages_tokens,
            summary_prompt=SUMMARY_PROMPT,
            trim_tokens_to_summarize=config.SUMMARY_CONTEXT_TOKENS,
            on_summary=self._record_summary if db is not None else None,
        )
        self._middleware = [
            _make_dynamic_prompt(prompt_builder),
            self._summary_middleware,
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
            timeout=self._config.MODEL_TIMEOUT,
            max_retries=2,
            max_tokens=self._config.MAX_OUTPUT_TOKENS,
            **({"use_responses_api": True} if provider == "openai" else {}),
            **({"reasoning": {"effort": "low"}} if model_name in REASONING_EFFORT_LOW else {}),
        )
        self._compile(model)
        logger.info("Agent compiled for %s (%s)", model_name, provider)

    # --- runtime ---------------------------------------------------------
    def _config_for(self, chat_id: str) -> dict:
        return {"configurable": {"thread_id": str(chat_id)}}

    def _record_summary(self, record: SummaryAuditRecord) -> None:
        """Write one checkpoint-confirmed summary audit record."""
        self._db.record_conversation_summary(
            chat_id=record.chat_id,
            summary_text=record.summary_text,
            summary_model=record.summary_model,
            before_message_count=record.before_message_count,
            after_message_count=record.after_message_count,
            before_tokens=record.before_tokens,
            after_tokens=record.after_tokens,
        )

    def _persist_checkpointed_summary_records(
        self,
        chat_id: str,
        context: AgentContext,
        final_messages: list[BaseMessage] | None,
    ) -> None:
        """Audit staged records only after their exact message ID is confirmed."""
        records = context.pending_summary_records
        try:
            if not records or self._summary_middleware.on_summary is None:
                return
            if final_messages is None:
                state = self._graph.get_state(self._config_for(chat_id))
                final_messages = state.values.get("messages", [])
            confirmed_ids = {
                str(message.id)
                for message in final_messages
                if message.id is not None
            }
            for pending in records:
                if pending.summary_message_id in confirmed_ids:
                    try:
                        self._summary_middleware.on_summary(pending.record)
                    except Exception:
                        logger.exception(
                            "Failed to persist summary audit record thread=%s",
                            pending.record.chat_id,
                        )
        except Exception:
            logger.exception(
                "Could not confirm checkpointed summary audit records thread=%s",
                chat_id,
            )
        finally:
            records.clear()

    async def run(self, chat_id, human_message, is_group, reply_context=None) -> str:
        if self._graph is None:
            raise CompletionError(
                f"❌ {PROVIDER_LABEL[self._provider]} API key is not set. "
                "Set it or switch models with /model."
            )
        context = AgentContext(
            is_group=is_group,
            reply_context=reply_context,
            thread_id=str(chat_id),
        )
        result = None
        empty_reply_ids: list[str] = []
        try:
            for attempt in range(EMPTY_RESPONSE_MAX_RETRIES + 1):
                result = await asyncio.to_thread(
                    self._graph.invoke,
                    {"messages": [human_message]},
                    config=self._config_for(chat_id),
                    context=context,
                )
                last_message = result["messages"][-1]
                response = _message_text(last_message)
                if response.strip():
                    return response
                logger.warning(
                    "Empty model reply for chat %s (attempt %s/%s)",
                    chat_id, attempt + 1, EMPTY_RESPONSE_MAX_RETRIES + 1,
                )
                if last_message.id is not None:
                    empty_reply_ids.append(last_message.id)
                    try:
                        self._graph.update_state(
                            self._config_for(chat_id),
                            {"messages": [RemoveMessage(id=last_message.id)]},
                        )
                    except Exception:
                        logger.exception(
                            "Failed to prune empty-reply message for chat %s", chat_id
                        )
                    else:
                        empty_reply_ids.remove(last_message.id)
            raise CompletionError(
                "⚠️ The model didn't return a reply after several attempts. "
                "Please try again or use /model to switch models."
            )
        except CompletionError:
            raise
        except Exception as e:
            raise _to_completion_error(e) from e
        finally:
            if empty_reply_ids:
                try:
                    self._graph.update_state(
                        self._config_for(chat_id),
                        {"messages": [RemoveMessage(id=mid) for mid in empty_reply_ids]},
                    )
                except Exception:
                    logger.exception(
                        "Failed to prune empty-reply messages for chat %s", chat_id
                    )
            self._persist_checkpointed_summary_records(
                str(chat_id),
                context,
                result["messages"] if result is not None else None,
            )

    def append_context_message(self, chat_id, human_message) -> None:
        """Append a non-triggering message to the thread (no model call)."""
        if self._graph is None:
            return
        try:
            self._graph.update_state(
                self._config_for(chat_id), {"messages": [human_message]}
            )
        except Exception as e:
            logger.error("Failed to append context message: %s", e, exc_info=True)

    def clear_thread(self, chat_id) -> None:
        self._checkpointer.delete_thread(str(chat_id))
