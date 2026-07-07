# LangChain Agent Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the linear `openai_client.py` request/response path with an agentic, model-agnostic LangChain agent (`create_agent`) that can call web-search tools in a loop, backed by a LangGraph `PostgresSaver` checkpointer for conversation memory, while preserving all Telegram-facing behavior and `/model` switching across OpenAI, xAI, and Gemini.

**Architecture:** A new `agent.py` builds a LangChain agent via `create_agent(model, tools, middleware, checkpointer, context_schema)`. The model is resolved from the DB-backed `active_model` through `init_chat_model` with a provider-prefixed id. A `@dynamic_prompt` middleware injects the personality/private-vs-group system prompt per invocation; a `@wrap_model_call` middleware trims history to the token budget non-destructively. Conversation state lives in a `PostgresSaver` keyed by `thread_id = chat_id`. The existing `messages` table becomes an audit/stats log only. `token_manager.py` and `openai_client.py` are deleted.

**Tech Stack:** Python 3.12, `langchain` 1.x, `langgraph` 1.x, `langgraph-checkpoint-postgres`, `langchain-openai` / `langchain-xai` / `langchain-google-genai`, `langchain-tavily` + `ddgs` (DuckDuckGo fallback), `tiktoken`, `psycopg` (psycopg3) + `psycopg-pool` for the checkpointer, `python-telegram-bot`, `pytest`.

## Global Constraints

- Target **Python 3.12+**; keep code async-compatible with existing `python-telegram-bot` handlers.
- **LangChain / LangGraph 1.0 LTS**: pin `langchain>=1.0,<2.0`, `langchain-core>=1.0,<2.0`, `langgraph>=1.0,<2.0`. Community tool packages (`langchain-tavily`, `ddgs`) use latest.
- **Persistence backend is unchanged**: single Neon Postgres via `DATABASE_URL`. No SQLite fallback. `DATABASE_URL` is required in every environment including local dev.
- **App schema stays Alembic-owned** (`messages`, `granted_users`, `personality`, `active_personality`, `active_model`). The checkpointer's tables (`checkpoints`, `checkpoint_blobs`, `checkpoint_writes`, `checkpoint_migrations`) are created by `PostgresSaver.setup()` as an explicit deploy step, **never implicitly at runtime**.
- **Required env vars** (startup fails only on these): `TELEGRAM_BOT_TOKEN`, `AUTHORIZED_USER_ID`, `OPENAI_API_KEY`, `DATABASE_URL`. All others have code defaults. A missing `XAI_API_KEY`/`GEMINI_API_KEY` must **not** fail startup even if `DEFAULT_MODEL` selects that provider — it surfaces as a clear runtime error on first use.
- **No observable change** to Telegram behavior except the one-time cutover: checkpoints start empty per thread (no backfill from `messages`).
- `CompletionError(user_message)` is the single error contract handlers rely on; its user-facing messages must stay equivalent to today's.
- Tests remain pure logic (no live DB / `.env` / network); reuse the existing `tests/` pytest style.



## Model → Provider Registry (used across tasks)

The bare model names stored in `active_model` and their `init_chat_model` provider prefix. This dict replaces the old `MODEL_REGISTRY` and is the source of truth for `/model` validation, provider-prefix construction, and API-key selection.

```python
MODEL_PROVIDERS: dict[str, str] = {
    # OpenAI
    "gpt-4o-mini": "openai",
    "gpt-4.1-mini": "openai",
    "gpt-5.4-mini": "openai",
    "gpt-5": "openai",
    # xAI Grok
    "grok-4.20-0309-reasoning": "xai",
    "grok-4.20-0309-non-reasoning": "xai",
    "grok-4-1-fast-reasoning": "xai",
    # Google Gemini
    "gemini-3.1-flash-lite-preview": "google_genai",
    "gemini-3-flash-preview": "google_genai",
}

PROVIDER_LABEL: dict[str, str] = {"openai": "OpenAI", "xai": "xAI", "google_genai": "Gemini"}
```



## File Structure

- **Create** `agent.py` — replaces `openai_client.py`. Holds: `CompletionError`, `MODEL_PROVIDERS`/`PROVIDER_LABEL`, token-counting + trimming middleware, model resolution, the `@dynamic_prompt` system-prompt middleware, the `AgentContext` dataclass, and the `Agent` class (compile/recompile, `run`, `append_context_message`, `clear_thread`, `count_tokens`).
- **Create** `tools.py` — `build_tools(config)` returning the web-search tool (Tavily when `TAVILY_API_KEY` set, else DuckDuckGo) plus a `fetch_url` tool.
- **Create** `scripts/setup_checkpointer.py` — one-shot `PostgresSaver(...).setup()` runner for deploy/local.
- **Modify** `prompt_builder.py` — keep `build_system_prompt` (unchanged logic); replace `format_messages` (OpenAI dicts) with `to_lc_human_message(...)` producing a LangChain `HumanMessage` with text/image content blocks and the group `[Name]:` prefix.
- **Modify** `config.py` — restructure `validate()` to the smaller required set and update defaults (`RESERVE_TOKENS_TEXT=2000`, `MAX_GROUP_CONTEXT_MESSAGES=500`, `DEFAULT_MODEL=gpt-5.4-mini`, add `TAVILY_API_KEY`).
- **Modify** `handlers.py` — call the `Agent` instead of `OpenAIClient`; audit-log to `messages`; non-triggering group messages go to the checkpoint via `agent.append_context_message`; `/clear` calls `agent.clear_thread`; `/model` uses `MODEL_PROVIDERS` + `agent.set_model`.
- **Modify** `bot.py` — construct the `psycopg_pool.ConnectionPool` + `PostgresSaver`, build `Agent`, drop `TokenManager`/`OpenAIClient`, close the pool on shutdown.
- **Modify** `.env.example`**,** `requirements.txt`**,** `requirements-dev.txt`**,** `start.sh`**,** `README.md`**,** `database.md`**.**
- **Delete** `token_manager.py`**,** `openai_client.py`**.**
- **Tests:** add `tests/test_config.py`, `tests/test_trimming.py`, `tests/test_model_resolution.py`, `tests/test_tools.py`, `tests/test_agent.py`; rewrite `tests/test_prompt_builder.py`; delete `tests/test_token_manager.py`, `tests/test_model_registry.py`. Keep `tests/test_extract_keyword.py`.

---



## Task 1: Dependencies

**Files:**

- Modify: `requirements.txt`
- Modify: `requirements-dev.txt`

**Interfaces:**

- Produces: the importable package set every later task relies on (`langchain.agents.create_agent`, `langchain.chat_models.init_chat_model`, `langgraph.checkpoint.postgres.PostgresSaver`, `langchain_tavily.TavilySearch`, `psycopg_pool.ConnectionPool`).

- [ ] **Step 1: Rewrite** `requirements.txt`

Replace the whole file with:

```
python-telegram-bot==21.7
tiktoken==0.12.0
python-dotenv==1.0.1

# Existing app schema layer (Alembic + psycopg2 sync pool)
psycopg2-binary==2.9.9
alembic==1.14.1
SQLAlchemy==2.0.37

# LangChain / LangGraph agent stack
langchain>=1.0,<2.0
langchain-core>=1.0,<2.0
langgraph>=1.0,<2.0
langgraph-checkpoint-postgres>=2.0,<3.0

# Model providers
langchain-openai>=0.3
langchain-xai>=0.2
langchain-google-genai>=2.0

# Checkpointer driver (psycopg3 + pool) — separate from the psycopg2 app layer
psycopg[binary]>=3.2
psycopg-pool>=3.2

# Tools
langchain-tavily>=0.1
ddgs>=6.0
```

Notes:

- `openai==2.9.0` is removed as a direct pin — `langchain-openai` manages a compatible `openai`.
- `psycopg2-binary` (app layer) and `psycopg` (checkpointer) intentionally coexist.

- [ ] **Step 2: Update** `requirements-dev.txt`

```
pytest==8.3.4
```

(unchanged; kept as its own step so the task is self-contained.)

- [ ] **Step 3: Install and verify imports**

Run:

```bash
pip install -r requirements.txt -r requirements-dev.txt
python3 -c "from langchain.agents import create_agent; from langchain.chat_models import init_chat_model; from langgraph.checkpoint.postgres import PostgresSaver; from langchain.agents.middleware import dynamic_prompt, wrap_model_call, ModelRequest, ModelResponse; from psycopg_pool import ConnectionPool; from langchain_tavily import TavilySearch; print('ok')"
```

Expected: prints `ok` with no ImportError.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt requirements-dev.txt
git commit -m "Add LangChain/LangGraph agent dependencies"
```

---



## Task 2: Configuration restructure

**Files:**

- Modify: `config.py`
- Modify: `.env.example`
- Test: `tests/test_config.py` (create)

**Interfaces:**

- Produces: `Config` with new defaults and `Config.validate()` requiring only `TELEGRAM_BOT_TOKEN`, `AUTHORIZED_USER_ID`, `OPENAI_API_KEY`, `DATABASE_URL`; new attribute `Config.TAVILY_API_KEY: str`. Defaults: `DEFAULT_MODEL="gpt-5.4-mini"`, `RESERVE_TOKENS_TEXT=2000`, `MAX_GROUP_CONTEXT_MESSAGES=500`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_config.py`:

```python
"""Config validation and defaults (no .env / DB required)."""
import importlib
import sys
import pytest


def _fresh_config(monkeypatch, env: dict):
    """Reload config.py with a controlled environment."""
    for key in [
        "TELEGRAM_BOT_TOKEN", "BOT_USERNAME", "OPENAI_API_KEY", "XAI_API_KEY",
        "GEMINI_API_KEY", "DEFAULT_MODEL", "OPENAI_TIMEOUT", "MAX_CONTEXT_TOKENS",
        "RESERVE_TOKENS_TEXT", "RESERVE_TOKENS_IMAGE", "MAX_GROUP_CONTEXT_MESSAGES",
        "TAVILY_API_KEY", "AUTHORIZED_USER_ID", "DATABASE_URL", "LOG_LEVEL",
    ]:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    sys.modules.pop("config", None)
    return importlib.import_module("config")


VALID = {
    "TELEGRAM_BOT_TOKEN": "t",
    "AUTHORIZED_USER_ID": "123",
    "OPENAI_API_KEY": "sk-x",
    "DATABASE_URL": "postgresql://u:p@h:5432/db",
}


def test_defaults_apply_when_optional_unset(monkeypatch):
    cfg = _fresh_config(monkeypatch, VALID)
    assert cfg.config.DEFAULT_MODEL == "gpt-5.4-mini"
    assert cfg.config.RESERVE_TOKENS_TEXT == 2000
    assert cfg.config.RESERVE_TOKENS_IMAGE == 3000
    assert cfg.config.MAX_CONTEXT_TOKENS == 16000
    assert cfg.config.MAX_GROUP_CONTEXT_MESSAGES == 500
    assert cfg.config.OPENAI_TIMEOUT == 60
    assert cfg.config.BOT_USERNAME == ""
    assert cfg.config.TAVILY_API_KEY == ""


def test_validate_passes_with_only_required(monkeypatch):
    cfg = _fresh_config(monkeypatch, VALID)
    cfg.config.validate()  # must not raise / sys.exit


def test_missing_required_exits(monkeypatch):
    env = dict(VALID)
    del env["OPENAI_API_KEY"]
    cfg = _fresh_config(monkeypatch, env)
    with pytest.raises(SystemExit):
        cfg.config.validate()


def test_missing_provider_key_does_not_fail_startup(monkeypatch):
    # DEFAULT_MODEL selects xAI but XAI_API_KEY absent — must still validate.
    env = dict(VALID, DEFAULT_MODEL="grok-4-1-fast-reasoning")
    cfg = _fresh_config(monkeypatch, env)
    cfg.config.validate()  # must not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — current `validate()` requires `BOT_USERNAME` and cross-checks provider keys; `DEFAULT_MODEL` default and `RESERVE_TOKENS_TEXT` differ.

- [ ] **Step 3: Update** `config.py`

Change the defaults and drop the provider cross-check. Edit these lines:

```python
    # Default model to use on first startup (persisted in DB after first run)
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-5.4-mini")

    OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", "60"))
    MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "16000"))

    # Token Reserve Configuration
    RESERVE_TOKENS_TEXT = int(os.getenv("RESERVE_TOKENS_TEXT", "2000"))
    RESERVE_TOKENS_IMAGE = int(os.getenv("RESERVE_TOKENS_IMAGE", "3000"))

    # Web search tool (Tavily); blank falls back to DuckDuckGo at runtime
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

    # Authorization
    AUTHORIZED_USER_ID = os.getenv("AUTHORIZED_USER_ID", "")

    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "")

    # Group chat settings
    MAX_GROUP_CONTEXT_MESSAGES = int(os.getenv("MAX_GROUP_CONTEXT_MESSAGES", "500"))
```

Replace the entire body of `validate()` with:

```python
    @classmethod
    def validate(cls):
        """Validate the small required set; optional vars fall back to defaults."""
        errors = []

        if not cls.TELEGRAM_BOT_TOKEN.strip():
            errors.append("TELEGRAM_BOT_TOKEN is required")

        if not cls.OPENAI_API_KEY.strip():
            errors.append("OPENAI_API_KEY is required")

        if not cls.AUTHORIZED_USER_ID:
            errors.append("AUTHORIZED_USER_ID is required")
        elif not cls.AUTHORIZED_USER_ID.isdigit():
            errors.append("AUTHORIZED_USER_ID must be numeric")

        if not cls.DATABASE_URL.strip():
            errors.append("DATABASE_URL is required")

        for name in ("OPENAI_TIMEOUT", "MAX_CONTEXT_TOKENS",
                     "RESERVE_TOKENS_TEXT", "RESERVE_TOKENS_IMAGE"):
            if getattr(cls, name) <= 0:
                errors.append(f"{name} must be positive")

        if cls.MAX_CONTEXT_TOKENS > 100000:
            logger.warning(
                f"MAX_CONTEXT_TOKENS is very large ({cls.MAX_CONTEXT_TOKENS}). "
                "Make sure this matches your model's actual context window limit."
            )

        if errors:
            logger.error("Configuration validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            sys.exit(1)

        logger.info("Configuration validated successfully")
```

`BOT_VERSION` stays; bump it to `"2.0.0"` (major cutover):

```python
    BOT_VERSION = "2.0.0"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Rewrite** `.env.example`

Replace the whole file with the layout from the spec (required vars keep placeholders; optional vars are blank with an inline "defaults to X"):

```dotenv
# Required
# --------

# Telegram Bot Configuration
# Get your bot token from @BotFather on Telegram
TELEGRAM_BOT_TOKEN=your_bot_token_here

# Authorization
# Get your Telegram user ID from @userinfobot
AUTHORIZED_USER_ID=your_telegram_user_id_here

# AI Provider API Keys
# OPENAI_API_KEY is the only required key.
OPENAI_API_KEY=your_openai_api_key_here

# Neon/Postgres connection string. Backs the messages/admin tables and the
# LangGraph checkpointer (the agent's conversation memory).
DATABASE_URL=postgresql://user:password@host:port/database?sslmode=require&channel_binding=require

# Optional
# --------

# Without this, @mention activation is disabled — only the "chatgpt" keyword
# triggers the bot. Defaults to empty.
BOT_USERNAME=

# Only needed if you use grok-* / gemini-* models. Defaults to empty.
XAI_API_KEY=
GEMINI_API_KEY=

# Seed value for a fresh database only — active_model in the DB wins after
# first run. Defaults to gpt-5.4-mini.
DEFAULT_MODEL=

# API timeout in seconds. Defaults to 60.
OPENAI_TIMEOUT=

# Maximum tokens to use for conversation context. Defaults to 16000.
MAX_CONTEXT_TOKENS=

# Tokens reserved for text responses. Defaults to 2000.
RESERVE_TOKENS_TEXT=

# Tokens reserved for image/vision responses. Defaults to 3000.
RESERVE_TOKENS_IMAGE=

# Maximum messages to store per group chat. Defaults to 500.
MAX_GROUP_CONTEXT_MESSAGES=

# Powers the agent's web search tool via Tavily. If left blank, web search
# automatically falls back to a DuckDuckGo-backed tool instead.
# Get a key at https://tavily.com. Defaults to empty.
TAVILY_API_KEY=

# Logging verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL). Defaults to INFO.
LOG_LEVEL=
```

- [ ] **Step 6: Commit**

```bash
git add config.py .env.example tests/test_config.py
git commit -m "Restructure config around smaller required set"
```

---



## Task 3: Token counting + trimming middleware

**Files:**

- Create: `agent.py` (first slice — module-level helpers only)
- Test: `tests/test_trimming.py` (create)

**Interfaces:**

- Consumes: `config` (for budgets), `MODEL_PROVIDERS` is not needed here.
- Produces:
  - `count_tokens(text: str) -> int` — tiktoken-based token count of a string (cl100k_base fallback). Used by handlers for audit logging and by the trimmer.
  - `count_message_tokens(message) -> int` — token count of one LangChain message (stringifies content blocks).
  - `make_trim_middleware(max_context_tokens: int, reserve_text: int, reserve_image: int)` — returns a `@wrap_model_call` middleware that replaces `request.messages` with a token-trimmed copy (non-destructive; checkpoint state is untouched). Reserves `reserve_image` tokens if any message carries an image block, else `reserve_text`. Always keeps the most recent message; never leaves a leading `ToolMessage` orphaned from its `AIMessage`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_trimming.py`:

```python
"""Token counting and the pre-model trimming middleware."""
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
import agent


def test_count_tokens_nonzero():
    assert agent.count_tokens("hello world") > 0
    assert agent.count_tokens("") == 0


def _run_trim(messages, max_context=1000, reserve_text=100, reserve_image=300):
    mw = agent.make_trim_middleware(max_context, reserve_text, reserve_image)
    captured = {}

    def handler(request):
        captured["messages"] = request.messages
        return AIMessage(content="ok")

    request = agent.ModelRequest(messages=list(messages))  # minimal shim; see impl note
    mw(request, handler)
    return captured["messages"]


def test_keeps_recent_and_drops_old_when_over_budget():
    # Many large messages; only the newest should survive a tiny budget.
    big = "word " * 500
    messages = [HumanMessage(content=big) for _ in range(10)]
    kept = _run_trim(messages, max_context=200, reserve_text=50)
    assert kept[-1] is messages[-1]
    assert len(kept) < len(messages)


def test_never_starts_with_orphan_tool_message():
    messages = [
        AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "1"}]),
        ToolMessage(content="result", tool_call_id="1"),
        HumanMessage(content="hi"),
    ]
    # Force a budget that would cut the AIMessage but keep the ToolMessage.
    kept = _run_trim(messages, max_context=agent.count_message_tokens(messages[1])
                     + agent.count_message_tokens(messages[2]) + 5, reserve_text=0)
    assert not (kept and isinstance(kept[0], ToolMessage))
```

> Implementation note for the test's `ModelRequest` shim: `agent.make_trim_middleware` must not depend on private `ModelRequest` internals beyond `.messages` (read) and `request.override(messages=...)` (write). If constructing a real `ModelRequest` in tests is impractical for the installed version, expose the pure trimming logic as `agent.trim_messages(messages, max_context, reserve_text, reserve_image) -> list` and have the middleware call it; test `trim_messages` directly and keep one thin middleware wrapper untested. Prefer this split — it keeps the tests independent of the middleware object's constructor.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_trimming.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent'`.

- [ ] **Step 3: Create** `agent.py` **first slice**

```python
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
    while kept and isinstance(kept[0], ToolMessage):
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
```

> If the installed `wrap_model_call` requires the middleware be a plain function (not returned from a factory), define `trim` at module scope reading budgets from `config` instead; the factory form is preferred for testability but either satisfies the interface.

- [ ] **Step 4: Adapt the test to the pure function (per the note)**

If the `ModelRequest` shim proves impractical, replace `_run_trim` calls with direct `agent.trim_messages(...)` assertions:

```python
def test_keeps_recent_and_drops_old_when_over_budget():
    big = "word " * 500
    messages = [HumanMessage(content=big) for _ in range(10)]
    kept = agent.trim_messages(messages, 200, 50, 300)
    assert kept[-1] is messages[-1]
    assert len(kept) < len(messages)
```

- [ ] **Step 5: Run tests to verify they pass**



Run: `pytest tests/test_trimming.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add agent.py tests/test_trimming.py
git commit -m "Add token counting and pre-model trimming middleware"
```

---



## Task 4: Model resolution

**Files:**

- Modify: `agent.py` (add registry + resolution)
- Test: `tests/test_model_resolution.py` (create)

**Interfaces:**

- Consumes: `count_tokens` etc. from Task 3 (same module).
- Produces:
  - `MODEL_PROVIDERS: dict[str, str]`, `PROVIDER_LABEL: dict[str, str]` (values as in the registry section above).
  - `resolve_model(name: str) -> tuple[str, str]` — returns `(provider, prefixed_id)` where `prefixed_id = f"{provider}:{name}"`; raises `KeyError` for unknown models.
  - `provider_api_key(provider: str, config) -> str` — returns the configured key for a provider (`openai`→`OPENAI_API_KEY`, `xai`→`XAI_API_KEY`, `google_genai`→`GEMINI_API_KEY`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_model_resolution.py`:

```python
"""Provider-prefixed model resolution replacing MODEL_REGISTRY validation."""
import pytest
import agent


def test_known_models_map_to_expected_providers():
    assert agent.resolve_model("gpt-5") == ("openai", "openai:gpt-5")
    assert agent.resolve_model("grok-4-1-fast-reasoning") == (
        "xai", "xai:grok-4-1-fast-reasoning")
    assert agent.resolve_model("gemini-3-flash-preview") == (
        "google_genai", "google_genai:gemini-3-flash-preview")


def test_every_registered_model_has_a_label():
    for provider in agent.MODEL_PROVIDERS.values():
        assert provider in agent.PROVIDER_LABEL


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        agent.resolve_model("does-not-exist")


class _Cfg:
    OPENAI_API_KEY = "o"
    XAI_API_KEY = "x"
    GEMINI_API_KEY = "g"


def test_provider_api_key_selection():
    assert agent.provider_api_key("openai", _Cfg) == "o"
    assert agent.provider_api_key("xai", _Cfg) == "x"
    assert agent.provider_api_key("google_genai", _Cfg) == "g"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_model_resolution.py -v`
Expected: FAIL — `AttributeError: module 'agent' has no attribute 'resolve_model'`.

- [ ] **Step 3: Add resolution to** `agent.py`

Append after the imports/`CompletionError`:

```python
MODEL_PROVIDERS: dict[str, str] = {
    "gpt-4o-mini": "openai",
    "gpt-4.1-mini": "openai",
    "gpt-5.4-mini": "openai",
    "gpt-5": "openai",
    "grok-4.20-0309-reasoning": "xai",
    "grok-4.20-0309-non-reasoning": "xai",
    "grok-4-1-fast-reasoning": "xai",
    "gemini-3.1-flash-lite-preview": "google_genai",
    "gemini-3-flash-preview": "google_genai",
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_model_resolution.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add agent.py tests/test_model_resolution.py
git commit -m "Add provider-prefixed model resolution"
```

---



## Task 5: prompt_builder — LangChain message objects

**Files:**

- Modify: `prompt_builder.py`
- Test: `tests/test_prompt_builder.py` (rewrite)

**Interfaces:**

- Consumes: nothing new.
- Produces:
  - `PromptBuilder.build_system_prompt(is_group, custom_system_prompt=None, reply_context=None) -> str` — **unchanged behavior**.
  - `PromptBuilder.to_lc_human_message(text=None, is_group=False, sender_name="Unknown", image_data_url=None) -> HumanMessage` — builds a LangChain `HumanMessage`. In group chats, user text is prefixed `[sender_name]:`  (unless it already starts with `[`). If `image_data_url` is given, content is a block list `[{"type": "text", "text": ...}, {"type": "image_url", "image_url": {"url": image_data_url}}]`; otherwise content is the plain (possibly prefixed) string.
  - Remove `format_messages` (no longer used).

- [ ] **Step 1: Write the failing test**

Replace `tests/test_prompt_builder.py` entirely:

```python
"""PromptBuilder: system prompt resolution + LangChain message construction."""
from langchain_core.messages import HumanMessage
from prompt_builder import PromptBuilder


def _pb(active=None, prompt_for=None):
    return PromptBuilder(
        default_private_prompt="PRIVATE",
        default_group_prompt="GROUP",
        get_active_personality=(lambda: active) if active else None,
        get_personality_prompt=(lambda name: prompt_for) if prompt_for is not None else None,
    )


def test_private_uses_default_private_prompt():
    out = _pb().build_system_prompt(is_group=False)
    assert "PRIVATE" in out


def test_group_uses_active_personality_prompt():
    out = _pb(active="villain", prompt_for="BE EVIL").build_system_prompt(is_group=True)
    assert "BE EVIL" in out


def test_group_falls_back_to_default_when_personality_missing():
    out = _pb(active="ghost", prompt_for=None).build_system_prompt(is_group=True)
    assert "GROUP" in out


def test_to_lc_human_message_plain_text():
    msg = _pb().to_lc_human_message(text="hello", is_group=False)
    assert isinstance(msg, HumanMessage)
    assert msg.content == "hello"


def test_group_message_gets_sender_prefix():
    msg = _pb().to_lc_human_message(text="hi", is_group=True, sender_name="Alice")
    assert msg.content == "[Alice]: hi"


def test_image_message_has_text_and_image_blocks():
    msg = _pb().to_lc_human_message(
        text="what is this?", image_data_url="data:image/jpeg;base64,AAAA"
    )
    types = [b["type"] for b in msg.content]
    assert types == ["text", "image_url"]
    assert msg.content[1]["image_url"]["url"] == "data:image/jpeg;base64,AAAA"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_prompt_builder.py -v`
Expected: FAIL — `to_lc_human_message` doesn't exist yet.

- [ ] **Step 3: Update** `prompt_builder.py`

Add at the top:

```python
from langchain_core.messages import HumanMessage
```

Delete the `format_messages` method and `_apply_group_sender_prefix`'s current usage inside it, then add:

```python
    @staticmethod
    def _group_prefix(text: str, sender_name: str) -> str:
        if text.startswith("["):
            return text
        return f"[{sender_name}]: {text}"

    def to_lc_human_message(
        self,
        text: str | None = None,
        is_group: bool = False,
        sender_name: str = "Unknown",
        image_data_url: str | None = None,
    ) -> HumanMessage:
        """Build a LangChain HumanMessage from an incoming Telegram message."""
        body = text or ""
        if is_group and body:
            body = self._group_prefix(body, sender_name)

        if image_data_url:
            return HumanMessage(content=[
                {"type": "text", "text": body or "What's in this image?"},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ])
        return HumanMessage(content=body)
```

Keep `_apply_group_sender_prefix` only if nothing else references it; otherwise delete it. `build_system_prompt`, `_resolve_group_personality_prompt`, and `_current_time_iso` are unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_prompt_builder.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add prompt_builder.py tests/test_prompt_builder.py
git commit -m "Build LangChain messages in PromptBuilder"
```

---



## Task 6: tools.py — web search + fetch

**Files:**

- Create: `tools.py`
- Test: `tests/test_tools.py` (create)

**Interfaces:**

- Consumes: `config` (`TAVILY_API_KEY`).
- Produces:
  - `build_tools(config) -> list` — returns `[search_tool, fetch_url]`. `search_tool` is a `TavilySearch(max_results=5)` when `config.TAVILY_API_KEY` is non-empty, else a DuckDuckGo-backed `@tool web_search`.
  - `fetch_url` — `@tool` fetching a URL via `httpx`, returning up to ~8000 chars of text.
  - `web_search_backend(config) -> str` — returns `"tavily"` or `"duckduckgo"` (pure helper for tests).

- [ ] **Step 1: Write the failing test**

Create `tests/test_tools.py`:

```python
"""Tool selection: Tavily when key present, DuckDuckGo fallback otherwise."""
import tools


class _CfgTavily:
    TAVILY_API_KEY = "tvly-x"


class _CfgNoKey:
    TAVILY_API_KEY = ""


def test_backend_selection():
    assert tools.web_search_backend(_CfgTavily) == "tavily"
    assert tools.web_search_backend(_CfgNoKey) == "duckduckgo"


def test_build_tools_returns_search_and_fetch():
    built = tools.build_tools(_CfgNoKey)
    names = {t.name for t in built}
    assert "fetch_url" in names
    assert any("search" in n for n in names)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tools.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools'`.

- [ ] **Step 3: Create** `tools.py`

```python
"""Agent tools: web search (Tavily or DuckDuckGo) and page fetch."""
from __future__ import annotations

import logging

import httpx
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def web_search_backend(config) -> str:
    """Which search backend will be used given the current config."""
    return "tavily" if getattr(config, "TAVILY_API_KEY", "").strip() else "duckduckgo"


@tool
def fetch_url(url: str) -> str:
    """Fetch the text content of a web page.

    Use this to read a specific URL returned by web search.

    Args:
        url: The full http(s) URL to fetch.
    """
    try:
        resp = httpx.get(url, timeout=15, follow_redirects=True,
                         headers={"User-Agent": "Mozilla/5.0 (telegram-gpt bot)"})
        resp.raise_for_status()
        text = resp.text
        return text[:8000]
    except Exception as e:  # tool errors are surfaced to the model, not the user
        return f"Failed to fetch {url}: {e}"


def _duckduckgo_search_tool():
    from ddgs import DDGS

    @tool
    def web_search(query: str) -> str:
        """Search the web for current information.

        Use when you need recent facts, news, or data you don't already know.

        Args:
            query: The search query (2-10 words works best).
        """
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5))
            if not results:
                return "No results found."
            return "\n\n".join(
                f"{r.get('title', '')}\n{r.get('href', '')}\n{r.get('body', '')}"
                for r in results
            )
        except Exception as e:
            return f"Search failed: {e}"

    return web_search


def build_tools(config) -> list:
    """Assemble the agent's tool set based on configuration."""
    if web_search_backend(config) == "tavily":
        from langchain_tavily import TavilySearch
        search = TavilySearch(max_results=5, tavily_api_key=config.TAVILY_API_KEY)
    else:
        logger.info("TAVILY_API_KEY not set — using DuckDuckGo web search")
        search = _duckduckgo_search_tool()
    return [search, fetch_url]
```

> Verify against the installed `langchain-tavily` that `TavilySearch(...).name` contains `"search"` (it is `"tavily_search"`). If a version names it differently, the `test_build_tools_returns_search_and_fetch` assertion `any("search" in n ...)` still holds; adjust only if the name lacks "search".

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tools.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add tools.py tests/test_tools.py
git commit -m "Add web search and page fetch tools"
```

---



## Task 7: agent.py — the Agent class

**Files:**

- Modify: `agent.py` (add `AgentContext`, dynamic prompt middleware, `Agent` class)
- Test: `tests/test_agent.py` (create)

**Interfaces:**

- Consumes: `PromptBuilder` (Task 5), `build_tools` (Task 6), `resolve_model`/`provider_api_key`/`PROVIDER_LABEL` (Task 4), `make_trim_middleware` (Task 3).
- Produces the `Agent` class:
  - `Agent(config, prompt_builder, checkpointer, model_name)` — builds tools once, builds the dynamic-prompt + trim middleware, compiles the agent for `model_name`.
  - `set_model(model_name: str) -> None` — resolves provider; if the provider key is blank, leaves the agent uncompiled and records the provider (so `run` raises a clear message); else recompiles with a fresh `init_chat_model` instance.
  - `async run(chat_id, human_message, is_group, reply_context=None) -> str` — invokes the compiled agent with `thread_id=chat_id` and `context=AgentContext(is_group, reply_context)` inside `asyncio.to_thread`; returns the final assistant text; maps failures to `CompletionError`.
  - `append_context_message(chat_id, human_message) -> None` — `graph.update_state` to append a non-triggering group message to the thread without a model call.
  - `clear_thread(chat_id) -> None` — `checkpointer.delete_thread(chat_id)`.
  - `count_tokens` is the module function (re-exported for handlers).

- [ ] **Step 1: Write the failing test**

Create `tests/test_agent.py`:

```python
"""Agent: fake-model tool invocation, key-missing handling, error mapping."""
import asyncio
import pytest

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

import agent as agent_mod
from prompt_builder import PromptBuilder


class _Cfg:
    OPENAI_API_KEY = "o"
    XAI_API_KEY = ""            # xAI key intentionally missing
    GEMINI_API_KEY = ""
    TAVILY_API_KEY = ""
    OPENAI_TIMEOUT = 60
    MAX_CONTEXT_TOKENS = 16000
    RESERVE_TOKENS_TEXT = 2000
    RESERVE_TOKENS_IMAGE = 3000


def _prompt_builder():
    return PromptBuilder(default_private_prompt="PRIVATE", default_group_prompt="GROUP")


def _agent_with_fake(fake_model):
    """Build an Agent, then swap in a fake compiled graph over a fake model."""
    a = agent_mod.Agent(
        config=_Cfg,
        prompt_builder=_prompt_builder(),
        checkpointer=InMemorySaver(),
        model_name="gpt-5",
    )
    a._compile(fake_model)  # test hook: recompile against an injected model
    return a


def test_run_returns_final_text():
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    fake = GenericFakeChatModel(messages=iter([AIMessage(content="hi there")]))
    a = _agent_with_fake(fake)
    out = asyncio.run(a.run("chat-1", HumanMessage(content="hello"), is_group=False))
    assert out == "hi there"


def test_missing_provider_key_raises_completion_error():
    a = agent_mod.Agent(
        config=_Cfg, prompt_builder=_prompt_builder(),
        checkpointer=InMemorySaver(), model_name="gpt-5",
    )
    a.set_model("grok-4-1-fast-reasoning")  # xAI key blank -> uncompiled
    with pytest.raises(agent_mod.CompletionError) as exc:
        asyncio.run(a.run("chat-1", HumanMessage(content="hi"), is_group=False))
    assert "xAI" in exc.value.user_message
```

> `_compile(model)` is a small internal method (Step 3) that builds the graph from an already-constructed chat model — used both by `set_model` (real model) and by tests (fake model). If `GenericFakeChatModel`'s import path differs in the installed version, use `langchain_core.language_models.fake_chat_models.GenericFakeChatModel` or `FakeMessagesListChatModel`; both accept a queued message list.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent.py -v`
Expected: FAIL — `Agent` doesn't exist.

- [ ] **Step 3: Add** `AgentContext`**, dynamic prompt, and** `Agent` **to** `agent.py`

Add imports at the top of `agent.py`:

```python
import asyncio
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
```

Append:

```python
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
            make_trim_middleware(
                config.MAX_CONTEXT_TOKENS,
                config.RESERVE_TOKENS_TEXT,
                config.RESERVE_TOKENS_IMAGE,
            ),
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
        )
        self._compile(model)
        logger.info("Agent compiled for %s (%s)", model_name, provider)

    # --- runtime ---------------------------------------------------------
    def _config_for(self, chat_id: str) -> dict:
        return {"configurable": {"thread_id": str(chat_id)}}

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
            return result["messages"][-1].content
        except CompletionError:
            raise
        except Exception as e:
            raise _to_completion_error(e) from e

    def append_context_message(self, chat_id, human_message) -> None:
        """Append a non-triggering group message to the thread (no model call)."""
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
```



Add the error-mapping helper near the top of the file (after `CompletionError`):

```python
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
        return CompletionError(
            "❌ Network error connecting to the API. "
            "Please check your internet connection."
        )
    logger.error("Unhandled agent error: %s", exc, exc_info=True)
    return CompletionError(
        "❌ An unexpected error occurred. Please try again or contact support."
    )
```

> `_make_dynamic_prompt` reads `request.runtime.context`. If the installed `@dynamic_prompt` passes a different object, fall back to `@wrap_model_call` and `request.override(system_prompt=...)` reading `request.runtime.context` — the TDD test in Step 1 will reveal the exact attribute; adjust the one accessor.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_agent.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Add a fake-model tool-invocation test**

Append to `tests/test_agent.py` (spec: verify which tool is called / argument parsing without a live API):

```python
def test_agent_invokes_a_tool_then_answers():
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage
    # First model turn asks to call fetch_url; second turn answers.
    fake = GenericFakeChatModel(messages=iter([
        AIMessage(content="", tool_calls=[
            {"name": "fetch_url", "args": {"url": "https://example.com"}, "id": "c1"}]),
        AIMessage(content="done"),
    ]))
    a = _agent_with_fake(fake)
    out = asyncio.run(a.run("chat-tool", HumanMessage(content="read example.com"),
                            is_group=False))
    assert out == "done"
```

> If `GenericFakeChatModel` does not replay queued `tool_calls` in the installed version, use `langchain.agents` test utilities or `FakeMessagesListChatModel`. If neither reliably drives a tool loop, keep this as the single test that may be marked `xfail` with a comment — the tool-selection logic is otherwise covered by `tests/test_tools.py`.

- [ ] **Step 6: Run and commit**

Run: `pytest tests/test_agent.py -v`
Expected: PASS (3 tests, or 2 pass + 1 xfail per the note).

```bash
git add agent.py tests/test_agent.py
git commit -m "Add Agent class wiring create_agent + checkpointer"
```

---



## Task 8: Cutover — bot.py + [handlers.py](http://handlers.py)

**Files:**

- Modify: `bot.py`
- Modify: `handlers.py`
- Delete: `openai_client.py`, `token_manager.py`
- Delete: `tests/test_token_manager.py`, `tests/test_model_registry.py`

**Interfaces:**

- Consumes: `Agent`, `CompletionError`, `MODEL_PROVIDERS`, `count_tokens` from `agent.py`; `PromptBuilder.to_lc_human_message` from Task 5.
- Produces: a running bot whose triggering messages go through `agent.run`, non-triggering group messages through `agent.append_context_message`, `/clear` through `agent.clear_thread`, `/model` through `agent.set_model` validated against `MODEL_PROVIDERS`; `messages` used only for audit/stats.

- [ ] **Step 1: Rewrite** `bot.py` **wiring**

Replace the imports:

```python
from config import config
from database import Database
from prompt_builder import PromptBuilder
from agent import Agent
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres import PostgresSaver
import handlers
```

Replace the default private/group prompt source (previously `OpenAIClient.SYSTEM_PROMPT`) with module constants in `agent.py`. Add these to `agent.py` (top-level constants, copied verbatim from the old `OpenAIClient`):

```python
SYSTEM_PROMPT = """You are Tze Foong's Assistant, an AI helper in Telegram.

Key behaviors:
- Be direct and concise - no unnecessary preambles
- Provide clear, helpful responses
- Never claim to be OpenAI or reference being a language model
- Respond naturally as a personal assistant"""

SYSTEM_PROMPT_GROUP = """You are Tze Foong's Assistant, an AI helper in Telegram group chats.

Key behaviors:
- Be direct and concise - no unnecessary preambles
- Provide clear, helpful responses
- Never claim to be OpenAI or reference being a language model
- Track conversation context from multiple participants
- Messages are formatted as [Name]: content - reply naturally without mimicking this format"""
```

In `bot.py` `main()`, replace the token-manager / OpenAI-client / prompt-builder block (steps 3–5 in the old file) with:

```python
        # 3. Build the Postgres checkpointer over a dedicated psycopg3 pool.
        #    Tables are created out-of-band by scripts/setup_checkpointer.py
        #    (deploy preDeployCommand); we do NOT call .setup() here.
        logger.info("Initializing checkpointer pool...")
        global checkpointer_pool
        checkpointer_pool = ConnectionPool(
            conninfo=config.DATABASE_URL,
            max_size=10,
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        )
        checkpointer = PostgresSaver(checkpointer_pool)

        # 4. Prompt builder (shared with the dynamic-prompt middleware).
        logger.info("Initializing prompt builder...")
        prompt_builder = PromptBuilder(
            default_private_prompt=agent_module.SYSTEM_PROMPT,
            default_group_prompt=agent_module.SYSTEM_PROMPT_GROUP,
            get_active_personality=db.get_active_personality,
            get_personality_prompt=db.get_personality_prompt,
        )

        # 5. Build the agent for the active model.
        logger.info("Building agent...")
        bot_agent = Agent(
            config=config,
            prompt_builder=prompt_builder,
            checkpointer=checkpointer,
            model_name=effective_model,
        )
```

Add `import agent as agent_module` alongside the imports, and declare the new globals near the top:

```python
db = None
application = None
bot_agent = None
checkpointer_pool = None
```

Remove `token_manager` and `openai_client` globals. Update the `handlers.init_handlers(...)` call:

```python
        handlers.init_handlers(config, db, bot_agent, prompt_builder, bot_username)
```

Update `post_shutdown` to close the pool:

```python
async def post_shutdown(app: Application):
    logger.info("Bot shutting down gracefully...")
    if db:
        db.close()
    if checkpointer_pool:
        checkpointer_pool.close()
```

The `db.init_active_model(config.DEFAULT_MODEL)` + `effective_model = db.get_active_model()` block stays as-is.

- [ ] **Step 2: Rewrite** `handlers.py` **module globals +** `init_handlers`

Replace the top of `handlers.py`:

```python
import logging
import re
import random
import base64
from telegram import Update
from telegram.ext import ContextTypes
from telegram.helpers import escape_markdown
from agent import MODEL_PROVIDERS, CompletionError, count_tokens

logger = logging.getLogger(__name__)

config = None
db = None
agent = None
prompt_builder = None
bot_username = None


def init_handlers(cfg, database, bot_agent, prompt_bldr, username=None):
    """Initialize handler dependencies."""
    global config, db, agent, prompt_builder, bot_username
    config = cfg
    db = database
    agent = bot_agent
    prompt_builder = prompt_bldr
    bot_username = username
```

- [ ] **Step 3: Rewrite the non-triggering group branch in** `message_handler`

Replace the `if is_group and not has_keyword:` block body with (audit-log to `messages` **and** append to the checkpoint thread):

```python
    if is_group and not has_keyword:
        try:
            db.add_message(
                chat_id=chat_id, role="user", content=message.text,
                user_id=user_id, message_id=message.message_id,
                token_count=count_tokens(message.text),
                sender_name=sender_name, sender_username=sender_username,
                is_group_chat=True,
            )
            agent.append_context_message(
                chat_id,
                prompt_builder.to_lc_human_message(
                    text=message.text, is_group=True, sender_name=sender_name),
            )
            if random.random() < 0.1:
                db.cleanup_old_group_messages(chat_id, config.MAX_GROUP_CONTEXT_MESSAGES)
        except Exception as e:
            logger.error(f"Failed to store group message: {e}")
        return
```

- [ ] **Step 4: Rewrite** `process_request`

Replace the whole `process_request` body:

```python
async def process_request(message, prompt, user_id, sender_name, sender_username,
                          is_group, reply_context=None):
    """Process a triggering text request through the agent."""
    chat_id = str(message.chat_id)
    try:
        # Audit-log the user message (context lives in the checkpoint).
        db.add_message(
            chat_id=chat_id, role="user", content=message.text,
            user_id=user_id, message_id=message.message_id,
            token_count=count_tokens(prompt),
            sender_name=sender_name, sender_username=sender_username,
            is_group_chat=is_group,
        )

        human = prompt_builder.to_lc_human_message(
            text=prompt, is_group=is_group, sender_name=sender_name)
        response = await agent.run(chat_id, human, is_group, reply_context=reply_context)

        db.add_message(
            chat_id=chat_id, role="assistant", content=response,
            token_count=count_tokens(response), is_group_chat=is_group,
        )
        await message.reply_text(response)
        logger.info(f"Response sent for chat {chat_id}")

    except CompletionError as e:
        await message.reply_text(e.user_message)
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        await message.reply_text(
            "Sorry, I encountered an error processing your request. Please try again."
        )
```

- [ ] **Step 5: Rewrite** `process_image_request`

Replace its body (split storage stays for the audit log; the image is sent to the agent for this turn only):

```python
async def process_image_request(message, prompt, user_id, sender_name, sender_username,
                                is_group, reply_context=None):
    """Process a triggering image request through the agent."""
    chat_id = str(message.chat_id)
    try:
        photo = message.photo[-1]
        photo_file = await photo.get_file()
        photo_bytes = await photo_file.download_as_bytearray()
        base64_image = base64.b64encode(photo_bytes).decode("utf-8")
        image_data_url = f"data:image/jpeg;base64,{base64_image}"

        # Audit-log a text marker only (never the base64 payload).
        caption_marker = f"[image] {message.caption}" if message.caption else "[image]"
        db.add_message(
            chat_id=chat_id, role="user", content=caption_marker,
            user_id=user_id, message_id=message.message_id,
            token_count=count_tokens(caption_marker),
            sender_name=sender_name, sender_username=sender_username,
            is_group_chat=is_group,
        )

        human = prompt_builder.to_lc_human_message(
            text=prompt, is_group=is_group, sender_name=sender_name,
            image_data_url=image_data_url)
        response = await agent.run(chat_id, human, is_group, reply_context=reply_context)

        db.add_message(
            chat_id=chat_id, role="assistant", content=response,
            token_count=count_tokens(response), is_group_chat=is_group,
        )
        await message.reply_text(response)
        logger.info(f"Image processed for chat {chat_id}")

    except CompletionError as e:
        await message.reply_text(e.user_message)
    except Exception as e:
        logger.error(f"Error processing image: {e}", exc_info=True)
        await message.reply_text(
            "Sorry, I encountered an error processing your image. Please try again."
        )
```



Remove the now-unused `import base64` inside the old function body (it moved to the module header in Step 2).

- [ ] **Step 6: Rewrite** `/clear` **and** `/model` **commands**

In `clear_command`, replace `db.clear_history(chat_id)` with the checkpointer delete (audit `messages` are intentionally retained):

```python
    chat_id = str(update.message.chat_id)
    try:
        agent.clear_thread(chat_id)
        await update.message.reply_text("✅ Conversation history cleared for this chat.")
        logger.info(f"History cleared for chat {chat_id}")
    except Exception as e:
        logger.error(f"Error clearing history: {e}", exc_info=True)
        await update.message.reply_text("❌ Failed to clear history. Please try again.")
```

In `model_command`, replace every `MODEL_REGISTRY` reference with `MODEL_PROVIDERS` and the client/token-manager updates with `agent.set_model`:

```python
    available = "\n".join(f"  `{m}`" for m in MODEL_PROVIDERS)

    if not context.args:
        current = db.get_active_model()
        await update.message.reply_text(
            f"Current model: `{current}`\n\nAvailable models:\n{available}\n\nUsage: `/model <name>`",
            parse_mode="Markdown",
        )
        return

    new_model = context.args[0].strip()
    if new_model not in MODEL_PROVIDERS:
        await update.message.reply_text(
            f"❌ Unknown model `{new_model}`.\n\nAvailable models:\n{available}",
            parse_mode="Markdown",
        )
        return

    try:
        db.set_active_model(new_model)
        agent.set_model(new_model)
        await update.message.reply_text(f"✅ Model switched to `{new_model}`", parse_mode="Markdown")
        logger.info(f"User {user_id} switched model to {new_model}")
    except Exception as e:
        logger.error(f"Error switching model: {e}", exc_info=True)
        await update.message.reply_text("❌ Failed to switch model. Please try again.")
```

- [ ] **Step 7: Delete retired modules and tests**

```bash
git rm openai_client.py token_manager.py tests/test_token_manager.py tests/test_model_registry.py
```

- [ ] **Step 8: Compile and run the full suite**

Run:

```bash
python3 -m py_compile *.py scripts/*.py
pytest tests/ -v
```

Expected: py_compile clean (no import of `openai_client`/`token_manager` remains); all tests pass (`test_extract_keyword`, `test_config`, `test_trimming`, `test_model_resolution`, `test_prompt_builder`, `test_tools`, `test_agent`).

- [ ] **Step 9: Grep for dangling references**

Run:

```bash
grep -rn "openai_client\|token_manager\|MODEL_REGISTRY\|OpenAIClient\|format_messages\|trim_to_fit\|get_messages_by_tokens" --include="*.py" . | grep -v tests/
```

Expected: no matches outside comments. (`db.get_messages_by_tokens`/`db.clear_history` may remain defined in `database.py` — that's fine; they're simply no longer called by the agent path. Leave them for audit/back-compat.)

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "Cut bot over to LangChain agent; retire openai_client and token_manager"
```

---



## Task 9: Checkpointer setup, start.sh, and docs

**Files:**

- Create: `scripts/setup_checkpointer.py`
- Modify: `start.sh`
- Modify: `README.md`, `database.md`, `CLAUDE.md`

**Interfaces:**

- Produces: an idempotent `PostgresSaver.setup()` runner and documentation for the new deploy step; no code imports depend on this task.

- [ ] **Step 1: Create** `scripts/setup_checkpointer.py`

```python
"""Create/upgrade the LangGraph PostgresSaver tables. Idempotent.

Run once per environment AFTER `alembic upgrade head` and BEFORE the bot
starts. The checkpointer tables are versioned independently by
langgraph-checkpoint-postgres and are intentionally NOT under Alembic.
"""
import logging

from config import config
from langgraph.checkpoint.postgres import PostgresSaver

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    if not config.DATABASE_URL.strip():
        raise SystemExit("DATABASE_URL is required to set up the checkpointer")
    with PostgresSaver.from_conn_string(config.DATABASE_URL) as checkpointer:
        checkpointer.setup()
    logger.info("Checkpointer tables are set up")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script runs against a database**

Run (requires a reachable `DATABASE_URL`; safe/idempotent):

```bash
python3 scripts/setup_checkpointer.py
```

Expected: logs "Checkpointer tables are set up"; re-running is a no-op. If no DB is available in the dev sandbox, skip execution and rely on the compile check in Task 8 Step 8.

- [ ] **Step 3: Update** `start.sh`

Insert the checkpointer setup immediately after the existing Alembic line so the block reads:

```bash
echo "Applying database migrations..."
alembic upgrade head || exit 1

echo "Setting up LangGraph checkpointer tables..."
python scripts/setup_checkpointer.py || exit 1

echo "Starting bot with fresh instance..."
python bot.py
```

- [ ] **Step 4: Update docs**

In `README.md` add a "Checkpointer setup" note: local dev and each Railway environment must run `python scripts/setup_checkpointer.py` once after `alembic upgrade head`; on Railway this belongs in each environment's `preDeployCommand` (`alembic upgrade head && python scripts/setup_checkpointer.py`). Document `TAVILY_API_KEY` (optional; DuckDuckGo fallback) and the new required set (`TELEGRAM_BOT_TOKEN`, `AUTHORIZED_USER_ID`, `OPENAI_API_KEY`, `DATABASE_URL`).

In `database.md` add a "LangGraph checkpointer (not Alembic-managed)" section listing `checkpoints`, `checkpoint_blobs`, `checkpoint_writes`, `checkpoint_migrations` as owned by `langgraph-checkpoint-postgres` via `setup()`, separate from the Alembic app schema.

In `CLAUDE.md`: update the provider-routing section to reference `agent.py`/`init_chat_model` and `MODEL_PROVIDERS` instead of `openai_client.py`/`MODEL_REGISTRY`; note `token_manager.py` is gone; add the checkpointer `preDeployCommand` step.

- [ ] **Step 5: Commit**

```bash
git add scripts/setup_checkpointer.py start.sh README.md database.md CLAUDE.md
git commit -m "Add checkpointer setup step and update docs"
```

- [ ] **Step 6: Update the Railway preDeployCommand (infra — not a repo change)**

For both the `production` and `dev` Railway environments, set the service `preDeployCommand` to:

```
alembic upgrade head && python scripts/setup_checkpointer.py
```

This is a Railway dashboard/config change, not a file in the repo. Verify by triggering a dev deploy and confirming both steps run in the deploy logs before the bot boots. (Cannot be done from this plan's code steps; record it as a required manual deploy-config change.)

---



## Task 10: End-to-end verification

**Files:** none (verification only)

- [ ] **Step 1: Full local validation**

Run:

```bash
pip install -r requirements.txt -r requirements-dev.txt
python3 -m py_compile *.py scripts/*.py
pytest tests/ -v
```

Expected: all pass.

- [ ] **Step 2: CLI simulator smoke (if it still applies)**

`scripts/chat_cli.py` drives the old client directly; if it imports `openai_client`, update it to build an `Agent` the same way `bot.py` does, or mark it out of scope. Run:

```bash
python3 scripts/chat_cli.py --chat-id test
```

Expected: either a working chat loop against the agent, or a clear note in the PR that the simulator was updated/deferred.

- [ ] **Step 3: Deploy to** `dev` **and verify against a real bot**

Push the branch, open a PR into `dev` (per repo workflow), let it deploy to the Railway `dev` environment, then in the dev bot verify: a `chatgpt` message returns an answer; a `chatgpt` message that needs current info triggers a web search; `/model` switches provider and the next message works; `/clear` empties context (next message has no memory of prior turns); `/stats` still reports counts from `messages`; a group non-triggering message is remembered as context on the next trigger.

- [ ] **Step 4: Promote via PR to** `main`

Open a PR `dev` → `main` (do not merge locally). Ensure CI passes. Confirm the `production` Railway `preDeployCommand` was updated (Task 9 Step 6) before merge so production runs `setup_checkpointer.py`.

---



## Assumptions / Notes Carried Into Implementation

- **Reasoning-effort/verbosity tuning is dropped.** The old client passed `reasoning={"effort":"low"}` / `text={"verbosity":"low"}` for some models. The spec does not require preserving this. If desired later, pass provider-specific kwargs to `init_chat_model`. Not implemented here.
- **Images are persisted in the checkpoint for the turn they arrive.** The spec defers the image-storage-window feature; this plan sends the image to the agent for the current turn and does not add logic to strip it from checkpoint state. Base64 growth is bounded in practice by the trimming middleware dropping old messages from the model view, and by `/clear`. Revisit in the image-storage-window spec.
- **Error normalization is best-effort by exception type/message** (`_to_completion_error`), because LangChain does not fully unify provider exception types across OpenAI/xAI/Gemini. User-facing messages match the old client's for the common cases (auth, rate limit, timeout, context length, connection).
- **Two Postgres drivers coexist**: `psycopg2` (existing `database.py`) and `psycopg` v3 (checkpointer). This is intentional and low-risk; a future cleanup could consolidate on psycopg3.
- **Fast-moving-library accessors to verify during TDD** (the failing test will pin the exact shape for the installed versions): `@dynamic_prompt` request object's `.runtime.context`; `ModelRequest.override(messages=...)`; `GenericFakeChatModel` replaying queued `tool_calls`; `TavilySearch` tool `.name`. Each has a documented fallback noted at its task.

