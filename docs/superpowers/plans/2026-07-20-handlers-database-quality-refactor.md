# Handlers & Database Code Quality Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the testability and separation of concerns of `handlers.py` and `database.py` — the two most-coupled, least-tested modules in the bot — without changing observable bot behavior at any point in the sequence.

**Architecture:** Characterize current behavior with regression tests first. Then replace `handlers.py`'s module-level globals with an explicit `HandlerDependencies` bundle, split it into `authorization.py` / `request_processor.py` / `message_handlers.py` / `command_handlers.py` behind a thin `handlers.py` facade that `bot.py` keeps registering unchanged. Then split `database.py` into a shared `ConnectionManager` plus four focused repositories behind a `Database` facade. Then remove the two confirmed-dead legacy read APIs, add scoped lint/type gates, and finish by extracting a single `app_factory.py` composition point so `bot.py` and `scripts/chat_cli.py` can no longer drift apart.

**Tech Stack:** Python 3.12, python-telegram-bot 21.7, psycopg2 (app schema), LangChain/LangGraph (agent), pytest (no live DB/Telegram/API calls in tests — see `tests/test_context_message_retention.py` and `tests/test_agent.py` for the existing fake/`SimpleNamespace` conventions this plan follows).

## Global Constraints

- No behavior change to the running bot at any task boundary — every task ends with `pytest tests/ -v` green and `python3 -m py_compile *.py` clean.
- Tests never hit a live database, Telegram API, or model API (per `CLAUDE.md` Testing Guidelines). Database tests use fake connection/cursor doubles, never a real `psycopg2` connection.
- `handlers.py` remains the module `bot.py` imports and registers Telegram callbacks from (`handlers.message_handler`, `handlers.clear_command`, etc.) and keeps `init_handlers(cfg, database, bot_agent, prompt_bldr, username=None)` as its entry point — this is a project-documented module (`CLAUDE.md` Project Structure) and this plan turns it into a facade, not a module to delete.
- Every new/changed public function or method signature used across module boundaries must match exactly what downstream tasks expect — see each task's **Interfaces** block.
- Commit after every green step, scoped to that step's files only.

---

## File Structure

New files, in the order they're introduced:

| File | Introduced in | Responsibility |
|---|---|---|
| `tests/test_database_characterization.py` | Task 1 | Regression coverage for current `Database` (deleted in Task 15, replaced by `tests/test_repositories.py`) |
| `tests/test_handlers_characterization.py` | Task 2 | Regression coverage for current `handlers.py` free-function/global wiring |
| `handler_deps.py` | Task 3 | `HandlerDependencies` dataclass |
| `authorization.py` | Task 3 | `is_authorized` / `is_main_authorized_user`, as plain functions taking `config`/`db` explicitly |
| `request_processor.py` | Task 4 | `typing_action` + `RequestProcessor` — the shared audit-log → agent.run → audit-log → reply workflow |
| `message_handlers.py` | Task 5 | `MessageHandlers` class (`message_handler`, `photo_handler`) + `extract_keyword` / `extract_reply_data` |
| `command_handlers.py` | Task 6 | `CommandHandlers` class (all ten admin commands) + `error_handler` |
| `handlers.py` (rewritten) | Task 7 | Thin facade: builds `HandlerDependencies` in `init_handlers`, re-exports bound-method wrappers |
| `model_registry.py` | Task 8 | `MODEL_PROVIDERS`, `PROVIDER_LABEL`, `resolve_model`, `provider_api_key` (moved out of `agent.py`) |
| `token_budget.py` | Task 9 | `count_tokens`, `count_message_tokens`, `count_messages_tokens`, `trim_messages`, `make_trim_middleware` (moved out of `agent.py`) |
| `db_connection.py` | Task 10 | `ConnectionManager` — pool + health-checked connection context manager |
| `message_repository.py` | Task 11 | `MessageRepository` — `messages` table CRUD |
| `access_repository.py` | Task 12 | `AccessRepository` — `granted_users` table CRUD |
| `settings_repository.py` | Task 13 | `SettingsRepository` — `personality` / `active_personality` / `active_model` CRUD |
| `summary_audit_repository.py` | Task 14 | `SummaryAuditRepository` — `conversation_summaries` insert-only audit |
| `database.py` (rewritten) | Task 15 | Thin facade delegating to the four repositories |
| `tests/test_repositories.py` | Task 15 | Replaces `tests/test_database_characterization.py`, targets repositories directly |
| `pyproject.toml` | Task 17 | Ruff + mypy config, scoped to the new modules |
| `app_factory.py` | Task 18 | `build_app_stack(config) -> AppStack` — single composition point for `bot.py` and `scripts/chat_cli.py` |

Modified existing files: `agent.py` (Tasks 8, 9 — re-exports moved names), `scripts/chat_cli.py` (Tasks 8, 18), `tests/test_context_message_retention.py` (Task 7), `.github/workflows/ci.yml` (Task 17), `requirements-dev.txt` (Task 17), `bot.py` (Task 18).

---

## Task 1: Database characterization tests

**Files:**
- Create: `tests/test_database_characterization.py`

**Interfaces:**
- Consumes: `database.Database` as it exists today (`__init__(db_url)`, `_get_connection()` contextmanager method, `_cache` attribute).
- Produces: nothing consumed by later tasks directly — this file is superseded and deleted in Task 15 once the repository split lands.

- [ ] **Step 1: Write the characterization tests**

```python
"""Database characterization tests: SQL executed, params, and cache
invalidation timing, using a fake connection double instead of a live DB
(CLAUDE.md: unit tests never hit a live database)."""
from contextlib import contextmanager

from cache import MISSING, TTLCache
from database import Database


class _FakeCursor:
    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):
        self._conn.executed.append((" ".join(sql.split()), params))

    def fetchone(self):
        return self._conn.results.pop(0) if self._conn.results else None

    def fetchall(self):
        return self._conn.results.pop(0) if self._conn.results else []

    @property
    def rowcount(self):
        return self._conn.rowcount


class _FakeConnection:
    def __init__(self, results=None, rowcount=0):
        self.executed = []
        self.results = list(results or [])
        self.rowcount = rowcount
        self.closed = False

    def cursor(self, cursor_factory=None):
        return _FakeCursor(self)

    def commit(self):
        pass

    def rollback(self):
        pass


def _make_db(results=None, rowcount=0):
    """Build a Database without opening a real connection pool."""
    db = Database.__new__(Database)
    db._cache = TTLCache(default_ttl=60.0)
    fake_conn = _FakeConnection(results=results, rowcount=rowcount)

    @contextmanager
    def fake_get_connection():
        yield fake_conn

    db._get_connection = fake_get_connection
    return db, fake_conn


def _broken_db():
    """A Database whose connection always fails to open."""
    db = Database.__new__(Database)
    db._cache = TTLCache(default_ttl=60.0)

    @contextmanager
    def fake_get_connection():
        raise RuntimeError("db down")
        yield  # pragma: no cover

    db._get_connection = fake_get_connection
    return db


def test_add_message_inserts_with_expected_params():
    db, conn = _make_db(results=[(42,)])
    msg_id = db.add_message(
        chat_id=123, role="user", content="hi", user_id=7, message_id=1,
        token_count=3, sender_name="Alice", sender_username="alice",
        is_group_chat=True,
    )
    assert msg_id == 42
    sql, params = conn.executed[0]
    assert "INSERT INTO messages" in sql
    assert params[0] == "123"
    assert params[1] == "user"
    assert params[2] == "hi"


def test_is_user_granted_caches_result_and_skips_second_query():
    db, conn = _make_db(results=[("55",)])
    first = db.is_user_granted(55)
    second = db.is_user_granted(55)
    assert first is True
    assert second is True
    assert len(conn.executed) == 1


def test_grant_access_invalidates_granted_cache():
    db, conn = _make_db(results=[None])
    db._cache.set("granted:99", False, ttl=120.0)
    was_granted = db.grant_access(99, first_name="Bob", username="bob")
    assert was_granted is True
    assert db._cache.get("granted:99") is MISSING


def test_set_active_personality_invalidates_related_cache_entries():
    db, conn = _make_db()
    db._cache.set("active_personality", "old", ttl=60.0)
    db._cache.set("personality_prompt:old", "prompt text", ttl=300.0)
    db.set_active_personality("villain")
    assert db._cache.get("active_personality") is MISSING
    assert db._cache.get("personality_prompt:old") is MISSING


def test_get_active_model_returns_default_when_no_row():
    db, conn = _make_db(results=[None])
    assert db.get_active_model() == "gpt-5.4-mini"


def test_get_stats_returns_na_when_no_messages():
    db, conn = _make_db(results=[{
        "total_messages": 0, "total_tokens": 0,
        "first_message": None, "last_message": None,
    }])
    stats = db.get_stats("123")
    assert stats == {
        "total_messages": 0, "total_tokens": 0,
        "first_message": "N/A", "last_message": "N/A",
    }


def test_get_stats_returns_zeroed_dict_on_query_failure():
    db = _broken_db()
    stats = db.get_stats("123")
    assert stats["total_messages"] == 0
    assert stats["first_message"] == "N/A"


def test_list_personalities_truncates_long_prompt_preview():
    long_prompt = "x" * 150
    db, conn = _make_db(results=[[
        {"personality": "villain", "prompt": long_prompt},
        {"personality": "normal", "prompt": "short"},
    ]])
    personalities = db.list_personalities()
    assert personalities[0] == ("villain", "x" * 100 + "...")
    assert personalities[1] == ("normal", "short")
```

- [ ] **Step 2: Run the tests**

Run: `pytest tests/test_database_characterization.py -v`
Expected: 8 passed (this characterizes existing behavior, not new behavior — nothing to implement).

- [ ] **Step 3: Commit**

```bash
git add tests/test_database_characterization.py
git commit -m "test: characterize Database cache and SQL behavior before refactor"
```

---

## Task 2: Handlers characterization tests

**Files:**
- Create: `tests/test_handlers_characterization.py`

**Interfaces:**
- Consumes: `handlers.init_handlers`, `handlers.message_handler`, `handlers.photo_handler`, `handlers.{clear,stats,grant,revoke,version,allowlist,personality,list_personality,model,help}_command`, `handlers.error_handler`, `handlers.is_authorized`, `handlers.is_main_authorized_user` as they exist today.
- Produces: nothing consumed by later tasks directly — this file's assertions get re-verified (not rewritten) once DI lands in Task 7, since `handlers.py` keeps the exact same public entry points.

- [ ] **Step 1: Write the characterization tests**

```python
"""Handler characterization tests: authorization gate, text/photo request
processing, and every admin command, against the current handlers.py
module-global wiring (SimpleNamespace/AsyncMock fakes only — no live
Telegram, database, or model calls, per CLAUDE.md)."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from agent import CompletionError
import handlers


class _Cfg:
    AUTHORIZED_USER_ID = "1"
    BOT_VERSION = "9.9.9"


def _init(db=None, agent=None, prompt_builder=None, config=None, username="mybot"):
    handlers.init_handlers(
        config or _Cfg,
        db or SimpleNamespace(),
        agent or SimpleNamespace(),
        prompt_builder or SimpleNamespace(),
        username,
    )


def _message(text=None, photo=None, caption=None, user_id=7, chat_id=123,
             chat_type="private", first_name="Alice", username="alice"):
    return SimpleNamespace(
        text=text,
        photo=photo,
        caption=caption,
        chat_id=chat_id,
        chat=SimpleNamespace(type=chat_type),
        from_user=SimpleNamespace(id=user_id, first_name=first_name, username=username),
        message_id=1,
        reply_to_message=None,
        reply_text=AsyncMock(),
    )


def _cmd_update(user_id=1, args=None):
    message = SimpleNamespace(
        from_user=SimpleNamespace(id=user_id),
        chat_id=123,
        reply_text=AsyncMock(),
    )
    update = SimpleNamespace(message=message)
    context = SimpleNamespace(args=args or [], bot=SimpleNamespace())
    return update, context, message


# --- authorization -------------------------------------------------------

def test_is_authorized_main_user_true():
    _init(config=_Cfg)
    assert handlers.is_authorized(1) is True


def test_is_authorized_granted_user_true():
    db = SimpleNamespace(is_user_granted=Mock(return_value=True))
    _init(db=db, config=_Cfg)
    assert handlers.is_authorized(42) is True
    db.is_user_granted.assert_called_once_with(42)


def test_is_authorized_unknown_user_false():
    db = SimpleNamespace(is_user_granted=Mock(return_value=False))
    _init(db=db, config=_Cfg)
    assert handlers.is_authorized(42) is False


def test_is_main_authorized_user():
    _init(config=_Cfg)
    assert handlers.is_main_authorized_user(1) is True
    assert handlers.is_main_authorized_user(2) is False


# --- text request processing ---------------------------------------------

def test_process_request_success_stores_and_replies():
    db = SimpleNamespace(add_message=Mock())
    agent = SimpleNamespace(run=AsyncMock(return_value="hi there"))
    prompt_builder = SimpleNamespace(to_lc_human_message=Mock(return_value="human"))
    _init(db=db, agent=agent, prompt_builder=prompt_builder, config=_Cfg)

    message = _message(text="chatgpt hello", user_id=1)
    update = SimpleNamespace(message=message)
    context = SimpleNamespace(bot=SimpleNamespace(send_chat_action=AsyncMock()))

    asyncio.run(handlers.message_handler(update, context))

    assert db.add_message.call_count == 2
    message.reply_text.assert_awaited_once_with("hi there")


def test_process_request_completion_error_replies_user_message():
    db = SimpleNamespace(add_message=Mock())
    agent = SimpleNamespace(run=AsyncMock(side_effect=CompletionError("rate limited")))
    prompt_builder = SimpleNamespace(to_lc_human_message=Mock(return_value="human"))
    _init(db=db, agent=agent, prompt_builder=prompt_builder, config=_Cfg)

    message = _message(text="chatgpt hello", user_id=1)
    update = SimpleNamespace(message=message)
    context = SimpleNamespace(bot=SimpleNamespace(send_chat_action=AsyncMock()))

    asyncio.run(handlers.message_handler(update, context))

    message.reply_text.assert_awaited_once_with("rate limited")


def test_process_request_generic_exception_replies_generic_message():
    db = SimpleNamespace(add_message=Mock())
    agent = SimpleNamespace(run=AsyncMock(side_effect=RuntimeError("boom")))
    prompt_builder = SimpleNamespace(to_lc_human_message=Mock(return_value="human"))
    _init(db=db, agent=agent, prompt_builder=prompt_builder, config=_Cfg)

    message = _message(text="chatgpt hello", user_id=1)
    update = SimpleNamespace(message=message)
    context = SimpleNamespace(bot=SimpleNamespace(send_chat_action=AsyncMock()))

    asyncio.run(handlers.message_handler(update, context))

    message.reply_text.assert_awaited_once_with(
        "Sorry, I encountered an error processing your request. Please try again."
    )


# --- photo request processing ---------------------------------------------

def test_process_image_request_success():
    db = SimpleNamespace(add_message=Mock())
    agent = SimpleNamespace(run=AsyncMock(return_value="nice photo"))
    prompt_builder = SimpleNamespace(to_lc_human_message=Mock(return_value="human"))
    _init(db=db, agent=agent, prompt_builder=prompt_builder, config=_Cfg)

    photo_file = SimpleNamespace(
        download_as_bytearray=AsyncMock(return_value=bytearray(b"fake-bytes"))
    )
    photo = SimpleNamespace(get_file=AsyncMock(return_value=photo_file))
    message = _message(photo=[photo], caption="chatgpt what is this", user_id=1)
    update = SimpleNamespace(message=message)
    context = SimpleNamespace(bot=SimpleNamespace(send_chat_action=AsyncMock()))

    asyncio.run(handlers.photo_handler(update, context))

    assert db.add_message.call_count == 2
    first_call_kwargs = db.add_message.call_args_list[0].kwargs
    assert first_call_kwargs["content"] == "[image] chatgpt what is this"
    message.reply_text.assert_awaited_once_with("nice photo")


def test_process_image_request_download_failure_replies_generic_image_message():
    db = SimpleNamespace(add_message=Mock())
    agent = SimpleNamespace(run=AsyncMock())
    prompt_builder = SimpleNamespace(to_lc_human_message=Mock())
    _init(db=db, agent=agent, prompt_builder=prompt_builder, config=_Cfg)

    photo = SimpleNamespace(get_file=AsyncMock(side_effect=RuntimeError("network down")))
    message = _message(photo=[photo], caption="chatgpt describe this", user_id=1)
    update = SimpleNamespace(message=message)
    context = SimpleNamespace(bot=SimpleNamespace(send_chat_action=AsyncMock()))

    asyncio.run(handlers.photo_handler(update, context))

    message.reply_text.assert_awaited_once_with(
        "Sorry, I encountered an error processing your image. Please try again."
    )
    db.add_message.assert_not_called()
    agent.run.assert_not_awaited()


# --- admin commands ---------------------------------------------------------

def test_clear_command_requires_main_user():
    _init(config=_Cfg)
    update, context, message = _cmd_update(user_id=2)
    asyncio.run(handlers.clear_command(update, context))
    message.reply_text.assert_awaited_once_with(
        "Sorry, only the main authorized user can clear history."
    )


def test_clear_command_success():
    agent = SimpleNamespace(clear_thread=Mock())
    _init(agent=agent, config=_Cfg)
    update, context, message = _cmd_update(user_id=1)
    asyncio.run(handlers.clear_command(update, context))
    agent.clear_thread.assert_called_once_with("123")
    message.reply_text.assert_awaited_once_with(
        "✅ Conversation history cleared for this chat."
    )


def test_stats_command_requires_main_user():
    _init(config=_Cfg)
    update, context, message = _cmd_update(user_id=2)
    asyncio.run(handlers.stats_command(update, context))
    message.reply_text.assert_awaited_once_with(
        "Sorry, only the main authorized user can view stats."
    )


def test_stats_command_formats_output():
    db = SimpleNamespace(get_stats=Mock(return_value={
        "total_messages": 5, "total_tokens": 1234,
        "first_message": "2026-01-01T00:00:00", "last_message": "2026-01-02T00:00:00",
    }))
    _init(db=db, config=_Cfg)
    update, context, message = _cmd_update(user_id=1)
    asyncio.run(handlers.stats_command(update, context))
    message.reply_text.assert_awaited_once_with(
        "📊 Chat Statistics:\nMessages: 5\nTotal tokens: 1,234\nSince: 2026-01-01"
    )


def test_grant_command_requires_args():
    _init(config=_Cfg)
    update, context, message = _cmd_update(user_id=1, args=[])
    asyncio.run(handlers.grant_command(update, context))
    message.reply_text.assert_awaited_once_with(
        "❌ Usage: /grant <user_id>\nExample: /grant 123456789"
    )


def test_grant_command_success():
    db = SimpleNamespace(grant_access=Mock(return_value=True))
    _init(db=db, config=_Cfg)
    update, context, message = _cmd_update(user_id=1, args=["555"])
    context.bot.get_chat = AsyncMock(return_value=SimpleNamespace(first_name="Bob", username="bobby"))
    asyncio.run(handlers.grant_command(update, context))
    db.grant_access.assert_called_once_with(555, first_name="Bob", username="bobby")
    message.reply_text.assert_awaited_once_with(
        "✅ Access granted to Bob (@bobby).\nThey can now use the bot with 'chatgpt' keyword."
    )


def test_grant_command_already_has_access():
    db = SimpleNamespace(grant_access=Mock(return_value=False))
    _init(db=db, config=_Cfg)
    update, context, message = _cmd_update(user_id=1, args=["555"])
    context.bot.get_chat = AsyncMock(return_value=SimpleNamespace(first_name="Bob", username="bobby"))
    asyncio.run(handlers.grant_command(update, context))
    message.reply_text.assert_awaited_once_with("ℹ️ User 555 already has access.")


def test_revoke_command_cannot_revoke_self():
    _init(config=_Cfg)
    update, context, message = _cmd_update(user_id=1, args=["1"])
    asyncio.run(handlers.revoke_command(update, context))
    message.reply_text.assert_awaited_once_with(
        "❌ Cannot revoke access from the main authorized user."
    )


def test_allowlist_command_lists_users():
    db = SimpleNamespace(get_granted_users=Mock(return_value=[
        (555, "2026-01-01T00:00:00", "Bob", "bobby"),
    ]))
    _init(db=db, config=_Cfg)
    update, context, message = _cmd_update(user_id=1)
    asyncio.run(handlers.allowlist_command(update, context))
    text = message.reply_text.call_args.args[0]
    assert "555" in text and "Bob" in text and "bobby" in text


def test_personality_command_shows_current_when_no_args():
    db = SimpleNamespace(get_active_personality=Mock(return_value="villain"))
    _init(db=db, config=_Cfg)
    update, context, message = _cmd_update(user_id=1, args=[])
    asyncio.run(handlers.personality_command(update, context))
    assert "villain" in message.reply_text.call_args.args[0]


def test_personality_command_unknown_personality():
    db = SimpleNamespace(personality_exists=Mock(return_value=False))
    _init(db=db, config=_Cfg)
    update, context, message = _cmd_update(user_id=1, args=["ghost"])
    asyncio.run(handlers.personality_command(update, context))
    message.reply_text.assert_awaited_once_with("❌ No personality 'ghost' found.")


def test_model_command_shows_current_when_no_args():
    db = SimpleNamespace(get_active_model=Mock(return_value="gpt-5.4-mini"))
    _init(db=db, config=_Cfg)
    update, context, message = _cmd_update(user_id=1, args=[])
    asyncio.run(handlers.model_command(update, context))
    assert "gpt-5.4-mini" in message.reply_text.call_args.args[0]


def test_model_command_unknown_model_rejected():
    _init(config=_Cfg)
    update, context, message = _cmd_update(user_id=1, args=["not-a-model"])
    asyncio.run(handlers.model_command(update, context))
    assert "Unknown model" in message.reply_text.call_args.args[0]


def test_model_command_switch_success():
    db = SimpleNamespace(set_active_model=Mock())
    agent = SimpleNamespace(set_model=Mock())
    _init(db=db, agent=agent, config=_Cfg)
    update, context, message = _cmd_update(user_id=1, args=["gpt-5.4"])
    asyncio.run(handlers.model_command(update, context))
    db.set_active_model.assert_called_once_with("gpt-5.4")
    agent.set_model.assert_called_once_with("gpt-5.4")
    message.reply_text.assert_awaited_once_with(
        "✅ Model switched to `gpt-5.4`", parse_mode="Markdown"
    )


def test_help_command_requires_main_user():
    _init(config=_Cfg)
    update, context, message = _cmd_update(user_id=2)
    asyncio.run(handlers.help_command(update, context))
    message.reply_text.assert_awaited_once_with(
        "Sorry, only the main authorized user can use this command."
    )


def test_error_handler_notifies_user_when_possible():
    message = SimpleNamespace(reply_text=AsyncMock())
    update = SimpleNamespace(message=message)
    context = SimpleNamespace(error=RuntimeError("boom"))
    asyncio.run(handlers.error_handler(update, context))
    message.reply_text.assert_awaited_once_with(
        "An error occurred while processing your request. "
        "The error has been logged."
    )
```

- [ ] **Step 2: Run the tests**

Run: `pytest tests/test_handlers_characterization.py -v`
Expected: 21 passed.

- [ ] **Step 3: Run the full suite to confirm no interference with existing tests**

Run: `pytest tests/ -v`
Expected: all tests pass (existing `tests/test_context_message_retention.py` and `tests/test_extract_keyword.py` unaffected).

- [ ] **Step 4: Commit**

```bash
git add tests/test_handlers_characterization.py
git commit -m "test: characterize handlers.py authorization, request, and command behavior"
```

---

## Task 3: HandlerDependencies dataclass and authorization module

**Files:**
- Create: `handler_deps.py`
- Create: `authorization.py`
- Create: `tests/test_authorization.py`

**Interfaces:**
- Produces: `HandlerDependencies(config, db, agent, prompt_builder, bot_username=None)` — consumed by Tasks 4–7.
- Produces: `is_authorized(user_id, config, db) -> bool` and `is_main_authorized_user(user_id, config) -> bool` — consumed by Tasks 5, 6.

- [ ] **Step 1: Write the failing test**

```python
"""authorization.py: pure access-control checks, no module globals."""
from types import SimpleNamespace
from unittest.mock import Mock

import authorization


class _Cfg:
    AUTHORIZED_USER_ID = "1"


def test_is_authorized_main_user_true():
    assert authorization.is_authorized(1, _Cfg, db=SimpleNamespace()) is True


def test_is_authorized_delegates_to_db_for_other_users():
    db = SimpleNamespace(is_user_granted=Mock(return_value=True))
    assert authorization.is_authorized(42, _Cfg, db) is True
    db.is_user_granted.assert_called_once_with(42)


def test_is_authorized_false_when_not_granted():
    db = SimpleNamespace(is_user_granted=Mock(return_value=False))
    assert authorization.is_authorized(42, _Cfg, db) is False


def test_is_main_authorized_user():
    assert authorization.is_main_authorized_user(1, _Cfg) is True
    assert authorization.is_main_authorized_user(2, _Cfg) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_authorization.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'authorization'`

- [ ] **Step 3: Write the implementation**

`handler_deps.py`:
```python
"""Typed dependency bundle shared by Telegram-facing handler classes,
replacing the module-level globals handlers.py used to expose."""
from dataclasses import dataclass
from typing import Any


@dataclass
class HandlerDependencies:
    config: Any
    db: Any
    agent: Any
    prompt_builder: Any
    bot_username: str | None = None
```

`authorization.py`:
```python
"""Access control checks shared by message and command handlers."""


def is_authorized(user_id: int, config, db) -> bool:
    """Check if user is authorized to use the bot."""
    if str(user_id) == config.AUTHORIZED_USER_ID:
        return True
    return db.is_user_granted(user_id)


def is_main_authorized_user(user_id: int, config) -> bool:
    """Check if user is the main authorized user (for admin commands)."""
    return str(user_id) == config.AUTHORIZED_USER_ID
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_authorization.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add handler_deps.py authorization.py tests/test_authorization.py
git commit -m "refactor: extract HandlerDependencies and authorization module"
```

---

## Task 4: Shared request processor

**Files:**
- Create: `request_processor.py`
- Create: `tests/test_request_processor.py`

**Interfaces:**
- Consumes: `HandlerDependencies` (Task 3).
- Produces: `typing_action(bot, chat_id)` async context manager and `RequestProcessor(deps).process(bot, message, *, user_id, sender_name, sender_username, is_group, build_payload, reply_context, generic_error_text, success_log, error_log_prefix)` — consumed by Task 5.

Note: `build_payload` is an async zero-arg callable returning `(content: str, token_count: int, human_message)`. It runs *inside* the guarded `try`/`typing_action` block so a failure while building the payload (e.g. a photo download error) is caught by the same generic-error path as an agent failure — this preserves `process_image_request`'s current behavior of replying with the generic image error message if the download itself fails, rather than letting that exception escape uncaught.

- [ ] **Step 1: Write the failing test**

```python
"""RequestProcessor: shared audit-log -> agent.run -> audit-log -> reply
workflow used by both text and photo handlers."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from agent import CompletionError
from handler_deps import HandlerDependencies
from request_processor import RequestProcessor


def _deps(db=None, agent=None):
    return HandlerDependencies(
        config=SimpleNamespace(), db=db or SimpleNamespace(add_message=Mock()),
        agent=agent or SimpleNamespace(), prompt_builder=SimpleNamespace(),
    )


def _message(chat_id=123):
    return SimpleNamespace(chat_id=chat_id, message_id=1, reply_text=AsyncMock())


def _bot():
    return SimpleNamespace(send_chat_action=AsyncMock())


async def _payload():
    return "content", 3, "human-message"


def test_process_success_stores_both_turns_and_replies():
    db = SimpleNamespace(add_message=Mock())
    agent = SimpleNamespace(run=AsyncMock(return_value="reply text"))
    processor = RequestProcessor(_deps(db=db, agent=agent))
    message = _message()

    asyncio.run(processor.process(
        _bot(), message, user_id=1, sender_name="Alice", sender_username="alice",
        is_group=False, build_payload=_payload, reply_context=None,
        generic_error_text="generic error", success_log="ok", error_log_prefix="err",
    ))

    assert db.add_message.call_count == 2
    message.reply_text.assert_awaited_once_with("reply text")


def test_process_completion_error_replies_user_message():
    agent = SimpleNamespace(run=AsyncMock(side_effect=CompletionError("rate limited")))
    processor = RequestProcessor(_deps(agent=agent))
    message = _message()

    asyncio.run(processor.process(
        _bot(), message, user_id=1, sender_name="Alice", sender_username="alice",
        is_group=False, build_payload=_payload, reply_context=None,
        generic_error_text="generic error", success_log="ok", error_log_prefix="err",
    ))

    message.reply_text.assert_awaited_once_with("rate limited")


def test_process_payload_build_failure_replies_generic_error():
    async def _failing_payload():
        raise RuntimeError("download failed")

    processor = RequestProcessor(_deps())
    message = _message()

    asyncio.run(processor.process(
        _bot(), message, user_id=1, sender_name="Alice", sender_username="alice",
        is_group=False, build_payload=_failing_payload, reply_context=None,
        generic_error_text="generic error", success_log="ok", error_log_prefix="err",
    ))

    message.reply_text.assert_awaited_once_with("generic error")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_request_processor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'request_processor'`

- [ ] **Step 3: Write the implementation**

```python
"""Shared agent-turn workflow for triggering text and image requests."""
import asyncio
import logging
from contextlib import asynccontextmanager

from telegram.constants import ChatAction

from agent import CompletionError, count_tokens
from handler_deps import HandlerDependencies

logger = logging.getLogger(__name__)


@asynccontextmanager
async def typing_action(bot, chat_id: str):
    """Keep the Telegram typing indicator active for the duration of the block."""
    async def _loop():
        while True:
            try:
                await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            except Exception as e:
                logger.debug(f"Failed to send typing action: {e}")
            await asyncio.sleep(4)
    task = asyncio.create_task(_loop())
    try:
        yield
    finally:
        task.cancel()


class RequestProcessor:
    """Runs the audit-log -> agent.run -> audit-log -> reply workflow shared
    by text and image requests, and maps failures to a user-facing reply."""

    def __init__(self, deps: HandlerDependencies):
        self._deps = deps

    async def process(
        self,
        bot,
        message,
        *,
        user_id: int,
        sender_name: str,
        sender_username: str,
        is_group: bool,
        build_payload,
        reply_context: tuple[str, str] | None,
        generic_error_text: str,
        success_log: str,
        error_log_prefix: str,
    ) -> None:
        chat_id = str(message.chat_id)
        db = self._deps.db
        agent = self._deps.agent
        try:
            async with typing_action(bot, chat_id):
                content, token_count, human_message = await build_payload()
                db.add_message(
                    chat_id=chat_id, role="user", content=content,
                    user_id=user_id, message_id=message.message_id,
                    token_count=token_count,
                    sender_name=sender_name, sender_username=sender_username,
                    is_group_chat=is_group,
                )
                response = await agent.run(
                    chat_id, human_message, is_group, reply_context=reply_context
                )
                db.add_message(
                    chat_id=chat_id, role="assistant", content=response,
                    token_count=count_tokens(response), is_group_chat=is_group,
                )
            await message.reply_text(response)
            logger.info(success_log)
        except CompletionError as e:
            await message.reply_text(e.user_message)
        except Exception as e:
            logger.error(f"{error_log_prefix}: {e}", exc_info=True)
            await message.reply_text(generic_error_text)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_request_processor.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add request_processor.py tests/test_request_processor.py
git commit -m "refactor: extract shared RequestProcessor for text/image agent turns"
```

---

## Task 5: Message handlers class

**Files:**
- Create: `message_handlers.py`
- Create: `tests/test_message_handlers.py`

**Interfaces:**
- Consumes: `HandlerDependencies` (Task 3), `is_authorized` (Task 3), `RequestProcessor` (Task 4).
- Produces: `extract_keyword(text, bot_username=None)`, `extract_reply_data(message)`, `MessageHandlers(deps, processor).message_handler(update, context)` / `.photo_handler(update, context)` — consumed by Task 7.

- [ ] **Step 1: Write the failing test**

```python
"""MessageHandlers: text/photo intake, activation parsing, and auth gate."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from handler_deps import HandlerDependencies
from message_handlers import MessageHandlers, extract_keyword, extract_reply_data
from request_processor import RequestProcessor


@pytest.mark.parametrize(
    "text, bot_username, expected_has_keyword, expected_prompt",
    [
        ("", None, False, ""),
        ("hello world", None, False, "hello world"),
        ("chatgpt what is 2+2", None, True, "what is 2+2"),
        ("@MyBot hello", "MyBot", True, "hello"),
    ],
)
def test_extract_keyword(text, bot_username, expected_has_keyword, expected_prompt):
    has_keyword, prompt = extract_keyword(text, bot_username)
    assert has_keyword is expected_has_keyword
    assert prompt == expected_prompt


def test_extract_reply_data_returns_none_without_reply():
    message = SimpleNamespace(reply_to_message=None)
    assert extract_reply_data(message) is None


class _Cfg:
    AUTHORIZED_USER_ID = "1"


def _handlers(db=None, agent=None, prompt_builder=None, username="mybot"):
    deps = HandlerDependencies(
        config=_Cfg, db=db or SimpleNamespace(), agent=agent or SimpleNamespace(),
        prompt_builder=prompt_builder or SimpleNamespace(), bot_username=username,
    )
    return MessageHandlers(deps, RequestProcessor(deps))


def _message(text=None, chat_id=123, chat_type="private", user_id=7):
    return SimpleNamespace(
        text=text, photo=None, caption=None, chat_id=chat_id,
        chat=SimpleNamespace(type=chat_type),
        from_user=SimpleNamespace(id=user_id, first_name="Alice", username="alice"),
        message_id=1, reply_to_message=None, reply_text=AsyncMock(),
    )


def test_non_triggering_message_stores_context_without_reply():
    db = SimpleNamespace(add_message=Mock())
    agent = SimpleNamespace(append_context_message=Mock(), run=AsyncMock())
    prompt_builder = SimpleNamespace(to_lc_human_message=Mock(return_value="human"))
    handlers_obj = _handlers(db=db, agent=agent, prompt_builder=prompt_builder)

    message = _message(text="ordinary message", chat_type="group")
    update = SimpleNamespace(message=message)
    context = SimpleNamespace()

    asyncio.run(handlers_obj.message_handler(update, context))

    db.add_message.assert_called_once()
    agent.append_context_message.assert_called_once_with("123", "human")
    agent.run.assert_not_awaited()


def test_unauthorized_triggering_message_replies_no_access():
    handlers_obj = _handlers()
    message = _message(text="chatgpt hi", user_id=99)
    update = SimpleNamespace(message=message)
    context = SimpleNamespace()

    asyncio.run(handlers_obj.message_handler(update, context))

    message.reply_text.assert_awaited_once_with("Sorry, you have no access to me.")


def test_authorized_empty_prompt_asks_for_request():
    handlers_obj = _handlers()
    message = _message(text="chatgpt", user_id=1)
    update = SimpleNamespace(message=message)
    context = SimpleNamespace()

    asyncio.run(handlers_obj.message_handler(update, context))

    message.reply_text.assert_awaited_once_with("Yes, what's your request?")


def test_authorized_triggering_message_processes_and_replies():
    db = SimpleNamespace(add_message=Mock())
    agent = SimpleNamespace(run=AsyncMock(return_value="hi there"))
    prompt_builder = SimpleNamespace(to_lc_human_message=Mock(return_value="human"))
    handlers_obj = _handlers(db=db, agent=agent, prompt_builder=prompt_builder)

    message = _message(text="chatgpt hello", user_id=1)
    update = SimpleNamespace(message=message)
    context = SimpleNamespace(bot=SimpleNamespace(send_chat_action=AsyncMock()))

    asyncio.run(handlers_obj.message_handler(update, context))

    message.reply_text.assert_awaited_once_with("hi there")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_message_handlers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'message_handlers'`

- [ ] **Step 3: Write the implementation**

```python
"""Telegram-facing text/photo intake: activation parsing, auth gate, and
handing off to the shared request processor."""
import base64
import logging
import re

from agent import count_tokens
from authorization import is_authorized
from handler_deps import HandlerDependencies
from request_processor import RequestProcessor

logger = logging.getLogger(__name__)


def extract_keyword(text: str, bot_username: str = None) -> tuple[bool, str]:
    """
    Check for activation keyword or @mention and extract prompt.

    Args:
        text: Message text
        bot_username: Bot's username (without @) for mention detection

    Returns:
        Tuple of (has_keyword, prompt_without_keyword)
    """
    if not text:
        return False, ""

    text_lower = text.lower()
    has_activation = False
    cleaned = text

    if "chatgpt" in text_lower:
        has_activation = True
        cleaned = re.sub(r'\bchatgpt\b', '', cleaned, flags=re.IGNORECASE)

    if bot_username:
        mention = f"@{bot_username}"
        if mention.lower() in text_lower:
            has_activation = True
            cleaned = re.sub(rf'@{re.escape(bot_username)}', '', cleaned, flags=re.IGNORECASE)

    prompt = cleaned.strip()
    return has_activation, prompt


def extract_reply_data(message) -> tuple[str, str] | None:
    """
    Extracts raw data from the message being replied to.

    Args:
        message: Telegram message object

    Returns:
        Tuple of (sender_name, content) or None if no valid reply
    """
    if not message.reply_to_message:
        return None

    reply = message.reply_to_message
    content = reply.text or reply.caption or ""
    if not content:
        return None

    sender = reply.from_user.first_name if reply.from_user else "Unknown"
    return (sender, content)


class MessageHandlers:
    """Text and photo Telegram handlers, bound to an explicit dependency set."""

    def __init__(self, deps: HandlerDependencies, processor: RequestProcessor):
        self._deps = deps
        self._processor = processor

    async def message_handler(self, update, context):
        message = update.message
        if not message or not message.text:
            return

        user_id = message.from_user.id
        chat_id = str(message.chat_id)
        is_group = message.chat.type in ["group", "supergroup"]
        sender_name = message.from_user.first_name or "Unknown"
        sender_username = message.from_user.username

        has_keyword, prompt = extract_keyword(message.text, self._deps.bot_username)

        if not has_keyword:
            try:
                self._deps.db.add_message(
                    chat_id=chat_id, role="user", content=message.text,
                    user_id=user_id, message_id=message.message_id,
                    token_count=count_tokens(message.text),
                    sender_name=sender_name, sender_username=sender_username,
                    is_group_chat=is_group,
                )
                self._deps.agent.append_context_message(
                    chat_id,
                    self._deps.prompt_builder.to_lc_human_message(
                        text=message.text, is_group=is_group, sender_name=sender_name),
                )
            except Exception as e:
                logger.error(f"Failed to store context message: {e}")
            return

        if not is_authorized(user_id, self._deps.config, self._deps.db):
            await message.reply_text("Sorry, you have no access to me.")
            return

        reply_data = extract_reply_data(message)

        if not prompt:
            await message.reply_text("Yes, what's your request?")
            return

        async def _build_payload():
            human = self._deps.prompt_builder.to_lc_human_message(
                text=prompt, is_group=is_group, sender_name=sender_name)
            return message.text, count_tokens(prompt), human

        await self._processor.process(
            context.bot, message,
            user_id=user_id, sender_name=sender_name, sender_username=sender_username,
            is_group=is_group, build_payload=_build_payload, reply_context=reply_data,
            generic_error_text=(
                "Sorry, I encountered an error processing your request. Please try again."
            ),
            success_log=f"Response sent for chat {chat_id}",
            error_log_prefix="Error processing request",
        )

    async def photo_handler(self, update, context):
        message = update.message
        if not message or not message.photo:
            return

        user_id = message.from_user.id
        chat_id = str(message.chat_id)
        is_group = message.chat.type in ["group", "supergroup"]
        sender_name = message.from_user.first_name or "Unknown"
        sender_username = message.from_user.username

        caption = message.caption or ""
        has_keyword, prompt = (
            extract_keyword(caption, self._deps.bot_username) if caption else (False, "")
        )

        if is_group and not has_keyword:
            return
        if not has_keyword:
            return

        if not is_authorized(user_id, self._deps.config, self._deps.db):
            await message.reply_text("Sorry, you have no access to me.")
            return

        reply_data = extract_reply_data(message)

        async def _build_payload():
            photo = message.photo[-1]
            photo_file = await photo.get_file()
            photo_bytes = await photo_file.download_as_bytearray()
            base64_image = base64.b64encode(photo_bytes).decode("utf-8")
            image_data_url = f"data:image/jpeg;base64,{base64_image}"
            caption_marker = f"[image] {message.caption}" if message.caption else "[image]"
            human = self._deps.prompt_builder.to_lc_human_message(
                text=prompt, is_group=is_group, sender_name=sender_name,
                image_data_url=image_data_url)
            return caption_marker, count_tokens(caption_marker), human

        await self._processor.process(
            context.bot, message,
            user_id=user_id, sender_name=sender_name, sender_username=sender_username,
            is_group=is_group, build_payload=_build_payload, reply_context=reply_data,
            generic_error_text=(
                "Sorry, I encountered an error processing your image. Please try again."
            ),
            success_log=f"Image processed for chat {chat_id}",
            error_log_prefix="Error processing image",
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_message_handlers.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add message_handlers.py tests/test_message_handlers.py
git commit -m "refactor: extract MessageHandlers with explicit dependencies"
```

---

## Task 6: Command handlers class

**Files:**
- Create: `command_handlers.py`
- Create: `tests/test_command_handlers.py`

**Interfaces:**
- Consumes: `HandlerDependencies` (Task 3), `is_main_authorized_user` (Task 3), `agent.MODEL_PROVIDERS`.
- Produces: `CommandHandlers(deps)` with methods `clear_command`, `stats_command`, `grant_command`, `revoke_command`, `version_command`, `allowlist_command`, `personality_command`, `list_personality_command`, `model_command`, `help_command`; module-level `error_handler(update, context)` — consumed by Task 7.

- [ ] **Step 1: Write the failing test**

```python
"""CommandHandlers: admin-only command surface, bound to explicit deps."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from command_handlers import CommandHandlers, error_handler
from handler_deps import HandlerDependencies


class _Cfg:
    AUTHORIZED_USER_ID = "1"
    BOT_VERSION = "9.9.9"


def _handlers(db=None, agent=None):
    deps = HandlerDependencies(
        config=_Cfg, db=db or SimpleNamespace(), agent=agent or SimpleNamespace(),
        prompt_builder=SimpleNamespace(),
    )
    return CommandHandlers(deps)


def _update(user_id=1, args=None):
    message = SimpleNamespace(
        from_user=SimpleNamespace(id=user_id), chat_id=123, reply_text=AsyncMock(),
    )
    update = SimpleNamespace(message=message)
    context = SimpleNamespace(args=args or [], bot=SimpleNamespace())
    return update, context, message


def test_clear_command_requires_main_user():
    handlers_obj = _handlers()
    update, context, message = _update(user_id=2)
    asyncio.run(handlers_obj.clear_command(update, context))
    message.reply_text.assert_awaited_once_with(
        "Sorry, only the main authorized user can clear history."
    )


def test_clear_command_success():
    agent = SimpleNamespace(clear_thread=Mock())
    handlers_obj = _handlers(agent=agent)
    update, context, message = _update(user_id=1)
    asyncio.run(handlers_obj.clear_command(update, context))
    agent.clear_thread.assert_called_once_with("123")
    message.reply_text.assert_awaited_once_with(
        "✅ Conversation history cleared for this chat."
    )


def test_model_command_switch_success():
    db = SimpleNamespace(set_active_model=Mock())
    agent = SimpleNamespace(set_model=Mock())
    handlers_obj = _handlers(db=db, agent=agent)
    update, context, message = _update(user_id=1, args=["gpt-5.4"])
    asyncio.run(handlers_obj.model_command(update, context))
    db.set_active_model.assert_called_once_with("gpt-5.4")
    agent.set_model.assert_called_once_with("gpt-5.4")
    message.reply_text.assert_awaited_once_with(
        "✅ Model switched to `gpt-5.4`", parse_mode="Markdown"
    )


def test_error_handler_notifies_user_when_possible():
    message = SimpleNamespace(reply_text=AsyncMock())
    update = SimpleNamespace(message=message)
    context = SimpleNamespace(error=RuntimeError("boom"))
    asyncio.run(error_handler(update, context))
    message.reply_text.assert_awaited_once_with(
        "An error occurred while processing your request. "
        "The error has been logged."
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_command_handlers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'command_handlers'`

- [ ] **Step 3: Write the implementation**

```python
"""Admin-only Telegram commands, bound to an explicit dependency set."""
import logging

from telegram.helpers import escape_markdown

from agent import MODEL_PROVIDERS
from authorization import is_main_authorized_user
from handler_deps import HandlerDependencies

logger = logging.getLogger(__name__)


class CommandHandlers:
    """All /commands gated by is_main_authorized_user, bound to deps."""

    def __init__(self, deps: HandlerDependencies):
        self._deps = deps

    async def clear_command(self, update, context):
        user_id = update.message.from_user.id
        if not is_main_authorized_user(user_id, self._deps.config):
            await update.message.reply_text("Sorry, only the main authorized user can clear history.")
            return

        chat_id = str(update.message.chat_id)
        try:
            self._deps.agent.clear_thread(chat_id)
            await update.message.reply_text("✅ Conversation history cleared for this chat.")
            logger.info(f"History cleared for chat {chat_id}")
        except Exception as e:
            logger.error(f"Error clearing history: {e}", exc_info=True)
            await update.message.reply_text("❌ Failed to clear history. Please try again.")

    async def stats_command(self, update, context):
        user_id = update.message.from_user.id
        if not is_main_authorized_user(user_id, self._deps.config):
            await update.message.reply_text("Sorry, only the main authorized user can view stats.")
            return

        chat_id = str(update.message.chat_id)
        try:
            stats = self._deps.db.get_stats(chat_id)
            first_msg = stats["first_message"]
            if first_msg != "N/A":
                first_msg = first_msg.split("T")[0]

            await update.message.reply_text(
                f"📊 Chat Statistics:\n"
                f"Messages: {stats['total_messages']}\n"
                f"Total tokens: {stats['total_tokens']:,}\n"
                f"Since: {first_msg}"
            )
            logger.info(f"Stats shown for chat {chat_id}")
        except Exception as e:
            logger.error(f"Error getting stats: {e}", exc_info=True)
            await update.message.reply_text("❌ Failed to retrieve statistics. Please try again.")

    async def grant_command(self, update, context):
        user_id = update.message.from_user.id
        if not is_main_authorized_user(user_id, self._deps.config):
            await update.message.reply_text("Sorry, only the main authorized user can grant access.")
            return

        if not context.args or len(context.args) == 0:
            await update.message.reply_text(
                "❌ Usage: /grant <user_id>\n"
                "Example: /grant 123456789"
            )
            return

        try:
            target_user_id = int(context.args[0])
            if target_user_id <= 0:
                await update.message.reply_text(
                    "❌ Invalid user ID. User IDs must be positive integers."
                )
                return

            if str(target_user_id) == self._deps.config.AUTHORIZED_USER_ID:
                await update.message.reply_text("ℹ️ You are already the main authorized user.")
                return

            first_name = None
            username = None
            try:
                chat = await context.bot.get_chat(target_user_id)
                first_name = chat.first_name
                username = chat.username
            except Exception as e:
                logger.warning(f"Could not fetch user info for {target_user_id}: {e}")

            was_granted = self._deps.db.grant_access(
                target_user_id, first_name=first_name, username=username
            )

            if was_granted:
                name_display = first_name or str(target_user_id)
                if username:
                    name_display += f" (@{username})"
                await update.message.reply_text(
                    f"✅ Access granted to {name_display}.\n"
                    f"They can now use the bot with 'chatgpt' keyword."
                )
                logger.info(f"User {user_id} granted access to {target_user_id}")
            else:
                await update.message.reply_text(f"ℹ️ User {target_user_id} already has access.")

        except ValueError:
            await update.message.reply_text(
                "❌ Invalid user ID. Please provide a numeric user ID.\n"
                "Example: /grant 123456789"
            )
        except Exception as e:
            logger.error(f"Error granting access: {e}", exc_info=True)
            await update.message.reply_text("❌ Failed to grant access. Please try again.")

    async def revoke_command(self, update, context):
        user_id = update.message.from_user.id
        if not is_main_authorized_user(user_id, self._deps.config):
            await update.message.reply_text("Sorry, only the main authorized user can revoke access.")
            return

        if not context.args or len(context.args) == 0:
            await update.message.reply_text(
                "❌ Usage: /revoke <user_id>\n"
                "Example: /revoke 123456789"
            )
            return

        try:
            target_user_id = int(context.args[0])
            if target_user_id <= 0:
                await update.message.reply_text(
                    "❌ Invalid user ID. User IDs must be positive integers."
                )
                return
            if str(target_user_id) == self._deps.config.AUTHORIZED_USER_ID:
                await update.message.reply_text(
                    "❌ Cannot revoke access from the main authorized user."
                )
                return

            was_revoked = self._deps.db.revoke_access(target_user_id)

            if was_revoked:
                await update.message.reply_text(f"✅ Access revoked from user {target_user_id}.")
                logger.info(f"User {user_id} revoked access from {target_user_id}")
            else:
                await update.message.reply_text(f"ℹ️ User {target_user_id} didn't have access.")

        except ValueError:
            await update.message.reply_text(
                "❌ Invalid user ID. Please provide a numeric user ID.\n"
                "Example: /revoke 123456789"
            )
        except Exception as e:
            logger.error(f"Error revoking access: {e}", exc_info=True)
            await update.message.reply_text("❌ Failed to revoke access. Please try again.")

    async def version_command(self, update, context):
        user_id = update.message.from_user.id
        if not is_main_authorized_user(user_id, self._deps.config):
            await update.message.reply_text("Sorry, only the main authorized user can view the bot version.")
            return

        await update.message.reply_text(f"Bot version: {self._deps.config.BOT_VERSION}")
        logger.info(f"Version shown for chat {update.message.chat_id}")

    async def allowlist_command(self, update, context):
        user_id = update.message.from_user.id
        if not is_main_authorized_user(user_id, self._deps.config):
            await update.message.reply_text("Sorry, only the main authorized user can see the allowlist.")
            return

        try:
            granted_users = self._deps.db.get_granted_users()

            message = "📋 **Bot Allowlist**\n\n"
            message += f"👑 **Main Admin:**\n- `{self._deps.config.AUTHORIZED_USER_ID}`\n\n"

            if granted_users:
                message += "👥 **Granted Users:**\n"
                for target_user_id, granted_at, first_name, username in granted_users:
                    parts = [f"`{target_user_id}`"]
                    name_parts = []
                    if first_name:
                        name_parts.append(escape_markdown(first_name))
                    if username:
                        name_parts.append(f"@{escape_markdown(username)}")
                    if name_parts:
                        parts.append(f"({' / '.join(name_parts)})")
                    parts.append(f"(granted: {granted_at.split('T')[0]})")
                    message += f"- {' '.join(parts)}\n"
            else:
                message += "👥 No other users have been granted access."

            await update.message.reply_text(message, parse_mode="Markdown")
            logger.info(f"Allowlist shown to admin user {user_id}")

        except Exception as e:
            logger.error(f"Error showing allowlist: {e}", exc_info=True)
            await update.message.reply_text("❌ Failed to retrieve allowlist. Please try again.")

    async def personality_command(self, update, context):
        user_id = update.message.from_user.id
        if not is_main_authorized_user(user_id, self._deps.config):
            await update.message.reply_text("Sorry, only the main authorized user can change personality.")
            return

        if not context.args or len(context.args) == 0:
            try:
                active_personality = self._deps.db.get_active_personality()
                await update.message.reply_text(
                    f"Current personality: **{active_personality}**\n\n"
                    f"Usage: /personality <name>\n"
                    f"Example: /personality villain",
                    parse_mode="Markdown"
                )
            except Exception as e:
                logger.error(f"Error getting active personality: {e}", exc_info=True)
                await update.message.reply_text(
                    "❌ Failed to retrieve active personality. Please try again."
                )
            return

        personality_name = context.args[0].strip()

        try:
            if not self._deps.db.personality_exists(personality_name):
                await update.message.reply_text(f"❌ No personality '{personality_name}' found.")
                return

            self._deps.db.set_active_personality(personality_name)
            await update.message.reply_text(f"✅ Personality changed to '{personality_name}'")
            logger.info(f"User {user_id} changed personality to {personality_name}")

        except Exception as e:
            logger.error(f"Error changing personality: {e}", exc_info=True)
            await update.message.reply_text("❌ Failed to change personality. Please try again.")

    async def list_personality_command(self, update, context):
        user_id = update.message.from_user.id
        if not is_main_authorized_user(user_id, self._deps.config):
            await update.message.reply_text("Sorry, only the main authorized user can view personalities.")
            return

        try:
            personalities = self._deps.db.list_personalities()
            active = self._deps.db.get_active_personality()

            if not personalities:
                await update.message.reply_text("No personalities found in database.")
                return

            message = f"**Available Personalities:**\n"
            message += f"Currently active: **{active}**\n\n"

            for name, prompt_preview in personalities:
                marker = "✓" if name == active else "-"
                message += f"{marker} `{name}`\n"
                message += f"  _{prompt_preview}_\n\n"

            await update.message.reply_text(message, parse_mode="Markdown")
            logger.info(f"Listed personalities for user {user_id}")

        except Exception as e:
            logger.error(f"Error listing personalities: {e}", exc_info=True)
            await update.message.reply_text("❌ Failed to list personalities. Please try again.")

    async def model_command(self, update, context):
        user_id = update.message.from_user.id
        if not is_main_authorized_user(user_id, self._deps.config):
            await update.message.reply_text("Sorry, only the main authorized user can change the model.")
            return

        available = "\n".join(f"  `{m}`" for m in MODEL_PROVIDERS)

        if not context.args:
            current = self._deps.db.get_active_model()
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
            self._deps.db.set_active_model(new_model)
            self._deps.agent.set_model(new_model)
            await update.message.reply_text(f"✅ Model switched to `{new_model}`", parse_mode="Markdown")
            logger.info(f"User {user_id} switched model to {new_model}")
        except Exception as e:
            logger.error(f"Error switching model: {e}", exc_info=True)
            await update.message.reply_text("❌ Failed to switch model. Please try again.")

    async def help_command(self, update, context):
        user_id = update.message.from_user.id
        if not is_main_authorized_user(user_id, self._deps.config):
            await update.message.reply_text("Sorry, only the main authorized user can use this command.")
            return

        help_text = (
            "📖 **Available Commands:**\n\n"
            "/help - Show this help message\n"
            "/clear - Clear conversation history for current chat\n"
            "/stats - Show message count, token usage, and first message date\n"
            "/grant <user\\_id> - Grant bot access to a user\n"
            "/revoke <user\\_id> - Revoke bot access from a user\n"
            "/allowlist - Show all authorized users\n"
            "/model - View or change the active AI model\n"
            "/personality <name> - View or change active personality\n"
            "/list\\_personality - List all available personalities\n"
            "/version - Show current bot version"
        )

        await update.message.reply_text(help_text, parse_mode="Markdown")
        logger.info(f"Help shown to admin user {user_id}")


async def error_handler(update, context):
    """Global error handler for unhandled exceptions."""
    logger.error("Exception while handling update:", exc_info=context.error)

    if update and update.message:
        try:
            await update.message.reply_text(
                "An error occurred while processing your request. "
                "The error has been logged."
            )
        except Exception:
            pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_command_handlers.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add command_handlers.py tests/test_command_handlers.py
git commit -m "refactor: extract CommandHandlers with explicit dependencies"
```

---

## Task 7: Rewrite handlers.py as a facade and remove module globals

**Files:**
- Modify: `handlers.py` (full rewrite)
- Modify: `tests/test_context_message_retention.py`

**Interfaces:**
- Consumes: `HandlerDependencies` (Task 3), `RequestProcessor` (Task 4), `MessageHandlers` (Task 5), `CommandHandlers` + `error_handler` (Task 6).
- Produces: `handlers.init_handlers(cfg, database, bot_agent, prompt_bldr, username=None)` and the same ten command callables / `message_handler` / `photo_handler` / `error_handler` `bot.py` already imports — unchanged public surface, so **no change to `bot.py` is required in this task**.

- [ ] **Step 1: Rewrite handlers.py**

```python
"""Backward-compatible facade: wires the DI-based handler classes to the
module-level callables bot.py registers with python-telegram-bot. See
message_handlers.py and command_handlers.py for the actual logic."""
import logging

from authorization import is_authorized, is_main_authorized_user
from command_handlers import CommandHandlers, error_handler
from handler_deps import HandlerDependencies
from message_handlers import MessageHandlers, extract_keyword, extract_reply_data
from request_processor import RequestProcessor

logger = logging.getLogger(__name__)

_message_handlers: MessageHandlers | None = None
_command_handlers: CommandHandlers | None = None


def init_handlers(cfg, database, bot_agent, prompt_bldr, username=None):
    """Initialize handler dependencies."""
    global _message_handlers, _command_handlers
    deps = HandlerDependencies(
        config=cfg, db=database, agent=bot_agent,
        prompt_builder=prompt_bldr, bot_username=username,
    )
    processor = RequestProcessor(deps)
    _message_handlers = MessageHandlers(deps, processor)
    _command_handlers = CommandHandlers(deps)


async def message_handler(update, context):
    return await _message_handlers.message_handler(update, context)


async def photo_handler(update, context):
    return await _message_handlers.photo_handler(update, context)


async def clear_command(update, context):
    return await _command_handlers.clear_command(update, context)


async def stats_command(update, context):
    return await _command_handlers.stats_command(update, context)


async def grant_command(update, context):
    return await _command_handlers.grant_command(update, context)


async def revoke_command(update, context):
    return await _command_handlers.revoke_command(update, context)


async def version_command(update, context):
    return await _command_handlers.version_command(update, context)


async def allowlist_command(update, context):
    return await _command_handlers.allowlist_command(update, context)


async def personality_command(update, context):
    return await _command_handlers.personality_command(update, context)


async def list_personality_command(update, context):
    return await _command_handlers.list_personality_command(update, context)


async def model_command(update, context):
    return await _command_handlers.model_command(update, context)


async def help_command(update, context):
    return await _command_handlers.help_command(update, context)
```

Note `error_handler`, `extract_keyword`, `extract_reply_data`, `is_authorized`, `is_main_authorized_user` are re-exported automatically because they're imported into this module's namespace — `handlers.error_handler`, `handlers.extract_keyword`, etc. all keep working, so `tests/test_extract_keyword.py` (`from handlers import extract_keyword`) needs no change, and `bot.py`'s `application.add_error_handler(handlers.error_handler)` needs no change.

- [ ] **Step 2: Update the one test that monkeypatches the old globals**

`handlers.config` / `handlers.db` / `handlers.agent` / `handlers.prompt_builder` / `handlers.bot_username` no longer exist as separate module globals (they're inside the `HandlerDependencies` instances held by `_message_handlers`/`_command_handlers`). Rewrite `tests/test_context_message_retention.py` to go through `handlers.init_handlers(...)` instead of monkeypatching five individual attributes:

```python
"""Non-triggering text context retention for group and private chats."""
import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import handlers


def _run_message_handler(message):
    database = SimpleNamespace(
        add_message=Mock(),
        cleanup_old_group_messages=Mock(),
    )
    bot_agent = SimpleNamespace(
        append_context_message=Mock(),
        run=Mock(),
    )
    prompt_builder = SimpleNamespace(to_lc_human_message=Mock(return_value="human"))
    config = SimpleNamespace(MAX_GROUP_CONTEXT_MESSAGES=500, AUTHORIZED_USER_ID="1")

    handlers.init_handlers(config, database, bot_agent, prompt_builder, "mybot")

    asyncio.run(
        handlers.message_handler(
            SimpleNamespace(message=message),
            SimpleNamespace(),
        )
    )
    return database, bot_agent, prompt_builder


def test_non_triggering_group_message_stores_context_without_cleanup():
    message = SimpleNamespace(
        text="ordinary group message",
        chat_id=-123,
        chat=SimpleNamespace(type="group"),
        from_user=SimpleNamespace(id=42, first_name="Alice", username="alice"),
        message_id=7,
        reply_to_message=None,
    )

    database, bot_agent, prompt_builder = _run_message_handler(message)

    database.add_message.assert_called_once()
    assert database.add_message.call_args.kwargs["is_group_chat"] is True
    prompt_builder.to_lc_human_message.assert_called_once_with(
        text="ordinary group message", is_group=True, sender_name="Alice",
    )
    bot_agent.append_context_message.assert_called_once_with("-123", "human")
    bot_agent.run.assert_not_called()
    database.cleanup_old_group_messages.assert_not_called()


def test_non_triggering_private_message_stores_context():
    message = SimpleNamespace(
        text="flight is at 6",
        chat_id=99,
        chat=SimpleNamespace(type="private"),
        from_user=SimpleNamespace(id=42, first_name="Alice", username="alice"),
        message_id=8,
        reply_to_message=None,
    )

    database, bot_agent, prompt_builder = _run_message_handler(message)

    database.add_message.assert_called_once()
    assert database.add_message.call_args.kwargs["is_group_chat"] is False
    prompt_builder.to_lc_human_message.assert_called_once_with(
        text="flight is at 6", is_group=False, sender_name="Alice",
    )
    bot_agent.append_context_message.assert_called_once_with("99", "human")
    bot_agent.run.assert_not_called()
```

(`reply_to_message=None` is added to both `message` `SimpleNamespace`s because `extract_reply_data` — now reached inside `message_handlers.py` for triggering messages — accesses `message.reply_to_message`; it wasn't needed before since these two tests only exercise the non-triggering path, but adding it keeps the fixture reusable and matches the real `telegram.Message` shape.)

- [ ] **Step 3: Run the full suite**

Run: `pytest tests/ -v`
Expected: all tests pass, including `tests/test_handlers_characterization.py` from Task 2 (unchanged — it only calls the public `handlers.*` entry points, which kept their signatures) and `tests/test_extract_keyword.py` (unchanged).

- [ ] **Step 4: Compile check**

Run: `python3 -m py_compile *.py`
Expected: no output, exit code 0.

- [ ] **Step 5: Commit**

```bash
git add handlers.py tests/test_context_message_retention.py
git commit -m "refactor: turn handlers.py into a facade over DI-based handler classes"
```

---

## Task 8: Extract model_registry.py

**Files:**
- Create: `model_registry.py`
- Create: `tests/test_model_registry.py`
- Modify: `agent.py`
- Modify: `command_handlers.py`
- Modify: `scripts/chat_cli.py`

**Interfaces:**
- Produces: `model_registry.MODEL_PROVIDERS`, `model_registry.PROVIDER_LABEL`, `model_registry.resolve_model(name)`, `model_registry.provider_api_key(provider, config)`.
- `agent.py` re-imports and re-exports these same names, so `agent.MODEL_PROVIDERS`, `agent.resolve_model`, etc. keep working unchanged — `tests/test_model_resolution.py` needs **no changes**.

- [ ] **Step 1: Write the failing test**

```python
"""Direct coverage for model_registry.py, independent of agent.py's re-export."""
import pytest

import model_registry


def test_resolve_model_known():
    assert model_registry.resolve_model("gpt-5.4") == ("openai", "openai:gpt-5.4")


def test_resolve_model_unknown_raises():
    with pytest.raises(KeyError):
        model_registry.resolve_model("does-not-exist")


class _Cfg:
    OPENAI_API_KEY = "o"
    XAI_API_KEY = "x"
    GEMINI_API_KEY = "g"


def test_provider_api_key_selection():
    assert model_registry.provider_api_key("openai", _Cfg) == "o"
    assert model_registry.provider_api_key("xai", _Cfg) == "x"
    assert model_registry.provider_api_key("google_genai", _Cfg) == "g"


def test_every_registered_model_has_a_label():
    for provider in model_registry.MODEL_PROVIDERS.values():
        assert provider in model_registry.PROVIDER_LABEL
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_model_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model_registry'`

- [ ] **Step 3: Create model_registry.py**

```python
"""Model -> provider mapping, independent of agent.py's LangChain wiring."""

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
```

- [ ] **Step 4: Update agent.py to re-export from model_registry**

In `agent.py`, delete the block currently at lines 100–130 (`MODEL_PROVIDERS`, `PROVIDER_LABEL`, `resolve_model`, `provider_api_key`) and replace it with an import:

```python
from model_registry import MODEL_PROVIDERS, PROVIDER_LABEL, resolve_model, provider_api_key
```

placed with `agent.py`'s other local imports near the top of the file (after `from tools import build_tools`).

- [ ] **Step 5: Reduce transport-to-agent coupling in command_handlers.py and scripts/chat_cli.py**

In `command_handlers.py`, change:
```python
from agent import MODEL_PROVIDERS
```
to:
```python
from model_registry import MODEL_PROVIDERS
```

In `scripts/chat_cli.py`, change:
```python
from agent import Agent, MODEL_PROVIDERS, CompletionError
```
to:
```python
from agent import Agent, CompletionError
from model_registry import MODEL_PROVIDERS
```

- [ ] **Step 6: Run the full suite**

Run: `pytest tests/ -v`
Expected: all tests pass, including `tests/test_model_resolution.py` and `tests/test_model_registry.py` unchanged.

- [ ] **Step 7: Commit**

```bash
git add model_registry.py tests/test_model_registry.py agent.py command_handlers.py scripts/chat_cli.py
git commit -m "refactor: extract model_registry.py from agent.py"
```

---

## Task 9: Extract token_budget.py

**Files:**
- Create: `token_budget.py`
- Create: `tests/test_token_budget.py`
- Modify: `agent.py`

**Interfaces:**
- Produces: `token_budget.count_tokens`, `token_budget.count_message_tokens`, `token_budget.count_messages_tokens`, `token_budget.trim_messages`, `token_budget.make_trim_middleware`, and the private `token_budget._message_text` used internally by `agent.py`.
- `agent.py` re-imports these; `agent.count_tokens`, `agent.trim_messages`, `agent.count_message_tokens` keep working unchanged, so `tests/test_trimming.py` needs **no changes**. `message_handlers.py`, `request_processor.py`, and `command_handlers.py` continue importing `count_tokens`/`CompletionError` from `agent` — no changes needed there either, since `agent.py` still exposes `count_tokens`.

- [ ] **Step 1: Write the failing test**

```python
"""Direct coverage for token_budget.py, independent of agent.py's re-export."""
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import token_budget


def test_count_tokens_nonzero_for_text():
    assert token_budget.count_tokens("hello world") > 0
    assert token_budget.count_tokens("") == 0


def test_trim_messages_keeps_last_message_even_over_budget():
    messages = [HumanMessage(content="a" * 50)]
    kept = token_budget.trim_messages(messages, max_context_tokens=1, reserve=0)
    assert kept == messages


def test_trim_messages_drops_orphaned_leading_tool_message():
    messages = [
        AIMessage(content="", tool_calls=[{"name": "f", "args": {}, "id": "1"}]),
        ToolMessage(content="result", tool_call_id="1"),
        HumanMessage(content="final"),
    ]
    kept = token_budget.trim_messages(messages, max_context_tokens=100000, reserve=0)
    assert not isinstance(kept[0], ToolMessage)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_token_budget.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'token_budget'`

- [ ] **Step 3: Create token_budget.py**

```python
"""Token counting and context-window trimming, independent of provider wiring."""
from __future__ import annotations

from collections.abc import Iterable

import tiktoken
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain_core.messages import BaseMessage, ToolMessage

# tiktoken encoding is model-independent for our budgeting purposes.
_ENCODING = tiktoken.get_encoding("cl100k_base")


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


def make_trim_middleware(max_context_tokens: int, reserve: int):
    """Build a wrap_model_call middleware that trims request.messages non-destructively."""

    @wrap_model_call
    def trim(request: ModelRequest, handler) -> ModelResponse:
        trimmed = trim_messages(list(request.messages), max_context_tokens, reserve)
        return handler(request.override(messages=trimmed))

    return trim
```

- [ ] **Step 4: Update agent.py to re-export from token_budget**

In `agent.py`:
- Delete the `import tiktoken` line and the `_ENCODING = tiktoken.get_encoding("cl100k_base")` line.
- Delete the `count_tokens`, `_message_text`, `count_message_tokens`, `count_messages_tokens`, `trim_messages`, and `make_trim_middleware` function definitions (currently lines 163–243).
- Remove the now-unused `from collections.abc import Iterable` import if nothing else in `agent.py` uses `Iterable` (confirm with `grep -n Iterable agent.py` after the deletions — it is only used by `count_messages_tokens`, which moved).
- Remove `ToolMessage` from the `from langchain_core.messages import BaseMessage, ToolMessage` import if nothing else in `agent.py` uses it (confirm with `grep -n ToolMessage agent.py` — it is only used by `trim_messages`, which moved; keep `BaseMessage` since `AgentContext`/`_message_text` usage elsewhere still needs it... actually `_message_text` moved too, so check whether `BaseMessage` is still referenced anywhere in `agent.py`, e.g. in type hints for `run()`'s `result["messages"][-1]` — if not, drop the whole import line).
- Add, alongside the other local imports:

```python
from token_budget import (
    _message_text,
    count_tokens,
    count_message_tokens,
    count_messages_tokens,
    trim_messages,
    make_trim_middleware,
)
```

`_message_text` is imported explicitly (despite the leading underscore) because `Agent.run()` still calls it directly to flatten the final response message.

- [ ] **Step 5: Run the full suite**

Run: `pytest tests/ -v`
Expected: all tests pass, including `tests/test_trimming.py` and `tests/test_agent.py` unchanged, plus the new `tests/test_token_budget.py`.

- [ ] **Step 6: Compile check**

Run: `python3 -m py_compile *.py`
Expected: no output, exit code 0 (confirms no leftover unused-but-broken imports in `agent.py`).

- [ ] **Step 7: Commit**

```bash
git add token_budget.py tests/test_token_budget.py agent.py
git commit -m "refactor: extract token_budget.py from agent.py"
```

---

## Task 10: ConnectionManager

**Files:**
- Create: `db_connection.py`
- Create: `tests/test_db_connection.py`

**Interfaces:**
- Produces: `ConnectionManager(db_url).connection()` (contextmanager yielding a psycopg2 connection, committing/rolling back/returning it to the pool) and `.close()` — consumed by Tasks 11–15.

Note: `psycopg2.pool.ThreadedConnectionPool` opens `minconn` real connections eagerly at construction time, so `ConnectionManager.__init__` itself is not unit-testable without a live database (this matches today's `Database.__init__`, which nothing in the test suite constructs directly either). Tests build a `ConnectionManager` via `__new__` and inject a fake `pool` object, exercising only `.connection()` and `.close()`.

- [ ] **Step 1: Write the failing test**

```python
"""ConnectionManager tests use a fake pool double; psycopg2 pools connect
eagerly on construction, so real pool creation is out of scope for unit
tests (CLAUDE.md: no live database in tests)."""
from unittest.mock import Mock

import psycopg2
import pytest

from db_connection import ConnectionManager


class _FakeConn:
    def __init__(self, closed=False, healthy=True):
        self.closed = closed
        self._healthy = healthy
        self.committed = False
        self.rolled_back = False

    def cursor(self):
        conn = self

        class _Cur:
            def __enter__(self):
                return self
            def __exit__(self, *exc):
                return False
            def execute(self, sql):
                if not conn._healthy:
                    raise psycopg2.OperationalError("connection failed")
        return _Cur()

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True


def _manager_with_pool(pool):
    manager = ConnectionManager.__new__(ConnectionManager)
    manager.db_url = "postgresql://fake"
    manager._last_health_check = 0.0
    manager.pool = pool
    return manager


def test_connection_yields_pooled_connection_and_commits():
    conn = _FakeConn()
    pool = Mock(getconn=Mock(return_value=conn), putconn=Mock())
    manager = _manager_with_pool(pool)

    with manager.connection() as yielded:
        assert yielded is conn

    assert conn.committed is True
    pool.putconn.assert_called_once_with(conn, close=False)


def test_connection_rolls_back_and_reraises_on_error():
    conn = _FakeConn()
    pool = Mock(getconn=Mock(return_value=conn), putconn=Mock())
    manager = _manager_with_pool(pool)

    with pytest.raises(RuntimeError):
        with manager.connection():
            raise RuntimeError("boom")

    assert conn.rolled_back is True


def test_close_closes_the_pool():
    pool = Mock()
    manager = _manager_with_pool(pool)
    manager.close()
    pool.closeall.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_db_connection.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'db_connection'`

- [ ] **Step 3: Write the implementation**

```python
"""Shared PostgreSQL connection pool and health-checked connection context
manager, used by every repository so they don't each open their own pool."""
import logging
import time
from contextlib import contextmanager

import psycopg2
from psycopg2 import pool

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Thread-safe PostgreSQL connection pool with keepalive + health checks."""

    def __init__(self, db_url: str):
        self.db_url = db_url
        self._last_health_check: float = 0.0
        try:
            self.pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=10,
                dsn=db_url,
                keepalives=1,
                keepalives_idle=30,
                keepalives_interval=10,
                keepalives_count=5,
            )
            logger.info("Database connection pool initialized with keepalives")
        except Exception as e:
            logger.error(f"Failed to create connection pool: {e}", exc_info=True)
            raise

    def close(self):
        try:
            if hasattr(self, "pool") and self.pool:
                self.pool.closeall()
                logger.info("Database connection pool closed")
        except Exception as e:
            logger.error(f"Failed to close connection pool: {e}", exc_info=True)

    @contextmanager
    def connection(self):
        """Thread-safe connection context manager using connection pool with validation."""
        conn = None
        try:
            conn = self.pool.getconn()
            if conn.closed:
                logger.warning("Retrieved closed connection from pool, discarding")
                self.pool.putconn(conn, close=True)
                conn = self.pool.getconn()

            now = time.monotonic()
            if now - self._last_health_check > 30.0:
                try:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1")
                    self._last_health_check = now
                except psycopg2.OperationalError:
                    logger.warning("Connection failed health check, replacing")
                    self.pool.putconn(conn, close=True)
                    conn = self.pool.getconn()
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1")
                    self._last_health_check = time.monotonic()

            yield conn
            conn.commit()
        except psycopg2.OperationalError:
            if conn and not conn.closed:
                try:
                    conn.rollback()
                except psycopg2.OperationalError:
                    pass
            raise
        except Exception:
            if conn and not conn.closed:
                conn.rollback()
            raise
        finally:
            if conn:
                self.pool.putconn(conn, close=conn.closed)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_db_connection.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add db_connection.py tests/test_db_connection.py
git commit -m "refactor: extract ConnectionManager from Database"
```

---

## Task 11: MessageRepository

**Files:**
- Create: `message_repository.py`

**Interfaces:**
- Consumes: `ConnectionManager` (Task 10).
- Produces: `MessageRepository(conn)` with `add_message`, `get_messages_by_tokens`, `get_recent_messages`, `clear_history`, `get_stats`, `cleanup_old_group_messages` — consumed by Task 15. (`get_messages_by_tokens`/`get_recent_messages`/`clear_history` are ported here unchanged and removed in Task 16 once confirmed callerless — this task's job is a behavior-preserving move, not deletion.)

No new characterization test in this task — Task 15 replaces `tests/test_database_characterization.py` with `tests/test_repositories.py`, which covers this repository (and the other two) once the `Database` facade routes through them. Writing repository-level tests now, then rewriting them again in Task 15 against the exact same repository classes, would just be double work; the safety net for this move is the *existing* `tests/test_database_characterization.py` from Task 1, which keeps exercising the still-unchanged `Database` class from Task 11 through Task 14 (the facade isn't touched until Task 15).

- [ ] **Step 1: Write the implementation**

```python
"""Message audit-log persistence: inserts, token-budget reads, and stats."""
import logging
from datetime import datetime

from psycopg2.extras import RealDictCursor

from db_connection import ConnectionManager

logger = logging.getLogger(__name__)


class MessageRepository:
    """CRUD for the `messages` audit table."""

    def __init__(self, conn: ConnectionManager):
        self._conn = conn

    def add_message(
        self,
        chat_id: str,
        role: str,
        content: str,
        user_id: int = None,
        message_id: int = None,
        token_count: int = 0,
        sender_name: str = None,
        sender_username: str = None,
        is_group_chat: bool = False,
    ) -> int:
        """Add a message to the database with atomic transaction."""
        try:
            chat_id = str(chat_id)
            timestamp = datetime.utcnow()

            with self._conn.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO messages
                        (chat_id, role, content, timestamp, user_id, message_id, token_count,
                         sender_name, sender_username, is_group_chat)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                        """,
                        (chat_id, role, content, timestamp, user_id, message_id, token_count,
                         sender_name, sender_username, is_group_chat),
                    )
                    msg_id = cur.fetchone()[0]

            logger.debug(
                f"Added message {msg_id} for chat {chat_id}: "
                f"{role} ({token_count} tokens)"
            )
            return msg_id

        except Exception as e:
            logger.error(f"Failed to add message: {e}", exc_info=True)
            raise

    def get_messages_by_tokens(self, chat_id: str, max_tokens: int) -> list:
        """
        Retrieve recent messages within token budget.

        Args:
            chat_id: Chat ID to retrieve messages from
            max_tokens: Maximum token budget

        Returns messages in chronological order (oldest first).
        For group chats, includes sender information in the format [Name]: message
        """
        try:
            chat_id = str(chat_id)
            messages = []
            total_tokens = 0

            with self._conn.connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    query = """
                        SELECT role, content, token_count, sender_name, sender_username, is_group_chat
                        FROM messages
                        WHERE chat_id = %s
                        ORDER BY timestamp DESC LIMIT 500
                    """
                    cur.execute(query, (chat_id,))

                    temp_messages = []
                    for row in cur.fetchall():
                        msg_tokens = row["token_count"] or 0
                        if not temp_messages or total_tokens + msg_tokens <= max_tokens:
                            temp_messages.append({
                                "role": row["role"],
                                "content": row["content"],
                                "sender_name": row["sender_name"],
                                "sender_username": row["sender_username"],
                                "is_group_chat": row["is_group_chat"],
                            })
                            total_tokens += msg_tokens
                        else:
                            break

                    messages = list(reversed(temp_messages))

            logger.debug(
                f"Retrieved {len(messages)} messages for chat {chat_id} "
                f"({total_tokens} tokens)"
            )
            return messages

        except Exception as e:
            logger.error(f"Failed to retrieve messages: {e}", exc_info=True)
            return self.get_recent_messages(chat_id, limit=50)

    def get_recent_messages(self, chat_id: str, limit: int = 100) -> list:
        """Fallback: get last N messages if token counting fails."""
        try:
            chat_id = str(chat_id)

            with self._conn.connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        """
                        SELECT role, content
                        FROM messages
                        WHERE chat_id = %s
                        ORDER BY timestamp ASC
                        LIMIT %s
                        """,
                        (chat_id, limit),
                    )
                    messages = [
                        {"role": row["role"], "content": row["content"]}
                        for row in cur.fetchall()
                    ]

            logger.debug(f"Retrieved {len(messages)} recent messages for chat {chat_id}")
            return messages

        except Exception as e:
            logger.error(f"Failed to retrieve recent messages: {e}", exc_info=True)
            return []

    def clear_history(self, chat_id: str):
        """Delete all messages for a chat."""
        try:
            chat_id = str(chat_id)

            with self._conn.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM messages WHERE chat_id = %s",
                        (chat_id,),
                    )
                    deleted_count = cur.rowcount

            logger.info(f"Cleared {deleted_count} messages for chat {chat_id}")

        except Exception as e:
            logger.error(f"Failed to clear history: {e}", exc_info=True)
            raise

    def get_stats(self, chat_id: str) -> dict:
        """Return statistics for monitoring."""
        try:
            chat_id = str(chat_id)

            with self._conn.connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        """
                        SELECT
                            COUNT(*) as total_messages,
                            COALESCE(SUM(token_count), 0) as total_tokens,
                            MIN(timestamp) as first_message,
                            MAX(timestamp) as last_message
                        FROM messages
                        WHERE chat_id = %s
                        """,
                        (chat_id,),
                    )
                    row = cur.fetchone()

                    return {
                        "total_messages": row["total_messages"] or 0,
                        "total_tokens": row["total_tokens"] or 0,
                        "first_message": row["first_message"].isoformat() if row["first_message"] else "N/A",
                        "last_message": row["last_message"].isoformat() if row["last_message"] else "N/A",
                    }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}", exc_info=True)
            return {
                "total_messages": 0,
                "total_tokens": 0,
                "first_message": "N/A",
                "last_message": "N/A",
            }

    def cleanup_old_group_messages(self, chat_id: str, keep_recent: int = 100):
        """
        Remove old messages from group chats to prevent unlimited growth.
        Keeps only the most recent N messages. Currently unused in the
        message-handling path (see CLAUDE.md Context Storage notes: the
        probabilistic cleanup call is intentionally disabled); kept for an
        eventual coordinated retention policy.

        Args:
            chat_id: Chat ID to clean up
            keep_recent: Number of recent messages to keep (default 100)
        """
        try:
            chat_id = str(chat_id)

            with self._conn.connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        "SELECT COUNT(*) as count FROM messages WHERE chat_id = %s AND is_group_chat = TRUE",
                        (chat_id,),
                    )
                    total = cur.fetchone()["count"]

                    if total > keep_recent:
                        cur.execute(
                            """
                            DELETE FROM messages
                            WHERE chat_id = %s AND is_group_chat = TRUE
                            AND id NOT IN (
                                SELECT id FROM messages
                                WHERE chat_id = %s AND is_group_chat = TRUE
                                ORDER BY timestamp DESC
                                LIMIT %s
                            )
                            """,
                            (chat_id, chat_id, keep_recent),
                        )
                        deleted = total - keep_recent
                        logger.info(f"Cleaned up {deleted} old messages from group chat {chat_id}")

        except Exception as e:
            logger.error(f"Failed to cleanup old messages: {e}", exc_info=True)
```

- [ ] **Step 2: Compile check**

Run: `python3 -m py_compile message_repository.py`
Expected: no output, exit code 0.

- [ ] **Step 3: Commit**

```bash
git add message_repository.py
git commit -m "refactor: add MessageRepository (Database still authoritative until Task 15)"
```

---

## Task 12: AccessRepository

**Files:**
- Create: `access_repository.py`

**Interfaces:**
- Consumes: `ConnectionManager` (Task 10), `cache.TTLCache`/`cache.MISSING`.
- Produces: `AccessRepository(conn, cache)` with `grant_access`, `revoke_access`, `is_user_granted`, `get_granted_users` — consumed by Task 15.

- [ ] **Step 1: Write the implementation**

```python
"""Allowlist persistence: grant/revoke/check access, cached."""
import logging
from datetime import datetime

from psycopg2.extras import RealDictCursor

from cache import MISSING, TTLCache
from db_connection import ConnectionManager

logger = logging.getLogger(__name__)


class AccessRepository:
    """CRUD for the `granted_users` allowlist table."""

    def __init__(self, conn: ConnectionManager, cache: TTLCache):
        self._conn = conn
        self._cache = cache

    def grant_access(self, user_id: int, first_name: str = None, username: str = None) -> bool:
        try:
            user_id_str = str(user_id)
            timestamp = datetime.utcnow()

            with self._conn.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT user_id FROM granted_users WHERE user_id = %s",
                        (user_id_str,),
                    )
                    existing = cur.fetchone()

                    if existing:
                        logger.info(f"User {user_id} already has access")
                        return False

                    cur.execute(
                        "INSERT INTO granted_users (user_id, granted_at, first_name, username) VALUES (%s, %s, %s, %s)",
                        (user_id_str, timestamp, first_name, username),
                    )

            self._cache.invalidate(f"granted:{user_id}")
            logger.info(f"Granted access to user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to grant access: {e}", exc_info=True)
            raise

    def revoke_access(self, user_id: int) -> bool:
        try:
            user_id_str = str(user_id)

            with self._conn.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM granted_users WHERE user_id = %s",
                        (user_id_str,),
                    )
                    deleted_count = cur.rowcount

            self._cache.invalidate(f"granted:{user_id}")
            if deleted_count > 0:
                logger.info(f"Revoked access from user {user_id}")
                return True
            else:
                logger.info(f"User {user_id} didn't have access")
                return False

        except Exception as e:
            logger.error(f"Failed to revoke access: {e}", exc_info=True)
            raise

    def is_user_granted(self, user_id: int) -> bool:
        cache_key = f"granted:{user_id}"
        cached = self._cache.get(cache_key)
        if cached is not MISSING:
            return cached

        try:
            user_id_str = str(user_id)

            with self._conn.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT user_id FROM granted_users WHERE user_id = %s",
                        (user_id_str,),
                    )
                    result = cur.fetchone()

            granted = result is not None
            self._cache.set(cache_key, granted, ttl=120.0)
            return granted

        except Exception as e:
            logger.error(f"Failed to check granted access: {e}", exc_info=True)
            return False

    def get_granted_users(self) -> list:
        try:
            with self._conn.connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        "SELECT user_id, granted_at, first_name, username FROM granted_users ORDER BY granted_at DESC"
                    )
                    users = [
                        (
                            row["user_id"],
                            row["granted_at"].isoformat() if row["granted_at"] else "N/A",
                            row["first_name"],
                            row["username"],
                        )
                        for row in cur.fetchall()
                    ]

            return users

        except Exception as e:
            logger.error(f"Failed to get granted users: {e}", exc_info=True)
            return []
```

- [ ] **Step 2: Compile check**

Run: `python3 -m py_compile access_repository.py`
Expected: no output, exit code 0.

- [ ] **Step 3: Commit**

```bash
git add access_repository.py
git commit -m "refactor: add AccessRepository (Database still authoritative until Task 15)"
```

---

## Task 13: SettingsRepository

**Files:**
- Create: `settings_repository.py`

**Interfaces:**
- Consumes: `ConnectionManager` (Task 10), `cache.TTLCache`/`cache.MISSING`.
- Produces: `SettingsRepository(conn, cache)` with `get_personality_prompt`, `get_active_personality`, `set_active_personality`, `personality_exists`, `list_personalities`, `init_active_model`, `get_active_model`, `set_active_model` — consumed by Task 15.

- [ ] **Step 1: Write the implementation**

```python
"""Personality and active-model global settings, cached."""
import logging
from datetime import datetime

from psycopg2.extras import RealDictCursor

from cache import MISSING, TTLCache
from db_connection import ConnectionManager

logger = logging.getLogger(__name__)


class SettingsRepository:
    """CRUD for `personality`, `active_personality`, and `active_model`."""

    def __init__(self, conn: ConnectionManager, cache: TTLCache):
        self._conn = conn
        self._cache = cache

    def get_personality_prompt(self, personality: str) -> str | None:
        cache_key = f"personality_prompt:{personality}"
        cached = self._cache.get(cache_key)
        if cached is not MISSING:
            return cached

        try:
            with self._conn.connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        "SELECT prompt FROM personality WHERE personality = %s",
                        (personality,)
                    )
                    row = cur.fetchone()
                    result = row["prompt"] if row else None

            self._cache.set(cache_key, result, ttl=300.0)
            return result

        except Exception as e:
            logger.error(f"Failed to get personality prompt: {e}", exc_info=True)
            return None

    def get_active_personality(self) -> str:
        cache_key = "active_personality"
        cached = self._cache.get(cache_key)
        if cached is not MISSING:
            return cached

        try:
            with self._conn.connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        "SELECT personality FROM active_personality WHERE id = 1"
                    )
                    row = cur.fetchone()
                    result = row["personality"] if row else "default"

            self._cache.set(cache_key, result, ttl=60.0)
            return result

        except Exception as e:
            logger.error(f"Failed to get active personality: {e}", exc_info=True)
            return "default"

    def set_active_personality(self, personality: str) -> None:
        try:
            timestamp = datetime.utcnow()
            with self._conn.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO active_personality (id, personality, updated_at)
                        VALUES (1, %s, %s)
                        ON CONFLICT (id) DO UPDATE
                        SET personality = EXCLUDED.personality,
                            updated_at = EXCLUDED.updated_at
                        """,
                        (personality, timestamp)
                    )

            self._cache.invalidate("active_personality")
            self._cache.invalidate_prefix("personality_prompt:")
            logger.info(f"Active personality set to: {personality}")

        except Exception as e:
            logger.error(f"Failed to set active personality: {e}", exc_info=True)
            raise

    def personality_exists(self, personality: str) -> bool:
        try:
            with self._conn.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT personality FROM personality WHERE personality = %s",
                        (personality,)
                    )
                    result = cur.fetchone()
                    return result is not None

        except Exception as e:
            logger.error(f"Failed to check personality existence: {e}", exc_info=True)
            return False

    def list_personalities(self) -> list[tuple[str, str]]:
        try:
            with self._conn.connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        "SELECT personality, prompt FROM personality ORDER BY personality"
                    )
                    personalities = [
                        (row["personality"], row["prompt"][:100] + "..." if len(row["prompt"]) > 100 else row["prompt"])
                        for row in cur.fetchall()
                    ]
            return personalities
        except Exception as e:
            logger.error(f"Failed to list personalities: {e}", exc_info=True)
            return []

    def init_active_model(self, default_model: str) -> None:
        try:
            with self._conn.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO active_model (id, model) VALUES (1, %s) ON CONFLICT DO NOTHING",
                        (default_model,)
                    )
        except Exception as e:
            logger.error(f"Failed to init active model: {e}", exc_info=True)

    def get_active_model(self) -> str:
        cache_key = "active_model"
        cached = self._cache.get(cache_key)
        if cached is not MISSING:
            return cached

        try:
            with self._conn.connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("SELECT model FROM active_model WHERE id = 1")
                    row = cur.fetchone()
                    result = row["model"] if row else "gpt-5.4-mini"

            self._cache.set(cache_key, result, ttl=60.0)
            return result

        except Exception as e:
            logger.error(f"Failed to get active model: {e}", exc_info=True)
            return "gpt-5.4-mini"

    def set_active_model(self, model: str) -> None:
        try:
            timestamp = datetime.utcnow()
            with self._conn.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO active_model (id, model, updated_at)
                        VALUES (1, %s, %s)
                        ON CONFLICT (id) DO UPDATE
                        SET model = EXCLUDED.model,
                            updated_at = EXCLUDED.updated_at
                        """,
                        (model, timestamp)
                    )

            self._cache.invalidate("active_model")
            logger.info(f"Active model set to: {model}")

        except Exception as e:
            logger.error(f"Failed to set active model: {e}", exc_info=True)
            raise
```

- [ ] **Step 2: Compile check**

Run: `python3 -m py_compile settings_repository.py`
Expected: no output, exit code 0.

- [ ] **Step 3: Commit**

```bash
git add settings_repository.py
git commit -m "refactor: add SettingsRepository (Database still authoritative until Task 15)"
```

---

## Task 14: SummaryAuditRepository

**Files:**
- Create: `summary_audit_repository.py`

**Interfaces:**
- Consumes: `ConnectionManager` (Task 10).
- Produces: `SummaryAuditRepository(conn).record_conversation_summary(...)` — consumed by Task 15 and, transitively, by `agent.Agent._record_summary`.

- [ ] **Step 1: Write the implementation**

```python
"""Audit-only persistence for confirmed checkpoint summaries."""
import logging

from db_connection import ConnectionManager

logger = logging.getLogger(__name__)


class SummaryAuditRepository:
    """Insert-only CRUD for the `conversation_summaries` audit table."""

    def __init__(self, conn: ConnectionManager):
        self._conn = conn

    def record_conversation_summary(
        self,
        chat_id: str,
        summary_text: str,
        summary_model: str,
        before_message_count: int,
        after_message_count: int,
        before_tokens: int,
        after_tokens: int,
    ) -> int:
        """Persist a permanent audit record of a successful checkpoint summary."""
        try:
            chat_id = str(chat_id)
            with self._conn.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO conversation_summaries
                        (chat_id, summary_text, summary_model, before_message_count,
                         after_message_count, before_tokens, after_tokens)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                        """,
                        (
                            chat_id,
                            summary_text,
                            summary_model,
                            before_message_count,
                            after_message_count,
                            before_tokens,
                            after_tokens,
                        ),
                    )
                    row_id = cur.fetchone()[0]

            logger.info(f"Recorded conversation summary {row_id} for chat {chat_id}")
            return row_id
        except Exception as e:
            logger.error(f"Failed to record conversation summary: {e}", exc_info=True)
            raise
```

- [ ] **Step 2: Compile check**

Run: `python3 -m py_compile summary_audit_repository.py`
Expected: no output, exit code 0.

- [ ] **Step 3: Commit**

```bash
git add summary_audit_repository.py
git commit -m "refactor: add SummaryAuditRepository (Database still authoritative until Task 15)"
```

---

## Task 15: Rewrite database.py as a facade

**Files:**
- Modify: `database.py` (full rewrite)
- Delete: `tests/test_database_characterization.py`
- Create: `tests/test_repositories.py`

**Interfaces:**
- Consumes: `ConnectionManager` (Task 10), `MessageRepository` (Task 11), `AccessRepository` (Task 12), `SettingsRepository` (Task 13), `SummaryAuditRepository` (Task 14).
- Produces: `Database(db_url)` with the exact same public method set it has today — `close`, `add_message`, `get_messages_by_tokens`, `get_recent_messages`, `clear_history`, `get_stats`, `cleanup_old_group_messages`, `grant_access`, `revoke_access`, `is_user_granted`, `get_granted_users`, `get_personality_prompt`, `get_active_personality`, `set_active_personality`, `personality_exists`, `list_personalities`, `init_active_model`, `get_active_model`, `set_active_model`, `record_conversation_summary`. `bot.py` and `scripts/chat_cli.py` need **no changes** in this task.

- [ ] **Step 1: Rewrite database.py**

```python
"""Thin facade preserving the public Database API, backed by focused
repositories that all share one ConnectionManager and one TTLCache."""
import logging

from access_repository import AccessRepository
from cache import TTLCache
from db_connection import ConnectionManager
from message_repository import MessageRepository
from settings_repository import SettingsRepository
from summary_audit_repository import SummaryAuditRepository

logger = logging.getLogger(__name__)


class Database:
    """Facade over MessageRepository, AccessRepository, SettingsRepository,
    and SummaryAuditRepository, preserving the pre-split public API."""

    def __init__(self, db_url: str):
        self._conn = ConnectionManager(db_url)
        self._cache = TTLCache(default_ttl=60.0)
        self._messages = MessageRepository(self._conn)
        self._access = AccessRepository(self._conn, self._cache)
        self._settings = SettingsRepository(self._conn, self._cache)
        self._summaries = SummaryAuditRepository(self._conn)

    def close(self):
        self._conn.close()

    # --- messages ------------------------------------------------------
    def add_message(self, *args, **kwargs) -> int:
        return self._messages.add_message(*args, **kwargs)

    def get_messages_by_tokens(self, chat_id: str, max_tokens: int) -> list:
        return self._messages.get_messages_by_tokens(chat_id, max_tokens)

    def get_recent_messages(self, chat_id: str, limit: int = 100) -> list:
        return self._messages.get_recent_messages(chat_id, limit)

    def clear_history(self, chat_id: str):
        return self._messages.clear_history(chat_id)

    def get_stats(self, chat_id: str) -> dict:
        return self._messages.get_stats(chat_id)

    def cleanup_old_group_messages(self, chat_id: str, keep_recent: int = 100):
        return self._messages.cleanup_old_group_messages(chat_id, keep_recent)

    # --- access ----------------------------------------------------------
    def grant_access(self, user_id: int, first_name: str = None, username: str = None) -> bool:
        return self._access.grant_access(user_id, first_name=first_name, username=username)

    def revoke_access(self, user_id: int) -> bool:
        return self._access.revoke_access(user_id)

    def is_user_granted(self, user_id: int) -> bool:
        return self._access.is_user_granted(user_id)

    def get_granted_users(self) -> list:
        return self._access.get_granted_users()

    # --- settings ----------------------------------------------------------
    def get_personality_prompt(self, personality: str) -> str | None:
        return self._settings.get_personality_prompt(personality)

    def get_active_personality(self) -> str:
        return self._settings.get_active_personality()

    def set_active_personality(self, personality: str) -> None:
        return self._settings.set_active_personality(personality)

    def personality_exists(self, personality: str) -> bool:
        return self._settings.personality_exists(personality)

    def list_personalities(self) -> list[tuple[str, str]]:
        return self._settings.list_personalities()

    def init_active_model(self, default_model: str) -> None:
        return self._settings.init_active_model(default_model)

    def get_active_model(self) -> str:
        return self._settings.get_active_model()

    def set_active_model(self, model: str) -> None:
        return self._settings.set_active_model(model)

    # --- summary audit ------------------------------------------------------
    def record_conversation_summary(self, *args, **kwargs) -> int:
        return self._summaries.record_conversation_summary(*args, **kwargs)
```

- [ ] **Step 2: Replace the characterization test with repository-targeted tests**

`tests/test_database_characterization.py` patched `Database._get_connection`, which no longer exists (connections now go through `ConnectionManager.connection()` inside each repository). Delete it and create `tests/test_repositories.py` covering the same behaviors directly against the repositories:

```bash
git rm tests/test_database_characterization.py
```

```python
"""Repository characterization tests: SQL executed, params, and cache
invalidation timing, using a fake connection double instead of a live DB."""
from contextlib import contextmanager

from access_repository import AccessRepository
from cache import MISSING, TTLCache
from db_connection import ConnectionManager
from message_repository import MessageRepository
from settings_repository import SettingsRepository


class _FakeCursor:
    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):
        self._conn.executed.append((" ".join(sql.split()), params))

    def fetchone(self):
        return self._conn.results.pop(0) if self._conn.results else None

    def fetchall(self):
        return self._conn.results.pop(0) if self._conn.results else []

    @property
    def rowcount(self):
        return self._conn.rowcount


class _FakeConn:
    def __init__(self, results=None, rowcount=0):
        self.executed = []
        self.results = list(results or [])
        self.rowcount = rowcount
        self.closed = False

    def cursor(self, cursor_factory=None):
        return _FakeCursor(self)

    def commit(self):
        pass

    def rollback(self):
        pass


def _fake_manager(results=None, rowcount=0):
    manager = ConnectionManager.__new__(ConnectionManager)
    conn = _FakeConn(results=results, rowcount=rowcount)

    @contextmanager
    def fake_connection():
        yield conn

    manager.connection = fake_connection
    return manager, conn


# --- MessageRepository ---------------------------------------------------

def test_add_message_inserts_with_expected_params():
    manager, conn = _fake_manager(results=[(42,)])
    repo = MessageRepository(manager)
    msg_id = repo.add_message(
        chat_id=123, role="user", content="hi", user_id=7, message_id=1,
        token_count=3, sender_name="Alice", sender_username="alice",
        is_group_chat=True,
    )
    assert msg_id == 42
    sql, params = conn.executed[0]
    assert "INSERT INTO messages" in sql
    assert params[0] == "123"
    assert params[1] == "user"
    assert params[2] == "hi"


def test_get_stats_returns_na_when_no_messages():
    manager, conn = _fake_manager(results=[{
        "total_messages": 0, "total_tokens": 0,
        "first_message": None, "last_message": None,
    }])
    repo = MessageRepository(manager)
    stats = repo.get_stats("123")
    assert stats == {
        "total_messages": 0, "total_tokens": 0,
        "first_message": "N/A", "last_message": "N/A",
    }


def test_get_stats_returns_zeroed_dict_on_query_failure():
    manager = ConnectionManager.__new__(ConnectionManager)

    @contextmanager
    def broken_connection():
        raise RuntimeError("db down")
        yield  # pragma: no cover

    manager.connection = broken_connection
    repo = MessageRepository(manager)
    stats = repo.get_stats("123")
    assert stats["total_messages"] == 0
    assert stats["first_message"] == "N/A"


# --- AccessRepository ------------------------------------------------------

def test_is_user_granted_caches_result_and_skips_second_query():
    manager, conn = _fake_manager(results=[("55",)])
    repo = AccessRepository(manager, TTLCache(default_ttl=60.0))
    first = repo.is_user_granted(55)
    second = repo.is_user_granted(55)
    assert first is True
    assert second is True
    assert len(conn.executed) == 1


def test_grant_access_invalidates_granted_cache():
    manager, conn = _fake_manager(results=[None])
    cache = TTLCache(default_ttl=60.0)
    cache.set("granted:99", False, ttl=120.0)
    repo = AccessRepository(manager, cache)
    was_granted = repo.grant_access(99, first_name="Bob", username="bob")
    assert was_granted is True
    assert cache.get("granted:99") is MISSING


# --- SettingsRepository -----------------------------------------------------

def test_set_active_personality_invalidates_related_cache_entries():
    manager, conn = _fake_manager()
    cache = TTLCache(default_ttl=60.0)
    cache.set("active_personality", "old", ttl=60.0)
    cache.set("personality_prompt:old", "prompt text", ttl=300.0)
    repo = SettingsRepository(manager, cache)
    repo.set_active_personality("villain")
    assert cache.get("active_personality") is MISSING
    assert cache.get("personality_prompt:old") is MISSING


def test_get_active_model_returns_default_when_no_row():
    manager, conn = _fake_manager(results=[None])
    repo = SettingsRepository(manager, TTLCache(default_ttl=60.0))
    assert repo.get_active_model() == "gpt-5.4-mini"


def test_list_personalities_truncates_long_prompt_preview():
    long_prompt = "x" * 150
    manager, conn = _fake_manager(results=[[
        {"personality": "villain", "prompt": long_prompt},
        {"personality": "normal", "prompt": "short"},
    ]])
    repo = SettingsRepository(manager, TTLCache(default_ttl=60.0))
    personalities = repo.list_personalities()
    assert personalities[0] == ("villain", "x" * 100 + "...")
    assert personalities[1] == ("normal", "short")
```

- [ ] **Step 3: Run the full suite**

Run: `pytest tests/ -v`
Expected: all tests pass, including `tests/test_repositories.py` and every other suite (`tests/test_agent.py` uses `Mock()`-based fake `db` objects, never a real `Database`, so it is unaffected by this rewrite).

- [ ] **Step 4: Compile check**

Run: `python3 -m py_compile *.py`
Expected: no output, exit code 0.

- [ ] **Step 5: Commit**

```bash
git add database.py tests/test_repositories.py
git commit -m "refactor: turn database.py into a facade over four repositories"
```

---

## Task 16: Remove confirmed-dead legacy read APIs

**Files:**
- Modify: `message_repository.py`
- Modify: `database.py`

**Interfaces:**
- Removes: `MessageRepository.get_messages_by_tokens`, `MessageRepository.get_recent_messages`, `MessageRepository.clear_history`, and their `Database` facade delegates. `Database.cleanup_old_group_messages` is kept — it is the intentionally-disabled retention hook `CLAUDE.md`'s Context Storage section documents as current, still-desired state, and `tests/test_context_message_retention.py` explicitly asserts it is *not* called from the message-handling path today; removing it would contradict both.

- [ ] **Step 1: Confirm zero callers**

Run:
```bash
grep -rn "get_messages_by_tokens\|get_recent_messages\|\.clear_history(" --include="*.py" . | grep -v "^\./venv/\|^\./.venv/\|/site-packages/\|message_repository.py:\|database.py:"
```
Expected: no output — `agent.clear_thread()` is what `handlers.clear_command` actually calls (see `command_handlers.py`), and `ChatCLI.clear_history` in `scripts/chat_cli.py` is an unrelated same-named method on a different class that calls `self.agent.clear_thread`, not `self.db.clear_history`.

- [ ] **Step 2: Remove the three methods from message_repository.py**

Delete the `get_messages_by_tokens`, `get_recent_messages`, and `clear_history` method definitions from the `MessageRepository` class in `message_repository.py`, leaving `add_message`, `get_stats`, and `cleanup_old_group_messages`.

- [ ] **Step 3: Remove the matching facade delegates from database.py**

Delete these three methods from the `Database` class in `database.py`:
```python
    def get_messages_by_tokens(self, chat_id: str, max_tokens: int) -> list:
        return self._messages.get_messages_by_tokens(chat_id, max_tokens)

    def get_recent_messages(self, chat_id: str, limit: int = 100) -> list:
        return self._messages.get_recent_messages(chat_id, limit)

    def clear_history(self, chat_id: str):
        return self._messages.clear_history(chat_id)
```

- [ ] **Step 4: Run the full suite**

Run: `pytest tests/ -v`
Expected: all tests pass (no test in the suite exercises these three methods, per Step 1's grep).

- [ ] **Step 5: Compile check**

Run: `python3 -m py_compile *.py`
Expected: no output, exit code 0.

- [ ] **Step 6: Commit**

```bash
git add message_repository.py database.py
git commit -m "refactor: remove get_messages_by_tokens/get_recent_messages/clear_history (checkpoint history superseded them, zero callers)"
```

---

## Task 17: Scoped Ruff + mypy gates

**Files:**
- Create: `pyproject.toml`
- Modify: `requirements-dev.txt`
- Modify: `.github/workflows/ci.yml`

**Interfaces:** none (tooling only, no runtime code changes).

This task intentionally scopes both tools to the modules this plan introduced or rewrote, rather than the whole pre-existing codebase — running Ruff/mypy repo-wide would surface a large, unrelated backlog of pre-existing style/typing issues that are out of scope for this refactor.

- [ ] **Step 1: Add ruff and mypy to requirements-dev.txt**

```
pytest==8.3.4
ruff==0.8.4
mypy==1.13.0
```

- [ ] **Step 2: Install and confirm the tools run**

Run: `pip install -r requirements-dev.txt`
Expected: `ruff` and `mypy` installed without error.

- [ ] **Step 3: Add pyproject.toml**

```toml
[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]

[tool.mypy]
python_version = "3.12"
ignore_missing_imports = true
disallow_untyped_defs = false
files = [
    "handler_deps.py",
    "authorization.py",
    "request_processor.py",
    "message_handlers.py",
    "command_handlers.py",
    "handlers.py",
    "model_registry.py",
    "token_budget.py",
    "db_connection.py",
    "message_repository.py",
    "access_repository.py",
    "settings_repository.py",
    "summary_audit_repository.py",
    "database.py",
]
```

- [ ] **Step 4: Run both tools locally and fix any findings**

Run:
```bash
ruff check handler_deps.py authorization.py request_processor.py message_handlers.py \
  command_handlers.py handlers.py model_registry.py token_budget.py db_connection.py \
  message_repository.py access_repository.py settings_repository.py \
  summary_audit_repository.py database.py
mypy
```
Expected: both exit 0. If either reports findings against code introduced in Tasks 1–16, fix them in place (e.g. an unused import Ruff's `F401` catches) and re-run — do not silence with inline `# noqa`/`# type: ignore` unless the finding is a false positive, in which case comment why.

- [ ] **Step 5: Wire into CI**

In `.github/workflows/ci.yml`, add two steps after the existing "Compile Python files" step and before "Run tests":

```yaml
      - name: Lint (new modules)
        run: >
          ruff check handler_deps.py authorization.py request_processor.py
          message_handlers.py command_handlers.py handlers.py model_registry.py
          token_budget.py db_connection.py message_repository.py
          access_repository.py settings_repository.py summary_audit_repository.py
          database.py

      - name: Type check (new modules)
        run: mypy
```

- [ ] **Step 6: Run the full suite**

Run: `pytest tests/ -v`
Expected: all tests pass (no runtime code changed in this task).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml requirements-dev.txt .github/workflows/ci.yml
git commit -m "ci: add scoped Ruff and mypy gates for the refactored modules"
```

---

## Task 18: Consolidate composition into app_factory.py

**Files:**
- Create: `app_factory.py`
- Modify: `bot.py`
- Modify: `scripts/chat_cli.py`

**Interfaces:**
- Produces: `build_app_stack(config) -> AppStack` where `AppStack` has fields `db`, `prompt_builder`, `agent`, `checkpointer_pool`. `bot.py` and `scripts/chat_cli.py` both call it instead of duplicating the db/checkpointer/prompt-builder/agent construction they each currently repeat (`bot.py` lines ~72–110; `scripts/chat_cli.py`'s `ChatCLI.__init__`).

Note: `scripts/chat_cli.py`'s current `ConnectionPool(...)` call omits `check=ConnectionPool.check_connection`, which `bot.py`'s call includes — an existing, undocumented drift between the two bootstraps that this task deliberately fixes by having both go through the same factory (chat_cli.py's checkpointer connections gain the same health check bot.py's already have).

- [ ] **Step 1: Create app_factory.py**

```python
"""Single composition point for the db/prompt-builder/agent stack, shared by
bot.py and scripts/chat_cli.py so their bootstrap can't drift apart."""
from dataclasses import dataclass

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver

import agent as agent_module
from agent import Agent
from database import Database
from prompt_builder import PromptBuilder


@dataclass
class AppStack:
    db: Database
    prompt_builder: PromptBuilder
    agent: Agent
    checkpointer_pool: ConnectionPool


def build_app_stack(config) -> AppStack:
    """Build the db/prompt-builder/agent stack shared by bot.py and chat_cli.py.

    Tables are created out-of-band by scripts/setup_checkpointer.py (deploy
    preDeployCommand); this does NOT call PostgresSaver.setup().
    """
    db = Database(config.DATABASE_URL)
    db.init_active_model(config.DEFAULT_MODEL)
    effective_model = db.get_active_model()

    checkpointer_pool = ConnectionPool(
        conninfo=config.DATABASE_URL,
        max_size=10,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        check=ConnectionPool.check_connection,
    )
    checkpointer = PostgresSaver(checkpointer_pool)

    prompt_builder = PromptBuilder(
        default_private_prompt=agent_module.SYSTEM_PROMPT,
        default_group_prompt=agent_module.SYSTEM_PROMPT_GROUP,
        get_active_personality=db.get_active_personality,
        get_personality_prompt=db.get_personality_prompt,
    )

    bot_agent = Agent(
        config=config,
        prompt_builder=prompt_builder,
        checkpointer=checkpointer,
        model_name=effective_model,
        db=db,
    )

    return AppStack(
        db=db,
        prompt_builder=prompt_builder,
        agent=bot_agent,
        checkpointer_pool=checkpointer_pool,
    )
```

- [ ] **Step 2: Update bot.py to use it**

Replace these imports at the top of `bot.py`:
```python
from config import config
from database import Database
from prompt_builder import PromptBuilder
from agent import Agent
import agent as agent_module
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres import PostgresSaver
import handlers
```
with:
```python
from config import config
from app_factory import build_app_stack
import handlers
```

Replace steps 2–5 of `main()` (currently the block from `# 2. Initialize database` through the end of `# 5. Build the agent for the active model.`) with:
```python
        # 2-5. Build the shared db/prompt-builder/agent stack.
        logger.info("Building application stack...")
        stack = build_app_stack(config)
        db = stack.db
        checkpointer_pool = stack.checkpointer_pool
        prompt_builder = stack.prompt_builder
        bot_agent = stack.agent
        logger.info(f"Active model: {bot_agent.model_name}")
```

The rest of `main()` (building the Telegram `Application`, calling `handlers.init_handlers(config, db, bot_agent, prompt_builder, bot_username)`, registering handlers, signal handling, `run_polling`) is unchanged — it already only references the `db`, `bot_agent`, `checkpointer_pool`, and `prompt_builder` names this block still defines.

- [ ] **Step 3: Update scripts/chat_cli.py to use it**

Replace these imports at the top of `scripts/chat_cli.py`:
```python
from config import config
from database import Database
from prompt_builder import PromptBuilder
from agent import Agent, CompletionError
from model_registry import MODEL_PROVIDERS
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres import PostgresSaver
```
with:
```python
from config import config
from agent import CompletionError
from app_factory import build_app_stack
from model_registry import MODEL_PROVIDERS
```

In `ChatCLI.__init__`, replace:
```python
        # Initialize components (mirroring bot.py setup)
        logger.info("Initializing components...")
        config.validate()

        # Database
        self.db = Database(config.DATABASE_URL)

        # Load persisted active model (seeds from DEFAULT_MODEL on first run)
        self.db.init_active_model(config.DEFAULT_MODEL)
        effective_model = self.db.get_active_model()

        # Checkpointer pool (tables created out-of-band; do NOT call .setup()).
        self.checkpointer_pool = ConnectionPool(
            conninfo=config.DATABASE_URL,
            max_size=10,
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        )
        checkpointer = PostgresSaver(self.checkpointer_pool)

        # Prompt builder + agent (mirrors bot.py wiring).
        self.prompt_builder = PromptBuilder(
            default_private_prompt=agent_module.SYSTEM_PROMPT,
            default_group_prompt=agent_module.SYSTEM_PROMPT_GROUP,
            get_active_personality=self.db.get_active_personality,
            get_personality_prompt=self.db.get_personality_prompt,
        )
        self.agent = Agent(
            config=config,
            prompt_builder=self.prompt_builder,
            checkpointer=checkpointer,
            model_name=effective_model,
            db=self.db,
        )

        logger.info(f"CLI initialized for chat_id={self.chat_id}, group={is_group}, test_mode={self.is_test_mode}")
```
with:
```python
        # Initialize components via the shared app_factory stack (mirrors bot.py).
        logger.info("Initializing components...")
        config.validate()

        stack = build_app_stack(config)
        self.db = stack.db
        self.checkpointer_pool = stack.checkpointer_pool
        self.prompt_builder = stack.prompt_builder
        self.agent = stack.agent

        logger.info(f"CLI initialized for chat_id={self.chat_id}, group={is_group}, test_mode={self.is_test_mode}")
```

- [ ] **Step 4: Update pyproject.toml's mypy files list**

Add `"app_factory.py",` to the `[tool.mypy]` `files` list in `pyproject.toml`, and add `app_factory.py` to the `ruff check` file list in both `.github/workflows/ci.yml`'s "Lint (new modules)" step and Task 17's local `ruff check` command.

- [ ] **Step 5: Run the full suite**

Run: `pytest tests/ -v`
Expected: all tests pass (no test constructs `bot.py`'s `main()` or `ChatCLI` directly, so nothing in the suite exercises `app_factory.py` besides import-time compilation).

- [ ] **Step 6: Compile check**

Run: `python3 -m py_compile *.py scripts/*.py alembic/env.py alembic/versions/*.py`
Expected: no output, exit code 0.

- [ ] **Step 7: Manual smoke test (requires a real `.env` and database — not run by CI)**

Run: `python3 scripts/chat_cli.py --chat-id test`
Expected: CLI starts, prints `Model: <active model>`, and `chatgpt hello` gets a real reply — confirms `build_app_stack` wires an equivalent, working stack.

- [ ] **Step 8: Commit**

```bash
git add app_factory.py bot.py scripts/chat_cli.py pyproject.toml .github/workflows/ci.yml
git commit -m "refactor: consolidate db/prompt-builder/agent composition into app_factory.py"
```

---

## Self-Review Notes

- **Spec coverage:** every bullet in the user's recommended approach maps to a task — characterization tests (1–2), remove handler globals (3–7), separate handler responsibilities into `message_handlers.py`/`command_handlers.py`/`authorization.py`/`request_processor.py`/facade `handlers.py` (3–7), reduce transport-to-agent coupling via `model_registry.py`/`token_budget.py` (8–9), split database into repositories behind a facade (10–15), remove legacy APIs (16), Ruff + type checker scoped initially to refactored modules (17), consolidate composition (18).
- **Deliberate deviation from the literal PR-sequence wording:** the user's recommended approach lists `handlers.py` as ending up as a "registration/public compatibility facade" — this plan treats that as the permanent target architecture (Task 7), not a temporary shim to delete later, since `CLAUDE.md` documents `handlers.py` as the module `bot.py` imports. `cleanup_old_group_messages` is kept rather than removed in Task 16, since no superseding retention design is in scope here (see Task 16's rationale) and deleting it would break `tests/test_context_message_retention.py`'s existing assertion that it stays uncalled.
- **Preserved a subtle existing quirk on purpose:** `message_handler`'s triggering path stores `content=message.text` (the raw text, keyword included) but sizes it with `count_tokens(prompt)` (the keyword-stripped text) — Task 5's `_build_payload` reproduces this exact mismatch rather than "fixing" it, since this plan's constraint is zero behavior change.
- **Type consistency check:** `RequestProcessor.process`'s `build_payload` contract (`async () -> (content, token_count, human_message)`) is defined once in Task 4 and consumed identically by both `message_handler` and `photo_handler` in Task 5; `HandlerDependencies` field names (`config`, `db`, `agent`, `prompt_builder`, `bot_username`) are defined in Task 3 and used identically in Tasks 4–7; repository constructor signatures (`MessageRepository(conn)`, `AccessRepository(conn, cache)`, `SettingsRepository(conn, cache)`, `SummaryAuditRepository(conn)`) are defined in Tasks 11–14 and match exactly how `database.py`'s `Database.__init__` constructs them in Task 15.

---

## Execution Handoff

Two execution options:

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
