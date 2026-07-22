# Image Persistence and Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist Telegram images durably in Postgres, keep a compact `[image #N] <summary>` marker in rolling checkpoint context, and give the agent a `get_image(N)` tool to pull the full image back when the summary is not enough.

**Architecture:** The triggering turn is unchanged — the raw image still goes to the reply model at full fidelity. After the reply is sent, a fail-open post-success hook runs a small vision model to describe the image, stores bytes+summary in a new `images` table, and rewrites the photo's checkpoint message in place (same message id) to the compact marker. A new `get_image` tool, scoped to the current chat via runtime context, returns the full image as a multimodal tool result on demand.

**Tech Stack:** Python 3.12+, LangChain 1.0 (`create_agent`, `@tool`, `ToolRuntime`, multimodal tool returns), LangGraph 1.0 `PostgresSaver` checkpointer, psycopg2 + Neon Postgres (`bytea`), Alembic migrations, pytest.

## Global Constraints

- Target Python 3.12+; match existing async patterns.
- LangChain/LangGraph pinned `>=1.0,<2.0` — no APIs outside that range.
- Only models present in `model_registry.MODEL_PROVIDERS` are valid anywhere.
- New model config follows the `SUMMARY_MODEL` precedent: fixed, independent of `/model`, and its provider key being absent must **not** block startup (fail-open).
- Every new persistence path that runs *after* a reply is sent must be fail-open: log and drop on any exception, never raise to the caller, never surface to the user.
- Tests are pure-logic only: no live DB, Telegram, or API calls. Use the fake-connection / fake-model doubles already established in `tests/`.
- 4-space indent, `snake_case` funcs/vars, `PascalCase` classes, `UPPER_CASE` constants; short docstrings on public methods.

---

## File Structure

- Create `alembic/versions/0003_images.py` — `images` table migration.
- Create `database/image_repository.py` — `ImageRecord` dataclass + `ImageRepository` (`save_image`, `get_image`).
- Modify `database/__init__.py` — wire `ImageRepository` into the `Database` facade.
- Modify `config.py` — add `VISION_SUMMARY_MODEL`.
- Modify `model_registry.py` — (no change; referenced only.)
- Create `image_store.py` — `make_image_summary()` helper + `build_image_tool(db)` returning the `get_image` tool.
- Modify `tools.py` — `build_tools(config, db=None)`, append `get_image` when `db` present.
- Modify `agent.py` — `make_vision_summary_model()` + fail-open builder, wire vision model into `Agent`, `Agent.persist_image()`, pass `db` to `build_tools`.
- Modify `prompt_builder.py` — `to_lc_human_message(message_id=...)` + `get_image` system-prompt hint.
- Modify `handlers/request_processor.py` — optional `post_success` hook.
- Modify `handlers/message_handlers.py` — `photo_handler` assigns a stable message id and passes a `post_success` closure.
- Modify `.env.example` and `CLAUDE.md` — document `VISION_SUMMARY_MODEL`.
- Tests: extend `tests/test_repositories.py`, `tests/test_config.py`, `tests/test_prompt_builder.py`, `tests/test_tools.py`, `tests/test_request_processor.py`; add `tests/test_image_store.py`, `tests/test_agent.py` cases.

---

## Task 1: `images` table + repository + facade

**Files:**
- Create: `alembic/versions/0003_images.py`
- Create: `database/image_repository.py`
- Modify: `database/__init__.py`
- Test: `tests/test_repositories.py`

**Interfaces:**
- Produces:
  - `database.image_repository.ImageRecord` — dataclass `(id: int, chat_id: str, mime_type: str, caption: str | None, summary: str, image_bytes: bytes)`.
  - `ImageRepository.save_image(chat_id: str, message_id: int | None, mime_type: str, caption: str | None, summary: str, image_bytes: bytes) -> int`
  - `ImageRepository.get_image(chat_id: str, image_id: int) -> ImageRecord | None`
  - `Database.save_image(...)` / `Database.get_image(...)` — same signatures, via the facade.

- [ ] **Step 1: Write the migration**

Create `alembic/versions/0003_images.py`:

```python
"""Add images table for durable image persistence

Revision ID: 0003
Revises: 0002
Create Date: 2026-07-23

"""
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE images (
            id BIGSERIAL PRIMARY KEY,
            chat_id TEXT NOT NULL,
            message_id BIGINT,
            mime_type TEXT NOT NULL,
            caption TEXT,
            summary TEXT NOT NULL,
            image_bytes BYTEA NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    op.execute("CREATE INDEX idx_images_chat_id ON images(chat_id, id)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_images_chat_id")
    op.execute("DROP TABLE IF EXISTS images")
```

- [ ] **Step 2: Write the failing repository test**

Add to `tests/test_repositories.py` (reuse the existing `_FakeConn` / `_FakeCursor` / a `ConnectionManager`-shaped double already in that file; follow how `MessageRepository` tests build one). Add at the end of the file:

```python
from database.image_repository import ImageRecord, ImageRepository


def test_save_image_inserts_and_returns_id():
    conn = _FakeConn(results=[(42,)])
    repo = ImageRepository(_FakeConnectionManager(conn))
    new_id = repo.save_image(
        chat_id="123", message_id=7, mime_type="image/jpeg",
        caption="a cat", summary="A tabby cat on a sofa.", image_bytes=b"\x00\x01",
    )
    assert new_id == 42
    sql, params = conn.executed[-1]
    assert "INSERT INTO images" in sql
    assert params == ("123", 7, "image/jpeg", "a cat", "A tabby cat on a sofa.", b"\x00\x01")


def test_get_image_returns_record_when_found():
    conn = _FakeConn(results=[("123", 7, "image/jpeg", "a cat", "summary", b"\x00\x01")])
    repo = ImageRepository(_FakeConnectionManager(conn))
    record = repo.get_image("123", 42)
    assert isinstance(record, ImageRecord)
    assert record.id == 42
    assert record.chat_id == "123"
    assert record.image_bytes == b"\x00\x01"
    sql, params = conn.executed[-1]
    assert "SELECT" in sql and "FROM images" in sql
    assert params == (42, "123")


def test_get_image_returns_none_when_missing_or_other_chat():
    conn = _FakeConn(results=[None])
    repo = ImageRepository(_FakeConnectionManager(conn))
    assert repo.get_image("123", 999) is None
```

If `tests/test_repositories.py` does not already define a `_FakeConnectionManager` (a double whose `.connection()` context manager yields the `_FakeConn`), add this helper near `_FakeConn` in that file:

```python
class _FakeConnectionManager:
    def __init__(self, conn):
        self._conn = conn

    @contextmanager
    def connection(self):
        yield self._conn
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `venv/bin/python3.12 -m pytest tests/test_repositories.py -k image -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'database.image_repository'`.

- [ ] **Step 4: Write the repository**

Create `database/image_repository.py`:

```python
"""Durable image blob persistence, keyed by chat for scoped retrieval."""
import logging
from dataclasses import dataclass

from psycopg2.extras import RealDictCursor

from .db_connection import ConnectionManager

logger = logging.getLogger(__name__)


@dataclass
class ImageRecord:
    id: int
    chat_id: str
    mime_type: str
    caption: str | None
    summary: str
    image_bytes: bytes


class ImageRepository:
    """CRUD for the `images` table. Retrieval is always chat-scoped."""

    def __init__(self, conn: ConnectionManager):
        self._conn = conn

    def save_image(
        self,
        chat_id: str,
        message_id: int | None,
        mime_type: str,
        caption: str | None,
        summary: str,
        image_bytes: bytes,
    ) -> int:
        """Persist one image and return its stable id."""
        chat_id = str(chat_id)
        with self._conn.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO images
                    (chat_id, message_id, mime_type, caption, summary, image_bytes)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (chat_id, message_id, mime_type, caption, summary, image_bytes),
                )
                row_id = cur.fetchone()[0]
        logger.info("Saved image %s for chat %s", row_id, chat_id)
        return row_id

    def get_image(self, chat_id: str, image_id: int) -> ImageRecord | None:
        """Fetch one image by id, scoped to chat_id. Returns None if the id
        does not exist OR belongs to a different chat (the isolation boundary)."""
        chat_id = str(chat_id)
        with self._conn.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT chat_id, message_id, mime_type, caption, summary, image_bytes
                    FROM images
                    WHERE id = %s AND chat_id = %s
                    """,
                    (image_id, chat_id),
                )
                row = cur.fetchone()
        if row is None:
            return None
        return ImageRecord(
            id=image_id,
            chat_id=row[0],
            mime_type=row[2],
            caption=row[3],
            summary=row[4],
            image_bytes=bytes(row[5]),
        )
```

- [ ] **Step 5: Wire the facade**

In `database/__init__.py`, add the import alongside the others:

```python
from .image_repository import ImageRepository
```

In `Database.__init__`, after `self._summaries = SummaryAuditRepository(self._conn)`:

```python
        self._images = ImageRepository(self._conn)
```

Add a new facade section after the summary-audit section:

```python
    # --- images -------------------------------------------------------------
    def save_image(self, *args, **kwargs) -> int:
        return self._images.save_image(*args, **kwargs)

    def get_image(self, *args, **kwargs):
        return self._images.get_image(*args, **kwargs)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `venv/bin/python3.12 -m pytest tests/test_repositories.py -k image -v`
Expected: PASS (3 tests).

- [ ] **Step 7: Compile check + commit**

```bash
venv/bin/python3.12 -m py_compile database/image_repository.py database/__init__.py alembic/versions/0003_images.py
git add alembic/versions/0003_images.py database/image_repository.py database/__init__.py tests/test_repositories.py
git commit -m "Add images table, repository, and Database facade methods"
```

---

## Task 2: `VISION_SUMMARY_MODEL` config + model builder

**Files:**
- Modify: `config.py:36` (near `SUMMARY_MODEL`)
- Modify: `agent.py` (add `make_vision_summary_model` + fail-open `_build_vision_model`)
- Test: `tests/test_config.py`, `tests/test_agent.py`

**Interfaces:**
- Consumes: `model_registry.resolve_model`, `model_registry.provider_api_key`, `config.MODEL_TIMEOUT`.
- Produces:
  - `config.Config.VISION_SUMMARY_MODEL: str` (default `"gpt-4.1-mini"`).
  - `agent.make_vision_summary_model(config)` — returns a chat model, raises `ValueError` for an unsupported model or a missing provider key.
  - `agent._build_vision_model(config)` — returns the model or `None` (fail-open), never raises.

- [ ] **Step 1: Write the failing config test**

Add to `tests/test_config.py`:

```python
def test_vision_summary_model_defaults(monkeypatch):
    monkeypatch.delenv("VISION_SUMMARY_MODEL", raising=False)
    import importlib
    import config as config_module
    importlib.reload(config_module)
    assert config_module.Config.VISION_SUMMARY_MODEL == "gpt-4.1-mini"
```

(If `tests/test_config.py` already has a reload helper/fixture for env-driven defaults, use that pattern instead of re-importing inline.)

- [ ] **Step 2: Run it to verify it fails**

Run: `venv/bin/python3.12 -m pytest tests/test_config.py -k vision -v`
Expected: FAIL with `AttributeError: type object 'Config' has no attribute 'VISION_SUMMARY_MODEL'`.

- [ ] **Step 3: Add the config var**

In `config.py`, immediately after line 36 (`SUMMARY_MODEL = os.getenv(...)`):

```python
    # Dedicated vision model that describes images on ingest so later turns
    # keep a text description. Fixed, independent of /model and SUMMARY_MODEL.
    VISION_SUMMARY_MODEL = os.getenv("VISION_SUMMARY_MODEL", "gpt-4.1-mini")
```

Do **not** add it to `validate()` — a missing provider key must not block startup (it degrades to fail-open at runtime).

- [ ] **Step 4: Run the config test to verify it passes**

Run: `venv/bin/python3.12 -m pytest tests/test_config.py -k vision -v`
Expected: PASS.

- [ ] **Step 5: Write the failing agent test**

Add to `tests/test_agent.py`:

```python
import agent as agent_module


class _CfgVision:
    VISION_SUMMARY_MODEL = "gpt-4.1-mini"
    OPENAI_API_KEY = ""
    XAI_API_KEY = ""
    GEMINI_API_KEY = ""
    MODEL_TIMEOUT = 60


def test_make_vision_summary_model_rejects_unsupported():
    cfg = _CfgVision()
    cfg.VISION_SUMMARY_MODEL = "not-a-real-model"
    import pytest
    with pytest.raises(ValueError):
        agent_module.make_vision_summary_model(cfg)


def test_build_vision_model_fail_open_on_missing_key():
    # Supported model but no provider key -> None, no exception.
    assert agent_module._build_vision_model(_CfgVision()) is None
```

- [ ] **Step 6: Run it to verify it fails**

Run: `venv/bin/python3.12 -m pytest tests/test_agent.py -k vision -v`
Expected: FAIL with `AttributeError: module 'agent' has no attribute 'make_vision_summary_model'`.

- [ ] **Step 7: Add the builders to `agent.py`**

Directly after `make_summary_model` (ends at `agent.py:138`), add:

```python
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
```

- [ ] **Step 8: Run the agent test to verify it passes**

Run: `venv/bin/python3.12 -m pytest tests/test_agent.py -k vision -v`
Expected: PASS (2 tests).

- [ ] **Step 9: Compile check + commit**

```bash
venv/bin/python3.12 -m py_compile config.py agent.py
git add config.py agent.py tests/test_config.py tests/test_agent.py
git commit -m "Add VISION_SUMMARY_MODEL config and fail-open model builders"
```

---

## Task 3: `image_store.py` — summary helper + `get_image` tool

**Files:**
- Create: `image_store.py`
- Test: `tests/test_image_store.py`

**Interfaces:**
- Consumes: `token_budget._message_text`, `database` `ImageRecord`-shaped objects from `db.get_image`.
- Produces:
  - `image_store.IMAGE_SUMMARY_PROMPT: str`
  - `image_store.make_image_summary(model, image_data_url: str) -> str` — sync; calls `model.invoke([...])`, returns flattened text.
  - `image_store.build_image_blocks(db, chat_id: str | None, image_id: int) -> list[dict]` — pure, directly-testable core of the tool.
  - `image_store.build_image_tool(db)` — returns a LangChain `@tool` named `get_image` taking `(image_id: int)` (+ injected `ToolRuntime`); delegates to `build_image_blocks`.

Note: `ToolRuntime` is an *injected* argument (supplied by the agent at call time, not part of the tool's input schema), so the tool wrapper itself is not conveniently unit-testable via `tool.invoke(...)`. All retrieval logic therefore lives in the plain `build_image_blocks` function, which the tests exercise directly; the tool wrapper only extracts `chat_id` from the runtime and delegates.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_image_store.py`:

```python
"""Image summary helper + get_image tool: chat-scoped retrieval, multimodal
return shape, and text-flattening of the vision model reply."""
from dataclasses import dataclass
from types import SimpleNamespace

from langchain_core.messages import AIMessage

import image_store


@dataclass
class _Rec:
    id: int
    chat_id: str
    mime_type: str
    caption: str | None
    summary: str
    image_bytes: bytes


class _FakeModel:
    def __init__(self, reply):
        self._reply = reply
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content=self._reply)


def test_make_image_summary_returns_text():
    model = _FakeModel("A tabby cat on a sofa.")
    out = image_store.make_image_summary(model, "data:image/jpeg;base64,AAAA")
    assert out == "A tabby cat on a sofa."
    # The image data-URL was sent as an image_url block.
    sent = model.calls[0][0].content
    assert any(b.get("type") == "image_url" for b in sent)


def test_build_image_blocks_returns_multimodal_for_matching_chat():
    db = SimpleNamespace(
        get_image=lambda chat_id, image_id: _Rec(
            image_id, chat_id, "image/jpeg", "a cat", "summary", b"\x00\x01")
    )
    result = image_store.build_image_blocks(db, "123", 42)
    types = [b["type"] for b in result]
    assert types == ["text", "image"]
    assert result[0]["text"] == "Image #42 (a cat):"
    assert result[1]["mime_type"] == "image/jpeg"
    assert result[1]["base64"] == "AAE="  # base64 of b"\x00\x01"


def test_build_image_blocks_not_found_returns_text_only():
    db = SimpleNamespace(get_image=lambda chat_id, image_id: None)
    result = image_store.build_image_blocks(db, "123", 999)
    assert result == [{"type": "text", "text": "Image #999 not found."}]


def test_build_image_blocks_none_chat_returns_unavailable():
    db = SimpleNamespace(get_image=lambda chat_id, image_id: None)
    result = image_store.build_image_blocks(db, None, 5)
    assert result == [{"type": "text", "text": "Image not available."}]


def test_build_image_tool_is_named_get_image():
    tool = image_store.build_image_tool(SimpleNamespace(get_image=lambda *a: None))
    assert tool.name == "get_image"
```

- [ ] **Step 2: Run to verify it fails**

Run: `venv/bin/python3.12 -m pytest tests/test_image_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'image_store'`.

- [ ] **Step 3: Write `image_store.py`**

Create `image_store.py`:

```python
"""Vision summary helper and the get_image agent tool.

Kept separate from tools.py (web search / fetch) so image concerns stay
in one focused module. build_image_tool binds a Database so the tool can
scope retrieval to the calling chat via runtime context."""
from __future__ import annotations

import base64
import logging

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import HumanMessage

from token_budget import _message_text

logger = logging.getLogger(__name__)

IMAGE_SUMMARY_PROMPT = (
    "Describe this image in 2-4 sentences. Note key objects, any visible text, "
    "layout, and notable details someone might ask about later. Be concise and factual."
)


def make_image_summary(model, image_data_url: str) -> str:
    """Run the vision model on one image and return its flattened text reply."""
    message = HumanMessage(content=[
        {"type": "text", "text": IMAGE_SUMMARY_PROMPT},
        {"type": "image_url", "image_url": {"url": image_data_url}},
    ])
    response = model.invoke([message])
    return _message_text(response).strip()


def build_image_blocks(db, chat_id, image_id: int) -> list[dict]:
    """Pure, directly-testable core of get_image: chat-scoped lookup -> blocks."""
    if chat_id is None:
        return [{"type": "text", "text": "Image not available."}]
    record = db.get_image(chat_id, image_id)
    if record is None:
        return [{"type": "text", "text": f"Image #{image_id} not found."}]
    b64 = base64.b64encode(record.image_bytes).decode("utf-8")
    caption = f" ({record.caption})" if record.caption else ""
    return [
        {"type": "text", "text": f"Image #{image_id}{caption}:"},
        {"type": "image", "base64": b64, "mime_type": record.mime_type},
    ]


def build_image_tool(db):
    """Return a get_image tool bound to db, scoped to the calling chat."""

    @tool
    def get_image(image_id: int, runtime: ToolRuntime) -> list[dict]:
        """Retrieve a previously shared image so you can see the full picture.

        Use this when an [image #N] marker's text description is not enough
        to answer a question about that image.

        Args:
            image_id: The numeric id from an [image #N] marker.
        """
        context = getattr(runtime, "context", None)
        chat_id = getattr(context, "thread_id", None)
        return build_image_blocks(db, chat_id, image_id)

    return get_image
```

- [ ] **Step 4: Run to verify it passes**

Run: `venv/bin/python3.12 -m pytest tests/test_image_store.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Compile check + commit**

```bash
venv/bin/python3.12 -m py_compile image_store.py
git add image_store.py tests/test_image_store.py
git commit -m "Add image_store: vision summary helper and get_image tool"
```

---

## Task 4: Wire `get_image` into the agent's tool set

**Files:**
- Modify: `tools.py:63-71`
- Modify: `agent.py:181`
- Test: `tests/test_tools.py`

**Interfaces:**
- Consumes: `image_store.build_image_tool(db)` (Task 3), `Agent._db`.
- Produces: `tools.build_tools(config, db=None) -> list` — includes `get_image` iff `db is not None`.

- [ ] **Step 1: Update the failing tool test**

In `tests/test_tools.py`, replace `test_build_tools_returns_search_and_fetch` and add a new case:

```python
def test_build_tools_returns_search_and_fetch():
    built = tools.build_tools(_CfgNoKey, db=None)
    names = {t.name for t in built}
    assert "fetch_url" in names
    assert any("search" in n for n in names)
    assert "get_image" not in names


def test_build_tools_includes_get_image_when_db_present():
    from types import SimpleNamespace
    built = tools.build_tools(_CfgNoKey, db=SimpleNamespace(get_image=lambda *a: None))
    names = {t.name for t in built}
    assert "get_image" in names
```

- [ ] **Step 2: Run to verify it fails**

Run: `venv/bin/python3.12 -m pytest tests/test_tools.py -v`
Expected: FAIL — `test_build_tools_includes_get_image_when_db_present` fails because `build_tools` takes no `db` param / `get_image` is absent.

- [ ] **Step 3: Update `build_tools`**

In `tools.py`, replace the `build_tools` function (lines 63-71):

```python
def build_tools(config, db=None) -> list:
    """Assemble the agent's tool set based on configuration.

    When db is provided, includes the get_image retrieval tool."""
    if web_search_backend(config) == "tavily":
        from langchain_tavily import TavilySearch
        search = TavilySearch(max_results=5, tavily_api_key=config.TAVILY_API_KEY)
    else:
        logger.info("TAVILY_API_KEY not set — using DuckDuckGo web search")
        search = _duckduckgo_search_tool()
    built = [search, fetch_url]
    if db is not None:
        from image_store import build_image_tool
        built.append(build_image_tool(db))
    return built
```

- [ ] **Step 4: Pass db from the agent**

In `agent.py:181`, change:

```python
        self._tools = build_tools(config)  # from tools.py
```

to:

```python
        self._tools = build_tools(config, db)  # from tools.py
```

(`db` is the `Agent.__init__` parameter, already in scope at that line.)

- [ ] **Step 5: Run to verify it passes**

Run: `venv/bin/python3.12 -m pytest tests/test_tools.py -v`
Expected: PASS.

- [ ] **Step 6: Compile check + commit**

```bash
venv/bin/python3.12 -m py_compile tools.py agent.py
git add tools.py agent.py tests/test_tools.py
git commit -m "Wire get_image tool into agent tool set when db is present"
```

---

## Task 5: `to_lc_human_message(message_id=...)` + system-prompt hint

**Files:**
- Modify: `prompt_builder.py:71-134`
- Test: `tests/test_prompt_builder.py`

**Interfaces:**
- Produces:
  - `PromptBuilder.to_lc_human_message(text=None, is_group=False, sender_name="Unknown", image_data_url=None, message_id=None) -> HumanMessage` — sets `HumanMessage.id` when `message_id` is provided.
  - `build_system_prompt(...)` output now contains a one-line hint about `[image #N]` markers and `get_image`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_prompt_builder.py`:

```python
def test_to_lc_human_message_sets_id_when_provided():
    msg = _pb().to_lc_human_message(text="hi", message_id="abc-123")
    assert msg.id == "abc-123"


def test_to_lc_human_message_no_id_by_default():
    msg = _pb().to_lc_human_message(text="hi")
    assert msg.id is None


def test_system_prompt_mentions_get_image():
    out = _pb().build_system_prompt(is_group=False)
    assert "get_image" in out
```

- [ ] **Step 2: Run to verify it fails**

Run: `venv/bin/python3.12 -m pytest tests/test_prompt_builder.py -k "id or get_image" -v`
Expected: FAIL — `to_lc_human_message` has no `message_id` kwarg / prompt lacks `get_image`.

- [ ] **Step 3: Add the `message_id` param**

In `prompt_builder.py`, replace `to_lc_human_message` (lines 117-134):

```python
    def to_lc_human_message(
        self,
        text: str | None = None,
        is_group: bool = False,
        sender_name: str = "Unknown",
        image_data_url: str | None = None,
        message_id: str | None = None,
    ) -> HumanMessage:
        """Build a LangChain HumanMessage from an incoming Telegram message.

        When message_id is given, it is set as the message's stable id so the
        message can later be rewritten in place in the checkpoint."""
        body = text or ""
        if is_group and body:
            body = self._group_prefix(body, sender_name)

        extra = {"id": message_id} if message_id is not None else {}
        if image_data_url:
            return HumanMessage(content=[
                {"type": "text", "text": body or "What's in this image?"},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ], **extra)
        return HumanMessage(content=body, **extra)
```

- [ ] **Step 4: Add the system-prompt hint**

In `prompt_builder.py`, inside `build_system_prompt`, insert this just before the final `return "".join(system_parts)` (after the `reply_context` block, ~line 108):

```python
        system_parts.append(
            "\n\nAn [image #N] marker refers to an image shared earlier in the "
            "conversation. Call get_image(N) to view the full image if the "
            "text description is not enough to answer."
        )
```

- [ ] **Step 5: Run to verify it passes**

Run: `venv/bin/python3.12 -m pytest tests/test_prompt_builder.py -v`
Expected: PASS (all, including pre-existing cases).

- [ ] **Step 6: Compile check + commit**

```bash
venv/bin/python3.12 -m py_compile prompt_builder.py
git add prompt_builder.py tests/test_prompt_builder.py
git commit -m "Support stable message id and add get_image system-prompt hint"
```

---

## Task 6: `Agent.persist_image` — vision summary → store → checkpoint rewrite

**Files:**
- Modify: `agent.py` (imports; `Agent.__init__`; new `_image_marker` + `Agent.persist_image`)
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: `image_store.make_image_summary` (Task 3), `Database.save_image` (Task 1), `_build_vision_model` (Task 2), `PromptBuilder.to_lc_human_message` id support (Task 5).
- Produces:
  - `Agent._vision_summary_model` (built in `__init__`, may be `None`).
  - `agent._image_marker(image_id: int, caption: str | None, summary: str) -> str`
  - `Agent.persist_image(chat_id, image_message_id: str, image_data_url: str, mime_type: str, caption: str | None, telegram_message_id: int | None) -> None` — async, fail-open, never raises.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_agent.py`:

```python
import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import agent as agent_module


def test_image_marker_with_and_without_caption():
    assert agent_module._image_marker(5, None, "a cat") == "[image #5] a cat"
    assert agent_module._image_marker(5, "pets", "a cat") == "[image #5] pets — a cat"


def _agent_for_persist(vision_model, db, graph):
    a = agent_module.Agent.__new__(agent_module.Agent)
    a._vision_summary_model = vision_model
    a._db = db
    a._graph = graph
    return a


def test_persist_image_stores_and_rewrites_checkpoint(monkeypatch):
    monkeypatch.setattr(agent_module, "make_image_summary", lambda m, url: "A tabby cat.")
    db = SimpleNamespace(save_image=Mock(return_value=77))
    graph = SimpleNamespace(update_state=Mock())
    a = _agent_for_persist(vision_model=object(), db=db, graph=graph)

    asyncio.run(a.persist_image(
        chat_id="123", image_message_id="mid-1",
        image_data_url="data:image/jpeg;base64,AAAA",
        mime_type="image/jpeg", caption="pets", telegram_message_id=9,
    ))

    db.save_image.assert_called_once()
    _, kwargs = db.save_image.call_args
    assert kwargs["summary"] == "A tabby cat."
    assert kwargs["chat_id"] == "123"
    graph.update_state.assert_called_once()
    _, rewrite = graph.update_state.call_args[0]
    rewritten = rewrite["messages"][0]
    assert rewritten.id == "mid-1"
    assert rewritten.content == "[image #77] pets — A tabby cat."


def test_persist_image_fail_open_when_no_vision_model():
    db = SimpleNamespace(save_image=Mock())
    graph = SimpleNamespace(update_state=Mock())
    a = _agent_for_persist(vision_model=None, db=db, graph=graph)
    asyncio.run(a.persist_image(
        chat_id="123", image_message_id="mid-1",
        image_data_url="data:image/jpeg;base64,AAAA",
        mime_type="image/jpeg", caption=None, telegram_message_id=9,
    ))
    db.save_image.assert_not_called()
    graph.update_state.assert_not_called()


def test_persist_image_fail_open_on_empty_summary(monkeypatch):
    monkeypatch.setattr(agent_module, "make_image_summary", lambda m, url: "   ")
    db = SimpleNamespace(save_image=Mock())
    graph = SimpleNamespace(update_state=Mock())
    a = _agent_for_persist(vision_model=object(), db=db, graph=graph)
    asyncio.run(a.persist_image(
        chat_id="123", image_message_id="mid-1",
        image_data_url="data:image/jpeg;base64,AAAA",
        mime_type="image/jpeg", caption=None, telegram_message_id=9,
    ))
    db.save_image.assert_not_called()
    graph.update_state.assert_not_called()
```

- [ ] **Step 2: Run to verify it fails**

Run: `venv/bin/python3.12 -m pytest tests/test_agent.py -k "image_marker or persist_image" -v`
Expected: FAIL — `_image_marker` / `persist_image` do not exist.

- [ ] **Step 3: Add imports**

In `agent.py`, add `import base64` to the stdlib imports (after `import asyncio`), and add `HumanMessage` to the `langchain_core.messages` import (line 12):

```python
from langchain_core.messages import BaseMessage, HumanMessage, RemoveMessage
```

Add, next to the other cross-module imports (after `from tools import build_tools`, line 21):

```python
from image_store import make_image_summary
```

- [ ] **Step 4: Build the vision model in `__init__`**

In `Agent.__init__`, right after `self._summary_model = summary_model or make_summary_model(config)` (line 183):

```python
        self._vision_summary_model = _build_vision_model(config)
```

- [ ] **Step 5: Add `_image_marker` and `persist_image`**

Add the module-level helper next to the other module helpers (e.g. after `_build_vision_model` from Task 2):

```python
def _image_marker(image_id: int, caption: str | None, summary: str) -> str:
    """Compact checkpoint marker that replaces a raw image after ingest."""
    prefix = f"[image #{image_id}]"
    if caption:
        return f"{prefix} {caption} — {summary}"
    return f"{prefix} {summary}"
```

Add `persist_image` as a method on `Agent`, after `append_context_message` (`agent.py:355-364`):

```python
    async def persist_image(
        self,
        chat_id,
        image_message_id: str,
        image_data_url: str,
        mime_type: str,
        caption: str | None,
        telegram_message_id: int | None,
    ) -> None:
        """Fail-open post-reply step: describe the image, store it durably, and
        replace its checkpoint message in place with a compact [image #id]
        marker. Never raises — a failure just leaves the raw image as-is."""
        if self._graph is None or self._vision_summary_model is None or self._db is None:
            return
        try:
            summary = await asyncio.to_thread(
                make_image_summary, self._vision_summary_model, image_data_url
            )
            if not summary:
                logger.warning(
                    "Empty image summary for chat %s; skipping image persist", chat_id
                )
                return
            raw = base64.b64decode(image_data_url.split(",", 1)[1])
            image_id = self._db.save_image(
                chat_id=str(chat_id),
                message_id=telegram_message_id,
                mime_type=mime_type,
                caption=caption,
                summary=summary,
                image_bytes=raw,
            )
            self._graph.update_state(
                self._config_for(chat_id),
                {"messages": [HumanMessage(
                    id=image_message_id,
                    content=_image_marker(image_id, caption, summary),
                )]},
            )
            logger.info("Persisted image %s for chat %s", image_id, chat_id)
        except Exception:
            logger.exception("Failed to persist image for chat %s", chat_id)
```

- [ ] **Step 6: Run to verify it passes**

Run: `venv/bin/python3.12 -m pytest tests/test_agent.py -k "image_marker or persist_image" -v`
Expected: PASS (4 tests).

- [ ] **Step 7: Compile check + commit**

```bash
venv/bin/python3.12 -m py_compile agent.py
git add agent.py tests/test_agent.py
git commit -m "Add Agent.persist_image: vision summary, store, checkpoint rewrite"
```

---

## Task 7: `RequestProcessor` post-success hook

**Files:**
- Modify: `handlers/request_processor.py:42-83`
- Test: `tests/test_request_processor.py`

**Interfaces:**
- Produces: `RequestProcessor.process(..., post_success: Callable[[], Awaitable[None]] | None = None)` — awaited after the reply is sent; its failures are logged and swallowed, never affecting the already-delivered reply.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_request_processor.py`:

```python
def test_process_runs_post_success_after_reply():
    calls = []
    db = SimpleNamespace(add_message=Mock())
    agent = SimpleNamespace(run=AsyncMock(return_value="reply text"))
    processor = RequestProcessor(_deps(db=db, agent=agent))
    message = _message()

    async def _hook():
        calls.append("ran")

    asyncio.run(processor.process(
        _bot(), message, user_id=1, sender_name="Alice", sender_username="alice",
        is_group=False, build_payload=_payload, reply_context=None,
        generic_error_text="generic error", success_log="ok", error_log_prefix="err",
        post_success=_hook,
    ))

    message.reply_text.assert_awaited_once_with("reply text")
    assert calls == ["ran"]


def test_process_post_success_failure_does_not_break_reply():
    db = SimpleNamespace(add_message=Mock())
    agent = SimpleNamespace(run=AsyncMock(return_value="reply text"))
    processor = RequestProcessor(_deps(db=db, agent=agent))
    message = _message()

    async def _hook():
        raise RuntimeError("boom")

    asyncio.run(processor.process(
        _bot(), message, user_id=1, sender_name="Alice", sender_username="alice",
        is_group=False, build_payload=_payload, reply_context=None,
        generic_error_text="generic error", success_log="ok", error_log_prefix="err",
        post_success=_hook,
    ))

    # Reply still went out exactly once; the generic error was NOT sent.
    message.reply_text.assert_awaited_once_with("reply text")
```

- [ ] **Step 2: Run to verify it fails**

Run: `venv/bin/python3.12 -m pytest tests/test_request_processor.py -k post_success -v`
Expected: FAIL — `process()` got an unexpected keyword argument `post_success`.

- [ ] **Step 3: Add the hook parameter**

In `handlers/request_processor.py`, add the parameter to `process` (after `error_log_prefix: str,` in the signature):

```python
        post_success=None,
```

Then, after `logger.info(success_log)` inside the `try` block, add:

```python
                if post_success is not None:
                    try:
                        await post_success()
                    except Exception:
                        logger.exception(
                            "post_success hook failed for chat %s", chat_id
                        )
```

Note: keep this **inside** the `try` but with its own inner `try/except`, so a hook failure can never fall through to the `generic_error_text` reply. Indentation must match the `await message.reply_text(response)` / `logger.info(success_log)` lines (they sit outside the `async with typing_action` block but inside the outer `try`).

- [ ] **Step 4: Run to verify it passes**

Run: `venv/bin/python3.12 -m pytest tests/test_request_processor.py -v`
Expected: PASS (all, including pre-existing cases).

- [ ] **Step 5: Compile check + commit**

```bash
venv/bin/python3.12 -m py_compile handlers/request_processor.py
git add handlers/request_processor.py tests/test_request_processor.py
git commit -m "Add fail-open post_success hook to RequestProcessor"
```

---

## Task 8: Wire `photo_handler` to assign a stable id and persist

**Files:**
- Modify: `handlers/message_handlers.py:3` (imports), `handlers/message_handlers.py:134-180` (`photo_handler`)
- Test: `tests/test_message_handlers.py`

**Interfaces:**
- Consumes: `PromptBuilder.to_lc_human_message(message_id=...)` (Task 5), `Agent.persist_image` (Task 6), `RequestProcessor.process(post_success=...)` (Task 7).
- Produces: no new public surface; `photo_handler` now passes a stable id into the photo `HumanMessage` and a `post_success` closure that calls `agent.persist_image`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_message_handlers.py` (follow the file's existing double-construction pattern for `MessageHandlers`/deps; the snippet below assumes a helper that builds a handler with mock deps — mirror whatever `test_message_handlers.py` already uses to instantiate the handler and a fake photo `message`):

```python
def test_photo_handler_passes_post_success_that_calls_persist_image():
    # Arrange a triggering photo message and capture the process() kwargs.
    captured = {}

    async def _fake_process(bot, message, **kwargs):
        captured.update(kwargs)
        # Simulate a successful turn: invoke the hook the processor would run.
        if kwargs.get("post_success") is not None:
            await kwargs["post_success"]()

    # ... build handler `mh` with deps whose agent has persist_image=AsyncMock(),
    #     prompt_builder.to_lc_human_message=Mock(), bot_username=None, and
    #     is_authorized returning True; set mh._processor.process = _fake_process.
    #     Build a fake `update` whose message.photo[-1].get_file()/download
    #     returns bytes, caption "chatgpt look", chat.type "private".

    # asyncio.run(mh.photo_handler(update, context))

    # Assert the hook wired through to persist_image with the same id used for
    # the HumanMessage.
    assert "post_success" in captured
    mh._deps.agent.persist_image.assert_awaited_once()
    call = mh._deps.agent.persist_image.await_args.kwargs
    id_used = mh._deps.prompt_builder.to_lc_human_message.call_args.kwargs["message_id"]
    assert call["image_message_id"] == id_used
    assert call["mime_type"] == "image/jpeg"
```

If `tests/test_message_handlers.py` has no existing harness that makes this practical, implement it using the same construction the file already uses for `text_handler` tests; the key assertions are (a) `process` received a `post_success`, and (b) invoking it calls `agent.persist_image` with `image_message_id` equal to the id passed to `to_lc_human_message`.

- [ ] **Step 2: Run to verify it fails**

Run: `venv/bin/python3.12 -m pytest tests/test_message_handlers.py -k photo -v`
Expected: FAIL — `photo_handler` does not pass `post_success` / does not set `message_id`.

- [ ] **Step 3: Add the `uuid` import**

In `handlers/message_handlers.py`, add to the imports at the top (after `import re`):

```python
import uuid
```

- [ ] **Step 4: Rewrite `photo_handler`'s payload + process call**

In `handlers/message_handlers.py`, replace the body from `reply_data = extract_reply_data(message)` (line 157) through the end of the `process(...)` call (line 180) with:

```python
        reply_data = extract_reply_data(message)
        image_message_id = str(uuid.uuid4())
        captured: dict[str, str] = {}

        async def _build_payload():
            photo = message.photo[-1]
            photo_file = await photo.get_file()
            photo_bytes = await photo_file.download_as_bytearray()
            base64_image = base64.b64encode(photo_bytes).decode("utf-8")
            image_data_url = f"data:image/jpeg;base64,{base64_image}"
            caption_marker = f"[image] {message.caption}" if message.caption else "[image]"
            human = self._deps.prompt_builder.to_lc_human_message(
                text=prompt, is_group=is_group, sender_name=sender_name,
                image_data_url=image_data_url, message_id=image_message_id)
            captured["image_data_url"] = image_data_url
            return caption_marker, count_tokens(caption_marker), human

        async def _post_success():
            image_data_url = captured.get("image_data_url")
            if not image_data_url:
                return
            await self._deps.agent.persist_image(
                chat_id=chat_id,
                image_message_id=image_message_id,
                image_data_url=image_data_url,
                mime_type="image/jpeg",
                caption=message.caption,
                telegram_message_id=message.message_id,
            )

        await self._processor.process(
            context.bot, message,
            user_id=user_id, sender_name=sender_name, sender_username=sender_username,
            is_group=is_group, build_payload=_build_payload, reply_context=reply_data,
            generic_error_text=(
                "Sorry, I encountered an error processing your image. Please try again."
            ),
            success_log=f"Image processed for chat {chat_id}",
            error_log_prefix="Error processing image",
            post_success=_post_success,
        )
```

- [ ] **Step 5: Run to verify it passes**

Run: `venv/bin/python3.12 -m pytest tests/test_message_handlers.py -v`
Expected: PASS.

- [ ] **Step 6: Full suite + compile check**

```bash
venv/bin/python3.12 -m py_compile handlers/message_handlers.py
venv/bin/python3.12 -m pytest tests/ -v
```
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add handlers/message_handlers.py tests/test_message_handlers.py
git commit -m "Wire photo_handler to persist images after reply"
```

---

## Task 9: Documentation

**Files:**
- Modify: `.env.example`
- Modify: `CLAUDE.md` (Configuration section)

**Interfaces:** none (docs only).

- [ ] **Step 1: Update `.env.example`**

Add near the `SUMMARY_MODEL` entry in `.env.example`:

```bash
# Dedicated vision model used to describe images on ingest so later turns keep
# a text description. Fixed; independent of /model and SUMMARY_MODEL. Must be a
# model listed in model_registry.MODEL_PROVIDERS. Defaults to gpt-4.1-mini.
VISION_SUMMARY_MODEL=
```

(If `.env.example` groups vars by section, place it in the model/summary group.)

- [ ] **Step 2: Update `CLAUDE.md`**

In `CLAUDE.md`, add `VISION_SUMMARY_MODEL` to the "Relevant environment variables" list (after `SUMMARY_CONTEXT_TOKENS`), and add a bullet under "Important notes":

```markdown
- `VISION_SUMMARY_MODEL` is the dedicated model that describes images on ingest;
  it is fixed and independent of `/model` and `SUMMARY_MODEL`. A missing provider
  key does not block startup — image persistence simply fails open.
```

Also add `images` to the "Expected tables" list in the Database Schema section.

- [ ] **Step 3: Commit**

```bash
git add .env.example CLAUDE.md
git commit -m "Document VISION_SUMMARY_MODEL and images table"
```

---

## Manual verification (after all tasks)

Requires a real DB + Telegram credentials (not part of CI). From the migration + checkpointer setup already in the deploy flow:

1. `alembic upgrade head` — confirm the `images` table is created.
2. Send a photo with a triggering caption (e.g. "chatgpt what is this?"). Confirm a normal reply.
3. Send a follow-up text turn that refers to the image ("what colour was it?") **without** re-sending the photo. Confirm the bot answers from the stored summary, and — if pressed for detail it lacks — that it calls `get_image` and answers from the full image.
4. `scripts/chat_cli.py --chat-id test` still starts and replies (regression check on `build_tools(config, db)` wiring).

---

## Self-Review Notes

- **Spec coverage:** storage table → T1; `save_image`/`get_image` scoping → T1; `VISION_SUMMARY_MODEL` + lenient validation → T2; ingest fail-open summary → T6; checkpoint same-id rewrite → T6; `get_image` multimodal tool + chat scoping → T3/T4; stable message id → T5; system-prompt hint → T5; post-reply hook (photo-only, unaffected text path) → T7/T8; token accounting (no change, `[image #N]` counted as text, tool image charged flat by existing `_message_text`) → covered by existing `token_budget`; docs + known-risk deferral to #21/#22 → T9 / spec. Non-goals (retention, non-triggering photos, multi-photo) → untouched by design.
- **Type consistency:** `save_image(chat_id, message_id, mime_type, caption, summary, image_bytes) -> int` and `get_image(chat_id, image_id) -> ImageRecord | None` are used identically in T1/T3/T6. `persist_image` kwargs match the `_post_success` call in T8. `to_lc_human_message(message_id=...)` set in T5 is the same id consumed in T6's rewrite and asserted in T8.
- **No placeholders:** every code step contains complete, runnable code. The only prose-guided step is T8's test harness, which defers to the file's existing handler-construction pattern rather than inventing an incompatible one.
