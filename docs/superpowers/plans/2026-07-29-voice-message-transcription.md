# Voice Message Transcription Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transcribe Telegram voice notes into conversation context so the model can see what was said, without ever replying to the voice note itself.

**Architecture:** A new `transcription.py` owns the single call to OpenAI's audio endpoint. A new passive `voice_handler()` in `handlers/message_handlers.py` downloads the ogg, transcribes it, and stores a `[voice] <transcript>` marker through the *existing* passive-text path (`db.add_message` + `agent.append_context_message`). No new database table, no migration, no Agent method, no checkpoint plumbing.

**Tech Stack:** Python (3.12+ target syntax; the local `venv` runs 3.11), python-telegram-bot 21.7, `openai` SDK (async), pytest.

**Spec:** `docs/superpowers/specs/2026-07-29-voice-message-transcription-design.md`
**Issue:** https://github.com/6felix9/telegram-gpt/issues/50
**Branch:** `feat/voice-message-transcription` (already created, off `dev`)

## Global Constraints

- Python 3.12+ target syntax; type hints in the `str | None` / `list[dict]` style.
- 4-space indent, `snake_case` functions, `PascalCase` classes, UPPER_CASE constants. No formatter is enforced — match surrounding style.
- **Run pytest as `venv/bin/python3 -m pytest`**, never bare `pytest`. A bare `pytest`/`python3` after `activate` can resolve to the wrong interpreter on this machine.
- Voice notes are **passive only**. `voice_handler` must never call `message.reply_text()`, never call `is_authorized()`, and never invoke the agent's reply path. It stores context and returns.
- Every voice path is **fail-open**: catch, log via `logger.exception`, return. A failure is never surfaced to the user.
- The transcription model is fixed and independent of `/model`, `SUMMARY_MODEL`, and `MODEL_PROVIDERS`. Do **not** add it to `MODEL_PROVIDERS` — it is not a LangChain chat model.
- Marker text is exactly `[voice] <transcript>`, or the bare `[voice]` when there is no transcript. The `[Name]: ` group prefix is applied by `prompt_builder.to_lc_human_message`, never hand-written into the marker.
- Tests use `SimpleNamespace` / `unittest.mock` fakes and `asyncio.run(...)`, matching `tests/test_message_handlers.py`. No live API, database, or Telegram calls.
- Commit after every task.

---

### Task 1: Configuration and dependency

Adds the two new settings and pins the `openai` SDK explicitly. Nothing consumes them yet.

**Files:**
- Modify: `config.py:58` (after the `VISION_SUMMARY_MODEL` block), `config.py:102-109` (positive-int validation loop)
- Modify: `.env.example:54` (after `VISION_SUMMARY_MODEL=`)
- Modify: `requirements.txt` (Model providers section)
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `config.config.TRANSCRIPTION_MODEL: str` (default `"gpt-transcribe"`), `config.config.MAX_VOICE_DURATION_SECONDS: int` (default `600`).

- [ ] **Step 1: Write the failing tests**

In `tests/test_config.py`, add `"TRANSCRIPTION_MODEL"` and `"MAX_VOICE_DURATION_SECONDS"` to the `monkeypatch.delenv` key list inside `_fresh_config` (it currently ends with `"VISION_SUMMARY_MODEL",`), then add these tests at the end of the file:

```python
def test_voice_defaults_apply_when_unset(monkeypatch):
    cfg = _fresh_config(monkeypatch, VALID)
    assert cfg.config.TRANSCRIPTION_MODEL == "gpt-transcribe"
    assert cfg.config.MAX_VOICE_DURATION_SECONDS == 600


def test_voice_settings_read_from_env(monkeypatch):
    cfg = _fresh_config(
        monkeypatch,
        {**VALID, "TRANSCRIPTION_MODEL": "whisper-1", "MAX_VOICE_DURATION_SECONDS": "90"},
    )
    assert cfg.config.TRANSCRIPTION_MODEL == "whisper-1"
    assert cfg.config.MAX_VOICE_DURATION_SECONDS == 90


def test_blank_max_voice_duration_falls_back_to_default(monkeypatch):
    cfg = _fresh_config(monkeypatch, {**VALID, "MAX_VOICE_DURATION_SECONDS": ""})
    assert cfg.config.MAX_VOICE_DURATION_SECONDS == 600


def test_non_positive_max_voice_duration_is_rejected(monkeypatch):
    cfg = _fresh_config(monkeypatch, {**VALID, "MAX_VOICE_DURATION_SECONDS": "0"})
    with pytest.raises(SystemExit):
        cfg.config.validate()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/bin/python3 -m pytest tests/test_config.py -v -k voice`
Expected: FAIL with `AttributeError: type object 'Config' has no attribute 'TRANSCRIPTION_MODEL'`

- [ ] **Step 3: Add the settings to `config.py`**

Insert immediately after the `VISION_SUMMARY_MODEL` line (`config.py:58`), before `SUMMARY_TRIGGER_TOKENS`:

```python
    # Dedicated speech-to-text model for voice notes. Uses OpenAI's audio
    # transcription endpoint, not init_chat_model — deliberately absent from
    # MODEL_PROVIDERS. Fixed and independent of /model and SUMMARY_MODEL.
    TRANSCRIPTION_MODEL = os.getenv("TRANSCRIPTION_MODEL", "gpt-transcribe")
    # Voice notes longer than this are skipped before download, so an
    # over-limit note costs nothing. At gpt-transcribe's $0.0045/min this caps
    # a single note at roughly $0.045.
    MAX_VOICE_DURATION_SECONDS = _int_env("MAX_VOICE_DURATION_SECONDS", 600)
```

Then add `"MAX_VOICE_DURATION_SECONDS",` to the positive-int validation tuple in `validate()` (`config.py:102-109`), after `"SUMMARY_CONTEXT_TOKENS",`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `venv/bin/python3 -m pytest tests/test_config.py -v`
Expected: PASS (all tests in the file, not just the new ones)

- [ ] **Step 5: Document the settings in `.env.example`**

Insert after the `VISION_SUMMARY_MODEL=` line (`.env.example:54`):

```
# Dedicated speech-to-text model used to transcribe voice notes into context.
# Uses OpenAI's audio transcription endpoint; fixed and independent of /model
# and SUMMARY_MODEL. Not a MODEL_PROVIDERS entry. Defaults to gpt-transcribe.
TRANSCRIPTION_MODEL=

# Voice notes longer than this many seconds are skipped before download and
# stored as a bare [voice] marker, so they cost nothing. Defaults to 600.
MAX_VOICE_DURATION_SECONDS=
```

- [ ] **Step 6: Pin the openai SDK**

In `requirements.txt`, under the `# Model providers` comment, add a line above `langchain-openai>=0.3`:

```
openai>=2.0
```

Rationale to keep in mind: `openai` is already installed as a transitive dependency of `langchain-openai`, but `transcription.py` imports it directly, so it must be a declared dependency.

- [ ] **Step 7: Verify nothing else broke and commit**

Run: `venv/bin/python3 -m py_compile config.py && venv/bin/python3 -m pytest tests/ -v`
Expected: PASS

```bash
git add config.py .env.example requirements.txt tests/test_config.py
git commit -m "Add transcription model and voice duration cap settings"
```

---

### Task 2: `transcription.py`

The only place in the codebase that talks to OpenAI's audio endpoint. Pure and independently testable — takes bytes, returns text or `None`.

**Files:**
- Create: `transcription.py`
- Test: `tests/test_transcription.py`

**Interfaces:**
- Consumes: `config.TRANSCRIPTION_MODEL`, `config.OPENAI_API_KEY` from Task 1.
- Produces: `async def transcribe(audio_bytes: bytes, filename: str, config, client=None) -> str | None`. Returns the stripped transcript, or `None` on any failure or empty result. **Never raises.** The optional `client` parameter exists purely for injection in tests; production callers omit it.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_transcription.py`:

```python
"""transcribe(): OpenAI audio endpoint wrapper, fail-open by contract."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from transcription import transcribe


class _Cfg:
    TRANSCRIPTION_MODEL = "gpt-transcribe"
    OPENAI_API_KEY = "sk-test"


def _client(create):
    """Minimal stand-in for AsyncOpenAI: only .audio.transcriptions.create."""
    return SimpleNamespace(
        audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create))
    )


def test_returns_stripped_transcript():
    create = AsyncMock(return_value=SimpleNamespace(text="  hi i am jack  "))
    result = asyncio.run(transcribe(b"ogg", "voice.ogg", _Cfg, client=_client(create)))
    assert result == "hi i am jack"


def test_passes_configured_model_and_file():
    create = AsyncMock(return_value=SimpleNamespace(text="ok"))
    asyncio.run(transcribe(b"ogg-bytes", "voice.ogg", _Cfg, client=_client(create)))
    kwargs = create.await_args.kwargs
    assert kwargs["model"] == "gpt-transcribe"
    assert kwargs["file"] == ("voice.ogg", b"ogg-bytes")


def test_api_error_returns_none():
    create = AsyncMock(side_effect=RuntimeError("api exploded"))
    result = asyncio.run(transcribe(b"ogg", "voice.ogg", _Cfg, client=_client(create)))
    assert result is None


def test_empty_transcript_returns_none():
    create = AsyncMock(return_value=SimpleNamespace(text="   "))
    result = asyncio.run(transcribe(b"ogg", "voice.ogg", _Cfg, client=_client(create)))
    assert result is None


def test_missing_text_attribute_returns_none():
    create = AsyncMock(return_value=SimpleNamespace(text=None))
    result = asyncio.run(transcribe(b"ogg", "voice.ogg", _Cfg, client=_client(create)))
    assert result is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/bin/python3 -m pytest tests/test_transcription.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'transcription'`

- [ ] **Step 3: Write the implementation**

Create `transcription.py`:

```python
"""Speech-to-text for Telegram voice notes.

Owns the one call in this codebase to OpenAI's audio transcription endpoint.
Deliberately outside agent.py: this is not a LangChain chat model, so it has no
place in MODEL_PROVIDERS or init_chat_model().
"""
from __future__ import annotations

import logging

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


async def transcribe(
    audio_bytes: bytes, filename: str, config, client=None
) -> str | None:
    """Transcribe audio to text, or return None if that is not possible.

    Fail-open by contract: callers store a bare marker instead of surfacing an
    error, so this never raises. `client` is injected only by tests.

    Args:
        audio_bytes: Raw audio file contents.
        filename: Name with extension (e.g. "voice.ogg") — the API infers the
            container format from it.
        config: Settings object supplying TRANSCRIPTION_MODEL and OPENAI_API_KEY.
        client: Optional AsyncOpenAI-compatible client.

    Returns:
        The stripped transcript, or None on failure or an empty result.
    """
    try:
        client = client or AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        response = await client.audio.transcriptions.create(
            model=config.TRANSCRIPTION_MODEL,
            file=(filename, audio_bytes),
        )
        text = (getattr(response, "text", None) or "").strip()
        if not text:
            logger.info("Empty transcript for %s", filename)
            return None
        return text
    except Exception:
        logger.exception("Transcription failed for %s", filename)
        return None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `venv/bin/python3 -m pytest tests/test_transcription.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add transcription.py tests/test_transcription.py
git commit -m "Add transcription module wrapping the OpenAI audio endpoint"
```

---

### Task 3: `build_voice_marker` and `voice_handler`

The handler itself. Passive: it stores context and returns, and never replies or checks authorization.

**Files:**
- Modify: `handlers/message_handlers.py` (imports at top; new module-level constant and function after `extract_reply_data`; new methods on `MessageHandlers`)
- Test: `tests/test_message_handlers.py`

**Interfaces:**
- Consumes: `transcribe(...)` from Task 2; `config.MAX_VOICE_DURATION_SECONDS` from Task 1; existing `db.add_message(...)`, `agent.append_context_message(chat_id, human_message)`, `prompt_builder.to_lc_human_message(text=..., is_group=..., sender_name=...)`, and `agent.count_tokens(...)`.
- Produces: `build_voice_marker(transcript: str | None) -> str` and `MessageHandlers.voice_handler(update, context) -> None` (async), consumed by Task 4.

- [ ] **Step 1: Write the failing tests**

In `tests/test_message_handlers.py`, extend the import line to pull in the new symbol:

```python
from handlers.message_handlers import (
    MessageHandlers, build_voice_marker, extract_keyword, extract_reply_data,
)
```

Add a voice-capable message factory and the tests at the end of the file. Note that the existing `_message()` factory has no `voice` attribute, so voice tests need their own:

```python
def _voice_message(duration=5, chat_id=123, chat_type="private", user_id=7):
    voice_file = SimpleNamespace(
        download_as_bytearray=AsyncMock(return_value=bytearray(b"ogg-bytes"))
    )
    voice = SimpleNamespace(duration=duration, get_file=AsyncMock(return_value=voice_file))
    return SimpleNamespace(
        text=None, photo=None, caption=None, voice=voice, chat_id=chat_id,
        chat=SimpleNamespace(type=chat_type),
        from_user=SimpleNamespace(id=user_id, first_name="Jack", username="jack"),
        message_id=1, reply_to_message=None, reply_text=AsyncMock(),
    )


class _VoiceCfg:
    AUTHORIZED_USER_ID = "1"
    MAX_VOICE_DURATION_SECONDS = 600


def _voice_handlers(db, agent, prompt_builder):
    deps = HandlerDependencies(
        config=_VoiceCfg, db=db, agent=agent,
        prompt_builder=prompt_builder, bot_username="mybot",
    )
    return MessageHandlers(deps, RequestProcessor(deps))


def test_build_voice_marker_with_transcript():
    assert build_voice_marker("hi i am jack") == "[voice] hi i am jack"


def test_build_voice_marker_strips_whitespace():
    assert build_voice_marker("  hi i am jack  ") == "[voice] hi i am jack"


def test_build_voice_marker_bare_without_transcript():
    assert build_voice_marker(None) == "[voice]"
    assert build_voice_marker("") == "[voice]"
    assert build_voice_marker("   ") == "[voice]"


def test_voice_message_stores_transcript_and_never_replies(monkeypatch):
    monkeypatch.setattr(
        "handlers.message_handlers.transcribe", AsyncMock(return_value="hi i am jack")
    )
    db = SimpleNamespace(add_message=Mock())
    agent = SimpleNamespace(append_context_message=Mock(), run=AsyncMock())
    prompt_builder = SimpleNamespace(to_lc_human_message=Mock(return_value="human"))
    handlers_obj = _voice_handlers(db, agent, prompt_builder)

    message = _voice_message()
    asyncio.run(handlers_obj.voice_handler(SimpleNamespace(message=message), SimpleNamespace()))

    assert db.add_message.call_args.kwargs["content"] == "[voice] hi i am jack"
    agent.append_context_message.assert_called_once_with("123", "human")
    assert prompt_builder.to_lc_human_message.call_args.kwargs["text"] == "[voice] hi i am jack"
    agent.run.assert_not_awaited()
    message.reply_text.assert_not_awaited()


def test_voice_message_in_group_passes_sender_for_prefix(monkeypatch):
    monkeypatch.setattr(
        "handlers.message_handlers.transcribe", AsyncMock(return_value="hi i am jack")
    )
    db = SimpleNamespace(add_message=Mock())
    agent = SimpleNamespace(append_context_message=Mock())
    prompt_builder = SimpleNamespace(to_lc_human_message=Mock(return_value="human"))
    handlers_obj = _voice_handlers(db, agent, prompt_builder)

    message = _voice_message(chat_type="group")
    asyncio.run(handlers_obj.voice_handler(SimpleNamespace(message=message), SimpleNamespace()))

    kwargs = prompt_builder.to_lc_human_message.call_args.kwargs
    assert kwargs["is_group"] is True
    assert kwargs["sender_name"] == "Jack"
    assert db.add_message.call_args.kwargs["is_group_chat"] is True


def test_voice_over_duration_cap_skips_download_and_stores_bare_marker(monkeypatch):
    stub = AsyncMock(return_value="should not be called")
    monkeypatch.setattr("handlers.message_handlers.transcribe", stub)
    db = SimpleNamespace(add_message=Mock())
    agent = SimpleNamespace(append_context_message=Mock())
    prompt_builder = SimpleNamespace(to_lc_human_message=Mock(return_value="human"))
    handlers_obj = _voice_handlers(db, agent, prompt_builder)

    message = _voice_message(duration=601)
    asyncio.run(handlers_obj.voice_handler(SimpleNamespace(message=message), SimpleNamespace()))

    stub.assert_not_awaited()
    message.voice.get_file.assert_not_awaited()
    assert db.add_message.call_args.kwargs["content"] == "[voice]"


def test_failed_transcription_stores_bare_marker(monkeypatch):
    monkeypatch.setattr(
        "handlers.message_handlers.transcribe", AsyncMock(return_value=None)
    )
    db = SimpleNamespace(add_message=Mock())
    agent = SimpleNamespace(append_context_message=Mock())
    prompt_builder = SimpleNamespace(to_lc_human_message=Mock(return_value="human"))
    handlers_obj = _voice_handlers(db, agent, prompt_builder)

    asyncio.run(handlers_obj.voice_handler(
        SimpleNamespace(message=_voice_message()), SimpleNamespace()))

    assert db.add_message.call_args.kwargs["content"] == "[voice]"


def test_download_failure_stores_nothing_and_does_not_raise(monkeypatch):
    monkeypatch.setattr("handlers.message_handlers.transcribe", AsyncMock(return_value="x"))
    db = SimpleNamespace(add_message=Mock())
    agent = SimpleNamespace(append_context_message=Mock())
    prompt_builder = SimpleNamespace(to_lc_human_message=Mock(return_value="human"))
    handlers_obj = _voice_handlers(db, agent, prompt_builder)

    message = _voice_message()
    message.voice.get_file = AsyncMock(side_effect=RuntimeError("telegram down"))

    asyncio.run(handlers_obj.voice_handler(SimpleNamespace(message=message), SimpleNamespace()))

    db.add_message.assert_not_called()
    agent.append_context_message.assert_not_called()


def test_non_voice_update_is_ignored():
    db = SimpleNamespace(add_message=Mock())
    agent = SimpleNamespace(append_context_message=Mock())
    handlers_obj = _voice_handlers(db, agent, SimpleNamespace())

    message = SimpleNamespace(voice=None)
    asyncio.run(handlers_obj.voice_handler(SimpleNamespace(message=message), SimpleNamespace()))

    db.add_message.assert_not_called()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/bin/python3 -m pytest tests/test_message_handlers.py -v -k voice`
Expected: FAIL with `ImportError: cannot import name 'build_voice_marker'`

- [ ] **Step 3: Add the marker builder**

In `handlers/message_handlers.py`, add the import below the existing `from agent import count_tokens` line (`message_handlers.py:8`):

```python
from transcription import transcribe
```

Then add this after `extract_reply_data` (i.e. just before `class MessageHandlers`):

```python
VOICE_MARKER = "[voice]"


def build_voice_marker(transcript: str | None) -> str:
    """Context marker for a voice note.

    Bare when there is no transcript, so context still records that a voice
    note happened. The group '[Name]: ' prefix is added later by
    PromptBuilder.to_lc_human_message, never here.
    """
    text = (transcript or "").strip()
    return f"{VOICE_MARKER} {text}" if text else VOICE_MARKER
```

- [ ] **Step 4: Add the handler methods**

Append these two methods to the `MessageHandlers` class (after `_resolve_reply_image_ref`, at the end of the file). Also update the module docstring on line 1-2 to read `"""Telegram-facing text/photo/voice intake: activation parsing, auth gate, and\nhanding off to the shared request processor."""`:

```python
    async def voice_handler(self, update, context):
        """Passively transcribe a voice note into context.

        Voice notes never trigger a reply — the user follows up with a normal
        text message, by which point the transcript is already in history. No
        auth gate for the same reason non-triggering text has none: nothing is
        being asked of the bot. Fully fail-open, mirroring _passive_persist.
        """
        message = update.message
        if not message or not message.voice:
            return

        chat_id = str(message.chat_id)
        is_group = message.chat.type in ["group", "supergroup"]
        sender_name = message.from_user.first_name or "Unknown"

        try:
            marker = build_voice_marker(await self._voice_transcript(message))
            self._deps.db.add_message(
                chat_id=chat_id, role="user", content=marker,
                user_id=message.from_user.id, message_id=message.message_id,
                token_count=count_tokens(marker),
                sender_name=sender_name,
                sender_username=message.from_user.username,
                is_group_chat=is_group,
            )
            self._deps.agent.append_context_message(
                chat_id,
                self._deps.prompt_builder.to_lc_human_message(
                    text=marker, is_group=is_group, sender_name=sender_name),
            )
        except Exception:
            logger.exception("Failed to persist voice message for chat %s", chat_id)

    async def _voice_transcript(self, message) -> str | None:
        """Download and transcribe a voice note.

        Returns None when the note is over the duration cap — checked against
        the duration Telegram ships in the update, so an over-limit note is
        skipped before any download and costs nothing. A download failure
        raises, which the caller turns into "store nothing".
        """
        max_seconds = self._deps.config.MAX_VOICE_DURATION_SECONDS
        duration = message.voice.duration or 0
        if duration > max_seconds:
            logger.info(
                "Voice note is %ss, over the %ss cap; storing a bare marker",
                duration, max_seconds,
            )
            return None
        voice_file = await message.voice.get_file()
        audio_bytes = bytes(await voice_file.download_as_bytearray())
        return await transcribe(audio_bytes, "voice.ogg", self._deps.config)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `venv/bin/python3 -m pytest tests/test_message_handlers.py -v`
Expected: PASS (all tests in the file — the existing text/photo tests must still pass)

- [ ] **Step 6: Commit**

```bash
git add handlers/message_handlers.py tests/test_message_handlers.py
git commit -m "Add passive voice handler that transcribes notes into context"
```

---

### Task 4: Wiring and the prompt convention

Registers the handler with Telegram and teaches the model what `[voice]` means. Until this task the handler exists but never runs.

**Files:**
- Modify: `handlers/__init__.py:67` (after the `photo_handler` passthrough)
- Modify: `bot.py:98-104` (after the PHOTO handler registration)
- Modify: `prompt_builder.py:37` (constants) and `prompt_builder.py:121-127` (`_conventions_section`)
- Test: `tests/test_prompt_builder.py`

**Interfaces:**
- Consumes: `MessageHandlers.voice_handler` from Task 3.
- Produces: module-level `handlers.voice_handler(update, context)`; `prompt_builder.VOICE_MARKER_CONVENTION`.

- [ ] **Step 1: Write the failing tests**

Add to the end of `tests/test_prompt_builder.py`:

```python
def test_conventions_explain_the_voice_marker():
    out = _pb().build_system_prompt(is_group=False)
    assert "[voice]" in out


def test_voice_convention_present_in_groups_too():
    out = _pb().build_system_prompt(is_group=True)
    assert "[voice]" in out
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `venv/bin/python3 -m pytest tests/test_prompt_builder.py -v -k voice`
Expected: FAIL with `AssertionError: assert '[voice]' in '...'`

- [ ] **Step 3: Add the convention**

In `prompt_builder.py`, add after `IMAGE_MARKER_CONVENTION` (ends at line 38):

```python
VOICE_MARKER_CONVENTION = (
    '"[voice] text" is the transcript of a voice message someone sent.'
)
```

Then in `_conventions_section`, add a line after the existing `lines.append(f"- {IMAGE_MARKER_CONVENTION}")`:

```python
        lines.append(f"- {VOICE_MARKER_CONVENTION}")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `venv/bin/python3 -m pytest tests/test_prompt_builder.py -v`
Expected: PASS

- [ ] **Step 5: Add the facade passthrough**

In `handlers/__init__.py`, add after the `photo_handler` function (ends at line 67):

```python
async def voice_handler(update, context):
    assert _message_handlers is not None, "init_handlers() must run before voice_handler()"
    return await _message_handlers.voice_handler(update, context)
```

- [ ] **Step 6: Register the handler in `bot.py`**

In `bot.py`, add after the PHOTO handler registration block (which ends at line 104), before the `# Command handlers` comment:

```python
        # Voice handler (voice notes — transcribed passively, never replied to)
        application.add_handler(
            MessageHandler(
                filters.VOICE,
                handlers.voice_handler
            )
        )
```

- [ ] **Step 7: Verify the whole suite and commit**

Run: `venv/bin/python3 -m py_compile *.py handlers/*.py && venv/bin/python3 -m pytest tests/ -v`
Expected: PASS

```bash
git add prompt_builder.py handlers/__init__.py bot.py tests/test_prompt_builder.py
git commit -m "Register voice handler and document the [voice] marker"
```

---

### Task 5: Documentation and end-to-end validation

**Files:**
- Modify: `README.md` (feature description near line 154, env table near line 200)
- Modify: `CLAUDE.md` and `AGENTS.md` — these two are mirrors of each other; make **identical** edits in both (Message Flow / Context Storage sections, the env var list near line 212, and the notes near line 230)

**Interfaces:**
- Consumes: everything from Tasks 1-4.
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Update `README.md`**

Add to the env var table (alongside the `VISION_SUMMARY_MODEL` row near line 200):

```markdown
| `TRANSCRIPTION_MODEL` | `gpt-transcribe` | Speech-to-text model for voice notes; uses OpenAI's audio endpoint, independent of `/model` and `SUMMARY_MODEL` |
| `MAX_VOICE_DURATION_SECONDS` | `600` | Voice notes longer than this are skipped before download and stored as a bare `[voice]` marker |
```

Add near the image behavior description (around line 154):

```markdown
- Voice notes are transcribed with `TRANSCRIPTION_MODEL` and stored as a `[voice] <transcript>` marker so later turns can reference what was said. They never trigger a reply on their own — ask about one with a normal `chatgpt` message
```

- [ ] **Step 2: Update `CLAUDE.md` and `AGENTS.md` identically**

Add `- \`TRANSCRIPTION_MODEL\`` and `- \`MAX_VOICE_DURATION_SECONDS\`` to the Configuration env var list (near line 212, after `VISION_SUMMARY_MODEL`).

Add to the "Important notes" list under Configuration (near line 230):

```markdown
- `TRANSCRIPTION_MODEL` is the dedicated speech-to-text model for voice notes. It calls OpenAI's audio transcription endpoint directly and is deliberately *not* in `MODEL_PROVIDERS`; it is fixed and independent of `/model` and `SUMMARY_MODEL`
- `MAX_VOICE_DURATION_SECONDS` bounds transcription cost: a longer note is skipped before download and recorded as a bare `[voice]` marker
```

Add a new subsection after "Image Handling":

```markdown
### Voice Handling

- `voice_handler()` runs on every voice note (`filters.VOICE`). Voice notes are **passive context only** — the bot never replies to one. To ask about a voice note, send a normal `chatgpt` message; the transcript is already in history.
- Like non-triggering text, voice notes are stored from anyone in the chat; no authorization check runs, because nothing is being asked of the bot.
- `transcription.transcribe()` owns the sole call to OpenAI's audio endpoint. It is not a LangChain chat model and is absent from `MODEL_PROVIDERS`.
- The transcript is stored as a `[voice] <transcript>` marker in both the `messages` audit table and the checkpoint, via the same `add_message` + `append_context_message` path non-triggering text uses. The group `[Name]: ` prefix is applied by `prompt_builder`, so a DM shows a bare `[voice] ...`.
- Raw audio is not persisted — only the transcript.
- Everything fails open. Over the duration cap, an API failure, or a silent recording all yield a bare `[voice]` marker; a Telegram download failure stores nothing. None of it is surfaced to the user.
```

Also add `tests/test_transcription.py` to the Testing Guidelines list of covered modules.

- [ ] **Step 3: Run the full validation suite**

```bash
venv/bin/python3 -m py_compile *.py handlers/*.py database/*.py
venv/bin/python3 -m pytest tests/ -v
```
Expected: PASS, with the new tests from Tasks 1-4 present in the output.

- [ ] **Step 4: Commit**

```bash
git add README.md CLAUDE.md AGENTS.md
git commit -m "Document voice message transcription"
```

- [ ] **Step 5: Push and open a PR into `dev`**

```bash
git push -u origin feat/voice-message-transcription
gh pr create --base dev --head feat/voice-message-transcription \
  --title "Add voice message transcription" \
  --body "Closes #50. Transcribes Telegram voice notes into passive conversation context via a new \`transcription.py\` and a \`voice_handler\`. Design: \`docs/superpowers/specs/2026-07-29-voice-message-transcription-design.md\`"
```

- [ ] **Step 6: Manual validation against the dev bot**

Merging to `dev` auto-deploys to the Railway `dev` environment. Once deployed, in a chat with the dev bot:

1. Send a short voice note (under 10 minutes) saying something distinctive.
2. Send a text message: `chatgpt what did I just say in my voice message?`
3. Expected: the bot answers with the transcript's content, confirming it reached context.
4. Confirm no reply was sent to the voice note itself in step 1.
5. If a voice note over 10 minutes is available, send one and confirm the bot does not error and treats it as an unheard `[voice]` note.

---

## Notes for the implementer

- **Do not** add `gpt-transcribe` to `MODEL_PROVIDERS` in `agent.py` or `model_registry.py`. It is not a chat model and `init_chat_model()` cannot build it. `/model` must not be able to select it.
- **Do not** add authorization checks or `reply_text` calls to `voice_handler`. Passive-only is a deliberate design decision, not an oversight — see the spec's Behavior section.
- The `openai` package version installed in `venv` is 2.44.0, so `AsyncOpenAI` and the tuple `file=(name, bytes)` form are both available without upgrading.
