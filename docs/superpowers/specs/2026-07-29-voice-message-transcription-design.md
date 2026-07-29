# Voice Message Transcription — Design

**Date:** 2026-07-29
**Issue:** [#50](https://github.com/6felix9/telegram-gpt/issues/50)
**Status:** Approved, ready for implementation planning

## Goal

Telegram voice notes currently vanish — the bot has no `filters.VOICE` handler, so
a voice message is neither stored nor visible to the model. This adds
speech-to-text so voice notes become ordinary conversation context.

## Behavior

Voice notes are **passive context only**. Every voice note in a chat the bot is in
is transcribed and stored; the bot never replies to one.

To ask about a voice note, the user sends a normal text message containing
`chatgpt` or `@BOT_USERNAME`. The transcript is already in history by then, so no
reply-context plumbing is needed — it works the same way asking about earlier text
does.

Authorization is unchanged. Like non-triggering text today, voice notes are stored
from anyone in the chat; the auth gate only applies when someone triggers a reply.

### Context format

Sender attribution stays in the existing `[Name]: ` prefix; the bracket marker only
labels the content type. This matches how images already work.

| | DM | Group |
|---|---|---|
| text | `hey what's up` | `[Jack]: hey what's up` |
| image | `[image #7] my dog — a golden retriever` | `[Jack]: [image #7] my dog — a golden retriever` |
| voice | `[voice] hi i am jack` | `[Jack]: [voice] hi i am jack` |

When there is no transcript (over the duration cap, API failure, or silence) the
marker is the bare `[voice]`, so context still records that a voice note happened.

## Architecture

Transcription lives in its own module rather than on `Agent`. `Agent.persist_image()`
sits on the agent because it needs `graph.update_state()` to rewrite a message in
place; voice only ever appends, which `Agent.append_context_message()` already does.
Transcription is also not a LangChain chat model — it is the raw OpenAI audio
endpoint — so it does not fit `init_chat_model()` or `MODEL_PROVIDERS`.

### New file: `transcription.py`

One public function:

```python
async def transcribe(audio_bytes: bytes, filename: str, config) -> str | None
```

Wraps `AsyncOpenAI().audio.transcriptions.create(model=config.TRANSCRIPTION_MODEL, ...)`.
Returns the stripped transcript, or `None` on any failure or empty result. It owns
the API key and model id; nothing else in the codebase touches the audio endpoint.

`openai` is pinned explicitly in `requirements.txt` — it is currently only a
transitive dependency of `langchain-openai`.

### `handlers/message_handlers.py` — `MessageHandlers.voice_handler()`

Structurally a sibling of `photo_handler`'s passive branch, but simpler because
there is no triggered path:

1. Guard: no `message.voice` → return.
2. `message.voice.duration > config.MAX_VOICE_DURATION_SECONDS` → store the bare
   `[voice]` marker and return. Telegram supplies the duration in the update, so an
   over-limit note is skipped before any download. Zero API cost.
3. Download the ogg via `voice.get_file()` / `download_as_bytearray()`.
4. `transcribe(...)` → text.
5. Build the marker: `[voice] {transcript}`, or bare `[voice]` if there is none.
6. `db.add_message(chat_id=..., role="user", content=marker, ...)` — audit row, with
   `sender_name` in its own column as usual.
7. `agent.append_context_message(chat_id, prompt_builder.to_lc_human_message(
   text=marker, is_group=..., sender_name=...))` — which applies the `[Jack]: `
   prefix in groups and nothing in DMs.

The whole body is wrapped fail-open: any exception is logged and swallowed, exactly
like `_passive_persist`. The user never sees an error, because they never asked the
bot for anything.

### Other touched files

- `handlers/__init__.py` — a `voice_handler` module-level passthrough, matching the
  existing facade pattern.
- `bot.py` — register `MessageHandler(filters.VOICE, handlers.voice_handler)`
  alongside the existing PHOTO handler.
- `config.py` / `.env.example` — two new settings (below).
- `prompt_builder.py` — one line in the conventions block explaining that
  `[voice] text` is a transcribed voice message, next to the existing `[image #N]`
  line.
- `README.md`, `AGENTS.md`, `CLAUDE.md` — document the new env vars and the voice
  path.

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `TRANSCRIPTION_MODEL` | `gpt-transcribe` | OpenAI's current transcription model ($0.0045/min, `v1/audio/transcriptions`). Fixed and independent of `/model` and `SUMMARY_MODEL`, like `VISION_SUMMARY_MODEL`. |
| `MAX_VOICE_DURATION_SECONDS` | `600` | Notes longer than this are skipped before download. At `gpt-transcribe` pricing this caps a single note at roughly $0.045. |

Neither blocks startup if unusable — transcription fails open. `OPENAI_API_KEY` is
already required by `config.validate()`, so a missing key cannot occur in practice.

## Error handling

Every failure degrades to "context is slightly poorer", never to a user-visible
error or a lost turn.

| Failure | Behavior |
|---|---|
| Over duration cap | Bare `[voice]` marker, no download, no API call |
| Telegram download fails | Logged, nothing stored |
| Transcription API error / timeout | Logged, bare `[voice]` marker stored |
| Empty transcript (silence) | Bare `[voice]` marker stored |
| DB or checkpoint write fails | Logged and swallowed, as in `_passive_persist` |

## Testing

Pure logic only — no Telegram, database, or live API calls, matching the existing
`tests/` convention.

- `tests/test_transcription.py` — `transcribe()` against a stubbed OpenAI client:
  happy path strips whitespace; API exception → `None`; empty or whitespace-only
  response → `None`.
- `tests/test_voice_marker.py` — the marker builder as a small pure function
  (transcript → `[voice] hi i am jack`; empty → `[voice]`), plus
  `to_lc_human_message` producing `[Jack]: [voice] ...` in a group and
  `[voice] ...` in a DM.
- A duration-cap test asserting the over-limit path yields the bare marker without
  calling `transcribe`.

### Manual validation

```bash
python3 -m py_compile *.py
pytest tests/ -v
```

Then, against the dev bot on Railway: send one voice note under the cap and one
over it, and confirm the transcript reached context by asking a follow-up text
question about it.

## Out of scope

- Text-to-speech replies
- Audio files (`filters.AUDIO`) and round video notes (`filters.VIDEO_NOTE`)
- Storing raw audio bytes (no consumer exists; would add to the unbounded-blob
  problem the `images` table already has)
- Replying directly to a voice note
- Re-transcription of stored notes
