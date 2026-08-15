<p align="center">
  <img src="assets/logo.png" alt="Telegram GPT Bot logo" width="160">
</p>

<h1 align="center">Telegram GPT Bot</h1>

<p align="center">
  An AI agent that lives directly in your Telegram.
</p>

A Telegram bot with persistent chat history, token-aware context trimming, image support, and a PostgreSQL/Neon backend. It's triggered by the keyword `chatgpt` or by directly mentioning the bot, supports multiple model providers, and persists the active model in the database so model switches survive restarts.

## Tech Stack

- **Language:** Python 3.12+
- **Bot framework:** [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot)
- **Agent orchestration:** LangChain / LangGraph (`create_agent`, `init_chat_model`)
- **Model providers:** OpenAI, xAI, Google Gemini
- **Database:** PostgreSQL / [Neon](https://neon.tech/)
- **Migrations:** Alembic
- **Token counting:** [tiktoken](https://github.com/openai/tiktoken)
- **Web search:** Tavily, falling back to DuckDuckGo
- **Deployment:** Docker, Railway

## Features

- Keyword or `@bot` activation
- Persistent conversation history in PostgreSQL / Neon
- Token-aware context trimming with `tiktoken`
- Group chat context storage for better follow-up answers
- Image handling through multimodal model requests
- Web search and page-fetch agent tools (Tavily, falling back to DuckDuckGo)
- Allowlist-based access control
- Global model switching with `/model`
- Global group personality switching with `/personality`
- Docker and local CLI testing support

## Supported Models

The current model list is defined in `agent.py` via `MODEL_PROVIDERS`, which routes each model name to its provider for `init_chat_model()`.

Supported today:

- OpenAI: `gpt-4.1-mini`, `gpt-5.4-nano`, `gpt-5.4-mini`, `gpt-5.4`, `gpt-5.6-luna`, `gpt-5.6-terra`
- xAI: `grok-4.20-0309-reasoning`, `grok-4.20-0309-non-reasoning`, `grok-4-1-fast-reasoning`
- Gemini: `gemini-3.1-flash-lite-preview`, `gemini-3.5-flash`

Each model is routed to its provider (`openai`, `xai`, or `google_genai`) and built with LangChain's `init_chat_model()`.

## Requirements

- Python 3.12+
- Telegram bot token
- Telegram bot username
- PostgreSQL / Neon database
- At least one provider API key matching your `DEFAULT_MODEL`

## Quick Start

### Local Development

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Copy the environment template:

```bash
cp .env.example .env
```

3. Fill in `.env`:

- `TELEGRAM_BOT_TOKEN`
- `BOT_USERNAME`
- `AUTHORIZED_USER_ID`
- `DATABASE_URL`
- `DEFAULT_MODEL`
- The API key required for that model's provider:
  - `OPENAI_API_KEY`
  - `XAI_API_KEY`
  - `GEMINI_API_KEY`

4. Apply database migrations:

```bash
alembic upgrade head
```

4a. Set up the LangGraph checkpointer tables (once per environment, idempotent):

```bash
python scripts/setup_checkpointer.py
```

5. Run the bot:

```bash
python3 bot.py
```

Or use the helper script:

```bash
./start.sh
```

`start.sh` creates or reuses `venv/`, installs dependencies, applies migrations, and starts the bot.

### Docker

Build and run with Compose:

```bash
docker compose up -d --build
```

View logs:

```bash
docker compose logs -f
```

Stop:

```bash
docker compose down
```

The current `docker-compose.yml` still mounts `./data:/app/data`, but the bot's persistent state lives in PostgreSQL, not local files.

## How Activation Works

### Private Chats

- The bot responds when the message contains `chatgpt` or `@BOT_USERNAME`
- Authorization is checked per user
- Text messages without activation are still stored for context (no reply)
- Private chats use the default private system prompt (not `/personality`)

### Group Chats

- The bot responds when a message contains `chatgpt` or `@BOT_USERNAME`
- Text messages without activation are still stored for context
- Authorization is still checked per user, not per chat
- Stored group messages are formatted internally as `[Name]: message`
- Group chats use the group system prompt or the active `/personality`

### Images

- Photo messages only trigger a reply when the caption contains `chatgpt` or `@BOT_USERNAME`; on a triggering photo the image itself is sent to the model at request time
- Every photo (triggering or not) is persisted on arrival: it is summarized with `VISION_SUMMARY_MODEL`, the raw bytes + summary are stored in the `images` table, and an `[image #<id>] <summary>` marker is written into the conversation so later turns can reference it
- The `messages` audit table stores a lightweight text marker instead of the raw image payload:
  `[image] <caption>` on arrival, rewritten to `[image #<id>] <caption> — <summary>` once the image is persisted
- The agent can call `get_image(<id>)` to pull a stored image back into context; replying to an earlier photo points the agent at that photo's `[image #<id>]`

### Voice Notes

- Voice notes are transcribed with `TRANSCRIPTION_MODEL` and stored as a `[voice] <transcript>` marker so later turns can reference what was said. They never trigger a reply on their own — ask about one with a normal `chatgpt` message

## Commands

All commands are restricted to the main admin (`AUTHORIZED_USER_ID`); granted users can chat with the bot but cannot run commands:

- `/clear` - Clear the current chat's checkpoint summary and recent messages (application audit rows are retained)
- `/stats` - Show message count and token usage for the current chat
- `/grant <user_id>` - Grant access to another user
- `/revoke <user_id>` - Revoke access from a granted user
- `/allowlist` - Show the current allowlist
- `/model [name]` - Show or change the globally active model
- `/personality [name]` - Show available personalities, or change the active group personality
- `/version` - Show the current bot version
- `/help` - Show the command reference

## Authorization Model

The bot uses a two-tier allowlist:

- `AUTHORIZED_USER_ID` is the main admin
- Additional users can be granted access with `/grant`

The main admin can use all commands. Granted users can talk to the bot but cannot run any command, including `/clear` and `/stats`.

## Configuration

Environment variables are loaded from `.env`.

| Variable | Default | Description |
|----------|---------|-------------|
| `TELEGRAM_BOT_TOKEN` | Required | Bot token from BotFather |
| `AUTHORIZED_USER_ID` | Required | Main admin Telegram user ID |
| `OPENAI_API_KEY` | Required | OpenAI API key (always required for config validation) |
| `DATABASE_URL` | Required | PostgreSQL / Neon connection string |
| `BOT_USERNAME` | Empty | Optional; bot username, with or without `@` |
| `XAI_API_KEY` | Empty | Required for Grok models |
| `GEMINI_API_KEY` | Empty | Required for Gemini models |
| `DEFAULT_MODEL` | `gpt-5.4-mini` | Initial model used to seed `active_model` on first run |
| `MODEL_TIMEOUT` | `60` | API timeout in seconds |
| `MAX_CONTEXT_TOKENS` | `16000` | Total history budget before reserve tokens |
| `MAX_OUTPUT_TOKENS` | `2048` | Max tokens per reply; also the trimming middleware's reserve |
| `SUMMARY_MODEL` | `gpt-5.6-luna` | Dedicated supported model used to compact checkpoint state into a rolling summary |
| `VISION_SUMMARY_MODEL` | `gpt-5.4-nano` | Dedicated supported model used to describe images on ingest; independent of `/model` and `SUMMARY_MODEL` |
| `SUMMARIZATION_TRIGGER` | `8000` | Compact the checkpoint when uncompacted conversation reaches this approximate token count (excluding rolling summary tokens) |
| `MAX_SUMMARY_OUTPUT` | `1000` | Hard output cap for one generated summary; must be less than `SUMMARIZATION_TRIGGER` |
| `MESSAGE_RETENTION_DAYS` | `30` | Age-based retention for the `messages` audit table; rows older than this many days are deleted by `scripts/cleanup_retention.py`. `0` disables cleanup |
| `TRANSCRIPTION_MODEL` | `gpt-transcribe` | Speech-to-text model for voice notes; uses OpenAI's audio endpoint, independent of `/model` and `SUMMARY_MODEL` |
| `MAX_VOICE_DURATION_SECONDS` | `600` | Voice notes longer than this are skipped before download and stored as a bare `[voice]` marker |
| `TAVILY_API_KEY` | Empty | Optional; powers the agent's web search tool. If blank, the search tool falls back to DuckDuckGo at runtime |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `LANGSMITH_TRACING` | Empty | Optional; set to `true` to enable automatic LangSmith agent tracing |
| `LANGSMITH_ENDPOINT` | Empty | Optional; LangSmith API endpoint URL |
| `LANGSMITH_API_KEY` | Empty | Optional; LangSmith API key for telemetry |
| `LANGSMITH_PROJECT` | Empty | Optional; LangSmith project name for environment isolation (e.g., `telegram-gpt-dev`, `telegram-gpt-prod`) |

Notes:

- The required set validated at startup is `TELEGRAM_BOT_TOKEN`, `AUTHORIZED_USER_ID`, `OPENAI_API_KEY`, and `DATABASE_URL`. `OPENAI_API_KEY` is required even if `DEFAULT_MODEL` targets another provider.
- `XAI_API_KEY`, `GEMINI_API_KEY`, and `TAVILY_API_KEY` are optional and only needed to use the corresponding provider/tool.
- `DEFAULT_MODEL` only matters when `active_model` has not been seeded yet.
- After first startup, the active model is read from the database and can be changed with `/model`.
- `config.py` validates that the correct provider key is present for the configured `DEFAULT_MODEL`.
- `SUMMARY_MODEL` must be listed in `agent.py`'s `MODEL_PROVIDERS`, and its provider key must be configured at startup.
- `MAX_SUMMARY_OUTPUT` must be less than `SUMMARIZATION_TRIGGER`, or a compaction could not bring state below the trigger and every subsequent message would re-trigger a summary call.
- `SUMMARIZATION_TRIGGER` is the only threshold governing checkpoint size. `MAX_CONTEXT_TOKENS` is unrelated: it bounds a single reply-model call, not stored state.
- Compaction runs before every checkpoint update, triggered or passive, so a chat that only ever receives non-triggering messages is still bounded. A passive message that crosses the threshold does cost one summary-model call with no reply to show for it.

## CLI Chat Simulator

Use the CLI to test the same prompt-building and context logic without Telegram.

Test mode, writes to the database:

```bash
python3 scripts/chat_cli.py --chat-id test
```

Read-only simulation against an existing chat:

```bash
python3 scripts/chat_cli.py --chat-id -5086459563 --group
```

Test mode with group formatting:

```bash
python3 scripts/chat_cli.py --chat-id test --group
```

CLI commands:

- `/clear`
- `/stats`
- `/model [name]`
- `/personality [name]`
- `/exit`
- `/quit`

## Architecture

Core modules:

- `bot.py` - Entry point, dependency wiring, Telegram application setup
- `config.py` - Env loading and validation
- `database/` - PostgreSQL connection pooling, persistence, cached lookups (`Database` facade + repositories)
- `handlers/` - Telegram handlers, authorization checks, command implementations
- `agent.py` - LangChain agent construction (`create_agent` + `init_chat_model`), provider/model routing (`MODEL_PROVIDERS`), checkpoint compaction (`_compact_if_needed`), and the token-trimming middleware
- `conversation_summary.py` - Fail-open summarization middleware, image sanitization for summary generation, and post-compaction audit callback wiring
- `tools.py` - Agent tools: `web_search` (Tavily or DuckDuckGo behind one stable name) and `fetch_url`
- `transcription.py` - Wraps OpenAI's audio transcription endpoint for passive voice-note handling
- `prompt_builder.py` - System prompt assembly (persona, generated tool section, conventions), the per-call context message, and outbound message formatting
- `cache.py` - Small in-memory TTL cache used by the database layer
- `scripts/chat_cli.py` - Local chat simulator

High-level flow:

1. Receive Telegram text, photo, or voice update.
2. Detect activation via `chatgpt` or `@BOT_USERNAME`.
3. Authorize the user.
4. Store the incoming message or image marker in PostgreSQL.
5. Before appending the incoming message, `Agent._compact_if_needed` replaces active checkpoint state with a rolling summary plus the last exchange if it has reached `SUMMARIZATION_TRIGGER`.
6. Request-time trimming keeps the reply-model input within the configured reserve.
7. Build the static system prompt (persona, then tools, then conventions), the provider-specific message payload, and append the per-call `## Current context` block after the trimmed history.
8. Call the active model provider (reply/tool loop).
9. Store the assistant response.
10. Reply back to Telegram.

## Database

Schema is managed with Alembic migrations in `alembic/versions/`. Run `alembic upgrade head` to apply pending migrations — this, along with checkpointer setup and retention cleanup, is done automatically by `start.sh` locally and by the Railway `preDeployCommand` in each environment.

Primary tables:

- `messages`
- `granted_users`
- `personality`
- `active_personality`
- `active_model`
- `conversation_summaries` (write-only audit of successful summaries; never read by the agent)

## Checkpointer Setup

The LangGraph agent's conversation checkpoints live in their own tables (`checkpoints`, `checkpoint_blobs`, `checkpoint_writes`, `checkpoint_migrations`), owned and versioned by `langgraph-checkpoint-postgres` — they are intentionally **not** part of the Alembic-managed schema above.

The latest checkpoint holds a rolling summary plus at most the last exchange.
Before any checkpoint update — a triggered reply or a passively stored message —
state at or above `SUMMARIZATION_TRIGGER` is compacted by `SUMMARY_MODEL` into a
single summary of at most `MAX_SUMMARY_OUTPUT` tokens. The incoming message is
appended afterward, so it is never swallowed by its own summary. Everything
older than the last exchange survives only as summarized prose.

Compaction is the only mechanism that bounds active checkpoint state. It is
fully fail-open: a summary failure leaves state unchanged and the reply still
proceeds, because request-time trimming is independent. A chat whose compaction
keeps failing can therefore grow its active state without limit — monitored via
the "Checkpoint compaction failed open" structured log rather than enforced with
a hard ceiling. Compaction rewrites the latest logical state; it does not
physically delete historical checkpoint rows — that's handled by the sweep
described below.

Current storage-growth handling:

- `scripts/cleanup_retention.py` prunes LangGraph's historical checkpoint rows down to the newest checkpoint per thread on every deploy, since the bot only ever reads the latest state.
- The same script deletes `messages` audit-log rows older than `MESSAGE_RETENTION_DAYS` (default 30 days; `0` disables it). `/stats`'s reported "Since" date reflects the oldest row currently retained, not necessarily the chat's true first message, once older rows have been pruned.
- `conversation_summaries` audit rows are still unbounded; audit insertion happens only after the exact generated summary ID is confirmed in result/checkpoint state, and audit failures never block compaction or replies.
- `/clear` removes the current checkpoint's summary and recent messages; it does not delete `messages` or `conversation_summaries` audit rows.

Run these once per environment, after `alembic upgrade head` and before the bot starts (both are idempotent — safe to re-run):

```bash
python scripts/setup_checkpointer.py
python scripts/cleanup_retention.py
```

- Locally, `start.sh` already runs these steps for you after migrations.
- On Railway, add them to each environment's `preDeployCommand` so they run before every deploy:

  ```
  alembic upgrade head && python scripts/setup_checkpointer.py && python scripts/cleanup_retention.py
  ```

## Validation

Minimum checks before merging changes:

```bash
python3 -m py_compile *.py
python3 -m py_compile alembic/env.py alembic/versions/*.py
alembic upgrade head  # requires a reachable local/dev DATABASE_URL
python3 scripts/chat_cli.py --chat-id test
```

If Telegram credentials are available, also verify:

- `/clear`
- `/stats`
- `/model`

## Troubleshooting

### Bot does not respond

- Confirm the message contains `chatgpt`, or mention the bot directly
- Confirm the sender is authorized
- Check logs with `docker compose logs -f` or local console output

### Authentication error

- Check the API key for the currently active provider
- Make sure the stored active model matches the key you configured
- If needed, switch models with `/model`

### Database error

- Verify `DATABASE_URL`
- Confirm the database is reachable
- Check logs for PostgreSQL / Neon connection failures

### Model changed unexpectedly after restart

- The bot reads `active_model` from the database on startup
- `DEFAULT_MODEL` is only the seed value for a fresh database

## Security Notes

- Do not commit `.env`
- Treat `DATABASE_URL`, `TELEGRAM_BOT_TOKEN`, and all provider keys as secrets
- Limit who receives `/grant` access

## Acknowledgments

- [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot)
- [OpenAI Python SDK](https://github.com/openai/openai-python)
- [tiktoken](https://github.com/openai/tiktoken)
- [Neon](https://neon.tech/)
