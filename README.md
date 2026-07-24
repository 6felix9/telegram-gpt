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
| `BOT_USERNAME` | Required | Bot username, with or without `@` |
| `OPENAI_API_KEY` | Required | OpenAI API key (always required for config validation) |
| `XAI_API_KEY` | Empty | Required for Grok models |
| `GEMINI_API_KEY` | Empty | Required for Gemini models |
| `TAVILY_API_KEY` | Empty | Optional; powers the agent's web search tool. If blank, the search tool falls back to DuckDuckGo at runtime |
| `DEFAULT_MODEL` | `gpt-5.4-mini` | Initial model used to seed `active_model` on first run |
| `MODEL_TIMEOUT` | `60` | API timeout in seconds |
| `MAX_CONTEXT_TOKENS` | `16000` | Total history budget before reserve tokens |
| `MAX_OUTPUT_TOKENS` | `2048` | Max tokens per reply; also the trimming middleware's reserve |
| `SUMMARY_MODEL` | `gpt-4.1-mini` | Dedicated supported model used for rolling checkpoint summaries |
| `VISION_SUMMARY_MODEL` | `gpt-5.4-nano` | Dedicated supported model used to describe images on ingest; independent of `/model` and `SUMMARY_MODEL` |
| `SUMMARY_TRIGGER_TOKENS` | `10000` | Summarize older active messages on the next triggered request at this approximate token count |
| `SUMMARY_KEEP_TOKENS` | `4000` | Approximate recent raw-message tokens retained after summarization |
| `SUMMARY_CONTEXT_TOKENS` | `14000` | Input token budget for the summary model call itself, independent of `MAX_CONTEXT_TOKENS` |
| `MAX_GROUP_CONTEXT_MESSAGES` | `500` | Reserved for future group message retention; cleanup is currently disabled |
| `AUTHORIZED_USER_ID` | Required | Main admin Telegram user ID |
| `DATABASE_URL` | Required | PostgreSQL / Neon connection string |
| `LOG_LEVEL` | `INFO` | Python logging level |

Notes:

- The required set validated at startup is `TELEGRAM_BOT_TOKEN`, `AUTHORIZED_USER_ID`, `OPENAI_API_KEY`, and `DATABASE_URL`. `OPENAI_API_KEY` is required even if `DEFAULT_MODEL` targets another provider.
- `XAI_API_KEY`, `GEMINI_API_KEY`, and `TAVILY_API_KEY` are optional and only needed to use the corresponding provider/tool.
- `DEFAULT_MODEL` only matters when `active_model` has not been seeded yet.
- After first startup, the active model is read from the database and can be changed with `/model`.
- `config.py` validates that the correct provider key is present for the configured `DEFAULT_MODEL`.
- `SUMMARY_MODEL` must be listed in `agent.py`'s `MODEL_PROVIDERS`, and its provider key must be configured at startup.
- `SUMMARY_KEEP_TOKENS` must be lower than `SUMMARY_TRIGGER_TOKENS`; `SUMMARY_CONTEXT_TOKENS` must be at least `SUMMARY_TRIGGER_TOKENS - SUMMARY_KEEP_TOKENS`.
- `SUMMARY_CONTEXT_TOKENS` bounds only the summary model's input and is unrelated to `MAX_CONTEXT_TOKENS`, which bounds the reply model's input instead.
- Passive non-triggering text is checkpointed without a model call. If it crosses the summary threshold, compaction waits for the next triggered request.

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
- `agent.py` - LangChain agent construction (`create_agent` + `init_chat_model`), provider/model routing (`MODEL_PROVIDERS`), rolling conversation summarization (`ResilientSummarizationMiddleware`), and the token-trimming middleware
- `conversation_summary.py` - Fail-open summarization middleware, image sanitization for summary generation, and post-compaction audit callback wiring
- `tools.py` - Agent tools
- `prompt_builder.py` - System prompt assembly and outbound message formatting
- `cache.py` - Small in-memory TTL cache used by the database layer
- `scripts/chat_cli.py` - Local chat simulator

High-level flow:

1. Receive Telegram text or photo update.
2. Detect activation via `chatgpt` or `@BOT_USERNAME`.
3. Authorize the user.
4. Store the incoming message or image marker in PostgreSQL.
5. On a triggered request, `ResilientSummarizationMiddleware` may compact active checkpoint history at or above `SUMMARY_TRIGGER_TOKENS` (at most one successful compaction per `Agent.run`/tool loop).
6. Request-time trimming keeps the reply-model input within the configured reserve.
7. Build the system prompt and provider-specific message payload.
8. Call the active model provider (reply/tool loop).
9. Store the assistant response.
10. Reply back to Telegram.

## Database

Schema is managed with Alembic migrations in `alembic/versions/`. Run `alembic upgrade head` to apply pending migrations — this is done automatically by `start.sh` locally and by the Railway `preDeployCommand` in each environment.

Primary tables:

- `messages`
- `granted_users`
- `personality`
- `active_personality`
- `active_model`
- `conversation_summaries` (write-only audit of successful summaries; never read by the agent)

## Checkpointer Setup

The LangGraph agent's conversation checkpoints live in their own tables (`checkpoints`, `checkpoint_blobs`, `checkpoint_writes`, `checkpoint_migrations`), owned and versioned by `langgraph-checkpoint-postgres` — they are intentionally **not** part of the Alembic-managed schema above.

The latest checkpoint uses rolling summaries plus recent raw messages. On a
triggered request, active history at or above `SUMMARY_TRIGGER_TOKENS` is
compacted by `SUMMARY_MODEL`; up to `SUMMARY_KEEP_TOKENS` of recent messages
remain verbatim. Summary failure leaves state unchanged and the normal
request-time trimming middleware still allows the reply to proceed.

Rolling summarization is the only mechanism that bounds active checkpoint
state; the previous fixed 500→400 message prune has been removed. A chat
whose summarization keeps failing open, or one that stays purely passive and
never triggers a reply, can grow its active checkpoint state without limit —
there is no message-count fallback. This is monitored via the "summary failed
open" structured log rather than enforced with a hard ceiling. Historical
checkpoint rows still accumulate regardless, and the application `messages`
audit table remains unbounded. Rolling summaries compact the latest logical
state; they do not physically delete historical checkpoint rows.

Current storage-growth limitations:

- LangGraph's historical checkpoint rows continue accumulating even though the bot only reads the latest state.
- The application `messages` audit-log table is also unbounded while its database cleanup remains disabled.
- `conversation_summaries` audit rows are also unbounded; audit insertion happens only after the exact generated summary ID is confirmed in result/checkpoint state, and audit failures never block compaction or replies.
- `/clear` removes the current checkpoint's summary and recent messages; it does not delete `messages` or `conversation_summaries` audit rows.

Run this once per environment, after `alembic upgrade head` and before the bot starts (it is idempotent — safe to re-run):

```bash
python scripts/setup_checkpointer.py
```

- Locally, `start.sh` already runs this step for you after migrations.
- On Railway, add it to each environment's `preDeployCommand` so it runs before every deploy:

  ```
  alembic upgrade head && python scripts/setup_checkpointer.py
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
