# Telegram GPT Bot

A Telegram bot with persistent chat history, token-aware context trimming, image support, and a PostgreSQL/Neon backend.

The bot is triggered by the keyword `chatgpt` or by directly mentioning the bot. It supports multiple model providers and persists the active model in the database so model switches survive restarts.

## Features

- Keyword or `@bot` activation
- Persistent conversation history in PostgreSQL / Neon
- Token-aware context trimming with `tiktoken`
- Group chat context storage for better follow-up answers
- Image handling through multimodal model requests
- Allowlist-based access control
- Global model switching with `/model`
- Global group personality switching with `/personality`
- Docker and local CLI testing support

## Supported Models

The current model list is defined in `openai_client.py` via `MODEL_REGISTRY`.

Supported today:

- OpenAI: `gpt-4o-mini`, `gpt-4.1-mini`, `gpt-5.4-mini`, `gpt-5`
- xAI: `grok-4.20-0309-reasoning`, `grok-4.20-0309-non-reasoning`, `grok-4-1-fast-reasoning`
- Gemini: `gemini-3.1-flash-lite-preview`, `gemini-3-flash-preview`

OpenAI and xAI models use the Responses API path. Gemini models use the Chat Completions path through the OpenAI-compatible Gemini endpoint.

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
- Messages without activation are ignored

### Group Chats

- The bot responds when a message contains `chatgpt` or `@BOT_USERNAME`
- Text messages without activation are still stored for context
- Authorization is still checked per user, not per chat
- Stored group messages are formatted internally as `[Name]: message`

### Images

- Photo messages only trigger when the caption contains `chatgpt` or `@BOT_USERNAME`
- The image itself is sent to the model at request time
- The database stores a lightweight text marker such as `[image] <caption>` instead of the raw image payload

## Commands

User-accessible commands:

Main-admin-only commands:

- `/clear` - Clear conversation history for the current chat
- `/stats` - Show message count and token usage for the current chat
- `/grant <user_id>` - Grant access to another user
- `/revoke <user_id>` - Revoke access from a granted user
- `/allowlist` - Show the current allowlist
- `/model [name]` - Show or change the globally active model
- `/personality [name]` - Show or change the active group personality
- `/list_personality` - List personalities stored in the database
- `/version` - Show the current bot version
- `/help` - Show the command reference

## Authorization Model

The bot uses a two-tier allowlist:

- `AUTHORIZED_USER_ID` is the main admin
- Additional users can be granted access with `/grant`

The main admin can use all commands. Granted users can talk to the bot but cannot change global settings.

## Configuration

Environment variables are loaded from `.env`.

| Variable | Default | Description |
|----------|---------|-------------|
| `TELEGRAM_BOT_TOKEN` | Required | Bot token from BotFather |
| `BOT_USERNAME` | Required | Bot username, with or without `@` |
| `OPENAI_API_KEY` | Empty | Required for OpenAI models |
| `XAI_API_KEY` | Empty | Required for Grok models |
| `GEMINI_API_KEY` | Empty | Required for Gemini models |
| `DEFAULT_MODEL` | `gpt-4.1-mini` | Initial model used to seed `active_model` on first run |
| `OPENAI_TIMEOUT` | `60` | API timeout in seconds |
| `MAX_CONTEXT_TOKENS` | `16000` | Total history budget before reserve tokens |
| `RESERVE_TOKENS_TEXT` | `1000` | Tokens reserved for text responses |
| `RESERVE_TOKENS_IMAGE` | `3000` | Tokens reserved for vision responses |
| `MAX_GROUP_CONTEXT_MESSAGES` | `100` | Group message retention target used by cleanup |
| `AUTHORIZED_USER_ID` | Required | Main admin Telegram user ID |
| `DATABASE_URL` | Required | PostgreSQL / Neon connection string |
| `LOG_LEVEL` | `INFO` | Python logging level |

Notes:

- `DEFAULT_MODEL` only matters when `active_model` has not been seeded yet.
- After first startup, the active model is read from the database and can be changed with `/model`.
- `config.py` validates that the correct provider key is present for the configured `DEFAULT_MODEL`.

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
- `/list_personality`
- `/exit`
- `/quit`

## Architecture

Core modules:

- `bot.py` - Entry point, dependency wiring, Telegram application setup
- `config.py` - Env loading and validation
- `database.py` - PostgreSQL connection pooling, schema init, persistence, cached lookups
- `handlers.py` - Telegram handlers, authorization checks, command implementations
- `openai_client.py` - Provider/model routing and API calls
- `prompt_builder.py` - System prompt assembly and outbound message formatting
- `token_manager.py` - Token counting and context trimming
- `cache.py` - Small in-memory TTL cache used by the database layer
- `scripts/chat_cli.py` - Local chat simulator

High-level flow:

1. Receive Telegram text or photo update.
2. Detect activation via `chatgpt` or `@BOT_USERNAME`.
3. Authorize the user.
4. Store the incoming message or image marker in PostgreSQL.
5. Retrieve recent history by token budget.
6. Trim history to fit the configured reserve.
7. Build the system prompt and provider-specific message payload.
8. Call the active model provider.
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

See `database.md` for the verified live schema summary.

## Validation

Minimum checks before merging changes:

```bash
python3 -m py_compile *.py
python3 -m py_compile alembic/env.py alembic/versions/*.py
alembic upgrade head
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
