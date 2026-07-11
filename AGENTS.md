## Project Overview

This repository contains a Telegram bot with:

- keyword / mention activation
- PostgreSQL / Neon-backed conversation history
- token-aware context trimming
- image support
- allowlist-based access control
- global active-model switching persisted in the database

The bot is no longer "OpenAI only". `agent.py` routes requests by model name to OpenAI, xAI, or Gemini via `MODEL_PROVIDERS` and LangChain's `init_chat_model()`.

## Project Structure & Module Organization

- Core runtime files are at repo root: `bot.py` (entrypoint), `handlers.py` (Telegram handlers and commands), `agent.py` (LangChain agent construction, provider/model routing via `MODEL_PROVIDERS`, and the trimming middleware), `tools.py` (agent tools: web search and page fetch), `database.py` (PostgreSQL/Neon persistence and global settings), `prompt_builder.py` (system prompt construction and message formatting), `cache.py` (TTL cache helpers), and `config.py` (env-driven settings).
- Operational docs live in `README.md` and `AGENTS.md`.
- Utility scripts are in `scripts/` (notably `scripts/chat_cli.py` for local chat simulation).
- Unit tests live in `tests/` (pytest; no database or `.env` required).

## Architecture

### Core Components

1. `bot.py` wires together config, database, the checkpointer, prompt builder, agent, and Telegram handlers.
2. `config.py` loads `.env` and validates required settings.
3. `database.py` owns PostgreSQL persistence, cached lookups, and global settings such as active model and active personality (schema itself is Alembic-managed — see Database Schema below).
4. `handlers.py` implements Telegram message handlers and bot commands.
5. `agent.py` builds the LangChain agent (`create_agent` + `init_chat_model`), maps the active model to a provider via `MODEL_PROVIDERS`, and applies the `wrap_model_call` trimming middleware before each model call.
6. `prompt_builder.py` builds system prompts and normalizes message payloads for the agent.
7. The checkpointer (`PostgresSaver`, keyed by chat_id thread) persists conversation state across turns; `agent.py` temporarily bounds the latest state with fixed 500→400 limits, while token counting and model-input trimming remain separate.
8. `cache.py` provides a small TTL cache used by the database layer.
9. `tools.py` builds the agent's tools: a web search tool (Tavily when `TAVILY_API_KEY` is set, else a DuckDuckGo fallback) and a page-fetch tool, wired into the agent via `create_agent`.

### Provider / API Routing

`agent.py` is the source of truth: `MODEL_PROVIDERS` maps each supported model name to its provider, and `resolve_model()` turns that into the provider-prefixed id (`"<provider>:<model>"`) passed to LangChain's `init_chat_model()`, which builds the actual chat model per provider. `openai_client.py` and `token_manager.py` have been retired — the agent (`agent.py`, built on `create_agent`) and its middleware now own model routing and context trimming.

- OpenAI models use `init_chat_model` with the `openai` provider
- xAI models use the `xai` provider
- Gemini models use the `google_genai` provider

Do not document or add models outside `MODEL_PROVIDERS` unless the code is updated as well.

### Message Flow

1. A text or photo message arrives.
2. `handlers.extract_keyword()` checks for `chatgpt` and optional `@BOT_USERNAME`.
3. Authorization is checked.
4. The incoming user message is stored in `messages`.
5. History is loaded from the checkpoint thread for the chat.
6. `agent.py`'s trimming middleware (`wrap_model_call`) keeps as much recent context as possible while reserving response tokens.
7. `prompt_builder` builds the system prompt and provider-specific message format.
8. `agent.run()` invokes the LangChain agent, which calls the active provider.
9. On success, the assistant response is stored and sent back to Telegram. API failures raise `CompletionError` and are shown to the user without persisting an assistant message.

### Group Chat Behavior

- Text messages in groups are stored even when they do not trigger the bot.
- This storage happens only for text messages; non-triggering photo posts are ignored.
- Group user messages are formatted as `[Name]: message` before model submission.
- Stored group messages in the application `messages` table currently have no retention limit. The previous probabilistic database cleanup remains disabled.
- Latest LangGraph checkpoint state uses a temporary 500→400 message cap. Historical checkpoint retention, summarization, and durable long-term memory remain future context-engineering work.

### Image Handling

- `photo_handler()` only processes images when the caption activates the bot.
- The database stores a text marker such as `[image] <caption>`.
- The actual image bytes are converted to a data URL and sent only in the outbound API request.

### Personality Behavior

- Private chats always use the default private system prompt.
- Group chats can use a database-backed personality prompt.
- `active_personality` is a single global setting, not per-chat.
- If the active personality has no matching row in `personality`, the default group prompt is used.

### Active Model Behavior

- On startup, `bot.py` calls `db.init_active_model(config.DEFAULT_MODEL)`.
- After that, the effective model comes from `active_model`, not directly from `.env`.
- `/model` updates the database, then calls `agent.set_model()` to rebuild the live chat model for the new provider.

## Build, Test, and Development Commands

- Install deps: `pip install -r requirements.txt`
- Install test deps: `pip install -r requirements-dev.txt`
- Apply database migrations: `alembic upgrade head`
- Run unit tests: `pytest tests/ -v`
- Run bot locally: `python3 bot.py`
- Start via helper script (creates/uses `venv`): `./start.sh`
- Run CLI simulator: `python3 scripts/chat_cli.py --chat-id test`
- Build container: `docker build -t telegram-gpt .`
- Run with compose/env file: `docker compose up -d --build`

### Local

```bash
python3 bot.py
```

### Startup Script

```bash
./start.sh
```

`start.sh` creates or reuses `venv/`, installs dependencies, applies migrations, and launches the bot.

### Docker

```bash
docker compose up -d --build
docker compose logs -f
docker compose down
```

### CLI Simulator

```bash
python3 scripts/chat_cli.py --chat-id test
python3 scripts/chat_cli.py --chat-id test --group
```

### Unit Tests

```bash
pip install -r requirements.txt -r requirements-dev.txt
python3 -m py_compile *.py
pytest tests/ -v
```

Tests cover pure logic only (no Telegram, database, or live API calls):

- `handlers.extract_keyword()` — activation keyword and `@mention` stripping
- `prompt_builder.PromptBuilder.format_messages()` — group prefixes and vision payload formatting
- `agent.trim_messages()` / `agent.count_message_tokens()` — context window trimming (`tests/test_trimming.py`)
- `agent.resolve_model()` / `MODEL_PROVIDERS` — model/provider validation used by `/model` (`tests/test_model_resolution.py`)
- `tests/test_config.py`, `tests/test_prompt_builder.py`, `tests/test_tools.py`, `tests/test_agent.py`, `tests/test_extract_keyword.py` cover config validation, prompt formatting, tools, agent wiring, and keyword extraction respectively

CI runs the same compile and pytest steps on pull requests and pushes to `main` (`.github/workflows/ci.yml`).

## Deployment

- Two Railway environments: `production` (tracks the `main` branch) and `dev` (tracks the `dev` branch), each with its own Telegram bot and Neon database branch.
- `.github/workflows/deploy-railway.yml` auto-deploys on push: `dev` → the Railway `dev` environment, `main` → `production`. No manual `railway up` needed for normal development.
- Each environment's Railway `preDeployCommand` runs `alembic upgrade head && python scripts/setup_checkpointer.py` before the bot starts — the first applies the Alembic-managed app schema, the second (idempotent) creates/upgrades the LangGraph checkpointer tables, which are versioned by `langgraph-checkpoint-postgres` and intentionally not part of Alembic.

## Branching & Release Workflow

- Two long-lived branches: `dev` (staging) and `main` (production). Do new work on `dev`, or on a short-lived branch merged into `dev`.
- Pushing to `dev` auto-deploys to the Railway `dev` environment (its own bot + Neon database branch) — use this to verify changes against a real bot before they reach users.
- Promote `dev` → `main` via a pull request (`gh pr create --base main --head dev`), not a local merge and direct push. This keeps a reviewable diff and CI status visible before anything reaches production.
- `main` is a protected branch: direct pushes are blocked and the `CI` workflow must pass before a PR can merge.

## Database Schema

Expected tables:

- `messages`
- `granted_users`
- `personality`
- `active_personality`
- `active_model`

Important details:

- `granted_users` includes `first_name` and `username`
- `active_model` persists the globally selected model
- `active_personality` is a single-row table
- Schema is version-controlled via Alembic migrations in `alembic/versions/`, applied with `alembic upgrade head` (not created automatically on boot)

## Configuration

Relevant environment variables:

- `TELEGRAM_BOT_TOKEN`
- `BOT_USERNAME`
- `OPENAI_API_KEY`
- `XAI_API_KEY`
- `GEMINI_API_KEY`
- `DEFAULT_MODEL`
- `OPENAI_TIMEOUT`
- `MAX_CONTEXT_TOKENS`
- `MAX_OUTPUT_TOKENS`
- `TAVILY_API_KEY`
- `MAX_GROUP_CONTEXT_MESSAGES`
- `AUTHORIZED_USER_ID`
- `DATABASE_URL`
- `LOG_LEVEL`

Important notes:

- `DEFAULT_MODEL` is only the seed value for a fresh database
- `config.py` validates the API key required by `DEFAULT_MODEL`
- The running bot may use a different model if `/model` has changed `active_model`
- `TAVILY_API_KEY` is optional; when blank, `tools.py` falls back to a DuckDuckGo-backed web search tool instead of Tavily

## Commands

All commands are main admin only (gated by `is_main_authorized_user()` in `handlers.py`); granted users can chat with the bot but cannot run any command:

- `/clear`
- `/stats`
- `/version`
- `/grant <user_id>`
- `/revoke <user_id>`
- `/allowlist`
- `/model [name]`
- `/personality [name]`
- `/list_personality`
- `/help`

## Coding Style & Naming Conventions

- Target Python 3.12+ and keep code compatible with async patterns already used.
- Use 4-space indentation, `snake_case` for functions/variables, `PascalCase` for classes, and UPPER_CASE for constants.
- Prefer type hints (`str | None`, `list[dict]`) and short docstrings on public methods.
- Keep modules focused; avoid scattering shared logic (for example, centralize prompt construction in one helper).
- No formatter/linter is currently enforced; match existing style and keep imports and logging consistent.

## Testing Guidelines

Minimum validation before PR:

```bash
pip install -r requirements.txt -r requirements-dev.txt
python3 -m py_compile *.py
pytest tests/ -v
python3 scripts/chat_cli.py --chat-id test
```

If Telegram credentials are available, verify:

- `/clear`
- `/stats`
- `/model`

For new pure logic (keyword parsing, prompt formatting, token trimming, model registry), add focused cases under `tests/` rather than broad refactors or integration harnesses.

## Commit & Pull Request Guidelines

- Follow the repository’s commit style: short, imperative subject lines (examples: `Fix /list_personality`, `Add VM auto-deploy workflow`, `Update db docs`).
- Keep commits scoped to one concern; avoid mixing feature work and cleanup.
- PRs should include:
  - What changed and why
  - Any env/config changes (for example `.env.example` updates)
  - Validation steps and observed results
  - Linked issue(s) when applicable

## Security & Configuration Tips

- Never commit secrets; use `.env` and keep `.env.example` as the template.
- Treat `DATABASE_URL`, `OPENAI_API_KEY`, and `TELEGRAM_BOT_TOKEN` as sensitive.
- Prefer least-privilege settings for deployment credentials and rotate keys if exposed.
