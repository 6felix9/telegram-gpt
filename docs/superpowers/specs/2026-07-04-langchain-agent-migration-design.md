# LangChain Agent Migration — Design

## Context

The bot currently routes every request through `openai_client.py`, which maps
a model name to a provider (OpenAI / xAI / Gemini) and API shape (Responses vs
Chat Completions) via a hand-maintained `MODEL_REGISTRY`, then makes a single
linear request/response call per Telegram message. There is no tool use: the
model can only answer from its own knowledge and the trimmed conversation
history built by `token_manager.py` and `prompt_builder.py`.

Goal: move to an agentic, model-agnostic framework (LangChain / LangGraph) so
the bot can call tools (web search) in a loop before answering, while still
supporting `/model` switching across OpenAI, xAI, and Gemini.

## Motivation

- **Tool use is the primary driver.** The bot should be able to search the
  web as a tool, reasoning over the results before responding — not just
  single-shot completions.
- **Model-agnosticism is a hard requirement, not a nice-to-have.** `/model`
  switching across three providers must keep working; this is the core
  reason LangChain (rather than a single-provider SDK) was chosen.
- The user is open to rewriting the surrounding memory/prompt/token layers if
  it produces a cleaner overall framework, provided the data continues to
  live in the existing Neon Postgres database.

## Non-Goals

- No multi-agent orchestration, planning, or subagent delegation (Deep
  Agents-style). This is a single agent with a fixed tool set.
- No custom LangGraph control flow beyond a standard tool-calling loop
  (reflection, branching, human-in-the-loop) unless a concrete need surfaces
  later.
- No change to Telegram-facing behavior: keyword/mention activation,
  allowlist authorization, group `[Name]:` formatting, personality selection,
  and admin commands are preserved as-is from the user's perspective — with
  one explicit, one-time exception: see **Cutover** below.
- No change of persistence backend — Railway dev/prod, and local development,
  stay on the existing Neon Postgres database. `DATABASE_URL` is a required
  variable (see Configuration); no SQLite fallback is introduced.
- No DB-read introspection tools (e.g. "what model/personality am I using").
  The agent doesn't need to re-fetch state that's already implicit in its
  system prompt (personality/model) or already present in checkpoint state
  (conversation history). Tool set stays limited to web search/fetch.

## Cutover

The new `PostgresSaver` checkpointer starts with empty state per thread —
it does not share rows with, or get backfilled from, the existing `messages`
table. This means every chat's model-facing context resets to empty at the
moment the migration ships; no backfill script will be written. This is a
one-time, accepted behavior change carved out of the Non-Goals above, not an
ongoing difference. `messages` itself is unaffected and keeps accumulating
history for audit/stats purposes as before (see Components).

## Architecture

`openai_client.py` is replaced by a new `agent.py` built on LangChain's
`create_agent(model, tools=[...])`, which runs on LangGraph internally.

- **Model resolution**: `init_chat_model()` with a provider-prefixed model id
  (`openai:gpt-5`, `xai:grok-4-1-fast-reasoning`,
  `google_genai:gemini-3-flash-preview`) replaces the manual
  `MODEL_REGISTRY` dispatch. `/model` continues to update `active_model` in
  the database; `agent.py` re-resolves the model string and recompiles the
  agent, the same way `OpenAIClient.set_model()` re-initializes its client
  today.
- **System prompt** is resolved dynamically, not baked into the compiled
  graph. `agent.py` attaches a dynamic system-prompt hook (LangChain's
  per-invocation prompt middleware) that, at call time, builds the prompt
  from the current chat's private-vs-group state and the current
  `active_personality` value via `prompt_builder.py`. This means `/personality`
  changes take effect on the next message with no agent recompilation —
  only `/model` changes require a recompile, since the underlying chat model
  object itself changes.
- **Conversation state** moves to a `PostgresSaver` checkpointer against the
  same `DATABASE_URL` (a required variable — see Configuration), keyed by
  `thread_id = chat_id`. This becomes the
  LLM-facing working memory, replacing `token_manager.trim_to_fit()`'s manual
  accounting with a pre-model trimming middleware — still token-aware, reusing
  the existing tiktoken-based counting and `MAX_CONTEXT_TOKENS` /
  `RESERVE_TOKENS_TEXT` / `RESERVE_TOKENS_IMAGE` budgets, just expressed as a
  middleware hook instead of a standalone trim step.
- **Tools** are plain LangChain `@tool`-decorated functions in a new
  `tools.py`, available globally (any triggered message may result in the
  agent invoking them):
  - Web search + page fetch

### Checkpointer Schema

The `PostgresSaver` checkpointer's tables (`checkpoints`, `checkpoint_blobs`,
`checkpoint_writes`, `checkpoint_migrations`) are **not** brought under
Alembic. `langgraph-checkpoint-postgres` ships its own internal, versioned
migration system (tracked via its `checkpoint_migrations` table) that evolves
independently across `langgraph` package upgrades; hand-copying its DDL into
an Alembic migration would drift the first time that package updates its
schema. Instead:

- Alembic continues to own the application schema exactly as today
  (`messages`, `granted_users`, `personality`, `active_personality`,
  `active_model`).
- `PostgresSaver.setup()` is called once as its own explicit step in each
  Railway environment's `preDeployCommand`, immediately after
  `alembic upgrade head` — not implicitly on bot boot or on first checkpoint
  write. This preserves the project's existing "nothing is created implicitly
  at runtime" convention (see `database.md`) while accepting the checkpointer
  as a separately-versioned subsystem with its own migration tool.
- Local development runs `PostgresSaver.setup()` manually (documented in
  `README.md`/`start.sh`) the same way `alembic upgrade head` is already run
  today.

## Components

- **`agent.py`** *(new, replaces `openai_client.py`)* — resolves the active
  model from the database, attaches tools, attaches the trimming middleware
  and the dynamic system-prompt hook, compiles the LangChain/LangGraph agent
  with the Postgres checkpointer. Recompiles only on `/model` change.
  Exposes a `get_completion()`-equivalent entry point so `handlers.py` needs
  minimal changes. Owns the `CompletionError` contract (see Error Handling).
- **`tools.py`** *(new)* — the web search/fetch tool described above.
- **`database.py`** — `granted_users`, `personality`, `active_personality`,
  `active_model` stay as-is; this is admin configuration, not part of the
  linear-model problem. The existing `messages` table is repurposed as an
  audit/stats log only — `/stats` and group `[Name]:` display keep reading
  from it, but the agent no longer uses it as its context source; the
  checkpointer does. `cleanup_old_group_messages()`'s existing 10%
  probabilistic cleanup keeps running unchanged against this table — it
  bounds audit-log size and is unrelated to the new checkpointer/context
  system.
- **`prompt_builder.py`** — slimmed to two responsibilities: (a) building the
  system prompt string from personality/private-vs-group state (logic
  unchanged), (b) converting a stored/incoming message into LangChain message
  objects (`HumanMessage` with text/image content blocks) instead of raw
  OpenAI-style dicts.
- **`token_manager.py`** — retired as a standalone module. Its token-counting
  logic is absorbed into the new trimming middleware in `agent.py`.

## Data Flow

1. A text or photo message arrives; `handlers.extract_keyword()` and
   authorization checks run unchanged.
2. **Non-triggering group text messages**: appended to checkpoint state via
   `graph.update_state()` (no model call), and still logged to `messages` for
   `/stats` and existing cleanup behavior — group context-building does not
   require an LLM round-trip.
3. **Triggering messages**: `agent.py` invokes the compiled graph with
   `thread_id=chat_id`. The dynamic system-prompt hook resolves the correct
   prompt for this chat, the trimming middleware prunes history to fit the
   configured token budget, the agent runs its tool-calling loop (web search
   as needed), and produces a final answer.
4. On success: the response is logged to `messages` (audit) and sent to
   Telegram. On failure: the same `CompletionError`-with-safe-message contract
   as today is raised and shown to the user; nothing is persisted to
   checkpoint state for that turn.
5. `/clear` deletes that thread's checkpoints via the Postgres checkpointer's
   delete API, instead of deleting rows from `messages`.

## Error Handling

Provider errors (auth, rate limit, timeout, bad request, connection,
internal server error) are normalized the same way as today: LangChain's
`init_chat_model` wraps each provider's SDK, so the existing
`except openai.AuthenticationError` / `RateLimitError` / etc. blocks are
replaced with LangChain's equivalent exception types, but map to the same
user-safe `CompletionError` messages `handlers.py` already expects. No
observable change to error messages shown in Telegram.

## Configuration

`.env.example` and `config.py`'s `validate()` are restructured around a much
smaller required set, reflecting that most settings have sane defaults and
only a few genuinely block startup.

**Required:**

- `TELEGRAM_BOT_TOKEN`
- `AUTHORIZED_USER_ID`
- `OPENAI_API_KEY` — now unconditionally required. Validation no longer
  cross-checks `OPENAI_API_KEY`/`XAI_API_KEY`/`GEMINI_API_KEY` against
  `DEFAULT_MODEL`'s provider; if `DEFAULT_MODEL` (or a later `/model` switch)
  selects a provider whose key is missing, that surfaces as a clear runtime
  error on first use (e.g. "xAI API key is not set"), not a startup failure.
- `DATABASE_URL` — no SQLite fallback; required for every environment,
  including local development. Backs `messages`/admin tables as today, and
  now also the LangGraph checkpointer.

**Optional, with defaults when unset:**

| Var | Default | Notes |
|---|---|---|
| `BOT_USERNAME` | `""` | Disables `@mention` activation; `chatgpt` keyword activation still works. `handlers.extract_keyword()` already no-ops mention detection when falsy. |
| `XAI_API_KEY` / `GEMINI_API_KEY` | `""` | Only needed if that provider's models are actually used. |
| `DEFAULT_MODEL` | `gpt-5.4-mini` | Seed value for a fresh database only; `active_model` in the DB wins after first run, unchanged from today. |
| `OPENAI_TIMEOUT` | `60` | Unchanged. |
| `MAX_CONTEXT_TOKENS` | `16000` | Unchanged. |
| `RESERVE_TOKENS_TEXT` | `2000` | Was `1000`. |
| `RESERVE_TOKENS_IMAGE` | `3000` | Unchanged. |
| `MAX_GROUP_CONTEXT_MESSAGES` | `500` | Was `100` in `config.py` / `300` in `.env.example` (previously inconsistent). |
| `TAVILY_API_KEY` | `""` | Powers the agent's web search tool. When unset/blank, web search falls back to a DuckDuckGo-backed tool automatically — no startup error either way. |
| `LOG_LEVEL` | `INFO` | Unchanged. |

**`.env.example` layout:** required vars keep instructive placeholder values
(e.g. `your_bot_token_here`) since they have no code default. Every optional
var is left blank and instead gets an inline comment naming its code
default — so a glance at the file shows exactly what's mandatory (has a
placeholder) versus optional (blank + "defaults to X"). Concretely:

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

## Testing

Same shape as today's `tests/` suite (pure logic, no database/`.env`/live
API calls):

- `handlers.extract_keyword()` — unchanged, no changes expected.
- `prompt_builder` — tests updated to assert LangChain message object output
  (text and image content blocks) instead of OpenAI dict shapes, plus the
  dynamic system-prompt hook: private chat → default prompt, group chat →
  current `active_personality` prompt, falling back to the default group
  prompt when the personality has no matching row.
- Trimming middleware — replaces `tests/test_token_manager.py`; same
  token-budget scenarios, adapted to the new middleware's interface.
- Model resolution — replaces the `MODEL_REGISTRY` validation tests; asserts
  provider-prefixed model ids resolve to the right provider for `/model`.
- Agent/tool tests — use LangChain's fake/test chat model utilities so tool
  invocation logic (which tool gets called, argument parsing) is verified
  without a live API or database, consistent with the existing test
  philosophy.
- Config validation — new tests asserting only `TELEGRAM_BOT_TOKEN`,
  `AUTHORIZED_USER_ID`, `OPENAI_API_KEY`, and `DATABASE_URL` are required;
  that a missing `XAI_API_KEY`/`GEMINI_API_KEY` no longer fails startup even
  when `DEFAULT_MODEL` selects that provider; and that defaults apply
  correctly when optional vars are unset (per the Configuration table
  above).

## Open Questions / Assumptions Carried Forward

- The image-storage-window feature (persisting images for reuse in context,
  bounded to a recent-messages window) is deferred to a separate future spec
  and is out of scope for this document.
