# LangChain Agent Migration — Design

## Context

The bot currently routes every request through `openai_client.py`, which maps
a model name to a provider (OpenAI / xAI / Gemini) and API shape (Responses vs
Chat Completions) via a hand-maintained `MODEL_REGISTRY`, then makes a single
linear request/response call per Telegram message. There is no tool use: the
model can only answer from its own knowledge and the trimmed conversation
history built by `token_manager.py` and `prompt_builder.py`.

Goal: move to an agentic, model-agnostic framework (LangChain / LangGraph) so
the bot can call tools (web search, code execution) in a loop before
answering, while still supporting `/model` switching across OpenAI, xAI, and
Gemini.

## Motivation

- **Tool use is the primary driver.** The bot should be able to search the
  web and execute code as tools, reasoning over the results before
  responding — not just single-shot completions.
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
- No change of persistence backend — stays on the existing Neon Postgres
  database.
- No DB-read introspection tools (e.g. "what model/personality am I using").
  The agent doesn't need to re-fetch state that's already implicit in its
  system prompt (personality/model) or already present in checkpoint state
  (conversation history). Tool set stays limited to web search/fetch and
  sandboxed code execution.

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
  same `DATABASE_URL`, keyed by `thread_id = chat_id`. This becomes the
  LLM-facing working memory, replacing `token_manager.trim_to_fit()`'s manual
  accounting with a pre-model trimming middleware — still token-aware, reusing
  the existing tiktoken-based counting and `MAX_CONTEXT_TOKENS` /
  `RESERVE_TOKENS_TEXT` / `RESERVE_TOKENS_IMAGE` budgets, just expressed as a
  middleware hook instead of a standalone trim step.
- **Tools** are plain LangChain `@tool`-decorated functions in a new
  `tools.py`, available globally (any triggered message may result in the
  agent invoking them):
  - Web search + page fetch
  - Sandboxed code execution

## Components

- **`agent.py`** *(new, replaces `openai_client.py`)* — resolves the active
  model from the database, attaches tools, attaches the trimming middleware
  and the dynamic system-prompt hook, compiles the LangChain/LangGraph agent
  with the Postgres checkpointer. Recompiles only on `/model` change.
  Exposes a `get_completion()`-equivalent entry point so `handlers.py` needs
  minimal changes. Owns the `CompletionError` contract (see Error Handling).
- **`tools.py`** *(new)* — the two tool groups described above.
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
   configured token budget, the agent runs its tool-calling loop (web search /
   code exec as needed), and produces a final answer.
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

## Open Questions / Assumptions Carried Forward

- Exact tool implementations (which web search provider, which code-exec
  sandbox) are left to the implementation plan, not fixed here.
