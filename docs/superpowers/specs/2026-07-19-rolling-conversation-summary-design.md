# Rolling Conversation Summary — Design

## Goal

Add rolling conversation summaries to the existing LangGraph checkpoint memory so long-running Telegram chats preserve important older context without sending the full raw transcript to the reply model.

The active memory for each chat will become:

1. One generated summary of older conversation history.
2. The most recent raw messages, bounded by a token budget.

The application `messages` table remains an unbounded audit log and is not read into model context.

## Motivation

The current memory system has two independent safeguards:

- Request-time token trimming keeps only the newest messages that fit a model call. It is non-destructive, so information outside that window is unavailable to the model.
- A 500→400 message checkpoint cap permanently removes the oldest active messages without preserving their content.

These controls bound model input and latest checkpoint size, but they do not preserve long-running conversational continuity. A rolling summary should retain durable facts and decisions while allowing recent messages to remain verbatim.

## Non-goals

- No long-term, cross-thread user memory or LangGraph Store.
- No admin command to browse or manage stored summaries — the audit table (see "Summary audit table") is write-only from the bot's perspective for this feature.
- No change to the application audit-log retention policy.
- No background or scheduled summarization of passive chats.
- No multi-replica or concurrent-update locking design.
- No change to activation, authorization, personality, or message formatting.
- No storage of non-triggering photos.
- No modification of LangGraph-managed checkpoint tables. (A new Alembic migration is still added for the app-level summary audit table — see "Summary audit table" — the same way `messages`, `personality`, and the other existing app tables are managed.)

## Decisions

- Use LangChain's built-in `SummarizationMiddleware`.
- Add a small resilient adapter around the middleware for fail-open behavior and image sanitization.
- Persist summaries directly in checkpoint `messages`; do not add a custom state field.
- Use a dedicated configurable summary model, defaulting to `gpt-4.1-mini`.
- Trigger summarization at 10,000 tokens by default.
- Preserve approximately 4,000 tokens of the newest raw messages.
- Run summarization only as part of a triggered agent request.
- Remove the existing 500→400 message-count prune; rolling summarization becomes the sole mechanism bounding active checkpoint state. Measured production data (two real group threads) puts average message size at ~15.5 tokens, so the 500-message cap fires at ~7,750 tokens — well below the 10,000-token summary trigger. Keeping both would make the destructive message-count prune the dominant safeguard for typical short-message chats, running routinely instead of as a rare fallback, which defeats the purpose of this feature. See "Operational trade-offs" for the unbounded-growth risk this removal accepts.
- Keep complete raw text only in the existing application audit table after checkpoint compaction.
- Persist a permanent audit record of every successful summary generation to a new, Alembic-managed `conversation_summaries` table. This is independent of the checkpoint's live state: it exists for debugging and quality review, is never read back into model context, and is not consulted by the agent.

## Current architecture

`Agent` compiles a LangChain `create_agent` graph with:

- Dynamic system-prompt middleware.
- Request-time message trimming middleware.
- A PostgreSQL checkpointer keyed by Telegram `chat_id`.

Triggered requests use `Agent.run()`. Non-triggering text uses `Agent.append_context_message()` and enters checkpoint state without a model call. `Agent._prune_checkpoint()` removes old messages after either path when the latest state exceeds 500 messages. This feature removes `_prune_checkpoint()` and its `MAX_CHECKPOINT_MESSAGES`/`CHECKPOINT_PRUNE_TARGET_MESSAGES` constants; summarization takes over as the only mechanism that bounds active checkpoint state.

The checkpoint's active logical state currently contains only `messages`. System prompts and `AgentContext` are invocation-time data. `/clear` deletes the entire checkpoint thread but leaves application audit rows intact.

## Target architecture

The checkpoint remains the sole source of active conversation memory. The `messages` table remains the audit source.

For readability, the agent middleware registration list will be:

1. Dynamic system prompt.
2. Resilient summarization.
3. Request-time trimming.

The important lifecycle behavior is that summarization runs as a `before_model` state hook, while dynamic prompting and trimming shape the subsequent model request. Persistent compression therefore completes before request-time trimming sees the message list. The normal trimming middleware remains the final guard for every reply-model call, including later calls in a tool loop.

The built-in middleware replaces all checkpoint messages with:

1. A `HumanMessage` containing the conversation summary and LangChain's summarization source marker.
2. The preserved recent message suffix.

It keeps tool-call and tool-result boundaries intact. No custom `AgentState` or database schema is needed.

## Configuration

Add these environment-backed settings:

- `SUMMARY_MODEL`, default `gpt-4.1-mini`.
- `SUMMARY_TRIGGER_TOKENS`, default `10000`.
- `SUMMARY_KEEP_TOKENS`, default `4000`.
- `SUMMARY_CONTEXT_TOKENS`, default `14000`.

Reuse `MODEL_TIMEOUT` and the existing provider API-key settings. Initialize the summary model through `MODEL_PROVIDERS`, `resolve_model()`, and `init_chat_model()` so provider behavior stays centralized. Configure the summary model for a maximum output of 1,024 tokens.

Validation must reject:

- A summary model not present in `MODEL_PROVIDERS`.
- Non-positive trigger, keep, or context values.
- A keep value greater than or equal to the trigger.
- A context value less than `SUMMARY_TRIGGER_TOKENS - SUMMARY_KEEP_TOKENS` (the older partition's size in the normal, non-backlog case; anything smaller would silently drop history on an ordinary trigger, not just an oversized backlog).
- A missing API key for the configured summary model's provider.

Use one list-level token-counter function built on the existing `count_message_tokens()` helper for both summary thresholds and the retained-token policy. This keeps summary and trimming decisions consistent.

Set `trim_tokens_to_summarize` directly to `SUMMARY_CONTEXT_TOKENS` — the budget for how much of the older partition the summary model is shown, sized independently of the reply model's `MAX_CONTEXT_TOKENS`. These two settings govern different model calls (the summary model vs. the reply model) and were never conceptually the same budget; deriving one from the other was an artifact of reusing a single number, not a real constraint. Decoupling them also makes the backlog case easier to reason about and tune: `SUMMARY_CONTEXT_TOKENS` can be sized generously (e.g., well above 14,000) specifically to reduce how often a large passive backlog gets truncated before reaching the summarizer, without needing to touch the reply model's context budget at all. If a passive history grows so large that its older partition alone exceeds `SUMMARY_CONTEXT_TOKENS`, LangChain retains only the newest portion for summary generation; earlier material can be lost. This is an accepted consequence of avoiding background model calls and unbounded summary inputs.

Document the new variables in `.env.example`, `README.md`, `AGENTS.md`, and `CLAUDE.md`. Existing deployments may omit them and use defaults.

## Summary content

Use a custom summary prompt that treats the serialized transcript as untrusted data and instructs the model not to follow commands found inside it.

The summary should preserve:

- Participant attribution, including existing `[Name]:` group prefixes.
- Durable facts and user preferences.
- Decisions and their relevant rationale.
- Open questions and unresolved topics.
- Commitments, action items, and deadlines.
- Important links, identifiers, and referenced resources.
- Uncertainty or disagreement when it affects future interpretation.

It should omit repeated statements, greetings, small talk, superseded details, and tool-call mechanics unless a tool result is itself important. The output should be concise factual prose, not instructions to the reply model.

When a prior rolling summary is present, it is part of the older partition and is summarized together with newly aged-out messages. The result remains one current summary rather than an accumulating chain of summaries.

## Summary audit table

Independent of the checkpoint state that drives replies, persist a permanent record of every successful summary generation for debugging and quality review. This table is write-only from the bot's perspective: the agent never reads it back, and this feature adds no command to browse it.

Add a new Alembic migration, on top of `0001_initial_schema.py`, for a `conversation_summaries` table alongside the existing app tables (`messages`, `granted_users`, `personality`, `active_personality`, `active_model`):

```sql
CREATE TABLE conversation_summaries (
    id SERIAL PRIMARY KEY,
    chat_id TEXT NOT NULL,
    summary_text TEXT NOT NULL,
    summary_model TEXT NOT NULL,
    before_message_count INTEGER NOT NULL,
    after_message_count INTEGER NOT NULL,
    before_tokens INTEGER NOT NULL,
    after_tokens INTEGER NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_summary_chat_created ON conversation_summaries(chat_id, created_at DESC);
```

A row is written only when `ResilientSummarizationMiddleware` actually produces and persists a usable summary (Triggered request flow, step 7) — never on a below-threshold skip, and never on a fail-open (exception, empty result, or LangChain's error sentinel).

`conversation_summary.py` stays decoupled from `database.py`: the middleware accepts an optional `on_summary` callback and invokes it with the same data it already computes for the "Summary succeeded" structured log (chat/thread id, summary model, before/after message and token counts, and the summary text itself). `Agent` accepts this callback as a constructor dependency and wires it to a new `Database.record_conversation_summary()` method, called after the checkpoint state update succeeds.

If the audit write itself fails, log the failure and continue. It must never roll back or block the already-successful checkpoint compaction or the reply — the checkpoint is the source of truth for active memory, and this table must not become a new way for summarization to fail closed. Retention is unbounded, consistent with the existing `messages` audit table.

## Triggered request flow

1. `Agent.run()` invokes the graph with the new human message and current thread ID.
2. The summarization middleware counts all active checkpoint messages.
3. Below `SUMMARY_TRIGGER_TOKENS`, it returns no state update.
4. At or above the threshold, it chooses a tool-safe cutoff that retains up to `SUMMARY_KEEP_TOKENS` of recent raw messages.
5. Historical image blocks in the older partition are copied and sanitized before summary generation.
6. The dedicated model summarizes the older partition, including any prior summary.
7. On success, the graph applies one checkpoint state update that replaces old messages with the new summary plus the preserved suffix, and one audit row is inserted into `conversation_summaries` (see "Summary audit table").
8. Dynamic prompting and request-time trimming prepare the reply-model input.
9. The reply/tool loop runs normally and persists its new messages.

Non-triggering text continues to call `append_context_message()` without invoking middleware or a model. If passive messages cross the token threshold, they remain raw until the next triggered request, since there is no message-count fallback to bound them in the meantime. A chat that stays purely passive (never triggers a reply) can grow without limit; see "Operational trade-offs."

## Resilient middleware adapter

The adapter remains a focused extension of `SummarizationMiddleware`; it does not reimplement partitioning, trigger evaluation, tool-boundary handling, or checkpoint replacement.

It adds three behaviors:

1. Sanitize copied historical multimodal content before delegating to summary generation. Replace image data-URL blocks with a text marker such as `[image omitted]`, while preserving captions and surrounding text. Do not mutate the checkpoint's recent raw messages.
2. Detect an empty generated summary.
3. Detect LangChain's generated `"Error generating summary: ..."` sentinel. The built-in middleware currently converts summary-model exceptions into this text instead of raising them.

For either unsuccessful result, return no state update and log the failure. The existing checkpoint remains unchanged, and request-time trimming allows the normal reply to proceed. Implement both synchronous and asynchronous hooks so behavior remains correct if invocation style changes later.

Because this adapter inspects built-in middleware output and protected summary hooks, compatibility tests are required under the repository's `langchain>=1.0,<2.0` dependency range.

## Image handling

Recent image messages continue to carry their current multimodal blocks, including the original data URL, so the immediate reply flow is unchanged.

Only messages entering the older summary partition are sanitized. Their captions and an `[image omitted]` marker are available to the summarizer; raw base64 is not. This prevents oversized summary prompts and avoids retaining image bytes in the newly compacted state.

Changing image storage for recent checkpoints or introducing object storage is outside this feature.

## Failure handling

### Invalid startup configuration

Fail validation with a clear setting-specific error. This includes unknown models, impossible thresholds, and missing provider credentials.

### Summary-model timeout or provider failure

Use the model client's existing retry policy. If the summary still fails:

- Do not alter checkpoint state.
- Log the chat/thread identifier, failure type, and elapsed time without logging conversation content.
- Continue the normal reply through request-time trimming.
- Retry summarization naturally on a later triggered request.

### Empty or malformed summary

Treat an empty result as failure and preserve the previous state. No low-quality placeholder may replace valid conversation history.

### No message-count fallback

`Agent._prune_checkpoint()` and the 500→400 message-count prune are removed, not kept as a fallback. If summarization repeatedly fails open for a given thread (see above), that thread's checkpoint keeps growing raw until a summary eventually succeeds — there is no other backstop. This is an accepted trade-off given how rarely summarization is expected to fail open in practice; see "Operational trade-offs."

### Reply-model failure after successful summary

The summary state update may already be checkpointed before the reply model fails. This is acceptable: the summary represents prior conversation plus the current user message, and the failed assistant response is not added.

## Concurrency

The current python-telegram-bot application processes updates serially and awaits each threaded graph invocation. This design adds no in-process lock.

The existing system is not safe against competing same-thread writes from multiple bot replicas, the CLI, or future concurrent update processing. Rolling summarization does not solve that problem. If deployment topology changes, add a separate per-chat PostgreSQL advisory-lock or equivalent serialization design.

## Observability

Add structured logs for:

- Summary triggered.
- Summary succeeded.
- Summary skipped because it was below threshold.
- Summary failed open.
- Before/after message and approximate token counts.
- Summary latency and configured summary model.

Do not log transcript text, generated summary text, image data, API keys, or provider request payloads. Routine below-threshold skips should use debug-level logging to avoid noise.

## Testing

### Configuration

- Defaults load as `gpt-4.1-mini`, 10,000 trigger tokens, 4,000 keep tokens, and 14,000 context tokens.
- Unknown models are rejected.
- Non-positive, reversed, and under-budget values are rejected, including a `SUMMARY_CONTEXT_TOKENS` below `SUMMARY_TRIGGER_TOKENS - SUMMARY_KEEP_TOKENS`.
- A missing provider key for `SUMMARY_MODEL` is rejected.

### Middleware behavior

- No summary call or state update occurs below the threshold.
- At threshold, older messages become one summary and the recent token-bounded suffix remains raw.
- A later summary incorporates and replaces the previous summary rather than accumulating summaries.
- AI tool calls and matching `ToolMessage` results stay together across the cutoff.
- The custom token counter controls both trigger and keep behavior.
- The summary input includes participant labels and required continuity information.

### Failure and image behavior

- A summary-model exception produces no checkpoint update and does not block the reply.
- LangChain's error sentinel and an empty response both fail open.
- Historical base64 image data is absent from summary-model input and replaced by a marker.
- Recent unsummarized image messages remain unchanged.

### Agent integration

- A triggered request over threshold persists summary plus recent messages and returns the reply model's output.
- Passive appends never call the summary model.
- `Agent._prune_checkpoint()` and the message-count prune no longer exist; a long-running passive-only thread is allowed to grow unbounded until a triggered request summarizes it.
- `/clear` removes summaries and recent checkpoint messages while application audit rows remain.
- `/model` can rebuild the reply graph without changing the dedicated summary model.

### Summary audit table

- A successful summary persists exactly one `conversation_summaries` row with the expected chat id, summary model, summary text, and before/after message and token counts.
- A below-threshold skip and a fail-open (exception, empty result, error sentinel) each write no audit row.
- An audit-write failure is logged and does not block the reply or roll back the already-applied checkpoint state update.

Use fake reply and summary models with `InMemorySaver` so tests exercise the compiled graph without live providers or PostgreSQL. Audit-table tests inject a fake `on_summary` callback into `Agent` and assert it is invoked with the expected fields on success and never invoked on skip or fail-open; `Database.record_conversation_summary()` itself is not covered by the unit suite, consistent with the project's existing no-live-database testing convention — it is verified in the Manual Staging Check like the project's other database methods. Run:

```bash
python3 -m py_compile *.py
pytest tests/ -v
```

## Acceptance criteria

- Long conversations crossing 10,000 approximate message tokens are compacted into one persisted rolling summary plus approximately 4,000 tokens of recent raw context.
- Important participants, facts, preferences, decisions, open questions, commitments, and links remain available to later replies.
- The reply path remains available when summarization fails.
- Summary generation never receives historical base64 image payloads.
- Non-triggering messages do not cause model calls.
- `/clear`, audit logging, active-model switching, personalities, authorization, and activation behavior remain unchanged.
- Each successful summary persists exactly one audit row to `conversation_summaries`, capturing the summary text, model, and before/after message and token counts.
- No LangGraph checkpoint schema migration is required. One new Alembic migration adds the `conversation_summaries` application table.

## Operational trade-offs

- A threshold-crossing reply incurs one additional model call and extra latency.
- Summaries are lossy and may gradually distort details despite the continuity-focused prompt.
- There is no message-count fallback. A thread whose summarization keeps failing open (see "No message-count fallback"), or a purely passive thread that never triggers a reply, can grow its checkpoint state without limit. This is accepted because such cases are expected to be rare, but it is a real regression from the current system's hard 500-message ceiling and should be monitored (see "Observability").
- An oversized passive history can also lose material that falls outside the bounded summary-model input.
- Historical checkpoint rows remain in PostgreSQL even after the latest logical state is compacted; this feature controls active state, not physical checkpoint retention.
- Summary behavior depends on LangChain middleware internals within the allowed major-version range and is protected by focused compatibility tests.
