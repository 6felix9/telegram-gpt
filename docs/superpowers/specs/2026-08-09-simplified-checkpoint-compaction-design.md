# Simplified Checkpoint Compaction

**Date:** 2026-08-09
**Status:** Approved for planning
**Replaces:** the rolling-summary design in `2026-07-19-rolling-conversation-summary-design.md`

## Problem

The current rolling summary keeps a summary plus a sliding window of recent raw
messages. Sizing that window needs three coupled settings
(`SUMMARY_TRIGGER_TOKENS`, `SUMMARY_KEEP_TOKENS`, `SUMMARY_CONTEXT_TOKENS`) with
two interdependent validation rules, and it runs as a `before_model` subclass of
LangChain's `SummarizationMiddleware`. Fitting a persistent state rewrite into a
per-request middleware hook forces a lot of machinery: message-ID snapshot and
restore, a per-invocation "already compacted" flag, and a staged audit record
that can only be written once the generated summary's exact ID is confirmed in
checkpoint state.

Summarization also only runs on triggered requests. A chat that stays passive —
messages stored but no reply ever triggered — grows its active checkpoint state
without bound.

## Goals

- Compaction runs on every checkpoint update, triggered or passive.
- Compacted state is a summary plus at most the last exchange, not a token-sized
  window.
- Fewer configuration settings and fewer interdependent validation rules.
- Less code: no LangChain middleware subclass, no ID-confirmation dance.

## Non-goals

- Changing request-time trimming. `make_trim_middleware` and
  `MAX_CONTEXT_TOKENS` bound what a single model call sees; that is a separate
  job from bounding stored state and is unchanged.
- Changing the `messages` audit table, `MESSAGE_RETENTION_DAYS`, or
  `scripts/cleanup_retention.py`.
- Changing image persistence or retrieval.

## Design

### Compaction is an explicit step, not middleware

Compaction moves out of the agent graph into an explicit
`compact_if_needed(chat_id)` step called at the top of both entry points:

```
triggering msg:  compact_if_needed(chat_id)  →  graph.invoke({messages: [new_human]})
passive msg:     compact_if_needed(chat_id)  →  graph.update_state({messages: [new_human]})
```

Compaction therefore operates on the pre-existing checkpoint only. The newly
arrived message is appended afterward and is never part of the summary input.

A `before_model` hook cannot express this. It never runs for passive messages,
because those never enter the graph; and LangGraph merges the incoming human
message into state before the hook fires, so "summarize the checkpoint, then
append the trigger" would require excluding the last element of state by
position. An explicit step states the intent directly.

### The algorithm

In `conversation_summary.py`:

```
msgs = graph.get_state(chat_id).messages
if count_messages_tokens(msgs) < SUMMARIZATION_TRIGGER:
    return                                              # append and move on
summary = summary_model(sanitize_summary_messages(msgs))
summary = validate(summary)                             # raises on unusable output
keep    = msgs[index of last HumanMessage :]
if count_messages_tokens(keep) > SUMMARIZATION_TRIGGER - MAX_SUMMARY_OUTPUT:
    keep = []
graph.update_state({messages: [RemoveMessage(REMOVE_ALL_MESSAGES), summary_msg, *keep]})
record_conversation_summary(...)
```

Four rules make this well-defined:

**Summary input is all of `msgs`.** That includes the previous summary and the
messages about to be kept raw. The summary is rolling, and nothing is lost if
the summary and the kept suffix disagree about what mattered.

**`keep` starts at the last `HumanMessage`.** This is exactly "the last
exchange", and starting on a `HumanMessage` guarantees the resulting list never
leads with a `ToolMessage` orphaned from its `AIMessage` tool call. If the
pre-existing state contains no `HumanMessage`, `keep` is empty.

**`keep` is dropped when oversized.** If the suffix alone exceeds
`SUMMARIZATION_TRIGGER - MAX_SUMMARY_OUTPUT`, it is discarded and only the
summary is kept. Without this, one very large message could leave post-compaction
state above the threshold and re-trigger a summary call on every subsequent
message.

**State is replaced wholesale.** `RemoveMessage(REMOVE_ALL_MESSAGES)` clears the
thread's message list in the same update that writes the summary and suffix, so
there is no intermediate state where the checkpoint is empty.

The summary message is a `HumanMessage` (the format every configured provider
accepts) carrying the summary text under a `## Conversation summary` heading.

### What this removes

From `conversation_summary.py`:

- `ResilientSummarizationMiddleware` is replaced by `ConversationCompactor`, a
  plain object holding the summary model, the threshold, and the audit callback.
  It does not subclass `SummarizationMiddleware` and is not middleware; the
  rename reflects that.
- `_snapshot_message_ids` / `_restore_message_ids` — nothing mutates message IDs
  in place anymore.
- `PendingSummaryAuditRecord` and the staged-record mechanism. Because the
  compactor issues its own `update_state`, a successful return *is* the
  confirmation the audit row needs.
- `before_model` / `abefore_model`.

From `agent.py`:

- `AgentContext.pending_summary_records` and `AgentContext.summary_compacted`.
- `_persist_checkpointed_summary_records`.
- The `SUMMARY_MAX_OUTPUT_TOKENS = 1024` module constant, replaced by
  `config.MAX_SUMMARY_OUTPUT`.
- The summary middleware entry in the `self._middleware` list. The dynamic
  prompt, trimming, and context middleware are unchanged and keep their order.

From `database/`:

- `Database.cleanup_old_group_messages` and
  `MessageRepository.cleanup_old_group_messages`. Already unused; the removal of
  `MAX_GROUP_CONTEXT_MESSAGES` leaves them with no caller and no parameter.

`sanitize_summary_messages`, `_validate_summary`, `SUMMARY_PROMPT`,
`SummaryGenerationError`, and `SummaryAuditRecord` are kept.

### Async

`compact_if_needed` performs a model call, so it must not block the event loop.
The summary call runs under `asyncio.to_thread`, matching how `Agent.run` and
`Agent.persist_image` already call into sync LangChain code.

`Agent.append_context_message` becomes `async` as a result. Its one call site,
`handlers/message_handlers.py:100`, gains an `await`.

### Where compaction does *not* run

`Agent.persist_image` writes an `[image #N]` marker via `update_state` without
compacting first. The marker is small (a caption plus a one-line description),
and it is compacted by the next `run` or `append_context_message` on that chat.
Adding a compaction check there would put a second summary-model call in the
photo path for negligible benefit.

## Configuration

| Change | Setting | Default |
|---|---|---|
| renamed from `SUMMARY_TRIGGER_TOKENS` | `SUMMARIZATION_TRIGGER` | `8000` (was `10000`) |
| new | `MAX_SUMMARY_OUTPUT` | `1000` |
| new default | `SUMMARY_MODEL` | `gpt-5.6-luna` (was `gpt-4.1-mini`) |
| removed | `SUMMARY_KEEP_TOKENS` | — |
| removed | `SUMMARY_CONTEXT_TOKENS` | — |
| removed | `MAX_GROUP_CONTEXT_MESSAGES` | — |

Both new settings are optional and read through the existing `_int_env` helper,
so a blank value in `.env` or Railway means "use the default".

`SUMMARY_CONTEXT_TOKENS` is removed because it is now implied. State is
compacted whenever it crosses `SUMMARIZATION_TRIGGER`, so the summary model's
input is inherently bounded at approximately that value; a separate input budget
has nothing left to protect against.

### Validation

The two coupled checks in `config.py:121-133` (`SUMMARY_KEEP_TOKENS <
SUMMARY_TRIGGER_TOKENS`, and `SUMMARY_CONTEXT_TOKENS >= SUMMARY_TRIGGER_TOKENS -
SUMMARY_KEEP_TOKENS`) are replaced by one:

> `MAX_SUMMARY_OUTPUT` must be less than `SUMMARIZATION_TRIGGER`.

Otherwise a compaction cannot bring state below the threshold and every
subsequent message re-triggers a summary call.

`SUMMARIZATION_TRIGGER` and `MAX_SUMMARY_OUTPUT` join the existing
"must be positive" list; `SUMMARY_TRIGGER_TOKENS`, `SUMMARY_KEEP_TOKENS`, and
`SUMMARY_CONTEXT_TOKENS` leave it.

### Summary model parameters

`make_summary_model` sends `max_tokens=MAX_SUMMARY_OUTPUT` as a hard cap, plus
`reasoning={"effort": "none"}` for OpenAI reasoning models — not the `"low"`
effort it currently inherits from `REASONING_EFFORT_LOW`. On the Responses API
the output cap counts reasoning tokens, so a hard 1000-token cap with low
reasoning effort could be consumed entirely by reasoning and return no visible
summary text.

**Open risk.** If the provider rejects `effort: "none"` for `gpt-5.6-luna`,
every summary call errors and fails open, and state grows unbounded. The first
implementation step is a live call verifying the parameter is accepted and that
a summary comes back within the cap. If it is rejected, fall back to the lowest
supported effort and raise the cap above `MAX_SUMMARY_OUTPUT` to leave room for
reasoning tokens; `MAX_SUMMARY_OUTPUT` then becomes the prompt's stated length
target rather than a hard cap, and the validation rule above still holds.

`REASONING_EFFORT_LOW` continues to govern the reply model in `Agent.set_model`
and is unchanged.

## Error handling

`compact_if_needed` is fully fail-open. A failure in `get_state`, the summary
model call, summary validation, or `update_state` is logged at ERROR with the
thread id, model name, and error type, then swallowed. The turn proceeds on
uncompacted state.

The consequences of sustained failure are bounded on the request side and
unbounded on the storage side: replies keep working, because request-time
trimming against `MAX_CONTEXT_TOKENS` is independent of compaction, but stored
checkpoint state keeps growing. This matches today's behavior. The distinct
ERROR log line is what makes it diagnosable.

Audit rows are written to `conversation_summaries` after `update_state` returns,
inside their own try/except, so an audit failure never affects the checkpoint or
the reply.

Empty and placeholder summaries are rejected by the existing `_validate_summary`
and are treated as failures, so an unusable summary can never replace valid
history.

## Testing

`tests/test_conversation_summary.py` is rewritten around the compactor:

- Below threshold: no summary call, no `update_state`.
- At and above threshold: one summary call whose input is the full message list;
  `update_state` receives `RemoveMessage(REMOVE_ALL_MESSAGES)`, the summary, then
  the kept suffix.
- `keep` selection starts at the last `HumanMessage`, including the case where
  trailing `AIMessage`/`ToolMessage` pairs follow it.
- `keep` is empty when the suffix exceeds
  `SUMMARIZATION_TRIGGER - MAX_SUMMARY_OUTPUT`, and when state contains no
  `HumanMessage`.
- Fail-open on a summary-model exception, on an empty summary, and on a
  LangChain placeholder summary: no `update_state`, no audit row, no raise.
- The audit row is written only after `update_state` returns successfully, and an
  audit failure does not propagate.
- Historical data-URL images are still replaced with `[image omitted]` in the
  summary input, and checkpoint messages are not mutated by that sanitization.

`tests/test_agent.py`: compaction runs before `graph.invoke` on a triggered run
and before the append in `append_context_message`; a compaction failure does not
block either path; the summary middleware is no longer in the middleware list.

`tests/test_config.py`: new defaults, removed settings, and the
`MAX_SUMMARY_OUTPUT < SUMMARIZATION_TRIGGER` rule.

`tests/test_message_handlers.py` and `tests/test_context_message_retention.py`:
`append_context_message` is awaited; `MAX_GROUP_CONTEXT_MESSAGES` and
`cleanup_old_group_messages` are gone from the fixtures.

## Migration and deployment

No database migration. Existing checkpoints need no conversion: a summary
message written by the old LangChain middleware is an ordinary message to the
new compactor and gets folded into the next summary.

`SUMMARY_TRIGGER_TOKENS`, `SUMMARY_KEEP_TOKENS`, `SUMMARY_CONTEXT_TOKENS`, and
`MAX_GROUP_CONTEXT_MESSAGES` are set in the Railway `dev` and `production`
environments. After deploy they are inert — nothing breaks if they linger, but
they should be deleted from both environments to avoid implying they still do
something.

Docs to update: `README.md`, `AGENTS.md`, `CLAUDE.md`, `.env.example`.

## Accepted trade-offs

**Added latency on triggered turns.** A turn that crosses the threshold now
pays a summary-model round-trip before the reply model runs, and the user waits
for both. This is a direct consequence of compacting the checkpoint before
appending the trigger message, which is the ordering this design is built on.

**Passive messages can cost a model call.** A non-triggering message that pushes
state over the threshold causes a summary call with no reply to show for it.
This is the point: it is what bounds passive-only chats, which grow without
limit today.

**Only the last exchange survives verbatim.** After a compaction, anything older
than the last exchange is available only as summarized prose. A follow-up
referring to specific wording from several turns back gets the summary's version
of it.
