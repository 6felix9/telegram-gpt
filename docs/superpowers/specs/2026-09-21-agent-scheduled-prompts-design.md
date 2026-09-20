# Agent-Scheduled Prompts

**Date:** 2026-09-21
**Status:** Approved for implementation
**Issue:** [#75](https://github.com/6felix9/telegram-gpt/issues/75)

## Problem

The bot is purely reactive: it replies only to an incoming message. There is
deliberately no scheduler in the codebase — `CLAUDE.md` notes this repeatedly
(open-access expiry, for instance, is evaluated lazily on read rather than by a
background job). Recurring work a user wants from the bot — a morning digest, a
standup nudge, a weekly summary — has nowhere to live.

Issue #75 proposes an admin `/schedule <prompt>` command that parses a cadence
out of free text, falling back to an inline keyboard when extraction fails. That
design spends most of its complexity on the parsing problem: a hand-rolled
mini-grammar for `every day at HH:MM` / `hourly` / `weekly`, plus the bot's
first `CallbackQueryHandler` to recover when the grammar misses.

This spec takes the other route. The agent already runs a tool loop on every
triggering message; giving it a scheduling tool lets the model do the
natural-language-to-cadence conversion, which is the one part of #75 a regex
cannot do well. The durable half of #75 — table, runner, fire path, SGT math,
missed-run policy — is unchanged and is specified here.

## Goals

- `chatgpt give me a good morning message at 8am every day` creates a durable
  schedule, with no new command and no callback handler.
- Scheduled firings go through the existing request path, so compaction,
  trimming, context assembly and persistence behave identically to a real
  message rather than drifting in a parallel implementation.
- Schedules survive restarts and redeploys; the database is the only source of
  truth.
- Creation, listing and cancellation are all conversational, and every creation
  forces a concrete next-run time back to the user.

## Non-goals

- A `/schedule` command, an inline keyboard, or a `CallbackQueryHandler`. The
  tool surface replaces #75's command surface entirely.
- Editing a schedule. Cancel and re-create.
- Any timezone other than Asia/Singapore.
- Backfilling missed firings after downtime.
- Distributed locking. Railway runs one instance per environment, and dev and
  production hold their schedules in separate Neon branches.

## Decisions taken

These were settled during design and are recorded because each closes an option
#75 left open:

| Question | Decision |
|---|---|
| Command or tool | Tools only — create, list, cancel |
| Who may schedule | Anyone already authorized to chat, including open-access users |
| What fires | The stored prompt, run through the full agent path |
| Cadence format | A 5-field cron string, interpreted in Asia/Singapore |
| One-shot reminders | In scope — `at=` instead of `cron=` |
| Runner | A 30s asyncio poll loop over `next_run_at` |

Allowing any authorized user to schedule is a deliberate widening of #75, which
was admin-only. Because open-access (`/openbot`) authorizes strangers for up to
a few hours, and a schedule they create outlives that window, the abuse surface
is carried by guardrails (below) and by `created_by`, rather than by an
authorization gate.

## Design

### Tool surface — new `scheduling.py`

A new module alongside `image_store.py`, following the same shape: a builder
that binds a `Database` and returns tools scoped to the calling chat through
`ToolRuntime.context.thread_id`. `tools.build_tools()` appends them when `db` is
present, so `scripts/chat_cli.py` gets them too.

```python
def build_schedule_tools(db) -> list:
    """Return schedule_prompt / list_schedules / cancel_schedule bound to db."""
```

**`schedule_prompt(prompt: str, label: str, cron: str | None, at: str | None) -> str`**

Exactly one of `cron` and `at` is supplied. `cron` is a 5-field expression read
in Asia/Singapore; `at` is an ISO-8601 local datetime (`2026-09-22T15:00`) for a
one-shot. `label` is the model's own human phrasing of the cadence ("every
weekday at 9am"), stored and replayed in listings so the user never has to read
cron.

Returns a confirmation the model is instructed to relay verbatim:

```
Scheduled #7 — every day at 8:00am. First run Mon 22 Sep, 8:00am SGT.
Will run: "Post a good morning message for Felix."
```

Because the tool's only success output is that string, a schedule cannot be
created without a concrete next-run time *and* the stored prompt reaching the
user. That is the structural answer to a model that schedules the wrong time,
stores the wrong prompt, or schedules silently.

The docstring carries the model-facing rules: cron is always Asia/Singapore;
only schedule when explicitly asked; always relay the returned confirmation;
prefer `at=` for anything that should happen once; and write `prompt` to be
self-contained, per below.

#### The stored prompt must be self-contained

The model, not the user, decides the exact text stored in `prompt` — it strips
the cadence out of the request and keeps the rest. That text is replayed cold
into the chat at fire time, when nobody is asking and the conversation around
the original request is long gone, so anything deictic is broken or wrong by
then:

| User says | Naive capture | Stored prompt |
|---|---|---|
| "give me a good morning message at 8am" | "give me a good morning message" | "Post a good morning message for Felix." |
| "summarise what we just discussed, every evening" | "summarise what we just discussed" | "Summarise the last 24 hours of messages in this chat." |
| "do that again every Monday" | "do that again" | The resolved action, spelled out |

`"give me"` has no *me* at fire time; `"what we just discussed"` refers to a
conversation that has since been compacted away; `"tomorrow"` means a different
day on every firing. The docstring therefore requires the model to resolve
person, time and reference words into absolutes before storing, and the
confirmation echoes the result so the user can catch a bad rewrite at creation
time rather than discovering it when the schedule fires.

**`list_schedules() -> str`** — id, label, next run in SGT, and a prompt preview,
for the calling chat only.

**`cancel_schedule(schedule_id: int) -> str`** — chat-scoped delete, returning
either a confirmation naming the cancelled label or `Schedule #N not found.`

### Validation and guardrails

Enforced inside `schedule_prompt`, as module constants in `scheduling.py`:

| Guardrail | Value | Reason |
|---|---|---|
| `MAX_SCHEDULES_PER_CHAT` | 10 | Bounds per-chat model spend |
| `MAX_PROMPT_CHARS` | 500 | A schedule is a prompt, not a document |
| `MIN_INTERVAL_MINUTES` | 60 | Rejects `* * * * *` and other runaway crons |
| One-shot horizon | 1 year | Rejects a typo'd year |

The minimum interval is checked by asking croniter for the cron's next *two*
occurrences and requiring at least an hour between them, which catches every
runaway shape without enumerating them.

An invalid cron, a past `at`, a breached guardrail, or supplying both/neither of
`cron` and `at` returns an explanatory string for the model to relay — tool
errors are surfaced to the model, never raised, matching `fetch_url` and
`get_image`.

**Duplicate protection.** A `(chat_id, cron, prompt)` that already exists
returns the existing id rather than inserting a twin, so a retried tool call is
idempotent.

### Storage — Alembic `0005_scheduled_prompts.py`

| Column | Type | Notes |
|---|---|---|
| `id` | serial PK | The id users cancel by |
| `chat_id` | text | Scopes every read and delete |
| `prompt` | text | Run verbatim at fire time |
| `label` | text | Model's human phrasing, shown in listings |
| `cron` | text NULL | NULL for a one-shot |
| `next_run_at` | timestamptz | Stored UTC, computed in SGT |
| `enabled` | boolean | Cleared by the failure cutout |
| `created_by` | bigint | Telegram user id, for traceability |
| `created_at` | timestamptz | |
| `last_run_at` | timestamptz NULL | |
| `consecutive_failures` | integer | Drives the cutout |

Indexed on `(enabled, next_run_at)` for the poll query and on `chat_id` for
listings.

A new `database/schedule_repository.py` sits behind the `Database` facade
alongside the five existing repositories, sharing the same `ConnectionManager`.
Schedule rows are not cached: the poll loop needs current data, and the row
count is trivial.

### Runner — new `scheduler.py`

An asyncio task started from `bot.py`'s `post_init` and cancelled in
`post_shutdown`. Every 30 seconds it selects `enabled AND next_run_at <= now()`,
fires each due row, and advances it.

A poll loop rather than PTB's `JobQueue` because the database is then the single
source of truth. There is no in-memory registry to re-register at startup or
mutate on every create and cancel, and nothing to drift out of sync with the
table. The cost is that a firing can be up to 30 seconds late, which does not
matter for a daily digest.

Advancing: `croniter(cron, now_sgt).get_next()` converted to UTC. A one-shot row
is deleted once it has fired. Because the next run is always computed from *now*
rather than from the missed fire time, downtime simply skips occurrences — no
backfill burst, as #75 specifies.

### Fire path — reusing the real request path

`RequestProcessor.process()` cannot be called as-is: it is shaped around a
Telegram `message` object, using `message.reply_text` to respond and
`message.message_id` for the audit row.

Its body is extracted into `process_agent_turn(chat_id, build_payload, send,
...)`, which takes a `send(text)` callable instead of a message. The existing
`process()` becomes a thin wrapper passing `message.reply_text`; the scheduled
runner passes `bot.send_message` bound to the chat. Both paths then share one
implementation of the audit-log → `agent.run` → audit-log → reply workflow, so
scheduled firings inherit compaction, trimming and context assembly rather than
reimplementing them.

Per firing:

1. Audit-log the user row: the stored prompt, `user_id=created_by`,
   `message_id=NULL`.
2. `agent.run(chat_id, human_message, is_group)`.
3. Audit-log the assistant row.
4. Send the reply into the chat.
5. Set `last_run_at`, clear `consecutive_failures`, advance `next_run_at`.

Scheduled runs bypass the per-user authorization check — nobody is asking at
fire time — but target a chat that was authorized when the schedule was created.
They use whatever `active_model` and `active_personality` are live at fire time,
and their `messages` rows are ordinary rows subject to `MESSAGE_RETENTION_DAYS`.

### Error handling

Everything about a firing fails quietly, because no user is waiting on it:

- A failed firing is logged, posts nothing to the chat, increments
  `consecutive_failures`, and still advances `next_run_at` so one bad day does
  not wedge the schedule. A one-shot is deleted whether or not its firing
  succeeded — it is never retried.
- At `MAX_CONSECUTIVE_FAILURES` (5) the row is disabled, so a permanently broken
  schedule cannot burn a model call every day forever. The next
  `list_schedules` shows it as disabled.
- An exception escaping a single firing is caught inside the loop, so it can
  never kill the runner task or take down the bot.

### Configuration

No new environment variables. The poll interval, guardrails and failure cutout
are module constants — none of them is something an operator would tune per
environment, and `config.py` stays focused on credentials and model settings.

One new dependency: `croniter` (small, pure-Python) in `requirements.txt`.

## Testing

`tests/test_scheduling.py`, pure logic with no database or network, matching the
existing suite's scope:

- Cron validation accepts the common shapes (`0 8 * * *`, `0 9 * * 1-5`,
  `0 */2 * * *`) and rejects malformed input.
- Each guardrail rejects: an eleventh schedule, an over-long prompt, a cron
  firing more often than hourly, a past `at`, and both/neither of `cron` and
  `at`.
- Next-run computation is correct across an SGT midnight boundary and converts
  to UTC correctly.
- Duplicate `(chat_id, cron, prompt)` returns the existing id.
- The confirmation string contains the next run in SGT and the stored prompt, so
  a regression that drops either from the tool's output fails a test.
- `cancel_schedule` and `list_schedules` refuse another chat's ids.
- `process_agent_turn` extraction: existing handler tests still pass unchanged,
  proving the wrapper preserved `process()`'s behaviour.

## Documentation

`CLAUDE.md` gains a "Scheduled Prompts" section. Two existing claims become
false and must be corrected in the same change: that there is no scheduler in
the codebase, and the open-access note explaining that expiry is lazy *because*
no scheduler exists (the expiry behaviour itself does not change). `README.md`
gains a usage example.

## Accepted trade-offs

- **The model owns both the cadence and the prompt text.** A wrong cron fires at
  the wrong time; a prompt left deictic ("give me...") reads oddly or misfires
  when replayed cold. Mitigated by the forced confirmation, which names the next
  run in SGT *and* echoes the stored prompt, so both are visible at creation
  time — and by cancellation being conversational.
- **Any authorized user can schedule.** An open-access stranger can leave a
  recurring prompt behind them. Mitigated by the per-chat cap, the hourly floor,
  the failure cutout, and `created_by`; an admin can see and cancel any of it.
- **Up to 30 seconds late.** Accepted for the durability the poll loop buys.
- **No per-schedule timezone.** Asia/Singapore is UTC+8 with no DST, so clock
  times are stable and the arithmetic has no ambiguous or skipped hours.
