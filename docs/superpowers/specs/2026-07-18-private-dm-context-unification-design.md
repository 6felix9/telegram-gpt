# Private DM Context Unification — Design

## Goal

Private DMs should accumulate non-triggering **text** into the LangGraph checkpoint the same way group chats already do, so the agent can use earlier conversation turns when the user later activates the bot. Reply activation, prompts, personality, and photo handling stay as they are today.

## Motivation

In private chats today, only keyword/`@mention` turns enter the checkpoint. Ordinary messages are dropped, so the model cannot see prior context the user already typed. Groups already store those lines via `append_context_message`. Extending that path to private DMs improves context quality with a small, localized change.

## Non-Goals

- No change to activation: replies still require `chatgpt` or `@BOT_USERNAME`.
- No unification of system prompts (`SYSTEM_PROMPT` vs `SYSTEM_PROMPT_GROUP`).
- No applying `/personality` to private DMs (remains group-only).
- No `[Name]:` prefix on private messages (remains group-only).
- No storing non-triggering photos (both chat types continue to ignore them without a caption trigger).
- No new retention/summarization architecture (existing 500→400 checkpoint prune and token trimming remain).
- No schema or Alembic changes.

## Decisions

| Topic | Decision |
|---|---|
| Scope | Unify non-triggering **text** context storage only |
| Auth on store | Match groups: store without an allowlist check; replies still require authorization |
| Private system prompt | Unchanged (no new “context vs request” wording unless a later regression needs it) |
| Message formatting | Private: `is_group=False` (plain text). Group: unchanged (`[Name]: …`) |
| Photos | Unchanged (ignore without keyword in caption) |

## Current behavior

```
Private non-trigger text  → drop (not in checkpoint)
Group non-trigger text    → db.add_message + agent.append_context_message; no reply
Any chat, trigger text    → authorize → agent.run → reply
Any chat, non-trigger photo → ignore
```

Relevant code: `handlers.message_handler` gates context storage with `if is_group and not has_keyword`.

## Target behavior

```
Any chat, non-trigger text → db.add_message + agent.append_context_message; no reply
Any chat, trigger text     → authorize → agent.run → reply  (unchanged)
Any chat, non-trigger photo → ignore  (unchanged)
```

### Before / after (private DM example)

```
Before:
  User: "flight is at 6"              → not stored
  User: "chatgpt what time was that?" → model has no prior "6"

After:
  User: "flight is at 6"              → checkpoint + audit log; no reply
  User: "chatgpt what time was that?" → model can see the earlier line
```

Group behavior for non-trigger text is unchanged in effect; it shares the same code path.

## Architecture

No new modules. Reuse the existing group context path in `handlers.py`:

1. Detect activation via `extract_keyword` (unchanged).
2. If **no** keyword: audit-log with `db.add_message(..., is_group_chat=<actual>)` and append to the thread with `agent.append_context_message` + `prompt_builder.to_lc_human_message(..., is_group=<actual>, sender_name=...)`.
3. Return without calling the model.
4. If keyword present: existing authorize → `process_request` / `process_image_request` flow.

`agent.append_context_message`, checkpoint prune (500→400), and `wrap_model_call` token trimming are already chat-type agnostic; they need no behavioral change.

### Prompt / personality (explicitly unchanged)

- Private runs continue to use `SYSTEM_PROMPT` via `PromptBuilder.build_system_prompt(is_group=False)`.
- Groups continue to resolve `active_personality` or `SYSTEM_PROMPT_GROUP`.
- Dynamic prompt middleware and `AgentContext.is_group` stay as they are.

## Error handling

Match today’s group path: failures while storing non-trigger context are logged; the handler returns without user-visible error (no reply was expected).

## Testing

- Extend coverage so a **private** non-triggering text message calls `db.add_message` and `agent.append_context_message`, and does not invoke `agent.run`.
- Keep existing group non-trigger retention test(s); adjust only if the shared condition changes their setup.
- Confirm triggering private/group paths still require the keyword and still authorize before reply (existing tests / smoke via `scripts/chat_cli.py` as needed).

## Docs

Update `AGENTS.md` / `README.md` group-context notes so they describe non-trigger text storage for **all** chat types, not groups only. Note that private prompts and personality remain distinct.

## Out of scope / follow-ups

- Coordinated retention policy for `messages` + checkpoint (already TODO in handlers).
- Optional private-prompt line clarifying that stored lines are not always requests — only if live behavior warrants it.
- Storing non-trigger image markers or bytes.
