# Neon Database Schema

Verified against the live Neon database on 2026-04-05.

This document summarizes the current public schema used by the bot. The live database currently contains these tables:

- `messages`
- `granted_users`
- `personality`
- `active_personality`
- `active_model`

## `messages`

| Column | Type | Nullable | Default |
|--------|------|----------|---------|
| `id` | `integer` | No | `nextval('messages_id_seq'::regclass)` |
| `chat_id` | `text` | No | none |
| `role` | `text` | No | none |
| `content` | `text` | No | none |
| `timestamp` | `timestamp without time zone` | No | `CURRENT_TIMESTAMP` |
| `user_id` | `bigint` | Yes | none |
| `message_id` | `bigint` | Yes | none |
| `token_count` | `integer` | Yes | `0` |
| `sender_name` | `text` | Yes | none |
| `sender_username` | `text` | Yes | none |
| `is_group_chat` | `boolean` | Yes | `false` |

Indexes:

- Primary key on `id`
- `idx_chat_timestamp` on `(chat_id, timestamp DESC)`

Purpose:

- Stores user and assistant message history
- Tracks token counts for history retrieval by budget
- Stores sender metadata for group-chat formatting

## `granted_users`

| Column | Type | Nullable | Default |
|--------|------|----------|---------|
| `user_id` | `text` | No | none |
| `granted_at` | `timestamp without time zone` | No | `CURRENT_TIMESTAMP` |
| `first_name` | `text` | Yes | none |
| `username` | `text` | Yes | none |

Indexes:

- Primary key on `user_id`

Purpose:

- Stores non-admin Telegram users who have been granted access
- Keeps optional Telegram profile metadata for `/allowlist`

## `personality`

| Column | Type | Nullable | Default |
|--------|------|----------|---------|
| `personality` | `text` | No | none |
| `prompt` | `text` | No | none |

Indexes:

- Primary key on `personality`

Purpose:

- Stores named group-chat system prompts

## `active_personality`

| Column | Type | Nullable | Default |
|--------|------|----------|---------|
| `id` | `integer` | No | `1` |
| `personality` | `text` | No | `'normal'` |
| `updated_at` | `timestamp without time zone` | No | `CURRENT_TIMESTAMP` |

Indexes:

- Primary key on `id`

Purpose:

- Single-row table tracking the globally active group personality

Note:

- The live database default is currently `'normal'`.
- The default seeded by `alembic/versions/0001_initial_schema.py` is `'default'`; the live production value was already changed to `'normal'` via `/personality` before Alembic was adopted, which is expected — the seed only applies to brand-new databases.

## `active_model`

| Column | Type | Nullable | Default |
|--------|------|----------|---------|
| `id` | `integer` | No | `1` |
| `model` | `text` | No | `'gpt-4o-mini'` |
| `updated_at` | `timestamp without time zone` | No | `CURRENT_TIMESTAMP` |

Indexes:

- Primary key on `id`

Purpose:

- Single-row table tracking the globally active model
- Used by `/model` and loaded on startup before requests are processed

## LangGraph checkpointer (not Alembic-managed)

The LangGraph agent persists conversation state (per-chat threads) using `PostgresSaver` from `langgraph-checkpoint-postgres`, backed by the same `DATABASE_URL`. This adds its own tables:

- `checkpoints`
- `checkpoint_blobs`
- `checkpoint_writes`
- `checkpoint_migrations`

These tables are owned and versioned by the `langgraph-checkpoint-postgres` package itself, not by this repo's Alembic migrations. They are created/upgraded out-of-band by running `python scripts/setup_checkpointer.py`, which calls `PostgresSaver.setup()`. This is idempotent and must be run once per environment, after `alembic upgrade head` and before the bot starts (`start.sh` does this locally; each Railway environment's `preDeployCommand` does this in deployment).

Do not add these tables to Alembic — schema changes to them are driven by upgrading `langgraph-checkpoint-postgres` and re-running `setup()`, not by writing migrations here.

## Operational Notes

- Schema changes to the application tables above are applied via Alembic migrations (`alembic upgrade head`), not created automatically on boot.
- The live database already includes `active_model`, even though older docs omitted it.
- `DEFAULT_MODEL` in `.env` is only a seed value for a fresh database; the runtime model is loaded from `active_model`.
- The LangGraph checkpointer tables described above are separate from this Alembic-managed schema; see the section above.
