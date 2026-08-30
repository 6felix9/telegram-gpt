# Image Retention

**Date:** 2026-08-30
**Status:** Approved for implementation
**Issue:** [#57](https://github.com/6felix9/telegram-gpt/issues/57)

## Problem

The `images` table has no retention or cleanup path, so it grows monotonically
for as long as the bot runs. Every photo, triggered or passive, is persisted by
`Agent.persist_image`, and nothing ever deletes the row:

- `MESSAGE_RETENTION_DAYS` / `scripts/cleanup_retention.py` prune `messages` rows
  and superseded historical checkpoint rows only; `images` is out of scope.
- Checkpoint compaction replaces the in-context `[image #N]` marker but never
  touches the underlying row.

Measured on 2026-08-09: 117 rows / 16 MB on `production`, 7 rows / 1.3 MB on
`dev`, against a 512 MB account limit the production branch was already using
~507 MB of. Rows accumulate at roughly 7 images and ~0.95 MB per day.

## Goals

- Bound the `images` table with an age-based policy, mirroring the shape of
  `MESSAGE_RETENTION_DAYS` so there is one obvious pattern for retention.
- Return the freed space rather than leaving it as reusable dead pages.
- Keep the existing fail-open behavior: retention must never block a deploy, and
  a pruned image must never surface as an error to a user.

## Non-goals

- Changing image ingest, `[image #N]` markers, or the `get_image` tool.
- Reclaiming space in the checkpoint tables or `messages`. That is the larger
  share of the 507 MB (see Accepted trade-offs) and belongs in its own issue.
- Any per-chat or count-based policy. A single global age threshold is enough.

## Design

### Configuration

One new optional setting, read through the existing `_int_env` helper so a blank
value in `.env` or Railway means "use the default":

| Setting | Default | Meaning |
|---|---|---|
| `IMAGE_RETENTION_DAYS` | `30` | Delete `images` rows older than this many days. `0` disables. |

Validation mirrors `MESSAGE_RETENTION_DAYS` exactly: `>= 0`, where `0` means
disabled. It is deliberately a separate knob from `MESSAGE_RETENTION_DAYS`
despite sharing a default, because image rows carry blobs and are the expensive
ones — the two thresholds should be able to diverge without a schema change.

### Deletion

`ImageRepository.delete_images_older_than(days) -> int`, mirroring
`MessageRepository.delete_messages_older_than`: one global `DELETE` against
`created_at`, returning `cur.rowcount`, logging, and re-raising on failure so the
caller owns the fail-open decision. Exposed through the `Database` facade under
the existing images section.

### Reclamation

`ConnectionManager.connection()` commits at the end of its context, so every
repository call runs inside a psycopg2 transaction — and `VACUUM` cannot run
inside a transaction block. Reclamation therefore lives in
`scripts/cleanup_retention.py`, on its own autocommit `psycopg` connection, the
same way `cleanup_checkpoints()` already works.

`cleanup_images()`:

```
if IMAGE_RETENTION_DAYS <= 0:   log and return
deleted = db.delete_images_older_than(IMAGE_RETENTION_DAYS)
if deleted > 0:                 vacuum_images()      # separate try/except
```

Two guards on the vacuum:

**It only runs when rows were actually deleted.** A routine deploy that prunes
nothing has no dead space to reclaim, and should not take an `ACCESS EXCLUSIVE`
lock for it.

**It runs in its own try/except, after the prune has already committed.** A lock
conflict or a failed rewrite must not discard a successful delete. A short
`lock_timeout` is set first so an overlapping deploy fails fast into the
fail-open path instead of stalling the deploy.

`VACUUM (FULL, ANALYZE) images` is used rather than a plain `VACUUM` because only
a full rewrite returns space; a plain vacuum marks pages reusable and leaves
Neon's reported size flat.

### Behavior after an image is pruned

No new degradation path is needed — both already exist and already fail open:

- `get_image(N)` returns `Image #N not found.` from `build_image_blocks`. The
  model still has the marker's text description, which lives in the checkpoint
  and in the rewritten `messages` audit row, not in the `images` row.
- A reply to a pruned photo finds nothing from `get_image_by_message_id` and
  falls through to `_resolve_reply_image_ref`'s existing re-persist path, which
  re-ingests the photo from Telegram under a fresh id. Deletion self-heals for
  exactly the case where someone cares about an old image again.

## Error handling

`cleanup_images()` is fail-open in both halves, consistent with the rest of the
script: a prune failure is logged with `logger.exception` and swallowed, and a
vacuum failure is logged and swallowed separately. Neither can fail the deploy.
`main()` calls it between `cleanup_messages()` and `cleanup_checkpoints()`.

## Testing

- `tests/test_repositories.py`: the delete issues one statement against `images`
  with the day count as its only parameter, and returns `rowcount`.
- `tests/test_config.py`: default of 30, and a negative value is rejected while
  `0` is accepted.
- `tests/test_cleanup_retention.py`: the prune is skipped when the setting is
  `0`; the vacuum is skipped when nothing was deleted; the vacuum runs when rows
  were deleted; a vacuum failure does not propagate; a prune failure does not
  propagate.

## Deployment

No migration — `images.created_at` already exists with a
`DEFAULT CURRENT_TIMESTAMP`. `scripts/cleanup_retention.py` is already wired into
each environment's Railway `preDeployCommand`, so the new step ships with the
existing invocation and needs no infrastructure change. `IMAGE_RETENTION_DAYS`
is optional in both environments.

On the first run against `production`, everything ingested between the table's
creation on 2026-07-23 and 30 days ago is deleted — roughly the first week of
images — and the vacuum returns that space.

Docs to update: `README.md`, `AGENTS.md`, `CLAUDE.md`, `.env.example`. The
existing "Persisted image bytes currently have no retention limit" bullet in
`AGENTS.md` / `CLAUDE.md` becomes wrong and must be replaced.

## Accepted trade-offs

**Pruned images cannot be viewed again unless re-sent.** Only the description
survives. `get_image` on a pruned id degrades to "not found", and the model
answers from the summary text.

**`VACUUM FULL` needs transient space roughly equal to the table.** It rebuilds
the heap and its TOAST before dropping the original, so it briefly needs a second
copy — tens of MB at the projected steady state. On a branch sitting near its
storage limit that is not obviously available, and Neon makes a branch read-only
when the limit is exceeded. Pruning before vacuuming and skipping the vacuum when
nothing was deleted keep the exposure small, but headroom is worth confirming
before the first production run.

**This does not fix the 507 MB.** At a projected ~28 MB steady state, `images` is
a small share of it; the rest sits in the checkpoint tables, `messages`, and dead
pages the existing sweeps freed but never returned. This bounds image growth,
which is what #57 asks for, but the headline number needs a separate look at the
checkpoint tables.
