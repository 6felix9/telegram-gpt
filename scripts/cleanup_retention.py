"""Prune bounded storage: age-based delete of old `messages` audit rows (#22)
and `images` rows (#57), plus a global sweep that keeps only the newest
LangGraph checkpoint per thread (#21). Idempotent — safe to re-run.

Run once per deploy, in each environment's Railway preDeployCommand, after
`alembic upgrade head && python scripts/setup_checkpointer.py`. Fail-open:
a failure in any part is logged but never raises, so it can't block a
deploy — consistent with this repo's treatment of non-critical maintenance
(image persistence, summary audit inserts) as fail-open elsewhere.
"""
import logging
import os
import sys

# Allow running as `python scripts/cleanup_retention.py`: put the repo root
# on sys.path so `config`/`database` are importable (see setup_checkpointer.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg

from config import config
from database import Database

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


VACUUM_LOCK_TIMEOUT = "5s"

CHECKPOINT_SWEEP_SQL = [
    # 1. Keep only the newest checkpoint per (thread_id, checkpoint_ns).
    #    checkpoint_id is a ULID-like, monotonically sortable string, so
    #    ordering by it descending correctly finds "newest".
    """
    WITH latest AS (
      SELECT DISTINCT ON (thread_id, checkpoint_ns) thread_id, checkpoint_ns, checkpoint_id
      FROM checkpoints
      ORDER BY thread_id, checkpoint_ns, checkpoint_id DESC
    )
    DELETE FROM checkpoints c
    USING latest l
    WHERE c.thread_id = l.thread_id AND c.checkpoint_ns = l.checkpoint_ns
      AND c.checkpoint_id <> l.checkpoint_id
    """,
    # 2. Drop writes for any checkpoint that no longer exists.
    """
    DELETE FROM checkpoint_writes cw
    WHERE NOT EXISTS (
      SELECT 1 FROM checkpoints c
      WHERE c.thread_id = cw.thread_id AND c.checkpoint_ns = cw.checkpoint_ns
        AND c.checkpoint_id = cw.checkpoint_id
    )
    """,
    # 3. Drop blobs not referenced by the surviving checkpoint's channel_versions
    #    (same jsonb_each_text join idiom LangGraph's own SELECT_SQL uses).
    """
    DELETE FROM checkpoint_blobs cb
    WHERE NOT EXISTS (
      SELECT 1 FROM checkpoints c, jsonb_each_text(c.checkpoint->'channel_versions') cv(channel, version)
      WHERE c.thread_id = cb.thread_id AND c.checkpoint_ns = cb.checkpoint_ns
        AND cv.channel = cb.channel AND cv.version = cb.version
    )
    """,
]


def cleanup_messages() -> None:
    if config.MESSAGE_RETENTION_DAYS <= 0:
        logger.info(
            "MESSAGE_RETENTION_DAYS=%s; skipping messages cleanup",
            config.MESSAGE_RETENTION_DAYS,
        )
        return
    db = None
    try:
        db = Database(config.DATABASE_URL)
        deleted = db.delete_messages_older_than(config.MESSAGE_RETENTION_DAYS)
        logger.info(
            "messages retention: deleted %d rows older than %d days",
            deleted, config.MESSAGE_RETENTION_DAYS,
        )
    except Exception:
        logger.exception("messages retention cleanup failed; continuing")
    finally:
        if db is not None:
            db.close()


def cleanup_images() -> None:
    if config.IMAGE_RETENTION_DAYS <= 0:
        logger.info(
            "IMAGE_RETENTION_DAYS=%s; skipping images cleanup",
            config.IMAGE_RETENTION_DAYS,
        )
        return
    db = None
    deleted = 0
    try:
        db = Database(config.DATABASE_URL)
        deleted = db.delete_images_older_than(config.IMAGE_RETENTION_DAYS)
        logger.info(
            "images retention: deleted %d rows older than %d days",
            deleted, config.IMAGE_RETENTION_DAYS,
        )
    except Exception:
        logger.exception("images retention cleanup failed; continuing")
    finally:
        if db is not None:
            db.close()
    if deleted > 0:
        reclaim_image_space()


def reclaim_image_space() -> None:
    """Rewrite `images` so deleted blobs are returned to the filesystem.

    A plain VACUUM only marks pages reusable, which leaves the reported database
    size flat; a full rewrite is what actually gives the space back. Two things
    shape this:

    VACUUM cannot run inside a transaction block, and every repository call
    commits one, hence the dedicated autocommit connection instead of a
    `Database` method.

    The prune has already committed by the time this runs, and is guarded
    separately, so a lock conflict here can never discard it. The lock_timeout
    makes that failure fast rather than a stalled deploy.

    Note the rewrite needs transient free space roughly equal to the table's
    current size, since the original is only dropped once the copy is complete.
    """
    try:
        with psycopg.connect(config.DATABASE_URL, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT set_config('lock_timeout', %s, false)",
                    (VACUUM_LOCK_TIMEOUT,),
                )
                cur.execute("VACUUM (FULL, ANALYZE) images")
        logger.info("images vacuum: rewrote table, reclaiming space from pruned rows")
    except Exception:
        logger.exception("images vacuum failed; prune already committed, continuing")


def cleanup_checkpoints() -> None:
    try:
        with psycopg.connect(config.DATABASE_URL, autocommit=True) as conn:
            with conn.cursor() as cur:
                counts = []
                for sql in CHECKPOINT_SWEEP_SQL:
                    cur.execute(sql)
                    counts.append(cur.rowcount)
        logger.info(
            "checkpoint sweep: deleted %d checkpoints, %d writes, %d blobs",
            *counts,
        )
    except Exception:
        logger.exception("checkpoint sweep failed; continuing")


def main() -> None:
    if not config.DATABASE_URL.strip():
        raise SystemExit("DATABASE_URL is required to run retention cleanup")
    cleanup_messages()
    cleanup_images()
    cleanup_checkpoints()
    logger.info("Retention cleanup complete")


if __name__ == "__main__":
    main()
