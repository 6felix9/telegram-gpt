"""Create/upgrade the LangGraph PostgresSaver tables. Idempotent.

Run once per environment AFTER `alembic upgrade head` and BEFORE the bot
starts. The checkpointer tables are versioned independently by
langgraph-checkpoint-postgres and are intentionally NOT under Alembic.
"""
import logging

from config import config
from langgraph.checkpoint.postgres import PostgresSaver

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    if not config.DATABASE_URL.strip():
        raise SystemExit("DATABASE_URL is required to set up the checkpointer")
    with PostgresSaver.from_conn_string(config.DATABASE_URL) as checkpointer:
        checkpointer.setup()
    logger.info("Checkpointer tables are set up")


if __name__ == "__main__":
    main()
