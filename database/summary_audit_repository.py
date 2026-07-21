"""Audit-only persistence for confirmed checkpoint summaries."""
import logging

from .db_connection import ConnectionManager

logger = logging.getLogger(__name__)


class SummaryAuditRepository:
    """Insert-only CRUD for the `conversation_summaries` audit table."""

    def __init__(self, conn: ConnectionManager):
        self._conn = conn

    def record_conversation_summary(
        self,
        chat_id: str,
        summary_text: str,
        summary_model: str,
        before_message_count: int,
        after_message_count: int,
        before_tokens: int,
        after_tokens: int,
    ) -> int:
        """Persist a permanent audit record of a successful checkpoint summary."""
        try:
            chat_id = str(chat_id)
            with self._conn.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO conversation_summaries
                        (chat_id, summary_text, summary_model, before_message_count,
                         after_message_count, before_tokens, after_tokens)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                        """,
                        (
                            chat_id,
                            summary_text,
                            summary_model,
                            before_message_count,
                            after_message_count,
                            before_tokens,
                            after_tokens,
                        ),
                    )
                    row_id = cur.fetchone()[0]

            logger.info(f"Recorded conversation summary {row_id} for chat {chat_id}")
            return row_id
        except Exception as e:
            logger.error(f"Failed to record conversation summary: {e}", exc_info=True)
            raise
