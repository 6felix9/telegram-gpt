"""Message audit-log persistence: inserts, token-budget reads, and stats."""
import logging
from datetime import datetime

from psycopg2.extras import RealDictCursor

from db_connection import ConnectionManager

logger = logging.getLogger(__name__)


class MessageRepository:
    """CRUD for the `messages` audit table."""

    def __init__(self, conn: ConnectionManager):
        self._conn = conn

    def add_message(
        self,
        chat_id: str,
        role: str,
        content: str,
        user_id: int = None,
        message_id: int = None,
        token_count: int = 0,
        sender_name: str = None,
        sender_username: str = None,
        is_group_chat: bool = False,
    ) -> int:
        """Add a message to the database with atomic transaction."""
        try:
            chat_id = str(chat_id)
            timestamp = datetime.utcnow()

            with self._conn.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO messages
                        (chat_id, role, content, timestamp, user_id, message_id, token_count,
                         sender_name, sender_username, is_group_chat)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                        """,
                        (chat_id, role, content, timestamp, user_id, message_id, token_count,
                         sender_name, sender_username, is_group_chat),
                    )
                    msg_id = cur.fetchone()[0]

            logger.debug(
                f"Added message {msg_id} for chat {chat_id}: "
                f"{role} ({token_count} tokens)"
            )
            return msg_id

        except Exception as e:
            logger.error(f"Failed to add message: {e}", exc_info=True)
            raise

    def get_stats(self, chat_id: str) -> dict:
        """Return statistics for monitoring."""
        try:
            chat_id = str(chat_id)

            with self._conn.connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        """
                        SELECT
                            COUNT(*) as total_messages,
                            COALESCE(SUM(token_count), 0) as total_tokens,
                            MIN(timestamp) as first_message,
                            MAX(timestamp) as last_message
                        FROM messages
                        WHERE chat_id = %s
                        """,
                        (chat_id,),
                    )
                    row = cur.fetchone()

                    return {
                        "total_messages": row["total_messages"] or 0,
                        "total_tokens": row["total_tokens"] or 0,
                        "first_message": row["first_message"].isoformat() if row["first_message"] else "N/A",
                        "last_message": row["last_message"].isoformat() if row["last_message"] else "N/A",
                    }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}", exc_info=True)
            return {
                "total_messages": 0,
                "total_tokens": 0,
                "first_message": "N/A",
                "last_message": "N/A",
            }

    def cleanup_old_group_messages(self, chat_id: str, keep_recent: int = 100):
        """
        Remove old messages from group chats to prevent unlimited growth.
        Keeps only the most recent N messages. Currently unused in the
        message-handling path (see CLAUDE.md Context Storage notes: the
        probabilistic cleanup call is intentionally disabled); kept for an
        eventual coordinated retention policy.

        Args:
            chat_id: Chat ID to clean up
            keep_recent: Number of recent messages to keep (default 100)
        """
        try:
            chat_id = str(chat_id)

            with self._conn.connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        "SELECT COUNT(*) as count FROM messages WHERE chat_id = %s AND is_group_chat = TRUE",
                        (chat_id,),
                    )
                    total = cur.fetchone()["count"]

                    if total > keep_recent:
                        cur.execute(
                            """
                            DELETE FROM messages
                            WHERE chat_id = %s AND is_group_chat = TRUE
                            AND id NOT IN (
                                SELECT id FROM messages
                                WHERE chat_id = %s AND is_group_chat = TRUE
                                ORDER BY timestamp DESC
                                LIMIT %s
                            )
                            """,
                            (chat_id, chat_id, keep_recent),
                        )
                        deleted = total - keep_recent
                        logger.info(f"Cleaned up {deleted} old messages from group chat {chat_id}")

        except Exception as e:
            logger.error(f"Failed to cleanup old messages: {e}", exc_info=True)
