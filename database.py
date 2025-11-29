"""SQLite database handler for message storage."""
import sqlite3
import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import shutil
import time

logger = logging.getLogger(__name__)


class Database:
    """Thread-safe SQLite message storage."""

    def __init__(self, db_path: str):
        """Initialize database connection with proper settings."""
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create tables and indexes if they don't exist."""
        try:
            with self._get_connection() as conn:
                # Create messages table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        chat_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        user_id INTEGER,
                        message_id INTEGER,
                        token_count INTEGER DEFAULT 0
                    )
                """)

                # Create index for efficient querying
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_chat_timestamp
                    ON messages(chat_id, timestamp DESC)
                """)

                logger.info(f"Database initialized at {self.db_path}")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}", exc_info=True)
            raise

    @contextmanager
    def _get_connection(self):
        """Thread-safe connection context manager with retry logic."""
        max_retries = 3
        retry_delay = 0.1  # 100ms

        for attempt in range(max_retries):
            try:
                conn = sqlite3.connect(self.db_path, timeout=10.0)
                conn.row_factory = sqlite3.Row

                # Enable WAL mode for better concurrency
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA busy_timeout=5000")  # 5 second timeout

                try:
                    yield conn
                    conn.commit()
                    return
                except Exception:
                    conn.rollback()
                    raise
                finally:
                    conn.close()

            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() and attempt < max_retries - 1:
                    logger.warning(f"Database locked, retrying... (attempt {attempt + 1})")
                    time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    raise

    def add_message(
        self,
        chat_id: str,
        role: str,
        content: str,
        user_id: int = None,
        message_id: int = None,
        token_count: int = 0,
    ) -> int:
        """Add a message to the database with atomic transaction."""
        try:
            # Ensure chat_id is string for consistency
            chat_id = str(chat_id)
            timestamp = datetime.utcnow().isoformat()

            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO messages
                    (chat_id, role, content, timestamp, user_id, message_id, token_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (chat_id, role, content, timestamp, user_id, message_id, token_count),
                )
                msg_id = cursor.lastrowid

            logger.debug(
                f"Added message {msg_id} for chat {chat_id}: "
                f"{role} ({token_count} tokens)"
            )
            return msg_id

        except Exception as e:
            logger.error(f"Failed to add message: {e}", exc_info=True)
            raise

    def get_messages_by_tokens(self, chat_id: str, max_tokens: int) -> list:
        """
        Retrieve recent messages within token budget.

        Returns messages in chronological order (oldest first).
        """
        try:
            chat_id = str(chat_id)
            messages = []
            total_tokens = 0

            with self._get_connection() as conn:
                # Get messages newest first
                cursor = conn.execute(
                    """
                    SELECT role, content, token_count
                    FROM messages
                    WHERE chat_id = ?
                    ORDER BY timestamp DESC
                    """,
                    (chat_id,),
                )

                # Accumulate messages while within token budget
                temp_messages = []
                for row in cursor:
                    msg_tokens = row["token_count"] or 0

                    # Always include at least the last message (user's prompt)
                    if not temp_messages or total_tokens + msg_tokens <= max_tokens:
                        temp_messages.append({
                            "role": row["role"],
                            "content": row["content"],
                        })
                        total_tokens += msg_tokens
                    else:
                        break

                # Reverse to chronological order (oldest first)
                messages = list(reversed(temp_messages))

            logger.debug(
                f"Retrieved {len(messages)} messages for chat {chat_id} "
                f"({total_tokens} tokens)"
            )
            return messages

        except Exception as e:
            logger.error(f"Failed to retrieve messages: {e}", exc_info=True)
            # Fallback to recent messages
            return self.get_recent_messages(chat_id, limit=50)

    def get_recent_messages(self, chat_id: str, limit: int = 100) -> list:
        """Fallback: get last N messages if token counting fails."""
        try:
            chat_id = str(chat_id)

            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT role, content
                    FROM messages
                    WHERE chat_id = ?
                    ORDER BY timestamp ASC
                    LIMIT ?
                    """,
                    (chat_id, limit),
                )

                messages = [
                    {"role": row["role"], "content": row["content"]}
                    for row in cursor
                ]

            logger.debug(f"Retrieved {len(messages)} recent messages for chat {chat_id}")
            return messages

        except Exception as e:
            logger.error(f"Failed to retrieve recent messages: {e}", exc_info=True)
            return []

    def clear_history(self, chat_id: str):
        """Delete all messages for a chat."""
        try:
            chat_id = str(chat_id)

            with self._get_connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM messages WHERE chat_id = ?",
                    (chat_id,),
                )
                deleted_count = cursor.rowcount

            logger.info(f"Cleared {deleted_count} messages for chat {chat_id}")

        except Exception as e:
            logger.error(f"Failed to clear history: {e}", exc_info=True)
            raise

    def get_stats(self, chat_id: str) -> dict:
        """Return statistics for monitoring."""
        try:
            chat_id = str(chat_id)

            with self._get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT
                        COUNT(*) as total_messages,
                        SUM(token_count) as total_tokens,
                        MIN(timestamp) as first_message,
                        MAX(timestamp) as last_message
                    FROM messages
                    WHERE chat_id = ?
                    """,
                    (chat_id,),
                )
                row = cursor.fetchone()

                return {
                    "total_messages": row["total_messages"] or 0,
                    "total_tokens": row["total_tokens"] or 0,
                    "first_message": row["first_message"] or "N/A",
                    "last_message": row["last_message"] or "N/A",
                }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}", exc_info=True)
            return {
                "total_messages": 0,
                "total_tokens": 0,
                "first_message": "N/A",
                "last_message": "N/A",
            }

    def backup_database(self):
        """Create a backup of the database file."""
        try:
            backup_path = f"{self.db_path}.bak"
            shutil.copy2(self.db_path, backup_path)
            logger.info(f"Database backed up to {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"Failed to backup database: {e}", exc_info=True)
            return None
