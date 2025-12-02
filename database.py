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

    def _migrate_schema(self, conn):
        """Add new columns to existing database if they don't exist."""
        try:
            # Check if new columns exist
            cursor = conn.execute("PRAGMA table_info(messages)")
            columns = {row[1] for row in cursor.fetchall()}

            # Add sender_name if missing
            if "sender_name" not in columns:
                conn.execute("ALTER TABLE messages ADD COLUMN sender_name TEXT")
                logger.info("Added sender_name column to messages table")

            # Add sender_username if missing
            if "sender_username" not in columns:
                conn.execute("ALTER TABLE messages ADD COLUMN sender_username TEXT")
                logger.info("Added sender_username column to messages table")

            # Add is_group_chat if missing
            if "is_group_chat" not in columns:
                conn.execute("ALTER TABLE messages ADD COLUMN is_group_chat INTEGER DEFAULT 0")
                logger.info("Added is_group_chat column to messages table")

            # Add has_image if missing
            if "has_image" not in columns:
                conn.execute("ALTER TABLE messages ADD COLUMN has_image INTEGER DEFAULT 0")
                logger.info("Added has_image column to messages table")

            # Add image_metadata if missing
            if "image_metadata" not in columns:
                conn.execute("ALTER TABLE messages ADD COLUMN image_metadata TEXT")
                logger.info("Added image_metadata column to messages table")

        except Exception as e:
            logger.warning(f"Schema migration warning: {e}")

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
                        token_count INTEGER DEFAULT 0,
                        sender_name TEXT,
                        sender_username TEXT,
                        is_group_chat INTEGER DEFAULT 0
                    )
                """)

                # Create index for efficient querying
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_chat_timestamp
                    ON messages(chat_id, timestamp DESC)
                """)

                # Create granted_users table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS granted_users (
                        user_id TEXT PRIMARY KEY,
                        granted_at TEXT NOT NULL
                    )
                """)

                # Migrate existing database if needed
                self._migrate_schema(conn)

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
        sender_name: str = None,
        sender_username: str = None,
        is_group_chat: bool = False,
        has_image: bool = False,
        image_metadata: str = None,
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
                    (chat_id, role, content, timestamp, user_id, message_id, token_count,
                     sender_name, sender_username, is_group_chat, has_image, image_metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (chat_id, role, content, timestamp, user_id, message_id, token_count,
                     sender_name, sender_username, 1 if is_group_chat else 0,
                     1 if has_image else 0, image_metadata),
                )
                msg_id = cursor.lastrowid

            logger.debug(
                f"Added message {msg_id} for chat {chat_id}: "
                f"{role} ({token_count} tokens){'[with image]' if has_image else ''}"
            )
            return msg_id

        except Exception as e:
            logger.error(f"Failed to add message: {e}", exc_info=True)
            raise

    def get_messages_by_tokens(self, chat_id: str, max_tokens: int, exclude_images: bool = False) -> list:
        """
        Retrieve recent messages within token budget.

        Args:
            chat_id: Chat ID to retrieve messages from
            max_tokens: Maximum token budget
            exclude_images: If True, exclude messages with images

        Returns messages in chronological order (oldest first).
        For group chats, includes sender information in the format [Name]: message
        """
        try:
            chat_id = str(chat_id)
            messages = []
            total_tokens = 0

            with self._get_connection() as conn:
                # Get messages newest first with sender info
                query = """
                    SELECT role, content, token_count, sender_name, sender_username, is_group_chat
                    FROM messages
                    WHERE chat_id = ?
                """
                if exclude_images:
                    query += " AND has_image = 0"
                query += " ORDER BY timestamp DESC"

                cursor = conn.execute(query, (chat_id,))

                # Accumulate messages while within token budget
                temp_messages = []
                for row in cursor:
                    msg_tokens = row["token_count"] or 0

                    # Always include at least the last message (user's prompt)
                    if not temp_messages or total_tokens + msg_tokens <= max_tokens:
                        temp_messages.append({
                            "role": row["role"],
                            "content": row["content"],
                            "sender_name": row["sender_name"],
                            "sender_username": row["sender_username"],
                            "is_group_chat": row["is_group_chat"],
                        })
                        total_tokens += msg_tokens
                    else:
                        break

                # Reverse to chronological order (oldest first)
                messages = list(reversed(temp_messages))

            logger.debug(
                f"Retrieved {len(messages)} messages for chat {chat_id} "
                f"({total_tokens} tokens){' [excluding images]' if exclude_images else ''}"
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

    def cleanup_old_group_messages(self, chat_id: str, keep_recent: int = 100):
        """
        Remove old messages from group chats to prevent unlimited growth.
        Keeps only the most recent N messages.

        Args:
            chat_id: Chat ID to clean up
            keep_recent: Number of recent messages to keep (default 100)
        """
        try:
            chat_id = str(chat_id)

            with self._get_connection() as conn:
                # Count total messages for this chat
                cursor = conn.execute(
                    "SELECT COUNT(*) as count FROM messages WHERE chat_id = ? AND is_group_chat = 1",
                    (chat_id,),
                )
                total = cursor.fetchone()["count"]

                if total > keep_recent:
                    # Delete older messages, keeping only recent ones
                    conn.execute(
                        """
                        DELETE FROM messages
                        WHERE chat_id = ? AND is_group_chat = 1
                        AND id NOT IN (
                            SELECT id FROM messages
                            WHERE chat_id = ? AND is_group_chat = 1
                            ORDER BY timestamp DESC
                            LIMIT ?
                        )
                        """,
                        (chat_id, chat_id, keep_recent),
                    )
                    deleted = total - keep_recent
                    logger.info(f"Cleaned up {deleted} old messages from group chat {chat_id}")

        except Exception as e:
            logger.error(f"Failed to cleanup old messages: {e}", exc_info=True)

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

    def grant_access(self, user_id: int) -> bool:
        """
        Grant access to a user.

        Args:
            user_id: Telegram user ID to grant access to

        Returns:
            True if access was granted, False if user already had access
        """
        try:
            user_id_str = str(user_id)
            timestamp = datetime.utcnow().isoformat()

            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT user_id FROM granted_users WHERE user_id = ?",
                    (user_id_str,),
                )
                existing = cursor.fetchone()

                if existing:
                    logger.info(f"User {user_id} already has access")
                    return False

                conn.execute(
                    "INSERT INTO granted_users (user_id, granted_at) VALUES (?, ?)",
                    (user_id_str, timestamp),
                )

            logger.info(f"Granted access to user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to grant access: {e}", exc_info=True)
            raise

    def revoke_access(self, user_id: int) -> bool:
        """
        Revoke access from a user.

        Args:
            user_id: Telegram user ID to revoke access from

        Returns:
            True if access was revoked, False if user didn't have access
        """
        try:
            user_id_str = str(user_id)

            with self._get_connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM granted_users WHERE user_id = ?",
                    (user_id_str,),
                )
                deleted_count = cursor.rowcount

            if deleted_count > 0:
                logger.info(f"Revoked access from user {user_id}")
                return True
            else:
                logger.info(f"User {user_id} didn't have access")
                return False

        except Exception as e:
            logger.error(f"Failed to revoke access: {e}", exc_info=True)
            raise

    def is_user_granted(self, user_id: int) -> bool:
        """
        Check if a user has been granted access.

        Args:
            user_id: Telegram user ID to check

        Returns:
            True if user has been granted access, False otherwise
        """
        try:
            user_id_str = str(user_id)

            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT user_id FROM granted_users WHERE user_id = ?",
                    (user_id_str,),
                )
                result = cursor.fetchone()

            return result is not None

        except Exception as e:
            logger.error(f"Failed to check granted access: {e}", exc_info=True)
            return False

    def get_granted_users(self) -> list:
        """
        Get list of all users with granted access.

        Returns:
            List of tuples (user_id, granted_at)
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT user_id, granted_at FROM granted_users ORDER BY granted_at DESC"
                )
                users = [(row["user_id"], row["granted_at"]) for row in cursor]

            return users

        except Exception as e:
            logger.error(f"Failed to get granted users: {e}", exc_info=True)
            return []
