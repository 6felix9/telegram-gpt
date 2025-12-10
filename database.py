"""PostgreSQL database handler for message storage."""
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
import logging
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger(__name__)


class Database:
    """Thread-safe PostgreSQL message storage with connection pooling."""

    def __init__(self, db_url: str):
        """Initialize database connection pool with proper settings."""
        self.db_url = db_url
        # Initialize connection pool with keepalive settings for Neon
        try:
            self.pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=10,
                dsn=db_url,
                keepalives=1,        # Enable TCP keepalives
                keepalives_idle=30,  # Start sending keepalives after 30 seconds of inactivity
                keepalives_interval=10,  # Send keepalive every 10 seconds
                keepalives_count=5   # Close connection after 5 failed keepalives
            )
            logger.info("Database connection pool initialized with keepalives")
        except Exception as e:
            logger.error(f"Failed to create connection pool: {e}", exc_info=True)
            raise

        self._init_db()

    def close(self):
        """Close all connections in the pool."""
        try:
            if hasattr(self, 'pool') and self.pool:
                self.pool.closeall()
                logger.info("Database connection pool closed")
        except Exception as e:
            logger.error(f"Failed to close connection pool: {e}", exc_info=True)

    def _init_db(self):
        """Create tables and indexes if they don't exist."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    # Create messages table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS messages (
                            id SERIAL PRIMARY KEY,
                            chat_id TEXT NOT NULL,
                            role TEXT NOT NULL,
                            content TEXT NOT NULL,
                            timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                            user_id BIGINT,
                            message_id BIGINT,
                            token_count INTEGER DEFAULT 0,
                            sender_name TEXT,
                            sender_username TEXT,
                            is_group_chat BOOLEAN DEFAULT FALSE,
                            has_image BOOLEAN DEFAULT FALSE,
                            image_metadata TEXT
                        )
                    """)

                    # Create index for efficient querying
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_chat_timestamp
                        ON messages(chat_id, timestamp DESC)
                    """)

                    # Create granted_users table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS granted_users (
                            user_id TEXT PRIMARY KEY,
                            granted_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                        )
                    """)

                    # Create personality table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS personality (
                            personality TEXT PRIMARY KEY,
                            prompt TEXT NOT NULL
                        )
                    """)

                    # Create active_personality table (single row table)
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS active_personality (
                            id INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
                            personality TEXT NOT NULL DEFAULT 'normal',
                            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                        )
                    """)

                    # Initialize active_personality if empty
                    cur.execute("""
                        INSERT INTO active_personality (id, personality, updated_at)
                        SELECT 1, 'normal', CURRENT_TIMESTAMP
                        WHERE NOT EXISTS (SELECT 1 FROM active_personality WHERE id = 1)
                    """)

                logger.info("Database tables initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}", exc_info=True)
            raise

    @contextmanager
    def _get_connection(self):
        """Thread-safe connection context manager using connection pool with validation."""
        conn = None
        try:
            conn = self.pool.getconn()
            # Validate connection is still alive
            if conn.closed:
                logger.warning("Retrieved closed connection from pool, discarding")
                self.pool.putconn(conn, close=True)
                conn = self.pool.getconn()

            # Actively probe the connection to avoid yielding a stale one
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
            except psycopg2.OperationalError:
                logger.warning("Connection failed health check, replacing")
                self.pool.putconn(conn, close=True)
                conn = self.pool.getconn()
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")

            yield conn
            conn.commit()
        except psycopg2.OperationalError as e:
            # Handle "connection already closed" and "SSL connection closed" errors
            if conn and not conn.closed:
                try:
                    conn.rollback()
                except psycopg2.OperationalError:
                    pass  # Connection is already closed, rollback not needed
            raise
        except Exception:
            if conn and not conn.closed:
                conn.rollback()
            raise
        finally:
            if conn:
                self.pool.putconn(conn, close=conn.closed)

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
            timestamp = datetime.utcnow()

            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO messages
                        (chat_id, role, content, timestamp, user_id, message_id, token_count,
                         sender_name, sender_username, is_group_chat, has_image, image_metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                        """,
                        (chat_id, role, content, timestamp, user_id, message_id, token_count,
                         sender_name, sender_username, is_group_chat, has_image, image_metadata),
                    )
                    msg_id = cur.fetchone()[0]

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
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Get messages newest first with sender info
                    query = """
                        SELECT role, content, token_count, sender_name, sender_username, is_group_chat
                        FROM messages
                        WHERE chat_id = %s
                    """
                    if exclude_images:
                        query += " AND has_image = FALSE"
                    # Fetch a bounded set of recent messages to avoid large loads
                    query += " ORDER BY timestamp DESC LIMIT 500"

                    cur.execute(query, (chat_id,))

                    # Accumulate messages while within token budget
                    temp_messages = []
                    for row in cur.fetchall():
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
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        """
                        SELECT role, content
                        FROM messages
                        WHERE chat_id = %s
                        ORDER BY timestamp ASC
                        LIMIT %s
                        """,
                        (chat_id, limit),
                    )

                    messages = [
                        {"role": row["role"], "content": row["content"]}
                        for row in cur.fetchall()
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
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM messages WHERE chat_id = %s",
                        (chat_id,),
                    )
                    deleted_count = cur.rowcount

            logger.info(f"Cleared {deleted_count} messages for chat {chat_id}")

        except Exception as e:
            logger.error(f"Failed to clear history: {e}", exc_info=True)
            raise

    def get_stats(self, chat_id: str) -> dict:
        """Return statistics for monitoring."""
        try:
            chat_id = str(chat_id)

            with self._get_connection() as conn:
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
        Keeps only the most recent N messages.

        Args:
            chat_id: Chat ID to clean up
            keep_recent: Number of recent messages to keep (default 100)
        """
        try:
            chat_id = str(chat_id)

            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Count total messages for this chat
                    cur.execute(
                        "SELECT COUNT(*) as count FROM messages WHERE chat_id = %s AND is_group_chat = TRUE",
                        (chat_id,),
                    )
                    total = cur.fetchone()["count"]

                    if total > keep_recent:
                        # Delete older messages, keeping only recent ones
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
            timestamp = datetime.utcnow()

            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT user_id FROM granted_users WHERE user_id = %s",
                        (user_id_str,),
                    )
                    existing = cur.fetchone()

                    if existing:
                        logger.info(f"User {user_id} already has access")
                        return False

                    cur.execute(
                        "INSERT INTO granted_users (user_id, granted_at) VALUES (%s, %s)",
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
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM granted_users WHERE user_id = %s",
                        (user_id_str,),
                    )
                    deleted_count = cur.rowcount

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
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT user_id FROM granted_users WHERE user_id = %s",
                        (user_id_str,),
                    )
                    result = cur.fetchone()

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
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        "SELECT user_id, granted_at FROM granted_users ORDER BY granted_at DESC"
                    )
                    users = [
                        (row["user_id"], row["granted_at"].isoformat() if row["granted_at"] else "N/A")
                        for row in cur.fetchall()
                    ]

            return users

        except Exception as e:
            logger.error(f"Failed to get granted users: {e}", exc_info=True)
            return []

    def get_personality_prompt(self, personality: str) -> str | None:
        """
        Get the prompt for a specific personality.

        Args:
            personality: Personality name to fetch

        Returns:
            Prompt text if found, None otherwise
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        "SELECT prompt FROM personality WHERE personality = %s",
                        (personality,)
                    )
                    row = cur.fetchone()
                    if row:
                        return row["prompt"]
                    return None

        except Exception as e:
            logger.error(f"Failed to get personality prompt: {e}", exc_info=True)
            return None

    def get_active_personality(self) -> str:
        """
        Get the currently active personality.

        Returns:
            Active personality name (defaults to 'normal')
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        "SELECT personality FROM active_personality WHERE id = 1"
                    )
                    row = cur.fetchone()
                    if row:
                        return row["personality"]
                    return "normal"

        except Exception as e:
            logger.error(f"Failed to get active personality: {e}", exc_info=True)
            return "normal"

    def set_active_personality(self, personality: str) -> None:
        """
        Set the active personality.

        Args:
            personality: Personality name to activate
        """
        try:
            timestamp = datetime.utcnow()
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO active_personality (id, personality, updated_at)
                        VALUES (1, %s, %s)
                        ON CONFLICT (id) DO UPDATE
                        SET personality = EXCLUDED.personality,
                            updated_at = EXCLUDED.updated_at
                        """,
                        (personality, timestamp)
                    )

            logger.info(f"Active personality set to: {personality}")

        except Exception as e:
            logger.error(f"Failed to set active personality: {e}", exc_info=True)
            raise

    def personality_exists(self, personality: str) -> bool:
        """
        Check if a personality exists in the database.

        Args:
            personality: Personality name to check

        Returns:
            True if personality exists, False otherwise
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT personality FROM personality WHERE personality = %s",
                        (personality,)
                    )
                    result = cur.fetchone()
                    return result is not None

        except Exception as e:
            logger.error(f"Failed to check personality existence: {e}", exc_info=True)
            return False
