"""Allowlist persistence: grant/revoke/check access, cached."""
import logging
from datetime import datetime

from psycopg2.extras import RealDictCursor

from cache import MISSING, TTLCache

from .db_connection import ConnectionManager

logger = logging.getLogger(__name__)


class AccessRepository:
    """CRUD for the `granted_users` allowlist table."""

    def __init__(self, conn: ConnectionManager, cache: TTLCache):
        self._conn = conn
        self._cache = cache

    def grant_access(
        self, user_id: int, first_name: str | None = None, username: str | None = None
    ) -> bool:
        try:
            user_id_str = str(user_id)
            timestamp = datetime.utcnow()

            with self._conn.connection() as conn:
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
                        "INSERT INTO granted_users (user_id, granted_at, first_name, username) "
                        "VALUES (%s, %s, %s, %s)",
                        (user_id_str, timestamp, first_name, username),
                    )

            self._cache.invalidate(f"granted:{user_id}")
            logger.info(f"Granted access to user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to grant access: {e}", exc_info=True)
            raise

    def revoke_access(self, user_id: int) -> bool:
        try:
            user_id_str = str(user_id)

            with self._conn.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM granted_users WHERE user_id = %s",
                        (user_id_str,),
                    )
                    deleted_count = cur.rowcount

            self._cache.invalidate(f"granted:{user_id}")
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
        cache_key = f"granted:{user_id}"
        cached = self._cache.get(cache_key)
        if cached is not MISSING:
            return cached

        try:
            user_id_str = str(user_id)

            with self._conn.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT user_id FROM granted_users WHERE user_id = %s",
                        (user_id_str,),
                    )
                    result = cur.fetchone()

            granted = result is not None
            self._cache.set(cache_key, granted, ttl=120.0)
            return granted

        except Exception as e:
            logger.error(f"Failed to check granted access: {e}", exc_info=True)
            return False

    def get_granted_users(self) -> list:
        try:
            with self._conn.connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        "SELECT user_id, granted_at, first_name, username "
                        "FROM granted_users ORDER BY granted_at DESC"
                    )
                    users = [
                        (
                            row["user_id"],
                            row["granted_at"].isoformat() if row["granted_at"] else "N/A",
                            row["first_name"],
                            row["username"],
                        )
                        for row in cur.fetchall()
                    ]

            return users

        except Exception as e:
            logger.error(f"Failed to get granted users: {e}", exc_info=True)
            return []
