"""Personality and active-model global settings, cached."""
import logging
from datetime import datetime

from psycopg2.extras import RealDictCursor

from cache import MISSING, TTLCache
from db_connection import ConnectionManager

logger = logging.getLogger(__name__)


class SettingsRepository:
    """CRUD for `personality`, `active_personality`, and `active_model`."""

    def __init__(self, conn: ConnectionManager, cache: TTLCache):
        self._conn = conn
        self._cache = cache

    def get_personality_prompt(self, personality: str) -> str | None:
        cache_key = f"personality_prompt:{personality}"
        cached = self._cache.get(cache_key)
        if cached is not MISSING:
            return cached

        try:
            with self._conn.connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        "SELECT prompt FROM personality WHERE personality = %s",
                        (personality,)
                    )
                    row = cur.fetchone()
                    result = row["prompt"] if row else None

            self._cache.set(cache_key, result, ttl=300.0)
            return result

        except Exception as e:
            logger.error(f"Failed to get personality prompt: {e}", exc_info=True)
            return None

    def get_active_personality(self) -> str:
        cache_key = "active_personality"
        cached = self._cache.get(cache_key)
        if cached is not MISSING:
            return cached

        try:
            with self._conn.connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        "SELECT personality FROM active_personality WHERE id = 1"
                    )
                    row = cur.fetchone()
                    result = row["personality"] if row else "default"

            self._cache.set(cache_key, result, ttl=60.0)
            return result

        except Exception as e:
            logger.error(f"Failed to get active personality: {e}", exc_info=True)
            return "default"

    def set_active_personality(self, personality: str) -> None:
        try:
            timestamp = datetime.utcnow()
            with self._conn.connection() as conn:
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

            self._cache.invalidate("active_personality")
            self._cache.invalidate_prefix("personality_prompt:")
            logger.info(f"Active personality set to: {personality}")

        except Exception as e:
            logger.error(f"Failed to set active personality: {e}", exc_info=True)
            raise

    def personality_exists(self, personality: str) -> bool:
        try:
            with self._conn.connection() as conn:
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

    def list_personalities(self) -> list[tuple[str, str]]:
        try:
            with self._conn.connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        "SELECT personality, prompt FROM personality ORDER BY personality"
                    )
                    personalities = [
                        (row["personality"], row["prompt"][:100] + "..." if len(row["prompt"]) > 100 else row["prompt"])
                        for row in cur.fetchall()
                    ]
            return personalities
        except Exception as e:
            logger.error(f"Failed to list personalities: {e}", exc_info=True)
            return []

    def init_active_model(self, default_model: str) -> None:
        try:
            with self._conn.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO active_model (id, model) VALUES (1, %s) ON CONFLICT DO NOTHING",
                        (default_model,)
                    )
        except Exception as e:
            logger.error(f"Failed to init active model: {e}", exc_info=True)

    def get_active_model(self) -> str:
        cache_key = "active_model"
        cached = self._cache.get(cache_key)
        if cached is not MISSING:
            return cached

        try:
            with self._conn.connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("SELECT model FROM active_model WHERE id = 1")
                    row = cur.fetchone()
                    result = row["model"] if row else "gpt-5.4-mini"

            self._cache.set(cache_key, result, ttl=60.0)
            return result

        except Exception as e:
            logger.error(f"Failed to get active model: {e}", exc_info=True)
            return "gpt-5.4-mini"

    def set_active_model(self, model: str) -> None:
        try:
            timestamp = datetime.utcnow()
            with self._conn.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO active_model (id, model, updated_at)
                        VALUES (1, %s, %s)
                        ON CONFLICT (id) DO UPDATE
                        SET model = EXCLUDED.model,
                            updated_at = EXCLUDED.updated_at
                        """,
                        (model, timestamp)
                    )

            self._cache.invalidate("active_model")
            logger.info(f"Active model set to: {model}")

        except Exception as e:
            logger.error(f"Failed to set active model: {e}", exc_info=True)
            raise
