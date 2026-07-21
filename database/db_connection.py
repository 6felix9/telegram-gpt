"""Shared PostgreSQL connection pool and health-checked connection context
manager, used by every repository so they don't each open their own pool."""
import logging
import time
from contextlib import contextmanager

import psycopg2
from psycopg2 import pool

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Thread-safe PostgreSQL connection pool with keepalive + health checks."""

    def __init__(self, db_url: str):
        self.db_url = db_url
        self._last_health_check: float = 0.0
        try:
            self.pool = pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=10,
                dsn=db_url,
                keepalives=1,
                keepalives_idle=30,
                keepalives_interval=10,
                keepalives_count=5,
            )
            logger.info("Database connection pool initialized with keepalives")
        except Exception as e:
            logger.error(f"Failed to create connection pool: {e}", exc_info=True)
            raise

    def close(self):
        try:
            if hasattr(self, "pool") and self.pool:
                self.pool.closeall()
                logger.info("Database connection pool closed")
        except Exception as e:
            logger.error(f"Failed to close connection pool: {e}", exc_info=True)

    @contextmanager
    def connection(self):
        """Thread-safe connection context manager using connection pool with validation."""
        conn = None
        try:
            conn = self.pool.getconn()
            if conn.closed:
                logger.warning("Retrieved closed connection from pool, discarding")
                self.pool.putconn(conn, close=True)
                conn = self.pool.getconn()

            now = time.monotonic()
            if now - self._last_health_check > 30.0:
                try:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1")
                    self._last_health_check = now
                except psycopg2.OperationalError:
                    logger.warning("Connection failed health check, replacing")
                    self.pool.putconn(conn, close=True)
                    conn = self.pool.getconn()
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1")
                    self._last_health_check = time.monotonic()

            yield conn
            conn.commit()
        except psycopg2.OperationalError:
            if conn and not conn.closed:
                try:
                    conn.rollback()
                except psycopg2.OperationalError:
                    pass
            raise
        except Exception:
            if conn and not conn.closed:
                conn.rollback()
            raise
        finally:
            if conn:
                self.pool.putconn(conn, close=conn.closed)
