"""ConnectionManager tests use a fake pool double; psycopg2 pools connect
eagerly on construction, so real pool creation is out of scope for unit
tests (CLAUDE.md: no live database in tests)."""
from unittest.mock import Mock

import psycopg2
import pytest

from db_connection import ConnectionManager


class _FakeConn:
    def __init__(self, closed=False, healthy=True):
        self.closed = closed
        self._healthy = healthy
        self.committed = False
        self.rolled_back = False

    def cursor(self):
        conn = self

        class _Cur:
            def __enter__(self):
                return self
            def __exit__(self, *exc):
                return False
            def execute(self, sql):
                if not conn._healthy:
                    raise psycopg2.OperationalError("connection failed")
        return _Cur()

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True


def _manager_with_pool(pool):
    manager = ConnectionManager.__new__(ConnectionManager)
    manager.db_url = "postgresql://fake"
    manager._last_health_check = 0.0
    manager.pool = pool
    return manager


def test_connection_yields_pooled_connection_and_commits():
    conn = _FakeConn()
    pool = Mock(getconn=Mock(return_value=conn), putconn=Mock())
    manager = _manager_with_pool(pool)

    with manager.connection() as yielded:
        assert yielded is conn

    assert conn.committed is True
    pool.putconn.assert_called_once_with(conn, close=False)


def test_connection_rolls_back_and_reraises_on_error():
    conn = _FakeConn()
    pool = Mock(getconn=Mock(return_value=conn), putconn=Mock())
    manager = _manager_with_pool(pool)

    with pytest.raises(RuntimeError):
        with manager.connection():
            raise RuntimeError("boom")

    assert conn.rolled_back is True


def test_close_closes_the_pool():
    pool = Mock()
    manager = _manager_with_pool(pool)
    manager.close()
    pool.closeall.assert_called_once()
