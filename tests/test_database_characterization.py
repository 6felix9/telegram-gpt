"""Database characterization tests: SQL executed, params, and cache
invalidation timing, using a fake connection double instead of a live DB
(CLAUDE.md: unit tests never hit a live database)."""
from contextlib import contextmanager

from cache import MISSING, TTLCache
from database import Database


class _FakeCursor:
    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):
        self._conn.executed.append((" ".join(sql.split()), params))

    def fetchone(self):
        return self._conn.results.pop(0) if self._conn.results else None

    def fetchall(self):
        return self._conn.results.pop(0) if self._conn.results else []

    @property
    def rowcount(self):
        return self._conn.rowcount


class _FakeConnection:
    def __init__(self, results=None, rowcount=0):
        self.executed = []
        self.results = list(results or [])
        self.rowcount = rowcount
        self.closed = False

    def cursor(self, cursor_factory=None):
        return _FakeCursor(self)

    def commit(self):
        pass

    def rollback(self):
        pass


def _make_db(results=None, rowcount=0):
    """Build a Database without opening a real connection pool."""
    db = Database.__new__(Database)
    db._cache = TTLCache(default_ttl=60.0)
    fake_conn = _FakeConnection(results=results, rowcount=rowcount)

    @contextmanager
    def fake_get_connection():
        yield fake_conn

    db._get_connection = fake_get_connection
    return db, fake_conn


def _broken_db():
    """A Database whose connection always fails to open."""
    db = Database.__new__(Database)
    db._cache = TTLCache(default_ttl=60.0)

    @contextmanager
    def fake_get_connection():
        raise RuntimeError("db down")
        yield  # pragma: no cover

    db._get_connection = fake_get_connection
    return db


def test_add_message_inserts_with_expected_params():
    db, conn = _make_db(results=[(42,)])
    msg_id = db.add_message(
        chat_id=123, role="user", content="hi", user_id=7, message_id=1,
        token_count=3, sender_name="Alice", sender_username="alice",
        is_group_chat=True,
    )
    assert msg_id == 42
    sql, params = conn.executed[0]
    assert "INSERT INTO messages" in sql
    assert params[0] == "123"
    assert params[1] == "user"
    assert params[2] == "hi"


def test_is_user_granted_caches_result_and_skips_second_query():
    db, conn = _make_db(results=[("55",)])
    first = db.is_user_granted(55)
    second = db.is_user_granted(55)
    assert first is True
    assert second is True
    assert len(conn.executed) == 1


def test_grant_access_invalidates_granted_cache():
    db, conn = _make_db(results=[None])
    db._cache.set("granted:99", False, ttl=120.0)
    was_granted = db.grant_access(99, first_name="Bob", username="bob")
    assert was_granted is True
    assert db._cache.get("granted:99") is MISSING


def test_set_active_personality_invalidates_related_cache_entries():
    db, conn = _make_db()
    db._cache.set("active_personality", "old", ttl=60.0)
    db._cache.set("personality_prompt:old", "prompt text", ttl=300.0)
    db.set_active_personality("villain")
    assert db._cache.get("active_personality") is MISSING
    assert db._cache.get("personality_prompt:old") is MISSING


def test_get_active_model_returns_default_when_no_row():
    db, conn = _make_db(results=[None])
    assert db.get_active_model() == "gpt-5.4-mini"


def test_get_stats_returns_na_when_no_messages():
    db, conn = _make_db(results=[{
        "total_messages": 0, "total_tokens": 0,
        "first_message": None, "last_message": None,
    }])
    stats = db.get_stats("123")
    assert stats == {
        "total_messages": 0, "total_tokens": 0,
        "first_message": "N/A", "last_message": "N/A",
    }


def test_get_stats_returns_zeroed_dict_on_query_failure():
    db = _broken_db()
    stats = db.get_stats("123")
    assert stats["total_messages"] == 0
    assert stats["first_message"] == "N/A"


def test_list_personalities_truncates_long_prompt_preview():
    long_prompt = "x" * 150
    db, conn = _make_db(results=[[
        {"personality": "villain", "prompt": long_prompt},
        {"personality": "normal", "prompt": "short"},
    ]])
    personalities = db.list_personalities()
    assert personalities[0] == ("villain", "x" * 100 + "...")
    assert personalities[1] == ("normal", "short")
