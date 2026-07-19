"""Repository characterization tests: SQL executed, params, and cache
invalidation timing, using a fake connection double instead of a live DB."""
from contextlib import contextmanager

from access_repository import AccessRepository
from cache import MISSING, TTLCache
from db_connection import ConnectionManager
from message_repository import MessageRepository
from settings_repository import SettingsRepository


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


class _FakeConn:
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


def _fake_manager(results=None, rowcount=0):
    manager = ConnectionManager.__new__(ConnectionManager)
    conn = _FakeConn(results=results, rowcount=rowcount)

    @contextmanager
    def fake_connection():
        yield conn

    manager.connection = fake_connection
    return manager, conn


# --- MessageRepository ---------------------------------------------------

def test_add_message_inserts_with_expected_params():
    manager, conn = _fake_manager(results=[(42,)])
    repo = MessageRepository(manager)
    msg_id = repo.add_message(
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


def test_get_stats_returns_na_when_no_messages():
    manager, conn = _fake_manager(results=[{
        "total_messages": 0, "total_tokens": 0,
        "first_message": None, "last_message": None,
    }])
    repo = MessageRepository(manager)
    stats = repo.get_stats("123")
    assert stats == {
        "total_messages": 0, "total_tokens": 0,
        "first_message": "N/A", "last_message": "N/A",
    }


def test_get_stats_returns_zeroed_dict_on_query_failure():
    manager = ConnectionManager.__new__(ConnectionManager)

    @contextmanager
    def broken_connection():
        raise RuntimeError("db down")
        yield  # pragma: no cover

    manager.connection = broken_connection
    repo = MessageRepository(manager)
    stats = repo.get_stats("123")
    assert stats["total_messages"] == 0
    assert stats["first_message"] == "N/A"


# --- AccessRepository ------------------------------------------------------

def test_is_user_granted_caches_result_and_skips_second_query():
    manager, conn = _fake_manager(results=[("55",)])
    repo = AccessRepository(manager, TTLCache(default_ttl=60.0))
    first = repo.is_user_granted(55)
    second = repo.is_user_granted(55)
    assert first is True
    assert second is True
    assert len(conn.executed) == 1


def test_grant_access_invalidates_granted_cache():
    manager, conn = _fake_manager(results=[None])
    cache = TTLCache(default_ttl=60.0)
    cache.set("granted:99", False, ttl=120.0)
    repo = AccessRepository(manager, cache)
    was_granted = repo.grant_access(99, first_name="Bob", username="bob")
    assert was_granted is True
    assert cache.get("granted:99") is MISSING


# --- SettingsRepository -----------------------------------------------------

def test_set_active_personality_invalidates_related_cache_entries():
    manager, conn = _fake_manager()
    cache = TTLCache(default_ttl=60.0)
    cache.set("active_personality", "old", ttl=60.0)
    cache.set("personality_prompt:old", "prompt text", ttl=300.0)
    repo = SettingsRepository(manager, cache)
    repo.set_active_personality("villain")
    assert cache.get("active_personality") is MISSING
    assert cache.get("personality_prompt:old") is MISSING


def test_get_active_model_returns_default_when_no_row():
    manager, conn = _fake_manager(results=[None])
    repo = SettingsRepository(manager, TTLCache(default_ttl=60.0))
    assert repo.get_active_model() == "gpt-5.4-mini"


def test_list_personalities_truncates_long_prompt_preview():
    long_prompt = "x" * 150
    manager, conn = _fake_manager(results=[[
        {"personality": "villain", "prompt": long_prompt},
        {"personality": "normal", "prompt": "short"},
    ]])
    repo = SettingsRepository(manager, TTLCache(default_ttl=60.0))
    personalities = repo.list_personalities()
    assert personalities[0] == ("villain", "x" * 100 + "...")
    assert personalities[1] == ("normal", "short")
