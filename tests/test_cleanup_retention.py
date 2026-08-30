"""Regression test for scripts/cleanup_retention.py invocation, plus unit
coverage of the images retention half (#57).

Mirrors test_setup_checkpointer.py: the script is run by path in Railway's
preDeployCommand, so `from config import config` / `from database import
Database` must resolve via the script's own sys.path bootstrap. It stops at
the DATABASE_URL guard, so it needs no DB or network.
"""
import importlib.util
import os
import subprocess
import sys
from contextlib import contextmanager

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_script_imports_config_when_run_by_path():
    env = dict(os.environ, DATABASE_URL="")
    result = subprocess.run(
        [sys.executable, "scripts/cleanup_retention.py"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    combined = result.stdout + result.stderr
    assert "No module named 'config'" not in combined
    assert "No module named 'database'" not in combined
    assert "ModuleNotFoundError" not in combined
    assert result.returncode != 0
    assert "DATABASE_URL is required" in combined


# --- images retention ------------------------------------------------------

@pytest.fixture
def script():
    """Load the script by path; it is not importable as a package member."""
    path = os.path.join(REPO_ROOT, "scripts", "cleanup_retention.py")
    spec = importlib.util.spec_from_file_location("_cleanup_retention", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeDatabase:
    """Stands in for Database, recording the retention call and the close."""

    def __init__(self, url, deleted=0, error=None):
        self.deleted = deleted
        self.error = error
        self.called_with = None
        self.closed = False

    def delete_images_older_than(self, days):
        self.called_with = days
        if self.error is not None:
            raise self.error
        return self.deleted

    def close(self):
        self.closed = True


class _FakePsycopg:
    """Minimal psycopg stand-in recording statements run on autocommit."""

    def __init__(self, error=None):
        self.error = error
        self.executed = []
        self.connect_calls = 0
        self.autocommit = None

    def connect(self, url, autocommit=False):
        self.connect_calls += 1
        self.autocommit = autocommit
        if self.error is not None:
            raise self.error
        return self._conn()

    @contextmanager
    def _conn(self):
        outer = self

        class _Cursor:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def execute(self, sql, params=None):
                outer.executed.append((" ".join(sql.split()), params))

        class _Conn:
            def cursor(self):
                return _Cursor()

        yield _Conn()


def _wire(script, monkeypatch, days=30, deleted=0, db_error=None, vacuum_error=None):
    db = _FakeDatabase("url", deleted=deleted, error=db_error)
    fake_psycopg = _FakePsycopg(error=vacuum_error)
    monkeypatch.setattr(script.config, "IMAGE_RETENTION_DAYS", days)
    monkeypatch.setattr(script.config, "DATABASE_URL", "postgresql://u:p@h:5432/db")
    monkeypatch.setattr(script, "Database", lambda url: db)
    monkeypatch.setattr(script, "psycopg", fake_psycopg)
    return db, fake_psycopg


def test_cleanup_images_skipped_when_disabled(script, monkeypatch):
    db, fake_psycopg = _wire(script, monkeypatch, days=0)
    script.cleanup_images()
    assert db.called_with is None
    assert fake_psycopg.connect_calls == 0


def test_cleanup_images_prunes_then_vacuums_when_rows_deleted(script, monkeypatch):
    db, fake_psycopg = _wire(script, monkeypatch, days=14, deleted=3)
    script.cleanup_images()
    assert db.called_with == 14
    assert db.closed
    # VACUUM cannot run in a transaction, so the connection must be autocommit.
    assert fake_psycopg.autocommit is True
    statements = [sql for sql, _ in fake_psycopg.executed]
    assert "VACUUM (FULL, ANALYZE) images" in statements
    # lock_timeout is set first so an overlapping deploy fails fast.
    assert statements.index("SELECT set_config('lock_timeout', %s, false)") < statements.index(
        "VACUUM (FULL, ANALYZE) images"
    )


def test_cleanup_images_skips_vacuum_when_nothing_deleted(script, monkeypatch):
    """An exclusive-lock table rewrite is not worth taking for zero dead rows."""
    db, fake_psycopg = _wire(script, monkeypatch, deleted=0)
    script.cleanup_images()
    assert db.called_with == 30
    assert fake_psycopg.connect_calls == 0


def test_vacuum_failure_does_not_propagate(script, monkeypatch):
    """The prune has already committed; a lock conflict must not surface."""
    db, fake_psycopg = _wire(
        script, monkeypatch, deleted=3, vacuum_error=RuntimeError("lock timeout")
    )
    script.cleanup_images()  # must not raise
    assert fake_psycopg.connect_calls == 1


def test_prune_failure_does_not_propagate_or_vacuum(script, monkeypatch):
    db, fake_psycopg = _wire(script, monkeypatch, db_error=RuntimeError("boom"))
    script.cleanup_images()  # fail-open: never blocks a deploy
    assert db.closed
    assert fake_psycopg.connect_calls == 0


def test_main_runs_images_cleanup_between_messages_and_checkpoints(script, monkeypatch):
    calls = []
    monkeypatch.setattr(script.config, "DATABASE_URL", "postgresql://u:p@h:5432/db")
    for name in ("cleanup_messages", "cleanup_images", "cleanup_checkpoints"):
        monkeypatch.setattr(script, name, lambda name=name: calls.append(name))
    script.main()
    assert calls == ["cleanup_messages", "cleanup_images", "cleanup_checkpoints"]
