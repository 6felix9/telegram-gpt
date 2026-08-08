"""Regression test for scripts/cleanup_retention.py invocation.

Mirrors test_setup_checkpointer.py: the script is run by path in Railway's
preDeployCommand, so `from config import config` / `from database import
Database` must resolve via the script's own sys.path bootstrap. It stops at
the DATABASE_URL guard, so it needs no DB or network.
"""
import os
import subprocess
import sys

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
