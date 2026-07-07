"""Regression test for scripts/setup_checkpointer.py invocation.

The script is run in deploy as `python scripts/setup_checkpointer.py` (Railway
preDeployCommand and start.sh). Running a script by path puts scripts/ on
sys.path, not the repo root, so `from config import config` used to fail with
ModuleNotFoundError at runtime — crashing the deploy. The script now inserts the
repo root onto sys.path itself; this test pins that by invoking it exactly as
deploy does. It stops at the DATABASE_URL guard, so it needs no DB or network.
"""
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_script_imports_config_when_run_by_path():
    # Empty DATABASE_URL makes the script exit at its own guard right after the
    # imports succeed, so no DB connection is attempted. config.load_dotenv()
    # uses override=False, so this empty value is not clobbered by a local .env.
    env = dict(os.environ, DATABASE_URL="")
    result = subprocess.run(
        [sys.executable, "scripts/setup_checkpointer.py"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    combined = result.stdout + result.stderr
    # The import must resolve — the old bug produced this exact message.
    assert "No module named 'config'" not in combined
    assert "ModuleNotFoundError" not in combined
    # And it must reach main() and hit the DATABASE_URL guard.
    assert result.returncode != 0
    assert "DATABASE_URL is required" in combined
