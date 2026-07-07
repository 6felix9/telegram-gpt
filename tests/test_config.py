"""Config validation and defaults (no .env / DB required)."""
import importlib
import sys
import pytest


def _fresh_config(monkeypatch, env: dict):
    """Reload config.py with a controlled environment."""
    for key in [
        "TELEGRAM_BOT_TOKEN", "BOT_USERNAME", "OPENAI_API_KEY", "XAI_API_KEY",
        "GEMINI_API_KEY", "DEFAULT_MODEL", "OPENAI_TIMEOUT", "MAX_CONTEXT_TOKENS",
        "RESERVE_TOKENS_TEXT", "RESERVE_TOKENS_IMAGE", "MAX_GROUP_CONTEXT_MESSAGES",
        "TAVILY_API_KEY", "AUTHORIZED_USER_ID", "DATABASE_URL", "LOG_LEVEL",
    ]:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    sys.modules.pop("config", None)
    return importlib.import_module("config")


VALID = {
    "TELEGRAM_BOT_TOKEN": "t",
    "AUTHORIZED_USER_ID": "123",
    "OPENAI_API_KEY": "sk-x",
    "DATABASE_URL": "postgresql://u:p@h:5432/db",
}


def test_defaults_apply_when_optional_unset(monkeypatch):
    cfg = _fresh_config(monkeypatch, VALID)
    assert cfg.config.DEFAULT_MODEL == "gpt-5.4-mini"
    assert cfg.config.RESERVE_TOKENS_TEXT == 2000
    assert cfg.config.RESERVE_TOKENS_IMAGE == 3000
    assert cfg.config.MAX_CONTEXT_TOKENS == 16000
    assert cfg.config.MAX_GROUP_CONTEXT_MESSAGES == 500
    assert cfg.config.OPENAI_TIMEOUT == 60
    assert cfg.config.BOT_USERNAME == ""
    assert cfg.config.TAVILY_API_KEY == ""


def test_validate_passes_with_only_required(monkeypatch):
    cfg = _fresh_config(monkeypatch, VALID)
    cfg.config.validate()  # must not raise / sys.exit


def test_missing_required_exits(monkeypatch):
    env = dict(VALID)
    del env["OPENAI_API_KEY"]
    cfg = _fresh_config(monkeypatch, env)
    with pytest.raises(SystemExit):
        cfg.config.validate()


def test_missing_provider_key_does_not_fail_startup(monkeypatch):
    # DEFAULT_MODEL selects xAI but XAI_API_KEY absent — must still validate.
    env = dict(VALID, DEFAULT_MODEL="grok-4-1-fast-reasoning")
    cfg = _fresh_config(monkeypatch, env)
    cfg.config.validate()  # must not raise
