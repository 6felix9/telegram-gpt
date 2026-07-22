"""Config validation and defaults (no .env / DB required)."""
import importlib
import sys
import pytest


def _fresh_config(monkeypatch, env: dict):
    """Reload config.py with a controlled environment."""
    for key in [
        "TELEGRAM_BOT_TOKEN", "BOT_USERNAME", "OPENAI_API_KEY", "XAI_API_KEY",
        "GEMINI_API_KEY", "DEFAULT_MODEL", "MODEL_TIMEOUT", "MAX_CONTEXT_TOKENS",
        "MAX_OUTPUT_TOKENS", "SUMMARY_MODEL", "SUMMARY_TRIGGER_TOKENS",
        "SUMMARY_KEEP_TOKENS", "SUMMARY_CONTEXT_TOKENS", "MAX_GROUP_CONTEXT_MESSAGES",
        "TAVILY_API_KEY", "AUTHORIZED_USER_ID", "DATABASE_URL", "LOG_LEVEL",
        "VISION_SUMMARY_MODEL",
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
    assert cfg.config.MAX_OUTPUT_TOKENS == 2048
    assert cfg.config.MAX_CONTEXT_TOKENS == 16000
    assert cfg.config.SUMMARY_MODEL == "gpt-4.1-mini"
    assert cfg.config.VISION_SUMMARY_MODEL == "gpt-4.1-mini"
    assert cfg.config.SUMMARY_TRIGGER_TOKENS == 10000
    assert cfg.config.SUMMARY_KEEP_TOKENS == 4000
    assert cfg.config.SUMMARY_CONTEXT_TOKENS == 14000
    assert cfg.config.MAX_GROUP_CONTEXT_MESSAGES == 500
    assert cfg.config.MODEL_TIMEOUT == 60
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


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"SUMMARY_TRIGGER_TOKENS": "0"}, "SUMMARY_TRIGGER_TOKENS must be positive"),
        ({"SUMMARY_KEEP_TOKENS": "0"}, "SUMMARY_KEEP_TOKENS must be positive"),
        ({"SUMMARY_CONTEXT_TOKENS": "0"}, "SUMMARY_CONTEXT_TOKENS must be positive"),
        (
            {"SUMMARY_TRIGGER_TOKENS": "4000", "SUMMARY_KEEP_TOKENS": "4000"},
            "SUMMARY_KEEP_TOKENS must be less than SUMMARY_TRIGGER_TOKENS",
        ),
        (
            {
                "SUMMARY_TRIGGER_TOKENS": "10000",
                "SUMMARY_KEEP_TOKENS": "4000",
                "SUMMARY_CONTEXT_TOKENS": "5000",
            },
            "SUMMARY_CONTEXT_TOKENS must be at least "
            "SUMMARY_TRIGGER_TOKENS - SUMMARY_KEEP_TOKENS",
        ),
    ],
)
def test_invalid_summary_limits_exit(monkeypatch, caplog, overrides, message):
    cfg = _fresh_config(monkeypatch, dict(VALID, **overrides))
    with pytest.raises(SystemExit):
        cfg.config.validate()
    assert message in caplog.text
