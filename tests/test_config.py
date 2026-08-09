"""Config validation and defaults (no .env / DB required)."""
import importlib
import sys
import pytest


def _fresh_config(monkeypatch, env: dict):
    """Reload config.py with a controlled environment."""
    for key in [
        "TELEGRAM_BOT_TOKEN", "BOT_USERNAME", "OPENAI_API_KEY", "XAI_API_KEY",
        "GEMINI_API_KEY", "DEFAULT_MODEL", "MODEL_TIMEOUT", "MAX_CONTEXT_TOKENS",
        "MAX_OUTPUT_TOKENS", "SUMMARY_MODEL", "SUMMARIZATION_TRIGGER",
        "MAX_SUMMARY_OUTPUT",
        "TAVILY_API_KEY", "AUTHORIZED_USER_ID", "DATABASE_URL", "LOG_LEVEL",
        "VISION_SUMMARY_MODEL", "MESSAGE_RETENTION_DAYS",
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
    assert cfg.config.SUMMARY_MODEL == "gpt-5.6-luna"
    assert cfg.config.VISION_SUMMARY_MODEL == "gpt-5.4-nano"
    assert cfg.config.SUMMARIZATION_TRIGGER == 8000
    assert cfg.config.MAX_SUMMARY_OUTPUT == 1000
    assert not hasattr(cfg.config, "SUMMARY_TRIGGER_TOKENS")
    assert not hasattr(cfg.config, "SUMMARY_KEEP_TOKENS")
    assert not hasattr(cfg.config, "SUMMARY_CONTEXT_TOKENS")
    assert not hasattr(cfg.config, "MAX_GROUP_CONTEXT_MESSAGES")
    assert cfg.config.MODEL_TIMEOUT == 60
    assert cfg.config.BOT_USERNAME == ""
    assert cfg.config.TAVILY_API_KEY == ""
    assert cfg.config.MESSAGE_RETENTION_DAYS == 30


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
        ({"SUMMARIZATION_TRIGGER": "0"}, "SUMMARIZATION_TRIGGER must be positive"),
        ({"MAX_SUMMARY_OUTPUT": "0"}, "MAX_SUMMARY_OUTPUT must be positive"),
        (
            {"SUMMARIZATION_TRIGGER": "1000", "MAX_SUMMARY_OUTPUT": "1000"},
            "MAX_SUMMARY_OUTPUT must be less than SUMMARIZATION_TRIGGER",
        ),
        (
            {"SUMMARIZATION_TRIGGER": "1000", "MAX_SUMMARY_OUTPUT": "2000"},
            "MAX_SUMMARY_OUTPUT must be less than SUMMARIZATION_TRIGGER",
        ),
    ],
)
def test_invalid_summary_limits_exit(monkeypatch, caplog, overrides, message):
    cfg = _fresh_config(monkeypatch, dict(VALID, **overrides))
    with pytest.raises(SystemExit):
        cfg.config.validate()
    assert message in caplog.text


def test_negative_message_retention_days_fails_validation(monkeypatch):
    cfg = _fresh_config(monkeypatch, dict(VALID, MESSAGE_RETENTION_DAYS="-1"))
    with pytest.raises(SystemExit):
        cfg.config.validate()


def test_zero_message_retention_days_is_valid(monkeypatch):
    cfg = _fresh_config(monkeypatch, dict(VALID, MESSAGE_RETENTION_DAYS="0"))
    cfg.config.validate()  # 0 disables cleanup, must not raise


def test_blank_int_vars_fall_back_to_defaults(monkeypatch):
    """.env.example ships these keys blank, so "" must mean "use the default".

    os.getenv's default only fires when a var is absent; a set-but-empty var
    (cp .env.example .env, or an empty Railway variable) used to crash import
    with ValueError: invalid literal for int().
    """
    cfg = _fresh_config(monkeypatch, dict(
        VALID,
        MODEL_TIMEOUT="",
        MAX_CONTEXT_TOKENS="",
        MAX_OUTPUT_TOKENS="",
        SUMMARIZATION_TRIGGER="",
        MAX_SUMMARY_OUTPUT="  ",
    ))
    assert cfg.config.MODEL_TIMEOUT == 60
    assert cfg.config.MAX_CONTEXT_TOKENS == 16000
    assert cfg.config.MAX_OUTPUT_TOKENS == 2048
    assert cfg.config.SUMMARIZATION_TRIGGER == 8000
    assert cfg.config.MAX_SUMMARY_OUTPUT == 1000
    cfg.config.validate()  # blank optionals must not fail validation


def test_non_numeric_int_var_falls_back_to_default(monkeypatch):
    cfg = _fresh_config(monkeypatch, dict(VALID, MODEL_TIMEOUT="banana"))
    assert cfg.config.MODEL_TIMEOUT == 60


def test_explicit_int_var_still_wins(monkeypatch):
    cfg = _fresh_config(monkeypatch, dict(VALID, MODEL_TIMEOUT="90"))
    assert cfg.config.MODEL_TIMEOUT == 90
