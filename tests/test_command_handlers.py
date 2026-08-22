"""CommandHandlers: admin-only command surface, bound to explicit deps."""
import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from handlers.command_handlers import CommandHandlers, error_handler
from handlers.handler_deps import HandlerDependencies

_NOW = datetime(2026, 8, 22, 12, 0, 0)


class _Cfg:
    AUTHORIZED_USER_ID = "1"
    BOT_VERSION = "9.9.9"


def _handlers(db=None, agent=None):
    deps = HandlerDependencies(
        config=_Cfg, db=db or SimpleNamespace(), agent=agent or SimpleNamespace(),
        prompt_builder=SimpleNamespace(),
    )
    return CommandHandlers(deps)


def _update(user_id=1, args=None):
    message = SimpleNamespace(
        from_user=SimpleNamespace(id=user_id), chat_id=123, reply_text=AsyncMock(),
    )
    update = SimpleNamespace(message=message)
    context = SimpleNamespace(args=args or [], bot=SimpleNamespace())
    return update, context, message


def test_clear_command_requires_main_user():
    handlers_obj = _handlers()
    update, context, message = _update(user_id=2)
    asyncio.run(handlers_obj.clear_command(update, context))
    message.reply_text.assert_awaited_once_with(
        "Sorry, only the main authorized user can clear history."
    )


def test_clear_command_success():
    agent = SimpleNamespace(clear_thread=Mock())
    handlers_obj = _handlers(agent=agent)
    update, context, message = _update(user_id=1)
    asyncio.run(handlers_obj.clear_command(update, context))
    agent.clear_thread.assert_called_once_with("123")
    message.reply_text.assert_awaited_once_with(
        "✅ Conversation history cleared for this chat."
    )


def test_model_command_switch_success():
    db = SimpleNamespace(set_active_model=Mock())
    agent = SimpleNamespace(set_model=Mock())
    handlers_obj = _handlers(db=db, agent=agent)
    update, context, message = _update(user_id=1, args=["gpt-5.4"])
    asyncio.run(handlers_obj.model_command(update, context))
    db.set_active_model.assert_called_once_with("gpt-5.4")
    agent.set_model.assert_called_once_with("gpt-5.4")
    message.reply_text.assert_awaited_once_with(
        "✅ Model switched to `gpt-5.4`", parse_mode="Markdown"
    )


def test_allowlist_command_shows_open_access_off():
    db = SimpleNamespace(
        get_granted_users=Mock(return_value=[]),
        get_open_access=Mock(return_value=(False, None)),
    )
    handlers_obj = _handlers(db=db)
    update, context, message = _update(user_id=1)
    asyncio.run(handlers_obj.allowlist_command(update, context))
    text = message.reply_text.await_args.args[0]
    assert "🔒 Open access: OFF" in text


def test_allowlist_command_shows_open_access_on_with_remaining_time():
    expires_at = _NOW + timedelta(hours=1)
    db = SimpleNamespace(
        get_granted_users=Mock(return_value=[]),
        get_open_access=Mock(return_value=(True, expires_at)),
    )
    handlers_obj = _handlers(db=db)
    update, context, message = _update(user_id=1)
    with patch("handlers.command_handlers.datetime") as mock_dt:
        mock_dt.utcnow.return_value = _NOW
        asyncio.run(handlers_obj.allowlist_command(update, context))
    text = message.reply_text.await_args.args[0]
    assert "🔓 Open access: ON (expires in 1h)" in text


def test_openbot_command_requires_main_user():
    handlers_obj = _handlers()
    update, context, message = _update(user_id=2)
    asyncio.run(handlers_obj.openbot_command(update, context))
    message.reply_text.assert_awaited_once_with(
        "Sorry, only the main authorized user can change open access."
    )


def test_openbot_command_bare_reports_off_state():
    db = SimpleNamespace(get_open_access=Mock(return_value=(False, None)))
    handlers_obj = _handlers(db=db)
    update, context, message = _update(user_id=1, args=[])
    asyncio.run(handlers_obj.openbot_command(update, context))
    message.reply_text.assert_awaited_once_with("🔒 Open access is OFF.")


def test_openbot_command_bare_reports_on_state_with_remaining_time():
    expires_at = _NOW + timedelta(hours=2, minutes=30)
    db = SimpleNamespace(get_open_access=Mock(return_value=(True, expires_at)))
    handlers_obj = _handlers(db=db)
    update, context, message = _update(user_id=1, args=[])
    with patch("handlers.command_handlers.datetime") as mock_dt:
        mock_dt.utcnow.return_value = _NOW
        asyncio.run(handlers_obj.openbot_command(update, context))
    message.reply_text.assert_awaited_once_with(
        "🔓 Open access is ON (expires in 2h 30m)."
    )


def test_openbot_command_on_default_duration():
    db = SimpleNamespace(set_open_access=Mock())
    handlers_obj = _handlers(db=db)
    update, context, message = _update(user_id=1, args=["on"])
    with patch("handlers.command_handlers.datetime") as mock_dt:
        mock_dt.utcnow.return_value = _NOW
        asyncio.run(handlers_obj.openbot_command(update, context))
    db.set_open_access.assert_called_once_with(True, _NOW + timedelta(hours=4))
    message.reply_text.assert_awaited_once_with(
        "🔓 Open access turned ON — expires in 4h."
    )


def test_openbot_command_on_explicit_duration():
    db = SimpleNamespace(set_open_access=Mock())
    handlers_obj = _handlers(db=db)
    update, context, message = _update(user_id=1, args=["on", "2h"])
    with patch("handlers.command_handlers.datetime") as mock_dt:
        mock_dt.utcnow.return_value = _NOW
        asyncio.run(handlers_obj.openbot_command(update, context))
    db.set_open_access.assert_called_once_with(True, _NOW + timedelta(hours=2))
    message.reply_text.assert_awaited_once_with(
        "🔓 Open access turned ON — expires in 2h."
    )


def test_openbot_command_on_invalid_duration():
    db = SimpleNamespace(set_open_access=Mock())
    handlers_obj = _handlers(db=db)
    update, context, message = _update(user_id=1, args=["on", "5x"])
    asyncio.run(handlers_obj.openbot_command(update, context))
    db.set_open_access.assert_not_called()
    message.reply_text.assert_awaited_once_with(
        "❌ Invalid duration `5x`. Use formats like `30m`, `2h`, or `1d`.",
        parse_mode="Markdown",
    )


def test_openbot_command_off():
    db = SimpleNamespace(set_open_access=Mock())
    handlers_obj = _handlers(db=db)
    update, context, message = _update(user_id=1, args=["off"])
    asyncio.run(handlers_obj.openbot_command(update, context))
    db.set_open_access.assert_called_once_with(False, None)
    message.reply_text.assert_awaited_once_with("🔒 Open access turned OFF.")


def test_openbot_command_unknown_arg_shows_usage():
    db = SimpleNamespace(set_open_access=Mock())
    handlers_obj = _handlers(db=db)
    update, context, message = _update(user_id=1, args=["frobnicate"])
    asyncio.run(handlers_obj.openbot_command(update, context))
    db.set_open_access.assert_not_called()
    message.reply_text.assert_awaited_once_with(
        "❌ Usage: `/openbot [on|off] [duration]`\nExample: `/openbot on 2h`",
        parse_mode="Markdown",
    )


def test_error_handler_notifies_user_when_possible():
    message = SimpleNamespace(reply_text=AsyncMock())
    update = SimpleNamespace(message=message)
    context = SimpleNamespace(error=RuntimeError("boom"))
    asyncio.run(error_handler(update, context))
    message.reply_text.assert_awaited_once_with(
        "An error occurred while processing your request. "
        "The error has been logged."
    )
