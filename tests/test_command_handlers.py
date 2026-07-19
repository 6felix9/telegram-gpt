"""CommandHandlers: admin-only command surface, bound to explicit deps."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from command_handlers import CommandHandlers, error_handler
from handler_deps import HandlerDependencies


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


def test_error_handler_notifies_user_when_possible():
    message = SimpleNamespace(reply_text=AsyncMock())
    update = SimpleNamespace(message=message)
    context = SimpleNamespace(error=RuntimeError("boom"))
    asyncio.run(error_handler(update, context))
    message.reply_text.assert_awaited_once_with(
        "An error occurred while processing your request. "
        "The error has been logged."
    )
