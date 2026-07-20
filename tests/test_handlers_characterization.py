"""Handler characterization tests: authorization gate, text/photo request
processing, and every admin command, against the current handlers.py
module-global wiring (SimpleNamespace/AsyncMock fakes only — no live
Telegram, database, or model calls, per CLAUDE.md)."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from agent import CompletionError
import handlers


class _Cfg:
    AUTHORIZED_USER_ID = "1"
    BOT_VERSION = "9.9.9"


def _init(db=None, agent=None, prompt_builder=None, config=None, username="mybot"):
    handlers.init_handlers(
        config or _Cfg,
        db or SimpleNamespace(),
        agent or SimpleNamespace(),
        prompt_builder or SimpleNamespace(),
        username,
    )


def _message(text=None, photo=None, caption=None, user_id=7, chat_id=123,
             chat_type="private", first_name="Alice", username="alice"):
    return SimpleNamespace(
        text=text,
        photo=photo,
        caption=caption,
        chat_id=chat_id,
        chat=SimpleNamespace(type=chat_type),
        from_user=SimpleNamespace(id=user_id, first_name=first_name, username=username),
        message_id=1,
        reply_to_message=None,
        reply_text=AsyncMock(),
    )


def _cmd_update(user_id=1, args=None):
    message = SimpleNamespace(
        from_user=SimpleNamespace(id=user_id),
        chat_id=123,
        reply_text=AsyncMock(),
    )
    update = SimpleNamespace(message=message)
    context = SimpleNamespace(args=args or [], bot=SimpleNamespace())
    return update, context, message


# --- authorization -------------------------------------------------------

def test_is_authorized_main_user_true():
    _init(config=_Cfg)
    assert handlers.is_authorized(1) is True


def test_is_authorized_granted_user_true():
    db = SimpleNamespace(is_user_granted=Mock(return_value=True))
    _init(db=db, config=_Cfg)
    assert handlers.is_authorized(42) is True
    db.is_user_granted.assert_called_once_with(42)


def test_is_authorized_unknown_user_false():
    db = SimpleNamespace(is_user_granted=Mock(return_value=False))
    _init(db=db, config=_Cfg)
    assert handlers.is_authorized(42) is False


def test_is_main_authorized_user():
    _init(config=_Cfg)
    assert handlers.is_main_authorized_user(1) is True
    assert handlers.is_main_authorized_user(2) is False


# --- text request processing ---------------------------------------------

def test_process_request_success_stores_and_replies():
    db = SimpleNamespace(add_message=Mock())
    agent = SimpleNamespace(run=AsyncMock(return_value="hi there"))
    prompt_builder = SimpleNamespace(to_lc_human_message=Mock(return_value="human"))
    _init(db=db, agent=agent, prompt_builder=prompt_builder, config=_Cfg)

    message = _message(text="chatgpt hello", user_id=1)
    update = SimpleNamespace(message=message)
    context = SimpleNamespace(bot=SimpleNamespace(send_chat_action=AsyncMock()))

    asyncio.run(handlers.message_handler(update, context))

    assert db.add_message.call_count == 2
    message.reply_text.assert_awaited_once_with("hi there")


def test_process_request_completion_error_replies_user_message():
    db = SimpleNamespace(add_message=Mock())
    agent = SimpleNamespace(run=AsyncMock(side_effect=CompletionError("rate limited")))
    prompt_builder = SimpleNamespace(to_lc_human_message=Mock(return_value="human"))
    _init(db=db, agent=agent, prompt_builder=prompt_builder, config=_Cfg)

    message = _message(text="chatgpt hello", user_id=1)
    update = SimpleNamespace(message=message)
    context = SimpleNamespace(bot=SimpleNamespace(send_chat_action=AsyncMock()))

    asyncio.run(handlers.message_handler(update, context))

    message.reply_text.assert_awaited_once_with("rate limited")


def test_process_request_generic_exception_replies_generic_message():
    db = SimpleNamespace(add_message=Mock())
    agent = SimpleNamespace(run=AsyncMock(side_effect=RuntimeError("boom")))
    prompt_builder = SimpleNamespace(to_lc_human_message=Mock(return_value="human"))
    _init(db=db, agent=agent, prompt_builder=prompt_builder, config=_Cfg)

    message = _message(text="chatgpt hello", user_id=1)
    update = SimpleNamespace(message=message)
    context = SimpleNamespace(bot=SimpleNamespace(send_chat_action=AsyncMock()))

    asyncio.run(handlers.message_handler(update, context))

    message.reply_text.assert_awaited_once_with(
        "Sorry, I encountered an error processing your request. Please try again."
    )


# --- photo request processing ---------------------------------------------

def test_process_image_request_success():
    db = SimpleNamespace(add_message=Mock())
    agent = SimpleNamespace(run=AsyncMock(return_value="nice photo"))
    prompt_builder = SimpleNamespace(to_lc_human_message=Mock(return_value="human"))
    _init(db=db, agent=agent, prompt_builder=prompt_builder, config=_Cfg)

    photo_file = SimpleNamespace(
        download_as_bytearray=AsyncMock(return_value=bytearray(b"fake-bytes"))
    )
    photo = SimpleNamespace(get_file=AsyncMock(return_value=photo_file))
    message = _message(photo=[photo], caption="chatgpt what is this", user_id=1)
    update = SimpleNamespace(message=message)
    context = SimpleNamespace(bot=SimpleNamespace(send_chat_action=AsyncMock()))

    asyncio.run(handlers.photo_handler(update, context))

    assert db.add_message.call_count == 2
    first_call_kwargs = db.add_message.call_args_list[0].kwargs
    assert first_call_kwargs["content"] == "[image] chatgpt what is this"
    message.reply_text.assert_awaited_once_with("nice photo")


def test_process_image_request_download_failure_replies_generic_image_message():
    db = SimpleNamespace(add_message=Mock())
    agent = SimpleNamespace(run=AsyncMock())
    prompt_builder = SimpleNamespace(to_lc_human_message=Mock())
    _init(db=db, agent=agent, prompt_builder=prompt_builder, config=_Cfg)

    photo = SimpleNamespace(get_file=AsyncMock(side_effect=RuntimeError("network down")))
    message = _message(photo=[photo], caption="chatgpt describe this", user_id=1)
    update = SimpleNamespace(message=message)
    context = SimpleNamespace(bot=SimpleNamespace(send_chat_action=AsyncMock()))

    asyncio.run(handlers.photo_handler(update, context))

    message.reply_text.assert_awaited_once_with(
        "Sorry, I encountered an error processing your image. Please try again."
    )
    db.add_message.assert_not_called()
    agent.run.assert_not_awaited()


# --- admin commands ---------------------------------------------------------

def test_clear_command_requires_main_user():
    _init(config=_Cfg)
    update, context, message = _cmd_update(user_id=2)
    asyncio.run(handlers.clear_command(update, context))
    message.reply_text.assert_awaited_once_with(
        "Sorry, only the main authorized user can clear history."
    )


def test_clear_command_success():
    agent = SimpleNamespace(clear_thread=Mock())
    _init(agent=agent, config=_Cfg)
    update, context, message = _cmd_update(user_id=1)
    asyncio.run(handlers.clear_command(update, context))
    agent.clear_thread.assert_called_once_with("123")
    message.reply_text.assert_awaited_once_with(
        "✅ Conversation history cleared for this chat."
    )


def test_stats_command_requires_main_user():
    _init(config=_Cfg)
    update, context, message = _cmd_update(user_id=2)
    asyncio.run(handlers.stats_command(update, context))
    message.reply_text.assert_awaited_once_with(
        "Sorry, only the main authorized user can view stats."
    )


def test_stats_command_formats_output():
    db = SimpleNamespace(get_stats=Mock(return_value={
        "total_messages": 5, "total_tokens": 1234,
        "first_message": "2026-01-01T00:00:00", "last_message": "2026-01-02T00:00:00",
    }))
    _init(db=db, config=_Cfg)
    update, context, message = _cmd_update(user_id=1)
    asyncio.run(handlers.stats_command(update, context))
    message.reply_text.assert_awaited_once_with(
        "📊 Chat Statistics:\nMessages: 5\nTotal tokens: 1,234\nSince: 2026-01-01"
    )


def test_grant_command_requires_args():
    _init(config=_Cfg)
    update, context, message = _cmd_update(user_id=1, args=[])
    asyncio.run(handlers.grant_command(update, context))
    message.reply_text.assert_awaited_once_with(
        "❌ Usage: /grant <user_id>\nExample: /grant 123456789"
    )


def test_grant_command_success():
    db = SimpleNamespace(grant_access=Mock(return_value=True))
    _init(db=db, config=_Cfg)
    update, context, message = _cmd_update(user_id=1, args=["555"])
    context.bot.get_chat = AsyncMock(return_value=SimpleNamespace(first_name="Bob", username="bobby"))
    asyncio.run(handlers.grant_command(update, context))
    db.grant_access.assert_called_once_with(555, first_name="Bob", username="bobby")
    message.reply_text.assert_awaited_once_with(
        "✅ Access granted to Bob (@bobby).\nThey can now use the bot with 'chatgpt' keyword."
    )


def test_grant_command_already_has_access():
    db = SimpleNamespace(grant_access=Mock(return_value=False))
    _init(db=db, config=_Cfg)
    update, context, message = _cmd_update(user_id=1, args=["555"])
    context.bot.get_chat = AsyncMock(return_value=SimpleNamespace(first_name="Bob", username="bobby"))
    asyncio.run(handlers.grant_command(update, context))
    message.reply_text.assert_awaited_once_with("ℹ️ User 555 already has access.")


def test_revoke_command_cannot_revoke_self():
    _init(config=_Cfg)
    update, context, message = _cmd_update(user_id=1, args=["1"])
    asyncio.run(handlers.revoke_command(update, context))
    message.reply_text.assert_awaited_once_with(
        "❌ Cannot revoke access from the main authorized user."
    )


def test_allowlist_command_lists_users():
    db = SimpleNamespace(get_granted_users=Mock(return_value=[
        (555, "2026-01-01T00:00:00", "Bob", "bobby"),
    ]))
    _init(db=db, config=_Cfg)
    update, context, message = _cmd_update(user_id=1)
    asyncio.run(handlers.allowlist_command(update, context))
    text = message.reply_text.call_args.args[0]
    assert "555" in text and "Bob" in text and "bobby" in text


def test_personality_command_shows_current_when_no_args():
    db = SimpleNamespace(get_active_personality=Mock(return_value="villain"))
    _init(db=db, config=_Cfg)
    update, context, message = _cmd_update(user_id=1, args=[])
    asyncio.run(handlers.personality_command(update, context))
    assert "villain" in message.reply_text.call_args.args[0]


def test_personality_command_unknown_personality():
    db = SimpleNamespace(personality_exists=Mock(return_value=False))
    _init(db=db, config=_Cfg)
    update, context, message = _cmd_update(user_id=1, args=["ghost"])
    asyncio.run(handlers.personality_command(update, context))
    message.reply_text.assert_awaited_once_with("❌ No personality 'ghost' found.")


def test_model_command_shows_current_when_no_args():
    db = SimpleNamespace(get_active_model=Mock(return_value="gpt-5.4-mini"))
    _init(db=db, config=_Cfg)
    update, context, message = _cmd_update(user_id=1, args=[])
    asyncio.run(handlers.model_command(update, context))
    assert "gpt-5.4-mini" in message.reply_text.call_args.args[0]


def test_model_command_unknown_model_rejected():
    _init(config=_Cfg)
    update, context, message = _cmd_update(user_id=1, args=["not-a-model"])
    asyncio.run(handlers.model_command(update, context))
    assert "Unknown model" in message.reply_text.call_args.args[0]


def test_model_command_switch_success():
    db = SimpleNamespace(set_active_model=Mock())
    agent = SimpleNamespace(set_model=Mock())
    _init(db=db, agent=agent, config=_Cfg)
    update, context, message = _cmd_update(user_id=1, args=["gpt-5.4"])
    asyncio.run(handlers.model_command(update, context))
    db.set_active_model.assert_called_once_with("gpt-5.4")
    agent.set_model.assert_called_once_with("gpt-5.4")
    message.reply_text.assert_awaited_once_with(
        "✅ Model switched to `gpt-5.4`", parse_mode="Markdown"
    )


def test_help_command_requires_main_user():
    _init(config=_Cfg)
    update, context, message = _cmd_update(user_id=2)
    asyncio.run(handlers.help_command(update, context))
    message.reply_text.assert_awaited_once_with(
        "Sorry, only the main authorized user can use this command."
    )


def test_error_handler_notifies_user_when_possible():
    message = SimpleNamespace(reply_text=AsyncMock())
    update = SimpleNamespace(message=message)
    context = SimpleNamespace(error=RuntimeError("boom"))
    asyncio.run(handlers.error_handler(update, context))
    message.reply_text.assert_awaited_once_with(
        "An error occurred while processing your request. "
        "The error has been logged."
    )
