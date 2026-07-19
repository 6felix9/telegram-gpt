"""Non-triggering text context retention for group and private chats."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import handlers


def _run_message_handler(message, *, monkeypatch):
    database = SimpleNamespace(
        add_message=Mock(),
        cleanup_old_group_messages=Mock(),
    )
    bot_agent = SimpleNamespace(
        append_context_message=AsyncMock(),
        run=Mock(),
    )
    prompt_builder = SimpleNamespace(to_lc_human_message=Mock(return_value="human"))

    monkeypatch.setattr(handlers, "config", SimpleNamespace(MAX_GROUP_CONTEXT_MESSAGES=500))
    monkeypatch.setattr(handlers, "db", database)
    monkeypatch.setattr(handlers, "agent", bot_agent)
    monkeypatch.setattr(handlers, "prompt_builder", prompt_builder)
    monkeypatch.setattr(handlers, "bot_username", "mybot")
    if hasattr(handlers, "random"):
        monkeypatch.setattr(handlers.random, "random", lambda: 0.0)

    asyncio.run(
        handlers.message_handler(
            SimpleNamespace(message=message),
            SimpleNamespace(),
        )
    )
    return database, bot_agent, prompt_builder


def test_non_triggering_group_message_stores_context_without_cleanup(monkeypatch):
    message = SimpleNamespace(
        text="ordinary group message",
        chat_id=-123,
        chat=SimpleNamespace(type="group"),
        from_user=SimpleNamespace(id=42, first_name="Alice", username="alice"),
        message_id=7,
    )

    database, bot_agent, prompt_builder = _run_message_handler(message, monkeypatch=monkeypatch)

    database.add_message.assert_called_once()
    assert database.add_message.call_args.kwargs["is_group_chat"] is True
    prompt_builder.to_lc_human_message.assert_called_once_with(
        text="ordinary group message", is_group=True, sender_name="Alice",
    )
    bot_agent.append_context_message.assert_awaited_once_with("-123", "human")
    bot_agent.run.assert_not_called()
    database.cleanup_old_group_messages.assert_not_called()


def test_non_triggering_private_message_stores_context(monkeypatch):
    message = SimpleNamespace(
        text="flight is at 6",
        chat_id=99,
        chat=SimpleNamespace(type="private"),
        from_user=SimpleNamespace(id=42, first_name="Alice", username="alice"),
        message_id=8,
    )

    database, bot_agent, prompt_builder = _run_message_handler(message, monkeypatch=monkeypatch)

    database.add_message.assert_called_once()
    assert database.add_message.call_args.kwargs["is_group_chat"] is False
    prompt_builder.to_lc_human_message.assert_called_once_with(
        text="flight is at 6", is_group=False, sender_name="Alice",
    )
    bot_agent.append_context_message.assert_awaited_once_with("99", "human")
    bot_agent.run.assert_not_called()
