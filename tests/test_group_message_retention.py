"""Group-message retention behavior."""
import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import handlers


def test_non_triggering_group_message_does_not_run_database_cleanup(monkeypatch):
    database = SimpleNamespace(
        add_message=Mock(),
        cleanup_old_group_messages=Mock(),
    )
    bot_agent = SimpleNamespace(append_context_message=Mock())
    prompt_builder = SimpleNamespace(to_lc_human_message=Mock(return_value="human"))

    monkeypatch.setattr(handlers, "config", SimpleNamespace(MAX_GROUP_CONTEXT_MESSAGES=500))
    monkeypatch.setattr(handlers, "db", database)
    monkeypatch.setattr(handlers, "agent", bot_agent)
    monkeypatch.setattr(handlers, "prompt_builder", prompt_builder)
    monkeypatch.setattr(handlers, "bot_username", "mybot")
    if hasattr(handlers, "random"):
        monkeypatch.setattr(handlers.random, "random", lambda: 0.0)

    message = SimpleNamespace(
        text="ordinary group message",
        chat_id=-123,
        chat=SimpleNamespace(type="group"),
        from_user=SimpleNamespace(id=42, first_name="Alice", username="alice"),
        message_id=7,
    )

    asyncio.run(
        handlers.message_handler(
            SimpleNamespace(message=message),
            SimpleNamespace(),
        )
    )

    database.add_message.assert_called_once()
    bot_agent.append_context_message.assert_called_once_with("-123", "human")
    database.cleanup_old_group_messages.assert_not_called()
