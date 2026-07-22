"""RequestProcessor: shared audit-log -> agent.run -> audit-log -> reply
workflow used by both text and photo handlers."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from agent import CompletionError
from handlers.handler_deps import HandlerDependencies
from handlers.request_processor import RequestProcessor


def _deps(db=None, agent=None):
    return HandlerDependencies(
        config=SimpleNamespace(), db=db or SimpleNamespace(add_message=Mock()),
        agent=agent or SimpleNamespace(), prompt_builder=SimpleNamespace(),
    )


def _message(chat_id=123):
    return SimpleNamespace(chat_id=chat_id, message_id=1, reply_text=AsyncMock())


def _bot():
    return SimpleNamespace(send_chat_action=AsyncMock())


async def _payload():
    return "content", 3, "human-message"


def test_process_success_stores_both_turns_and_replies():
    db = SimpleNamespace(add_message=Mock())
    agent = SimpleNamespace(run=AsyncMock(return_value="reply text"))
    processor = RequestProcessor(_deps(db=db, agent=agent))
    message = _message()

    asyncio.run(processor.process(
        _bot(), message, user_id=1, sender_name="Alice", sender_username="alice",
        is_group=False, build_payload=_payload, reply_context=None,
        generic_error_text="generic error", success_log="ok", error_log_prefix="err",
    ))

    assert db.add_message.call_count == 2
    message.reply_text.assert_awaited_once_with("reply text")


def test_process_runs_post_success_after_reply():
    calls = []
    db = SimpleNamespace(add_message=Mock())
    agent = SimpleNamespace(run=AsyncMock(return_value="reply text"))
    processor = RequestProcessor(_deps(db=db, agent=agent))
    message = _message()

    async def _hook():
        calls.append("ran")

    asyncio.run(processor.process(
        _bot(), message, user_id=1, sender_name="Alice", sender_username="alice",
        is_group=False, build_payload=_payload, reply_context=None,
        generic_error_text="generic error", success_log="ok", error_log_prefix="err",
        post_success=_hook,
    ))

    message.reply_text.assert_awaited_once_with("reply text")
    assert calls == ["ran"]


def test_process_post_success_failure_does_not_break_reply():
    db = SimpleNamespace(add_message=Mock())
    agent = SimpleNamespace(run=AsyncMock(return_value="reply text"))
    processor = RequestProcessor(_deps(db=db, agent=agent))
    message = _message()

    async def _hook():
        raise RuntimeError("boom")

    asyncio.run(processor.process(
        _bot(), message, user_id=1, sender_name="Alice", sender_username="alice",
        is_group=False, build_payload=_payload, reply_context=None,
        generic_error_text="generic error", success_log="ok", error_log_prefix="err",
        post_success=_hook,
    ))

    # Reply still went out exactly once; the generic error was NOT sent.
    message.reply_text.assert_awaited_once_with("reply text")


def test_process_completion_error_replies_user_message():
    agent = SimpleNamespace(run=AsyncMock(side_effect=CompletionError("rate limited")))
    processor = RequestProcessor(_deps(agent=agent))
    message = _message()

    asyncio.run(processor.process(
        _bot(), message, user_id=1, sender_name="Alice", sender_username="alice",
        is_group=False, build_payload=_payload, reply_context=None,
        generic_error_text="generic error", success_log="ok", error_log_prefix="err",
    ))

    message.reply_text.assert_awaited_once_with("rate limited")


def test_process_payload_build_failure_replies_generic_error():
    async def _failing_payload():
        raise RuntimeError("download failed")

    processor = RequestProcessor(_deps())
    message = _message()

    asyncio.run(processor.process(
        _bot(), message, user_id=1, sender_name="Alice", sender_username="alice",
        is_group=False, build_payload=_failing_payload, reply_context=None,
        generic_error_text="generic error", success_log="ok", error_log_prefix="err",
    ))

    message.reply_text.assert_awaited_once_with("generic error")
