"""MessageHandlers: text/photo intake, activation parsing, and auth gate."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from handlers.handler_deps import HandlerDependencies
from handlers.message_handlers import MessageHandlers, extract_keyword, extract_reply_data
from handlers.request_processor import RequestProcessor


@pytest.mark.parametrize(
    "text, bot_username, expected_has_keyword, expected_prompt",
    [
        ("", None, False, ""),
        ("hello world", None, False, "hello world"),
        ("chatgpt what is 2+2", None, True, "what is 2+2"),
        ("@MyBot hello", "MyBot", True, "hello"),
    ],
)
def test_extract_keyword(text, bot_username, expected_has_keyword, expected_prompt):
    has_keyword, prompt = extract_keyword(text, bot_username)
    assert has_keyword is expected_has_keyword
    assert prompt == expected_prompt


def test_extract_reply_data_returns_none_without_reply():
    message = SimpleNamespace(reply_to_message=None)
    assert extract_reply_data(message) is None


class _Cfg:
    AUTHORIZED_USER_ID = "1"


def _handlers(db=None, agent=None, prompt_builder=None, username="mybot"):
    if db is None:
        db = SimpleNamespace(is_user_granted=Mock(return_value=False))
    deps = HandlerDependencies(
        config=_Cfg, db=db, agent=agent or SimpleNamespace(),
        prompt_builder=prompt_builder or SimpleNamespace(), bot_username=username,
    )
    return MessageHandlers(deps, RequestProcessor(deps))


def _message(text=None, chat_id=123, chat_type="private", user_id=7):
    return SimpleNamespace(
        text=text, photo=None, caption=None, chat_id=chat_id,
        chat=SimpleNamespace(type=chat_type),
        from_user=SimpleNamespace(id=user_id, first_name="Alice", username="alice"),
        message_id=1, reply_to_message=None, reply_text=AsyncMock(),
    )


def test_non_triggering_message_stores_context_without_reply():
    db = SimpleNamespace(add_message=Mock())
    agent = SimpleNamespace(append_context_message=Mock(), run=AsyncMock())
    prompt_builder = SimpleNamespace(to_lc_human_message=Mock(return_value="human"))
    handlers_obj = _handlers(db=db, agent=agent, prompt_builder=prompt_builder)

    message = _message(text="ordinary message", chat_type="group")
    update = SimpleNamespace(message=message)
    context = SimpleNamespace()

    asyncio.run(handlers_obj.message_handler(update, context))

    db.add_message.assert_called_once()
    agent.append_context_message.assert_called_once_with("123", "human")
    agent.run.assert_not_awaited()


def test_unauthorized_triggering_message_replies_no_access():
    handlers_obj = _handlers()
    message = _message(text="chatgpt hi", user_id=99)
    update = SimpleNamespace(message=message)
    context = SimpleNamespace()

    asyncio.run(handlers_obj.message_handler(update, context))

    message.reply_text.assert_awaited_once_with("Sorry, you have no access to me.")


def test_authorized_empty_prompt_asks_for_request():
    handlers_obj = _handlers()
    message = _message(text="chatgpt", user_id=1)
    update = SimpleNamespace(message=message)
    context = SimpleNamespace()

    asyncio.run(handlers_obj.message_handler(update, context))

    message.reply_text.assert_awaited_once_with("Yes, what's your request?")


def test_authorized_triggering_message_processes_and_replies():
    db = SimpleNamespace(add_message=Mock())
    agent = SimpleNamespace(run=AsyncMock(return_value="hi there"))
    prompt_builder = SimpleNamespace(to_lc_human_message=Mock(return_value="human"))
    handlers_obj = _handlers(db=db, agent=agent, prompt_builder=prompt_builder)

    message = _message(text="chatgpt hello", user_id=1)
    update = SimpleNamespace(message=message)
    context = SimpleNamespace(bot=SimpleNamespace(send_chat_action=AsyncMock()))

    asyncio.run(handlers_obj.message_handler(update, context))

    message.reply_text.assert_awaited_once_with("hi there")


def _photo_message(caption="chatgpt look", user_id=1):
    message = _message(chat_type="private", user_id=user_id)
    photo_file = SimpleNamespace(
        download_as_bytearray=AsyncMock(return_value=bytearray(b"\xff\xd8jpeg")))
    message.photo = [SimpleNamespace(get_file=AsyncMock(return_value=photo_file))]
    message.caption = caption
    return message


def test_photo_handler_passes_post_success_that_calls_persist_image():
    agent = SimpleNamespace(run=AsyncMock(return_value="a cat"),
                            persist_image=AsyncMock())
    prompt_builder = SimpleNamespace(to_lc_human_message=Mock(return_value="human"))
    db = SimpleNamespace(add_message=Mock())
    handlers_obj = _handlers(db=db, agent=agent, prompt_builder=prompt_builder)

    captured = {}

    async def _fake_process(bot, message, **kwargs):
        # Simulate a successful turn: build the payload, then run the hook.
        await kwargs["build_payload"]()
        captured.update(kwargs)
        if kwargs.get("post_success") is not None:
            await kwargs["post_success"]()

    handlers_obj._processor.process = _fake_process

    message = _photo_message()
    update = SimpleNamespace(message=message)
    context = SimpleNamespace(bot=SimpleNamespace(send_chat_action=AsyncMock()))

    asyncio.run(handlers_obj.photo_handler(update, context))

    assert captured.get("post_success") is not None
    agent.persist_image.assert_awaited_once()
    call = agent.persist_image.await_args.kwargs
    id_used = prompt_builder.to_lc_human_message.call_args.kwargs["message_id"]
    assert call["image_message_id"] == id_used
    assert call["mime_type"] == "image/jpeg"
    assert call["image_data_url"].startswith("data:image/jpeg;base64,")
    assert call["caption"] == "chatgpt look"


def test_photo_handler_ignores_non_triggering_caption():
    agent = SimpleNamespace(persist_image=AsyncMock())
    handlers_obj = _handlers(agent=agent)
    message = _photo_message(caption="just a plain caption")
    update = SimpleNamespace(message=message)
    context = SimpleNamespace()

    asyncio.run(handlers_obj.photo_handler(update, context))

    agent.persist_image.assert_not_awaited()
    message.reply_text.assert_not_awaited()
