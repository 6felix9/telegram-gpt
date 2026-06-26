"""Tests for PromptBuilder message formatting."""
import pytest

from prompt_builder import PromptBuilder


@pytest.fixture
def builder():
    return PromptBuilder(default_private_prompt="private", default_group_prompt="group")


def test_private_chat_user_content_unchanged(builder):
    messages = [{"role": "user", "content": "hello", "sender_name": "Alice"}]
    result = builder.format_messages(messages, is_group=False)
    assert result == [{"role": "user", "content": "hello"}]


def test_group_chat_adds_sender_prefix(builder):
    messages = [{"role": "user", "content": "hello", "sender_name": "Alice"}]
    result = builder.format_messages(messages, is_group=True)
    assert result == [{"role": "user", "content": "[Alice]: hello"}]


def test_group_chat_skips_already_prefixed_content(builder):
    messages = [{"role": "user", "content": "[image] caption", "sender_name": "Alice"}]
    result = builder.format_messages(messages, is_group=True)
    assert result == [{"role": "user", "content": "[image] caption"}]


def test_assistant_messages_not_prefixed_in_group(builder):
    messages = [{"role": "assistant", "content": "reply", "sender_name": "Bot"}]
    result = builder.format_messages(messages, is_group=True)
    assert result == [{"role": "assistant", "content": "reply"}]


def test_multimodal_responses_api_format(builder):
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "describe this"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc"}},
        ],
        "sender_name": "Alice",
    }]
    result = builder.format_messages(messages, is_group=False, api_format="responses")
    assert result[0]["content"] == [
        {"type": "input_text", "text": "describe this"},
        {"type": "input_image", "image_url": "data:image/jpeg;base64,abc"},
    ]


def test_multimodal_chat_completions_format(builder):
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "describe this"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc"}},
        ],
        "sender_name": "Alice",
    }]
    result = builder.format_messages(messages, is_group=False, api_format="chat_completions")
    assert result[0]["content"] == [
        {"type": "text", "text": "describe this"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc"}},
    ]


def test_multimodal_group_prefix_on_text_part(builder):
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "describe this"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc"}},
        ],
        "sender_name": "Alice",
    }]
    result = builder.format_messages(messages, is_group=True, api_format="responses")
    assert result[0]["content"][0] == {"type": "input_text", "text": "[Alice]: describe this"}
