"""Tests for extract_keyword activation logic."""
import pytest

from handlers import extract_keyword


@pytest.mark.parametrize(
    "text, bot_username, expected_has_keyword, expected_prompt",
    [
        ("", None, False, ""),
        ("   ", None, False, ""),
        ("hello world", None, False, "hello world"),
        ("chatgpt what is 2+2", None, True, "what is 2+2"),
        ("ChatGPT help me", None, True, "help me"),
        ("chatgpt123", None, True, "chatgpt123"),
        ("@MyBot hello", "MyBot", True, "hello"),
        ("@mybot hello", "MyBot", True, "hello"),
        ("chatgpt @MyBot summarize this", "MyBot", True, "summarize this"),
        ("chatgpt", None, True, ""),
    ],
)
def test_extract_keyword(text, bot_username, expected_has_keyword, expected_prompt):
    has_keyword, prompt = extract_keyword(text, bot_username)
    assert has_keyword is expected_has_keyword
    assert prompt == expected_prompt
