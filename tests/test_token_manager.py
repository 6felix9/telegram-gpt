"""Tests for TokenManager context trimming."""
import pytest

from token_manager import TokenManager


@pytest.fixture
def manager():
    return TokenManager("gpt-4o-mini", max_tokens=100)


def test_trim_empty_list(manager):
    assert manager.trim_to_fit([]) == []


def test_trim_single_message_always_kept(manager):
    messages = [{"role": "user", "content": "hello"}]
    result = manager.trim_to_fit(messages, reserve_tokens=50)
    assert result == messages


def test_trim_drops_oldest_when_over_budget():
    manager = TokenManager("gpt-4o-mini", max_tokens=40)
    messages = [
        {"role": "user", "content": "a" * 100},
        {"role": "assistant", "content": "b" * 100},
        {"role": "user", "content": "c" * 100},
        {"role": "assistant", "content": "d" * 100},
        {"role": "user", "content": "latest prompt"},
    ]
    result = manager.trim_to_fit(messages, reserve_tokens=10)
    assert result[-1] == messages[-1]
    assert len(result) < len(messages)
    assert result[0] != messages[0]


def test_trim_last_message_always_present(manager):
    messages = [
        {"role": "user", "content": "x" * 200},
        {"role": "user", "content": "must keep this"},
    ]
    result = manager.trim_to_fit(messages, reserve_tokens=10)
    assert result[-1]["content"] == "must keep this"
