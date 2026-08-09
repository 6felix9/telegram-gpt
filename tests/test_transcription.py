"""transcribe(): OpenAI audio endpoint wrapper, fail-open by contract."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

import transcription
from transcription import transcribe


@pytest.fixture(autouse=True)
def _reset_client_singleton():
    """The lazy module-level client must not leak state across tests."""
    transcription._client = None
    yield
    transcription._client = None


class _Cfg:
    TRANSCRIPTION_MODEL = "gpt-transcribe"
    OPENAI_API_KEY = "sk-test"
    MODEL_TIMEOUT = 60


def _client(create):
    """Minimal stand-in for AsyncOpenAI: only .audio.transcriptions.create."""
    return SimpleNamespace(
        audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create))
    )


def test_returns_stripped_transcript():
    create = AsyncMock(return_value=SimpleNamespace(text="  hi i am jack  "))
    result = asyncio.run(transcribe(b"ogg", "voice.ogg", _Cfg, client=_client(create)))
    assert result == "hi i am jack"


def test_passes_configured_model_and_file():
    create = AsyncMock(return_value=SimpleNamespace(text="ok"))
    asyncio.run(transcribe(b"ogg-bytes", "voice.ogg", _Cfg, client=_client(create)))
    kwargs = create.await_args.kwargs
    assert kwargs["model"] == "gpt-transcribe"
    assert kwargs["file"] == ("voice.ogg", b"ogg-bytes")


def test_api_error_returns_none():
    create = AsyncMock(side_effect=RuntimeError("api exploded"))
    result = asyncio.run(transcribe(b"ogg", "voice.ogg", _Cfg, client=_client(create)))
    assert result is None


def test_empty_transcript_returns_none():
    create = AsyncMock(return_value=SimpleNamespace(text="   "))
    result = asyncio.run(transcribe(b"ogg", "voice.ogg", _Cfg, client=_client(create)))
    assert result is None


def test_missing_text_attribute_returns_none():
    create = AsyncMock(return_value=SimpleNamespace(text=None))
    result = asyncio.run(transcribe(b"ogg", "voice.ogg", _Cfg, client=_client(create)))
    assert result is None


def test_default_client_constructed_with_configured_timeout():
    with patch("transcription.AsyncOpenAI") as mock_cls:
        mock_cls.return_value = _client(AsyncMock(return_value=SimpleNamespace(text="ok")))
        asyncio.run(transcribe(b"ogg", "voice.ogg", _Cfg))
        mock_cls.assert_called_once_with(api_key="sk-test", timeout=60)


def test_default_client_is_a_reused_singleton():
    with patch("transcription.AsyncOpenAI") as mock_cls:
        mock_cls.return_value = _client(AsyncMock(return_value=SimpleNamespace(text="ok")))
        asyncio.run(transcribe(b"ogg", "voice.ogg", _Cfg))
        asyncio.run(transcribe(b"ogg", "voice.ogg", _Cfg))
        assert mock_cls.call_count == 1


def test_injected_client_never_touches_the_singleton():
    create = AsyncMock(return_value=SimpleNamespace(text="ok"))
    with patch("transcription.AsyncOpenAI") as mock_cls:
        asyncio.run(transcribe(b"ogg", "voice.ogg", _Cfg, client=_client(create)))
        mock_cls.assert_not_called()
    assert transcription._client is None
