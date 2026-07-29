"""transcribe(): OpenAI audio endpoint wrapper, fail-open by contract."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from transcription import transcribe


class _Cfg:
    TRANSCRIPTION_MODEL = "gpt-transcribe"
    OPENAI_API_KEY = "sk-test"


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
