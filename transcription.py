"""Speech-to-text for Telegram voice notes.

Owns the one call in this codebase to OpenAI's audio transcription endpoint.
Deliberately outside agent.py: this is not a LangChain chat model, so it has no
place in MODEL_PROVIDERS or init_chat_model().
"""
from __future__ import annotations

import logging

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


_client = None


def _default_client(config):
    """Lazily build and reuse one AsyncOpenAI client (and its connection pool)."""
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=config.OPENAI_API_KEY, timeout=config.MODEL_TIMEOUT)
    return _client


async def transcribe(
    audio_bytes: bytes, filename: str, config, client=None, chat_id: str | None = None
) -> str | None:
    """Transcribe audio to text, or return None if that is not possible.

    Fail-open by contract: callers store a bare marker instead of surfacing an
    error, so this never raises. `client` is injected only by tests; in
    production a single lazily-built client is reused across calls.

    Args:
        audio_bytes: Raw audio file contents.
        filename: Name with extension (e.g. "voice.ogg") — the API infers the
            container format from it.
        config: Settings object supplying TRANSCRIPTION_MODEL, OPENAI_API_KEY,
            and MODEL_TIMEOUT.
        client: Optional AsyncOpenAI-compatible client.
        chat_id: Optional chat id, included in failure logs for context.

    Returns:
        The stripped transcript, or None on failure or an empty result.
    """
    try:
        client = client or _default_client(config)
        response = await client.audio.transcriptions.create(
            model=config.TRANSCRIPTION_MODEL,
            file=(filename, audio_bytes),
        )
        text = (getattr(response, "text", None) or "").strip()
        if not text:
            logger.info("Empty transcript for chat %s, %s", chat_id, filename)
            return None
        return text
    except Exception:
        logger.warning(
            "Transcription failed for chat %s, %s (model=%s)",
            chat_id, filename, getattr(config, "TRANSCRIPTION_MODEL", "?"),
            exc_info=True,
        )
        return None
