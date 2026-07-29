"""Speech-to-text for Telegram voice notes.

Owns the one call in this codebase to OpenAI's audio transcription endpoint.
Deliberately outside agent.py: this is not a LangChain chat model, so it has no
place in MODEL_PROVIDERS or init_chat_model().
"""
from __future__ import annotations

import logging

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


async def transcribe(
    audio_bytes: bytes, filename: str, config, client=None
) -> str | None:
    """Transcribe audio to text, or return None if that is not possible.

    Fail-open by contract: callers store a bare marker instead of surfacing an
    error, so this never raises. `client` is injected only by tests.

    Args:
        audio_bytes: Raw audio file contents.
        filename: Name with extension (e.g. "voice.ogg") — the API infers the
            container format from it.
        config: Settings object supplying TRANSCRIPTION_MODEL and OPENAI_API_KEY.
        client: Optional AsyncOpenAI-compatible client.

    Returns:
        The stripped transcript, or None on failure or an empty result.
    """
    try:
        client = client or AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        response = await client.audio.transcriptions.create(
            model=config.TRANSCRIPTION_MODEL,
            file=(filename, audio_bytes),
        )
        text = (getattr(response, "text", None) or "").strip()
        if not text:
            logger.info("Empty transcript for %s", filename)
            return None
        return text
    except Exception:
        logger.exception("Transcription failed for %s", filename)
        return None
