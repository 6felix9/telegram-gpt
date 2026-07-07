"""Centralized prompt and message construction for model requests."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable
from zoneinfo import ZoneInfo

from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Build system prompts and normalize outbound message payloads."""

    def __init__(
        self,
        default_private_prompt: str,
        default_group_prompt: str,
        get_active_personality: Callable[[], str] | None = None,
        get_personality_prompt: Callable[[str], str | None] | None = None,
        timezone_name: str = "Asia/Singapore",
        fallback_timezone_name: str = "UTC",
    ):
        self.default_private_prompt = default_private_prompt
        self.default_group_prompt = default_group_prompt
        self._get_active_personality = get_active_personality
        self._get_personality_prompt = get_personality_prompt
        self.timezone_name = timezone_name
        self.fallback_timezone_name = fallback_timezone_name

    def _current_time_iso(self) -> str:
        """Return ISO timestamp using configured timezone with fallback."""
        try:
            return datetime.now(ZoneInfo(self.timezone_name)).isoformat(timespec="seconds")
        except Exception as e:
            logger.warning(
                "Failed to use timezone %s, falling back to %s: %s",
                self.timezone_name,
                self.fallback_timezone_name,
                e,
            )
            return datetime.now(ZoneInfo(self.fallback_timezone_name)).isoformat(timespec="seconds")

    def _resolve_group_personality_prompt(self) -> str | None:
        """Resolve active group personality prompt from storage if available."""
        if not self._get_active_personality or not self._get_personality_prompt:
            logger.debug("PromptBuilder: no personality resolvers configured")
            return None

        try:
            active_personality = self._get_active_personality()
            custom_prompt = self._get_personality_prompt(active_personality)
            if custom_prompt:
                logger.debug(
                    "PromptBuilder: using custom group personality '%s'",
                    active_personality,
                )
                return custom_prompt

            logger.warning(
                "PromptBuilder: personality '%s' not found, using default group prompt",
                active_personality,
            )
            return None
        except Exception as e:
            logger.error("PromptBuilder: failed to resolve group personality: %s", e, exc_info=True)
            return None

    def build_system_prompt(
        self,
        is_group: bool,
        custom_system_prompt: str | None = None,
        reply_context: tuple[str, str] | None = None,
    ) -> str:
        """
        Build final system prompt with time context and optional reply context.

        Args:
            is_group: Whether this is a group chat
            custom_system_prompt: Optional custom system prompt override
            reply_context: Optional tuple of (sender_name, content) being replied to

        Returns:
            Complete system prompt string
        """
        if custom_system_prompt:
            prompt_body = custom_system_prompt
            logger.debug("PromptBuilder: using explicit custom system prompt override")
        elif is_group:
            prompt_body = self._resolve_group_personality_prompt() or self.default_group_prompt
            if prompt_body == self.default_group_prompt:
                logger.debug("PromptBuilder: using default group prompt")
        else:
            prompt_body = self.default_private_prompt
            logger.debug("PromptBuilder: using default private prompt")

        now_iso = self._current_time_iso()
        system_parts = [f'Current date/time: {now_iso}\n\n"{prompt_body}"']

        # Add reply context if present
        if reply_context:
            sender_name, content = reply_context
            reply_note = f'\nNote: The user is replying to a previous message from {sender_name}: "{content}"'
            system_parts.append(reply_note)
            logger.debug("PromptBuilder: added reply context to system prompt")

        return "".join(system_parts)

    @staticmethod
    def _group_prefix(text: str, sender_name: str) -> str:
        if text.startswith("["):
            return text
        return f"[{sender_name}]: {text}"

    def to_lc_human_message(
        self,
        text: str | None = None,
        is_group: bool = False,
        sender_name: str = "Unknown",
        image_data_url: str | None = None,
    ) -> HumanMessage:
        """Build a LangChain HumanMessage from an incoming Telegram message."""
        body = text or ""
        if is_group and body:
            body = self._group_prefix(body, sender_name)

        if image_data_url:
            return HumanMessage(content=[
                {"type": "text", "text": body or "What's in this image?"},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ])
        return HumanMessage(content=body)

