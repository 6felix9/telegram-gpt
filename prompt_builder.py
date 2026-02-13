"""Centralized prompt and message construction for model requests."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable
from zoneinfo import ZoneInfo

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
            if active_personality == "normal":
                logger.debug("PromptBuilder: active personality is normal, using default group prompt")
                return None

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
    def _apply_group_sender_prefix(role: str, text: str, sender_name: str) -> str:
        """Prefix group user messages with sender name when missing."""
        if role != "user":
            return text
        if text.startswith("["):
            return text
        return f"[{sender_name}]: {text}"

    def format_messages(self, messages: list[dict], is_group: bool) -> list[dict]:
        """Format messages for Responses API and group context semantics."""
        formatted_messages: list[dict] = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            sender_name = msg.get("sender_name", "Unknown")

            if isinstance(content, str):
                formatted_content = content
                if is_group:
                    formatted_content = self._apply_group_sender_prefix(role, formatted_content, sender_name)

                formatted_messages.append({"role": role, "content": formatted_content})
                continue

            if isinstance(content, list):
                updated_content = []
                for part in content:
                    part_type = part.get("type")
                    if part_type in {"text", "input_text"}:
                        text = part.get("text", "")
                        if is_group:
                            text = self._apply_group_sender_prefix(role, text, sender_name)
                        updated_content.append({"type": "input_text", "text": text})
                    elif part_type in {"image_url", "input_image"}:
                        if part_type == "image_url":
                            image_url_obj = part.get("image_url", {})
                            url = image_url_obj.get("url", "") if isinstance(image_url_obj, dict) else str(image_url_obj)
                        else:
                            url = part.get("image_url", "")
                        updated_content.append({"type": "input_image", "image_url": url})
                    else:
                        logger.debug("PromptBuilder: skipping unsupported part type '%s'", part_type)

                formatted_messages.append({"role": role, "content": updated_content})
                continue

            logger.warning("PromptBuilder: unsupported content type %s for role %s", type(content), role)

        return formatted_messages

