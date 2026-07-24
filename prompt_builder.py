"""Centralized prompt and message construction for model requests."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable
from zoneinfo import ZoneInfo

from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

# One-line usage policy per tool. The full parameter docs already ship in each
# tool's schema on every request, so this only carries the "when to reach for
# it" guidance the schema can't express. Names are matched against the tools
# actually bound to the agent — never hardcoded into the prompt text.
TOOL_USAGE = {
    "web_search": (
        "current events, facts after your training cutoff, anything you would "
        "otherwise guess at. Not for general knowledge you already have."
    ),
    "fetch_url": "read one specific page, normally a URL from web_search results.",
    "get_image": (
        "view an image shared earlier as [image #N], when the marker's text "
        "description is not enough."
    ),
}

MARKDOWN_CONVENTION = (
    "Do not use Markdown formatting (no **bold**, *italics*, headers, or bullet "
    "asterisks); Telegram shows it literally."
)
GROUP_PREFIX_CONVENTION = (
    'Messages arrive as "[Name]: content". Reply naturally; never write that '
    "prefix yourself."
)
IMAGE_MARKER_CONVENTION = (
    '"[image #N] caption — description" stands in for an image shared earlier.'
)


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
        """Resolve active group personality prompt from storage if available.

        The returned prompt replaces the persona only; the Tools and Conventions
        sections are still appended after it. See the comment above
        agent.SYSTEM_PROMPT for what a personality row should and shouldn't say.
        """
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

    @staticmethod
    def _tools_section(tool_names: list[str]) -> str:
        """Render the ## Tools section from the tools actually bound to the agent.

        Unknown names are listed bare — their own schema still describes them,
        so a newly added tool degrades to "mentioned" rather than "invisible".
        """
        lines = [
            "## Tools",
            "Use a tool when it gives a better answer than replying from memory; "
            "otherwise just answer.",
        ]
        for name in tool_names:
            usage = TOOL_USAGE.get(name)
            lines.append(f"- {name} — {usage}" if usage else f"- {name}")
        return "\n".join(lines)

    def _conventions_section(self, is_group: bool) -> str:
        """Render the ## Conventions section: transport rules a persona can't override."""
        lines = ["## Conventions", f"- {MARKDOWN_CONVENTION}"]
        if is_group:
            lines.append(f"- {GROUP_PREFIX_CONVENTION}")
        lines.append(f"- {IMAGE_MARKER_CONVENTION}")
        return "\n".join(lines)

    def build_system_prompt(
        self,
        is_group: bool,
        custom_system_prompt: str | None = None,
        tool_names: list[str] | None = None,
    ) -> str:
        """
        Build the static system prompt: persona, then tools, then conventions.

        Deliberately free of per-call data (timestamp, reply context) so the whole
        string is byte-identical between requests and can hit provider prefix
        caches — see build_context_message() for the volatile half. Conventions
        come last so a custom personality can't countermand transport rules.

        Args:
            is_group: Whether this is a group chat
            custom_system_prompt: Optional custom system prompt override
            tool_names: Names of the tools bound to the agent, in bound order

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

        sections = [prompt_body]
        if tool_names:
            sections.append(self._tools_section(tool_names))
        sections.append(self._conventions_section(is_group))

        return "\n\n".join(sections)

    def build_context_message(
        self,
        reply_context: tuple[str, str] | None = None,
    ) -> SystemMessage:
        """Build the per-call context block appended after the conversation history.

        Kept out of the system prompt so that prompt stays cacheable. A
        SystemMessage rather than a HumanMessage so the model doesn't read it as
        the user speaking.

        Args:
            reply_context: Optional tuple of (sender_name, content) being replied to
        """
        lines = ["## Current context", f"Date/time: {self._current_time_iso()}"]
        if reply_context:
            sender_name, content = reply_context
            lines.append(
                f'Replying to a previous message from {sender_name}: "{content}"'
            )
            logger.debug("PromptBuilder: added reply context to context message")
        return SystemMessage(content="\n".join(lines))

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
        message_id: str | None = None,
    ) -> HumanMessage:
        """Build a LangChain HumanMessage from an incoming Telegram message.

        When message_id is given, it is set as the message's stable id so the
        message can later be rewritten in place in the checkpoint."""
        body = text or ""
        if is_group and body:
            body = self._group_prefix(body, sender_name)

        extra = {"id": message_id} if message_id is not None else {}
        if image_data_url:
            return HumanMessage(content=[
                {"type": "text", "text": body or "What's in this image?"},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ], **extra)
        return HumanMessage(content=body, **extra)

