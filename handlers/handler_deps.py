"""Typed dependency bundle shared by Telegram-facing handler classes,
replacing the module-level globals handlers.py used to expose."""
from dataclasses import dataclass
from typing import Any


@dataclass
class HandlerDependencies:
    config: Any
    db: Any
    agent: Any
    prompt_builder: Any
    bot_username: str | None = None
