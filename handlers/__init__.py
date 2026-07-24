"""Backward-compatible facade: wires the DI-based handler classes to the
module-level callables bot.py registers with python-telegram-bot. See
message_handlers.py and command_handlers.py for the actual logic."""
import logging

from .authorization import is_authorized as _authz_is_authorized
from .authorization import is_main_authorized_user as _authz_is_main_authorized_user
from .command_handlers import CommandHandlers
from .command_handlers import error_handler as error_handler  # re-exported for bot.py/tests
from .handler_deps import HandlerDependencies
from .message_handlers import MessageHandlers
from .message_handlers import extract_keyword as extract_keyword  # re-exported for tests
from .message_handlers import extract_reply_data as extract_reply_data  # re-exported for tests
from .request_processor import RequestProcessor

logger = logging.getLogger(__name__)

# NOTE: deviates from the plan's brief for this task. The brief's Step 1 had
# this module do `from .authorization import is_authorized,
# is_main_authorized_user` as a raw re-export. That breaks
# tests/test_handlers_characterization.py, which (unchanged since Task 2)
# calls `handlers.is_authorized(user_id)` / `handlers.is_main_authorized_user
# (user_id)` with a single argument — the pre-refactor, module-global-based
# calling convention. authorization.py (Task 3) deliberately dropped module
# globals in favor of explicit `config`/`db` parameters, so the raw
# re-export has the wrong arity for that frozen test. Fix (coordinator-
# approved): keep one typed `_deps: HandlerDependencies | None` module
# global (not five loose globals) and give these two functions thin
# wrappers that read config/db off it, preserving the single-arg call
# signature the characterization tests rely on.
_deps: HandlerDependencies | None = None
_message_handlers: MessageHandlers | None = None
_command_handlers: CommandHandlers | None = None


def init_handlers(cfg, database, bot_agent, prompt_bldr, username=None):
    """Initialize handler dependencies."""
    global _deps, _message_handlers, _command_handlers
    _deps = HandlerDependencies(
        config=cfg, db=database, agent=bot_agent,
        prompt_builder=prompt_bldr, bot_username=username,
    )
    processor = RequestProcessor(_deps)
    _message_handlers = MessageHandlers(_deps, processor)
    _command_handlers = CommandHandlers(_deps)


def is_authorized(user_id: int) -> bool:
    """Check if user is authorized to use the bot."""
    assert _deps is not None, "init_handlers() must run before is_authorized()"
    return _authz_is_authorized(user_id, _deps.config, _deps.db)


def is_main_authorized_user(user_id: int) -> bool:
    """Check if user is the main authorized user (for admin commands)."""
    assert _deps is not None, "init_handlers() must run before is_main_authorized_user()"
    return _authz_is_main_authorized_user(user_id, _deps.config)


async def message_handler(update, context):
    assert _message_handlers is not None, "init_handlers() must run before message_handler()"
    return await _message_handlers.message_handler(update, context)


async def photo_handler(update, context):
    assert _message_handlers is not None, "init_handlers() must run before photo_handler()"
    return await _message_handlers.photo_handler(update, context)


async def clear_command(update, context):
    assert _command_handlers is not None, "init_handlers() must run before clear_command()"
    return await _command_handlers.clear_command(update, context)


async def stats_command(update, context):
    assert _command_handlers is not None, "init_handlers() must run before stats_command()"
    return await _command_handlers.stats_command(update, context)


async def grant_command(update, context):
    assert _command_handlers is not None, "init_handlers() must run before grant_command()"
    return await _command_handlers.grant_command(update, context)


async def revoke_command(update, context):
    assert _command_handlers is not None, "init_handlers() must run before revoke_command()"
    return await _command_handlers.revoke_command(update, context)


async def version_command(update, context):
    assert _command_handlers is not None, "init_handlers() must run before version_command()"
    return await _command_handlers.version_command(update, context)


async def allowlist_command(update, context):
    assert _command_handlers is not None, "init_handlers() must run before allowlist_command()"
    return await _command_handlers.allowlist_command(update, context)


async def personality_command(update, context):
    assert _command_handlers is not None, "init_handlers() must run before personality_command()"
    return await _command_handlers.personality_command(update, context)


async def model_command(update, context):
    assert _command_handlers is not None, "init_handlers() must run before model_command()"
    return await _command_handlers.model_command(update, context)


async def help_command(update, context):
    assert _command_handlers is not None, "init_handlers() must run before help_command()"
    return await _command_handlers.help_command(update, context)
