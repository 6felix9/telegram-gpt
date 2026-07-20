"""Thin facade preserving the public Database API, backed by focused
repositories that all share one ConnectionManager and one TTLCache."""
import logging

from access_repository import AccessRepository
from cache import TTLCache
from db_connection import ConnectionManager
from message_repository import MessageRepository
from settings_repository import SettingsRepository
from summary_audit_repository import SummaryAuditRepository

logger = logging.getLogger(__name__)


class Database:
    """Facade over MessageRepository, AccessRepository, SettingsRepository,
    and SummaryAuditRepository, preserving the pre-split public API."""

    def __init__(self, db_url: str):
        self._conn = ConnectionManager(db_url)
        self._cache = TTLCache(default_ttl=60.0)
        self._messages = MessageRepository(self._conn)
        self._access = AccessRepository(self._conn, self._cache)
        self._settings = SettingsRepository(self._conn, self._cache)
        self._summaries = SummaryAuditRepository(self._conn)

    def close(self):
        self._conn.close()

    # --- messages ------------------------------------------------------
    def add_message(self, *args, **kwargs) -> int:
        return self._messages.add_message(*args, **kwargs)

    def get_stats(self, chat_id: str) -> dict:
        return self._messages.get_stats(chat_id)

    def cleanup_old_group_messages(self, chat_id: str, keep_recent: int = 100):
        return self._messages.cleanup_old_group_messages(chat_id, keep_recent)

    # --- access ----------------------------------------------------------
    def grant_access(self, user_id: int, first_name: str = None, username: str = None) -> bool:
        return self._access.grant_access(user_id, first_name=first_name, username=username)

    def revoke_access(self, user_id: int) -> bool:
        return self._access.revoke_access(user_id)

    def is_user_granted(self, user_id: int) -> bool:
        return self._access.is_user_granted(user_id)

    def get_granted_users(self) -> list:
        return self._access.get_granted_users()

    # --- settings ----------------------------------------------------------
    def get_personality_prompt(self, personality: str) -> str | None:
        return self._settings.get_personality_prompt(personality)

    def get_active_personality(self) -> str:
        return self._settings.get_active_personality()

    def set_active_personality(self, personality: str) -> None:
        return self._settings.set_active_personality(personality)

    def personality_exists(self, personality: str) -> bool:
        return self._settings.personality_exists(personality)

    def list_personalities(self) -> list[tuple[str, str]]:
        return self._settings.list_personalities()

    def init_active_model(self, default_model: str) -> None:
        return self._settings.init_active_model(default_model)

    def get_active_model(self) -> str:
        return self._settings.get_active_model()

    def set_active_model(self, model: str) -> None:
        return self._settings.set_active_model(model)

    # --- summary audit ------------------------------------------------------
    def record_conversation_summary(self, *args, **kwargs) -> int:
        return self._summaries.record_conversation_summary(*args, **kwargs)
