"""authorization.py: pure access-control checks, no module globals."""
from types import SimpleNamespace
from unittest.mock import Mock

from handlers import authorization


class _Cfg:
    AUTHORIZED_USER_ID = "1"


def test_is_authorized_main_user_true():
    assert authorization.is_authorized(1, _Cfg, db=SimpleNamespace()) is True


def test_is_authorized_delegates_to_db_for_other_users():
    db = SimpleNamespace(is_user_granted=Mock(return_value=True))
    assert authorization.is_authorized(42, _Cfg, db) is True
    db.is_user_granted.assert_called_once_with(42)


def test_is_authorized_false_when_not_granted():
    db = SimpleNamespace(is_user_granted=Mock(return_value=False))
    assert authorization.is_authorized(42, _Cfg, db) is False


def test_is_main_authorized_user():
    assert authorization.is_main_authorized_user(1, _Cfg) is True
    assert authorization.is_main_authorized_user(2, _Cfg) is False
