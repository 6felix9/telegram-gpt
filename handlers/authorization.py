"""Access control checks shared by message and command handlers."""


def is_authorized(user_id: int, config, db) -> bool:
    """Check if user is authorized to use the bot."""
    if str(user_id) == config.AUTHORIZED_USER_ID:
        return True
    return db.is_user_granted(user_id)


def is_main_authorized_user(user_id: int, config) -> bool:
    """Check if user is the main authorized user (for admin commands)."""
    return str(user_id) == config.AUTHORIZED_USER_ID
