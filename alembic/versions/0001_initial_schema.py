"""Initial schema: messages, granted_users, personality, active_personality, active_model

Revision ID: 0001
Revises:
Create Date: 2026-07-05

"""
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE messages (
            id SERIAL PRIMARY KEY,
            chat_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            user_id BIGINT,
            message_id BIGINT,
            token_count INTEGER DEFAULT 0,
            sender_name TEXT,
            sender_username TEXT,
            is_group_chat BOOLEAN DEFAULT FALSE
        )
    """)
    op.execute("""
        CREATE INDEX idx_chat_timestamp
        ON messages(chat_id, timestamp DESC)
    """)
    op.execute("""
        CREATE TABLE granted_users (
            user_id TEXT PRIMARY KEY,
            granted_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            first_name TEXT,
            username TEXT
        )
    """)
    op.execute("""
        CREATE TABLE personality (
            personality TEXT PRIMARY KEY,
            prompt TEXT NOT NULL
        )
    """)
    op.execute("""
        CREATE TABLE active_personality (
            id INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
            personality TEXT NOT NULL DEFAULT 'default',
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    op.execute("""
        INSERT INTO active_personality (id, personality, updated_at)
        VALUES (1, 'default', CURRENT_TIMESTAMP)
    """)
    op.execute("""
        CREATE TABLE active_model (
            id INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
            model TEXT NOT NULL DEFAULT 'gpt-4o-mini',
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS active_model")
    op.execute("DROP TABLE IF EXISTS active_personality")
    op.execute("DROP TABLE IF EXISTS personality")
    op.execute("DROP TABLE IF EXISTS granted_users")
    op.execute("DROP INDEX IF EXISTS idx_chat_timestamp")
    op.execute("DROP TABLE IF EXISTS messages")
