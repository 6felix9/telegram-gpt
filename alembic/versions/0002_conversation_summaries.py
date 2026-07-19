"""Add conversation_summaries audit table

Revision ID: 0002
Revises: 0001
Create Date: 2026-07-20

"""
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE conversation_summaries (
            id SERIAL PRIMARY KEY,
            chat_id TEXT NOT NULL,
            summary_text TEXT NOT NULL,
            summary_model TEXT NOT NULL,
            before_message_count INTEGER NOT NULL,
            after_message_count INTEGER NOT NULL,
            before_tokens INTEGER NOT NULL,
            after_tokens INTEGER NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    op.execute("""
        CREATE INDEX idx_summary_chat_created
        ON conversation_summaries(chat_id, created_at DESC)
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_summary_chat_created")
    op.execute("DROP TABLE IF EXISTS conversation_summaries")
