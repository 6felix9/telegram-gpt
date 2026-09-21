"""Add scheduled_prompts table for agent-created recurring prompts

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-21

"""
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE scheduled_prompts (
            id SERIAL PRIMARY KEY,
            chat_id TEXT NOT NULL,
            prompt TEXT NOT NULL,
            label TEXT NOT NULL,
            cron TEXT,
            next_run_at TIMESTAMPTZ NOT NULL,
            enabled BOOLEAN NOT NULL DEFAULT TRUE,
            created_by BIGINT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            last_run_at TIMESTAMPTZ,
            consecutive_failures INTEGER NOT NULL DEFAULT 0
        )
    """)
    op.execute(
        "CREATE INDEX idx_scheduled_prompts_due "
        "ON scheduled_prompts (enabled, next_run_at)"
    )
    op.execute(
        "CREATE INDEX idx_scheduled_prompts_chat ON scheduled_prompts (chat_id)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS scheduled_prompts")
