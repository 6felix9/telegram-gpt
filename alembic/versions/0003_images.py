"""Add images table for durable image persistence

Revision ID: 0003
Revises: 0002
Create Date: 2026-07-23

"""
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE images (
            id BIGSERIAL PRIMARY KEY,
            chat_id TEXT NOT NULL,
            message_id BIGINT,
            mime_type TEXT NOT NULL,
            caption TEXT,
            summary TEXT NOT NULL,
            image_bytes BYTEA NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    op.execute("CREATE INDEX idx_images_chat_id ON images(chat_id, id)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_images_chat_id")
    op.execute("DROP TABLE IF EXISTS images")
