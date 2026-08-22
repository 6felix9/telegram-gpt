"""Add open_access table for the /openbot allowlist-bypass toggle

Revision ID: 0004
Revises: 0003
Create Date: 2026-08-22

"""
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE open_access (
            id INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
            enabled BOOLEAN NOT NULL DEFAULT FALSE,
            expires_at TIMESTAMP,
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS open_access")
