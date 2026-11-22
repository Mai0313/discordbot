"""Create pending_public_message table

Revision ID: 1b342af15323
Revises:
Create Date: 2026-11-22 18:12:57.163694
"""

from alembic import op
import sqlalchemy as sa

revision = "1b342af15323"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Applies this revision."""
    op.create_table(
        "pending_public_message",
        sa.Column("message_id", sa.Integer(), nullable=False),
        sa.Column("channel_id", sa.Integer(), nullable=False),
        sa.Column("guild_name", sa.Text(), nullable=True),
        sa.Column("channel_name", sa.Text(), nullable=True),
        sa.Column("user_name", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.Text(),
            server_default=sa.text(text="(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("message_id"),
    )
