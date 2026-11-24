"""Create guild_settings table

Revision ID: e4b7d2c19a05
Revises:
Create Date: 2026-11-25 03:08:00.000000
"""

from alembic import op
import sqlalchemy as sa

revision = "e4b7d2c19a05"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Applies this revision."""
    op.create_table(
        "guild_settings",
        sa.Column("guild_id", sa.Integer(), nullable=False),
        sa.Column("auto_expand_twitter", sa.Boolean(), nullable=True),
        sa.Column("auto_expand_threads", sa.Boolean(), nullable=True),
        sa.Column("auto_expand_facebook", sa.Boolean(), nullable=True),
        sa.Column("auto_expand_instagram", sa.Boolean(), nullable=True),
        sa.Column("auto_expand_douyin", sa.Boolean(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("guild_id"),
    )
