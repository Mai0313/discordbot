"""Baseline: the schema as the models declared it

Revision ID: 17c51ec8c25f
Revises:
Create Date: 2026-11-07 04:52:04.048150
"""

from alembic import op
import sqlalchemy as sa

revision = "17c51ec8c25f"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Applies this revision."""
    op.create_table(
        "messages",
        sa.Column("id", sa.Integer(), nullable=True),
        sa.Column("discord_message_id", sa.Text(), nullable=True),
        sa.Column("source_type", sa.Text(), nullable=False),
        sa.Column("author", sa.Text(), nullable=True),
        sa.Column("author_id", sa.Text(), nullable=True),
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column("created_at", sa.Text(), nullable=True),
        sa.Column("channel_name", sa.Text(), nullable=True),
        sa.Column("channel_id", sa.Text(), nullable=True),
        sa.Column("attachments", sa.Text(), nullable=True),
        sa.Column("stickers", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sqlite_autoincrement=True,
    )
    with op.batch_alter_table(table_name="messages") as batch_op:
        batch_op.create_index(
            index_name="ix_messages_author_id_created_at",
            columns=["author_id", "created_at"],
            unique=False,
        )
        batch_op.create_index(
            index_name="ix_messages_channel_id_created_at",
            columns=["channel_id", "created_at"],
            unique=False,
        )
        batch_op.create_index(
            index_name="ix_messages_created_at", columns=["created_at"], unique=False
        )
        batch_op.create_index(
            index_name="ix_messages_discord_message_id",
            columns=["discord_message_id"],
            unique=True,
            sqlite_where=sa.text(text="discord_message_id IS NOT NULL"),
        )
