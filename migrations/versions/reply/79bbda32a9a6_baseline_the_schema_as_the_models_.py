"""Baseline: the schema as the models declared it

Revision ID: 79bbda32a9a6
Revises:
Create Date: 2026-11-07 04:52:04.040295
"""

from alembic import op
import sqlalchemy as sa

revision = "79bbda32a9a6"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Applies this revision."""
    op.create_table(
        "research",
        sa.Column("thread_id", sa.Integer(), nullable=False),
        sa.Column("owner_id", sa.Integer(), nullable=False),
        sa.Column("channel_id", sa.Integer(), nullable=False),
        sa.Column("guild_id", sa.Integer(), nullable=True),
        sa.Column("source_message_id", sa.Integer(), nullable=False),
        sa.Column("agent", sa.String(length=64), nullable=False),
        sa.Column("interaction_id", sa.String(length=256), nullable=True),
        sa.Column("brief", sa.String(length=16384), nullable=False),
        sa.Column("phase", sa.String(length=16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("thread_id"),
    )
    op.create_table(
        "ask_turn",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("channel_id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("message_id", sa.Integer(), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table(table_name="ask_turn") as batch_op:
        batch_op.create_index(
            index_name="ix_ask_turn_conversation",
            columns=["channel_id", "user_id", "id"],
            unique=False,
        )

    op.create_table(
        "pending_expansion",
        sa.Column("message_id", sa.Integer(), nullable=False),
        sa.Column("channel_id", sa.Integer(), nullable=False),
        sa.Column("source_message_id", sa.Integer(), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("message_id"),
    )
    with op.batch_alter_table(table_name="pending_expansion") as batch_op:
        batch_op.create_index(
            index_name="ix_pending_expansion_source",
            columns=["source", "created_at"],
            unique=False,
        )

    op.create_table(
        "memory_job",
        sa.Column("scope", sa.String(length=128), nullable=False),
        sa.Column("flavor", sa.String(length=16), nullable=False),
        sa.Column("subject", sa.String(length=128), nullable=False),
        sa.Column("transcript", sa.Text(), nullable=True),
        sa.Column("identity", sa.String(length=256), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("token", sa.Integer(), nullable=False),
        sa.Column("last_error", sa.String(length=512), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("scope"),
    )
    op.create_table(
        "memory_token_clock",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("high_watermark", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
