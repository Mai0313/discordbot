"""Baseline: the schema as the models declared it

Revision ID: 663df6fa7d1b
Revises:
Create Date: 2026-11-07 04:52:04.029966
"""

from alembic import op
import sqlalchemy as sa

from discordbot.utils.stored_integer import StoredInteger

revision = "663df6fa7d1b"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Applies this revision."""
    op.create_table(
        "blackjack_round_result",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("round_id", sa.String(length=36), nullable=False),
        sa.Column("channel_id", sa.Integer(), nullable=False),
        sa.Column("guild_id", sa.Integer(), nullable=False),
        sa.Column("message_id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("user_name", sa.String(length=128), nullable=False),
        sa.Column("is_bot", sa.Boolean(), nullable=False),
        sa.Column("is_vip", sa.Boolean(), nullable=False),
        sa.Column("bet", StoredInteger(), nullable=False),
        sa.Column("outcome", sa.String(length=32), nullable=False),
        sa.Column("delta", StoredInteger(), nullable=False),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table(table_name="blackjack_round_result") as batch_op:
        batch_op.create_index(
            index_name="ix_blackjack_round_result_round", columns=["round_id"], unique=False
        )
        batch_op.create_index(
            index_name="ix_blackjack_round_result_user_created",
            columns=["user_id", "created_at"],
            unique=False,
        )

    op.create_table(
        "pending_game_message",
        sa.Column("message_id", sa.Integer(), nullable=True),
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
