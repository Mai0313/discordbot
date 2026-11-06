"""Baseline: the schema as the models declared it, plus the singleton rows the ledger assumes.

Revision ID: 9e1621821fad
Revises:
Create Date: 2026-11-07 04:52:04.019767
"""

from alembic import op
import sqlalchemy as sa

from discordbot.utils.timezone import database_now
from discordbot.utils.stored_integer import StoredInteger

revision = "9e1621821fad"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Applies this revision."""
    op.create_table(
        "casino_account",
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("day_started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("daily_loss", StoredInteger(), nullable=False),
        sa.Column("daily_win", StoredInteger(), nullable=False),
        sa.Column("daily_net", StoredInteger(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("user_id"),
    )
    with op.batch_alter_table(table_name="casino_account") as batch_op:
        batch_op.create_index(
            index_name="ix_casino_account_day_loss",
            columns=["day_started_at", "daily_loss"],
            unique=False,
        )

    op.create_table(
        "casino_ledger",
        sa.Column("ledger_id", sa.String(length=32), nullable=False),
        sa.Column("balance", StoredInteger(), nullable=False),
        sa.Column("total_earned", StoredInteger(), nullable=False),
        sa.Column("total_spent", StoredInteger(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("ledger_id"),
    )
    op.create_table(
        "central_bank_ledger",
        sa.Column("ledger_id", sa.String(length=32), nullable=False),
        sa.Column("balance", StoredInteger(), nullable=False),
        sa.Column("total_earned", StoredInteger(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("ledger_id"),
    )
    op.create_table(
        "guild_participant",
        sa.Column("guild_id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("guild_id", "user_id"),
    )
    op.create_table(
        "jackpot_pool",
        sa.Column("game_id", sa.String(length=32), nullable=False),
        sa.Column("pool_balance", StoredInteger(), nullable=False),
        sa.Column("total_contributed", StoredInteger(), nullable=False),
        sa.Column("total_claimed", StoredInteger(), nullable=False),
        sa.Column("seeded_amount", StoredInteger(), nullable=False),
        sa.Column("generation", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("game_id"),
    )
    op.create_table(
        "loan_contract",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("proposal_id", sa.Integer(), nullable=True),
        sa.Column("lender_type", sa.String(length=16), nullable=False),
        sa.Column("lender_id", sa.Integer(), nullable=True),
        sa.Column("lender_name", sa.String(length=128), nullable=False),
        sa.Column("lender_avatar_url", sa.String(length=2048), nullable=False),
        sa.Column("borrower_id", sa.Integer(), nullable=False),
        sa.Column("borrower_name", sa.String(length=128), nullable=False),
        sa.Column("borrower_avatar_url", sa.String(length=2048), nullable=False),
        sa.Column("original_principal", StoredInteger(), nullable=False),
        sa.Column("principal_remaining", StoredInteger(), nullable=False),
        sa.Column("interest_due", StoredInteger(), nullable=False),
        sa.Column("total_interest_paid", StoredInteger(), nullable=False),
        sa.Column("total_principal_paid", StoredInteger(), nullable=False),
        sa.Column("monthly_rate_bps", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("opened_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_interest_accrued_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table(table_name="loan_contract") as batch_op:
        batch_op.create_index(
            index_name="ix_loan_contract_borrower_status",
            columns=["borrower_id", "status"],
            unique=False,
        )
        batch_op.create_index(
            index_name="ix_loan_contract_lender_status",
            columns=["lender_id", "status"],
            unique=False,
        )
        batch_op.create_index(
            index_name="ix_loan_contract_lender_type_status",
            columns=["lender_type", "status"],
            unique=False,
        )

    op.create_table(
        "loan_proposal",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("lender_type", sa.String(length=16), nullable=False),
        sa.Column("borrower_id", sa.Integer(), nullable=False),
        sa.Column("borrower_name", sa.String(length=128), nullable=False),
        sa.Column("borrower_avatar_url", sa.String(length=2048), nullable=False),
        sa.Column("lender_id", sa.Integer(), nullable=True),
        sa.Column("lender_name", sa.String(length=128), nullable=False),
        sa.Column("lender_avatar_url", sa.String(length=2048), nullable=False),
        sa.Column("creator_id", sa.Integer(), nullable=False),
        sa.Column("amount", StoredInteger(), nullable=False),
        sa.Column("monthly_rate_bps", sa.Integer(), nullable=False),
        sa.Column("escrow_amount", StoredInteger(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table(table_name="loan_proposal") as batch_op:
        batch_op.create_index(
            index_name="ix_loan_proposal_borrower_status",
            columns=["borrower_id", "status"],
            unique=False,
        )
        batch_op.create_index(
            index_name="ix_loan_proposal_lender_status",
            columns=["lender_id", "status"],
            unique=False,
        )
        batch_op.create_index(
            index_name="ix_loan_proposal_status_kind", columns=["status", "kind"], unique=False
        )

    op.create_table(
        "user_account",
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("avatar_url", sa.String(length=2048), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("is_vip", sa.Boolean(), nullable=False),
        sa.Column("is_admin", sa.Boolean(), nullable=False),
        sa.Column("is_central_banker", sa.Boolean(), server_default="0", nullable=False),
        sa.Column("hide_from_leaderboard", sa.Boolean(), server_default="0", nullable=False),
        sa.PrimaryKeyConstraint("user_id"),
    )
    op.create_table(
        "user_wallet",
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("balance", StoredInteger(), nullable=False),
        sa.Column("total_earned", StoredInteger(), nullable=False),
        sa.Column("total_spent", StoredInteger(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("user_id"),
    )
    with op.batch_alter_table(table_name="user_wallet") as batch_op:
        batch_op.create_index(
            index_name="ix_user_wallet_balance", columns=["balance"], unique=False
        )
    now = database_now()
    op.bulk_insert(
        table=sa.table(
            "jackpot_pool",
            sa.column("game_id", sa.String()),
            sa.column("pool_balance", StoredInteger()),
            sa.column("total_contributed", StoredInteger()),
            sa.column("total_claimed", StoredInteger()),
            sa.column("seeded_amount", StoredInteger()),
            sa.column("generation", sa.Integer()),
            sa.column("updated_at", sa.DateTime(timezone=True)),
        ),
        rows=[
            {
                "game_id": "dragon_gate",
                "pool_balance": 1_000,
                "total_contributed": 0,
                "total_claimed": 0,
                "seeded_amount": 1_000,
                "generation": 0,
                "updated_at": now,
            }
        ],
    )
    op.bulk_insert(
        table=sa.table(
            "casino_ledger",
            sa.column("ledger_id", sa.String()),
            sa.column("balance", StoredInteger()),
            sa.column("total_earned", StoredInteger()),
            sa.column("total_spent", StoredInteger()),
            sa.column("updated_at", sa.DateTime(timezone=True)),
        ),
        rows=[
            {
                "ledger_id": "casino",
                "balance": 0,
                "total_earned": 0,
                "total_spent": 0,
                "updated_at": now,
            }
        ],
    )
    op.bulk_insert(
        table=sa.table(
            "central_bank_ledger",
            sa.column("ledger_id", sa.String()),
            sa.column("balance", StoredInteger()),
            sa.column("total_earned", StoredInteger()),
            sa.column("updated_at", sa.DateTime(timezone=True)),
        ),
        rows=[{"ledger_id": "central_bank", "balance": 0, "total_earned": 0, "updated_at": now}],
    )
