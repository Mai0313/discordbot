"""Drop the dead central-banker flag, proposal escrow and wallet balance index

Revision ID: 0af4d831a349
Revises: 9e1621821fad
Create Date: 2026-11-07 04:58:29.698680
"""

from alembic import op

revision = "0af4d831a349"
down_revision = "9e1621821fad"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Applies this revision."""
    with op.batch_alter_table(table_name="loan_proposal") as batch_op:
        batch_op.drop_column(column_name="escrow_amount")

    with op.batch_alter_table(table_name="user_account") as batch_op:
        batch_op.drop_column(column_name="is_central_banker")

    with op.batch_alter_table(table_name="user_wallet") as batch_op:
        batch_op.drop_index(index_name="ix_user_wallet_balance")
