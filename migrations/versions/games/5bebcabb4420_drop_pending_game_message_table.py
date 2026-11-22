"""Drop pending_game_message table

Revision ID: 5bebcabb4420
Revises: 663df6fa7d1b
Create Date: 2026-11-22 18:12:57.198533
"""

from alembic import op

revision = "5bebcabb4420"
down_revision = "663df6fa7d1b"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Applies this revision."""
    op.drop_table(table_name="pending_game_message")
