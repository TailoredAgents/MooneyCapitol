"""Add copy target daily trade limit.

Revision ID: 0003_copy_limits
Revises: 0002_copy_target_sizing_fields
Create Date: 2026-05-17
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0003_copy_limits"
down_revision = "0002_copy_target_sizing_fields"
branch_labels = None
depends_on = None


def _columns(table_name: str) -> set[str]:
    return {column["name"] for column in sa.inspect(op.get_bind()).get_columns(table_name)}


def upgrade() -> None:
    if "max_daily_trades" not in _columns("copy_target_accounts"):
        op.add_column(
            "copy_target_accounts",
            sa.Column("max_daily_trades", sa.Integer(), nullable=False, server_default="0"),
        )
        op.alter_column("copy_target_accounts", "max_daily_trades", server_default=None)


def downgrade() -> None:
    if "max_daily_trades" in _columns("copy_target_accounts"):
        op.drop_column("copy_target_accounts", "max_daily_trades")
