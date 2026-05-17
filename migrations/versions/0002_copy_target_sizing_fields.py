"""Add copy target percent-equity sizing fields.

Revision ID: 0002_copy_target_sizing_fields
Revises: 0001_initial_schema
Create Date: 2026-05-17
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0002_copy_target_sizing_fields"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None


def _columns(table_name: str) -> set[str]:
    return {column["name"] for column in sa.inspect(op.get_bind()).get_columns(table_name)}


def _add_column_if_missing(table_name: str, column: sa.Column) -> None:
    if column.name not in _columns(table_name):
        op.add_column(table_name, column)


def upgrade() -> None:
    _add_column_if_missing("copy_target_accounts", sa.Column("equity", sa.Float(), nullable=True))
    _add_column_if_missing(
        "copy_target_accounts",
        sa.Column("min_notional", sa.Float(), nullable=False, server_default="0"),
    )
    _add_column_if_missing(
        "copy_target_accounts",
        sa.Column("max_position_pct", sa.Float(), nullable=False, server_default="0"),
    )
    op.alter_column("copy_target_accounts", "min_notional", server_default=None)
    op.alter_column("copy_target_accounts", "max_position_pct", server_default=None)


def downgrade() -> None:
    existing = _columns("copy_target_accounts")
    for column_name in ["max_position_pct", "min_notional", "equity"]:
        if column_name in existing:
            op.drop_column("copy_target_accounts", column_name)
