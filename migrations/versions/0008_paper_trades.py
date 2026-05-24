"""Add AI paper trades.

Revision ID: 0008_paper_trades
Revises: 0007_shadow_decisions
Create Date: 2026-05-24
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0008_paper_trades"
down_revision = "0007_shadow_decisions"
branch_labels = None
depends_on = None


def _has_table(name: str) -> bool:
    return name in sa.inspect(op.get_bind()).get_table_names()


def _has_index(table_name: str, index_name: str) -> bool:
    if not _has_table(table_name):
        return False
    return index_name in {idx["name"] for idx in sa.inspect(op.get_bind()).get_indexes(table_name)}


def upgrade() -> None:
    if not _has_table("paper_trades"):
        op.create_table(
            "paper_trades",
            sa.Column("id", sa.BigInteger(), primary_key=True),
            sa.Column("shadow_decision_id", sa.BigInteger(), sa.ForeignKey("shadow_decisions.id"), nullable=True, unique=True),
            sa.Column("alert_id", sa.BigInteger(), sa.ForeignKey("alerts.id"), nullable=True),
            sa.Column("setup_id", sa.BigInteger(), sa.ForeignKey("setups.id"), nullable=True),
            sa.Column("symbol", sa.String(length=16), nullable=False),
            sa.Column("direction", sa.String(length=8), nullable=False),
            sa.Column("status", sa.String(length=16), nullable=False, server_default="open"),
            sa.Column("opened_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("entry_price", sa.Float(), nullable=False),
            sa.Column("stop_price", sa.Float(), nullable=True),
            sa.Column("target_price", sa.Float(), nullable=True),
            sa.Column("exit_price", sa.Float(), nullable=True),
            sa.Column("exit_reason", sa.String(length=32), nullable=True),
            sa.Column("size_pct", sa.Float(), nullable=False, server_default="0"),
            sa.Column("account_equity", sa.Float(), nullable=False, server_default="0"),
            sa.Column("notional", sa.Float(), nullable=False, server_default="0"),
            sa.Column("qty", sa.Float(), nullable=False, server_default="0"),
            sa.Column("unrealized_pnl", sa.Float(), nullable=False, server_default="0"),
            sa.Column("realized_pnl", sa.Float(), nullable=True),
            sa.Column("realized_r", sa.Float(), nullable=True),
            sa.Column("last_price", sa.Float(), nullable=True),
            sa.Column("last_mark_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("model_version", sa.String(length=64), nullable=False, server_default="paper-v1"),
            sa.Column("reason", sa.Text(), nullable=True),
            sa.Column("raw_context", postgresql.JSONB(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )

    for index_name, columns in [
        ("ix_paper_trades_shadow_decision_id", ["shadow_decision_id"]),
        ("ix_paper_trades_alert_id", ["alert_id"]),
        ("ix_paper_trades_setup_id", ["setup_id"]),
        ("ix_paper_trades_symbol", ["symbol"]),
        ("ix_paper_trades_status", ["status"]),
        ("ix_paper_trades_opened_at", ["opened_at"]),
        ("ix_paper_trades_closed_at", ["closed_at"]),
        ("ix_paper_trades_created_at", ["created_at"]),
        ("idx_paper_trade_symbol_status", ["symbol", "status"]),
        ("idx_paper_trade_status_opened", ["status", "opened_at"]),
    ]:
        if not _has_index("paper_trades", index_name):
            op.create_index(index_name, "paper_trades", columns)


def downgrade() -> None:
    if _has_table("paper_trades"):
        op.drop_table("paper_trades")
