"""Add PnL monitoring tables.

Revision ID: 0004_pnl_monitoring
Revises: 0003_copy_target_daily_trade_limit
Create Date: 2026-05-17
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0004_pnl_monitoring"
down_revision = "0003_copy_target_daily_trade_limit"
branch_labels = None
depends_on = None


def _has_table(name: str) -> bool:
    return name in sa.inspect(op.get_bind()).get_table_names()


def _create_table_if_missing(name: str, *columns, **kwargs) -> None:
    if not _has_table(name):
        op.create_table(name, *columns, **kwargs)


def upgrade() -> None:
    _create_table_if_missing(
        "account_snapshots",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("account_ref", sa.String(length=128), nullable=False),
        sa.Column("account_name", sa.String(length=64), nullable=False),
        sa.Column("account_type", sa.String(length=20), nullable=False),
        sa.Column("broker", sa.String(length=32), nullable=False, server_default="webull"),
        sa.Column("snapshot_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("market_session", sa.String(length=20), nullable=False, server_default="unknown"),
        sa.Column("cash_balance", sa.Float(), nullable=True),
        sa.Column("equity_value", sa.Float(), nullable=True),
        sa.Column("total_value", sa.Float(), nullable=False),
        sa.Column("buying_power", sa.Float(), nullable=True),
        sa.Column("day_trades_used", sa.Integer(), nullable=True),
        sa.Column("day_trades_remaining", sa.Integer(), nullable=True),
        sa.Column("unrealized_pnl", sa.Float(), nullable=False, server_default="0"),
        sa.Column("realized_pnl_today", sa.Float(), nullable=False, server_default="0"),
        sa.Column("total_pnl_today", sa.Float(), nullable=False, server_default="0"),
        sa.Column("max_drawdown_today", sa.Float(), nullable=False, server_default="0"),
        sa.Column("max_profit_today", sa.Float(), nullable=False, server_default="0"),
        sa.Column("position_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_exposure", sa.Float(), nullable=False, server_default="0"),
        sa.Column("risk_level", sa.String(length=20), nullable=False, server_default="normal"),
        sa.Column("data_source", sa.String(length=50), nullable=False, server_default="webull_api"),
        sa.Column("raw_payload", postgresql.JSONB(), nullable=True),
    )
    _create_table_if_missing(
        "position_snapshots",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("account_ref", sa.String(length=128), nullable=False),
        sa.Column("account_name", sa.String(length=64), nullable=False),
        sa.Column("symbol", sa.String(length=16), nullable=False),
        sa.Column("snapshot_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("qty", sa.Float(), nullable=False),
        sa.Column("avg_price", sa.Float(), nullable=True),
        sa.Column("current_price", sa.Float(), nullable=True),
        sa.Column("market_value", sa.Float(), nullable=False, server_default="0"),
        sa.Column("cost_basis", sa.Float(), nullable=False, server_default="0"),
        sa.Column("unrealized_pnl", sa.Float(), nullable=False, server_default="0"),
        sa.Column("unrealized_pnl_pct", sa.Float(), nullable=True),
        sa.Column("side", sa.String(length=10), nullable=False, server_default="long"),
        sa.Column("raw_payload", postgresql.JSONB(), nullable=True),
    )
    _create_table_if_missing(
        "pnl_alerts",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("account_ref", sa.String(length=128), nullable=False),
        sa.Column("account_name", sa.String(length=64), nullable=False),
        sa.Column("account_type", sa.String(length=20), nullable=False),
        sa.Column("alert_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("alert_type", sa.String(length=50), nullable=False),
        sa.Column("severity", sa.String(length=20), nullable=False),
        sa.Column("threshold_value", sa.Float(), nullable=True),
        sa.Column("actual_value", sa.Float(), nullable=True),
        sa.Column("account_value", sa.Float(), nullable=True),
        sa.Column("unrealized_pnl", sa.Float(), nullable=False, server_default="0"),
        sa.Column("realized_pnl", sa.Float(), nullable=False, server_default="0"),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("triggered_by", sa.String(length=50), nullable=False, server_default="pnl_monitor"),
        sa.Column("action_taken", sa.String(length=100), nullable=True),
        sa.Column("acknowledged", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("acknowledged_by", sa.String(length=128), nullable=True),
        sa.Column("acknowledged_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("raw_context", postgresql.JSONB(), nullable=True),
    )
    _create_table_if_missing(
        "trading_sessions",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("account_ref", sa.String(length=128), nullable=False),
        sa.Column("account_name", sa.String(length=64), nullable=False),
        sa.Column("trade_date", sa.Date(), nullable=False),
        sa.Column("session_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("session_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("starting_value", sa.Float(), nullable=False),
        sa.Column("ending_value", sa.Float(), nullable=True),
        sa.Column("realized_pnl", sa.Float(), nullable=False, server_default="0"),
        sa.Column("unrealized_pnl", sa.Float(), nullable=False, server_default="0"),
        sa.Column("total_pnl", sa.Float(), nullable=False, server_default="0"),
        sa.Column("max_drawdown", sa.Float(), nullable=False, server_default="0"),
        sa.Column("max_profit", sa.Float(), nullable=False, server_default="0"),
        sa.Column("trades_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.UniqueConstraint("account_ref", "trade_date", name="uq_trading_session_account_date"),
    )

    for name, table, columns in [
        ("idx_account_snapshot_ref_time", "account_snapshots", ["account_ref", "snapshot_time"]),
        ("idx_account_snapshot_name_time", "account_snapshots", ["account_name", "snapshot_time"]),
        ("idx_position_snapshot_account_symbol_time", "position_snapshots", ["account_ref", "symbol", "snapshot_time"]),
        ("idx_pnl_alert_account_time", "pnl_alerts", ["account_ref", "alert_time"]),
    ]:
        if _has_table(table):
            indexes = {idx["name"] for idx in sa.inspect(op.get_bind()).get_indexes(table)}
            if name not in indexes:
                op.create_index(name, table, columns)


def downgrade() -> None:
    for table in ["trading_sessions", "pnl_alerts", "position_snapshots", "account_snapshots"]:
        if _has_table(table):
            op.drop_table(table)
