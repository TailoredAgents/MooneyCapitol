"""Initial schema with copier foundation.

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-05-10
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def _has_table(name: str) -> bool:
    return name in sa.inspect(op.get_bind()).get_table_names()


def _create_table_if_missing(name: str, *columns, **kwargs) -> None:
    if not _has_table(name):
        op.create_table(name, *columns, **kwargs)


def _has_index(table_name: str, index_name: str) -> bool:
    return index_name in {idx["name"] for idx in sa.inspect(op.get_bind()).get_indexes(table_name)}


def _create_index_if_missing(index_name: str, table_name: str, columns: list[str], unique: bool = False) -> None:
    if _has_table(table_name) and not _has_index(table_name, index_name):
        op.create_index(index_name, table_name, columns, unique=unique)


def upgrade() -> None:
    _create_table_if_missing(
        "symbols",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("ticker", sa.String(length=16), nullable=False),
        sa.Column("exchange", sa.String(length=16), nullable=True),
        sa.UniqueConstraint("ticker"),
    )
    _create_table_if_missing(
        "watchlist_entries",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("trade_date", sa.Date(), nullable=False),
        sa.Column("ticker", sa.String(length=16), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("gap_pct", sa.Float(), nullable=False),
        sa.Column("direction", sa.String(length=4), nullable=False),
        sa.Column("premkt_volume", sa.BigInteger(), nullable=True),
        sa.Column("price", sa.Float(), nullable=True),
        sa.Column("source", sa.String(length=16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("trade_date", "ticker", "source", name="uq_watchlist_trade_ticker_source"),
    )
    _create_table_if_missing(
        "candles",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("symbol_id", sa.Integer(), sa.ForeignKey("symbols.id"), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("tf", sa.String(length=8), nullable=False),
        sa.Column("o", sa.Float(), nullable=False),
        sa.Column("h", sa.Float(), nullable=False),
        sa.Column("l", sa.Float(), nullable=False),
        sa.Column("c", sa.Float(), nullable=False),
        sa.Column("v", sa.BigInteger(), nullable=False),
        sa.UniqueConstraint("symbol_id", "ts", "tf", name="uq_candles_symbol_ts_tf"),
    )
    _create_table_if_missing(
        "l2_snapshots",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("symbol_id", sa.Integer(), sa.ForeignKey("symbols.id"), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("bid_total", sa.Float(), nullable=False),
        sa.Column("ask_total", sa.Float(), nullable=False),
        sa.Column("levels", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("imbalance", sa.Float(), nullable=False),
        sa.UniqueConstraint("symbol_id", "ts", name="uq_l2_symbol_ts"),
    )
    _create_table_if_missing(
        "boxes",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("symbol_id", sa.Integer(), sa.ForeignKey("symbols.id"), nullable=False),
        sa.Column("tf", sa.String(length=8), nullable=False),
        sa.Column("start_ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("end_ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("hi", sa.Float(), nullable=False),
        sa.Column("lo", sa.Float(), nullable=False),
        sa.Column("bars", sa.Integer(), nullable=False),
        sa.Column("height", sa.Float(), nullable=False),
        sa.Column("quality_score", sa.Float(), nullable=False),
        sa.Column("rvol", sa.Float(), nullable=True),
        sa.Column("spread_cents", sa.Float(), nullable=True),
    )
    _create_table_if_missing(
        "setups",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("box_id", sa.BigInteger(), sa.ForeignKey("boxes.id"), nullable=False),
        sa.Column("symbol_id", sa.Integer(), sa.ForeignKey("symbols.id"), nullable=False),
        sa.Column("tf", sa.String(length=8), nullable=False),
        sa.Column("direction", sa.String(length=8), nullable=False),
        sa.Column("detected_ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("entry_price", sa.Float(), nullable=True),
        sa.Column("invalidation", sa.Float(), nullable=True),
        sa.Column("targets", postgresql.ARRAY(sa.Float()), nullable=True),
        sa.Column("rr_min", sa.Float(), nullable=True),
        sa.Column("score", sa.Integer(), nullable=True),
        sa.Column("l2_confirm", sa.Boolean(), nullable=False),
        sa.Column("state", sa.String(length=12), nullable=False),
        sa.Column("payload_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    _create_table_if_missing(
        "alerts",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("setup_id", sa.BigInteger(), sa.ForeignKey("setups.id"), nullable=True),
        sa.Column("sent_ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("channel", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("ack_by", sa.String(length=64), nullable=True),
        sa.Column("ack_ts", sa.DateTime(timezone=True), nullable=True),
        sa.Column("type", sa.String(length=16), nullable=False),
        sa.Column("symbol", sa.String(length=16), nullable=True),
        sa.Column("direction", sa.String(length=8), nullable=True),
        sa.Column("slack_thread_ts", sa.String(length=32), nullable=True),
        sa.Column("slack_message_ts", sa.String(length=32), nullable=True),
        sa.Column("payload_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    _create_table_if_missing(
        "fills",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("ext_trade_id", sa.String(length=128), nullable=True),
        sa.Column("account", sa.String(length=64), nullable=True),
        sa.Column("symbol", sa.String(length=16), nullable=True),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("side", sa.String(length=8), nullable=False),
        sa.Column("qty", sa.Integer(), nullable=False),
        sa.Column("price", sa.Float(), nullable=False),
        sa.Column("fee", sa.Float(), nullable=True),
        sa.Column("setup_id", sa.BigInteger(), sa.ForeignKey("setups.id"), nullable=True),
    )
    _create_table_if_missing(
        "trades",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("setup_id", sa.BigInteger(), sa.ForeignKey("setups.id"), nullable=False),
        sa.Column("account", sa.String(length=64), nullable=True),
        sa.Column("symbol", sa.String(length=16), nullable=True),
        sa.Column("open_ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("close_ts", sa.DateTime(timezone=True), nullable=True),
        sa.Column("qty", sa.Integer(), nullable=False),
        sa.Column("basis", sa.Float(), nullable=True),
        sa.Column("p_and_l", sa.Float(), nullable=True),
        sa.Column("realized_r", sa.Float(), nullable=True),
        sa.Column("exit_reason", sa.String(length=32), nullable=True),
    )
    _create_table_if_missing(
        "kv_store",
        sa.Column("key", sa.String(length=128), primary_key=True),
        sa.Column("value_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("value_bytes", sa.LargeBinary(), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    _create_table_if_missing(
        "copy_target_accounts",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column("broker", sa.String(length=32), nullable=False),
        sa.Column("environment", sa.String(length=16), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("account_ref", sa.String(length=128), nullable=True),
        sa.Column("equity", sa.Float(), nullable=True),
        sa.Column("sizing_mode", sa.String(length=32), nullable=False),
        sa.Column("sizing_value", sa.Float(), nullable=False),
        sa.Column("min_notional", sa.Float(), nullable=False),
        sa.Column("max_notional_per_trade", sa.Float(), nullable=False),
        sa.Column("max_position_pct", sa.Float(), nullable=False),
        sa.Column("max_daily_notional", sa.Float(), nullable=False),
        sa.Column("max_daily_trades", sa.Integer(), nullable=False),
        sa.Column("regular_hours_only", sa.Boolean(), nullable=False),
        sa.Column("shorting_enabled", sa.Boolean(), nullable=False),
        sa.Column("allowlist", postgresql.ARRAY(sa.String(length=16)), nullable=True),
        sa.Column("blocklist", postgresql.ARRAY(sa.String(length=16)), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("name"),
    )
    _create_table_if_missing(
        "master_executions",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("broker", sa.String(length=32), nullable=False),
        sa.Column("account_ref", sa.String(length=128), nullable=True),
        sa.Column("broker_execution_id", sa.String(length=128), nullable=False),
        sa.Column("broker_order_id", sa.String(length=128), nullable=True),
        sa.Column("symbol", sa.String(length=16), nullable=False),
        sa.Column("side", sa.String(length=8), nullable=False),
        sa.Column("qty", sa.Float(), nullable=False),
        sa.Column("price", sa.Float(), nullable=False),
        sa.Column("asset_class", sa.String(length=32), nullable=False),
        sa.Column("executed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("raw_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.UniqueConstraint("broker", "account_ref", "broker_execution_id", name="uq_master_exec_broker_account_exec"),
    )
    _create_table_if_missing(
        "copy_orders",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("master_execution_id", sa.BigInteger(), sa.ForeignKey("master_executions.id"), nullable=False),
        sa.Column("target_account_id", sa.BigInteger(), sa.ForeignKey("copy_target_accounts.id"), nullable=False),
        sa.Column("broker", sa.String(length=32), nullable=False),
        sa.Column("client_order_id", sa.String(length=128), nullable=False),
        sa.Column("broker_order_id", sa.String(length=128), nullable=True),
        sa.Column("symbol", sa.String(length=16), nullable=False),
        sa.Column("side", sa.String(length=8), nullable=False),
        sa.Column("qty", sa.Float(), nullable=False),
        sa.Column("order_type", sa.String(length=16), nullable=False),
        sa.Column("time_in_force", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("accepted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("filled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("filled_qty", sa.Float(), nullable=True),
        sa.Column("avg_fill_price", sa.Float(), nullable=True),
        sa.Column("reject_reason", sa.String(length=512), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("raw_submit_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("raw_response_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.UniqueConstraint("client_order_id"),
    )
    _create_table_if_missing(
        "copy_order_events",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("copy_order_id", sa.BigInteger(), sa.ForeignKey("copy_orders.id"), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=True),
        sa.Column("event_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("raw_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    _create_table_if_missing(
        "copy_reconciliations",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("target_account_id", sa.BigInteger(), sa.ForeignKey("copy_target_accounts.id"), nullable=True),
        sa.Column("symbol", sa.String(length=16), nullable=True),
        sa.Column("severity", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("message", sa.String(length=1024), nullable=False),
        sa.Column("detected_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("raw_context", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    _create_table_if_missing(
        "copier_audit_events",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("actor", sa.String(length=128), nullable=True),
        sa.Column("target_account_id", sa.BigInteger(), sa.ForeignKey("copy_target_accounts.id"), nullable=True),
        sa.Column("message", sa.String(length=1024), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )

    _create_index_if_missing("ix_symbols_ticker", "symbols", ["ticker"])
    _create_index_if_missing("ix_watchlist_entries_trade_date", "watchlist_entries", ["trade_date"])
    _create_index_if_missing("ix_watchlist_entries_ticker", "watchlist_entries", ["ticker"])
    _create_index_if_missing("ix_candles_symbol_id", "candles", ["symbol_id"])
    _create_index_if_missing("ix_candles_ts", "candles", ["ts"])
    _create_index_if_missing("ix_candles_tf", "candles", ["tf"])
    _create_index_if_missing("ix_l2_snapshots_symbol_id", "l2_snapshots", ["symbol_id"])
    _create_index_if_missing("ix_l2_snapshots_ts", "l2_snapshots", ["ts"])
    _create_index_if_missing("ix_boxes_symbol_id", "boxes", ["symbol_id"])
    _create_index_if_missing("ix_boxes_start_ts", "boxes", ["start_ts"])
    _create_index_if_missing("ix_setups_box_id", "setups", ["box_id"])
    _create_index_if_missing("ix_setups_symbol_id", "setups", ["symbol_id"])
    _create_index_if_missing("ix_setups_detected_ts", "setups", ["detected_ts"])
    _create_index_if_missing("ix_setups_state", "setups", ["state"])
    _create_index_if_missing("ix_alerts_setup_id", "alerts", ["setup_id"])
    _create_index_if_missing("ix_alerts_sent_ts", "alerts", ["sent_ts"])
    _create_index_if_missing("ix_alerts_symbol", "alerts", ["symbol"])
    _create_index_if_missing("ix_alerts_slack_thread_ts", "alerts", ["slack_thread_ts"])
    _create_index_if_missing("ix_alerts_slack_message_ts", "alerts", ["slack_message_ts"])
    _create_index_if_missing("ix_fills_ext_trade_id", "fills", ["ext_trade_id"])
    _create_index_if_missing("ix_fills_account", "fills", ["account"])
    _create_index_if_missing("ix_fills_symbol", "fills", ["symbol"])
    _create_index_if_missing("ix_fills_ts", "fills", ["ts"])
    _create_index_if_missing("ix_fills_setup_id", "fills", ["setup_id"])
    _create_index_if_missing("ix_trades_setup_id", "trades", ["setup_id"])
    _create_index_if_missing("ix_trades_account", "trades", ["account"])
    _create_index_if_missing("ix_trades_symbol", "trades", ["symbol"])
    _create_index_if_missing("ix_copy_target_accounts_name", "copy_target_accounts", ["name"])
    _create_index_if_missing("ix_master_executions_account_ref", "master_executions", ["account_ref"])
    _create_index_if_missing("ix_master_executions_symbol", "master_executions", ["symbol"])
    _create_index_if_missing("ix_master_executions_executed_at", "master_executions", ["executed_at"])
    _create_index_if_missing("ix_copy_orders_master_execution_id", "copy_orders", ["master_execution_id"])
    _create_index_if_missing("ix_copy_orders_target_account_id", "copy_orders", ["target_account_id"])
    _create_index_if_missing("ix_copy_orders_client_order_id", "copy_orders", ["client_order_id"], unique=True)
    _create_index_if_missing("ix_copy_orders_broker_order_id", "copy_orders", ["broker_order_id"])
    _create_index_if_missing("ix_copy_orders_symbol", "copy_orders", ["symbol"])
    _create_index_if_missing("ix_copy_orders_status", "copy_orders", ["status"])
    _create_index_if_missing("ix_copy_order_events_copy_order_id", "copy_order_events", ["copy_order_id"])
    _create_index_if_missing("ix_copy_order_events_received_at", "copy_order_events", ["received_at"])
    _create_index_if_missing("ix_copy_reconciliations_target_account_id", "copy_reconciliations", ["target_account_id"])
    _create_index_if_missing("ix_copy_reconciliations_symbol", "copy_reconciliations", ["symbol"])
    _create_index_if_missing("ix_copy_reconciliations_status", "copy_reconciliations", ["status"])
    _create_index_if_missing("ix_copy_reconciliations_detected_at", "copy_reconciliations", ["detected_at"])
    _create_index_if_missing("ix_copier_audit_events_event_type", "copier_audit_events", ["event_type"])
    _create_index_if_missing("ix_copier_audit_events_target_account_id", "copier_audit_events", ["target_account_id"])
    _create_index_if_missing("ix_copier_audit_events_created_at", "copier_audit_events", ["created_at"])


def downgrade() -> None:
    for table_name in [
        "copier_audit_events",
        "copy_reconciliations",
        "copy_order_events",
        "copy_orders",
        "master_executions",
        "copy_target_accounts",
        "kv_store",
        "trades",
        "fills",
        "alerts",
        "setups",
        "boxes",
        "l2_snapshots",
        "candles",
        "watchlist_entries",
        "symbols",
    ]:
        if _has_table(table_name):
            op.drop_table(table_name)
