"""Add broker-neutral V2 futures foundation tables.

Revision ID: 0009_v2_futures_foundation
Revises: 0008_paper_trades
Create Date: 2026-10-05
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0009_v2_futures_foundation"
down_revision = "0008_paper_trades"
branch_labels = None
depends_on = None

MONEY = sa.Numeric(24, 10)
JSON = postgresql.JSONB()


def _create(name: str, *columns, **kwargs) -> None:
    if name not in sa.inspect(op.get_bind()).get_table_names():
        op.create_table(name, *columns, **kwargs)


def _index(name: str, table: str, columns: list[str]) -> None:
    inspector = sa.inspect(op.get_bind())
    if table in inspector.get_table_names() and name not in {item["name"] for item in inspector.get_indexes(table)}:
        op.create_index(name, table, columns)


def upgrade() -> None:
    _create(
        "v2_futures_contracts",
        sa.Column("contract_id", sa.String(96), primary_key=True),
        sa.Column("product_code", sa.String(16), nullable=False),
        sa.Column("exchange", sa.String(16), nullable=False),
        sa.Column("expiration", sa.Date(), nullable=False),
        sa.Column("point_value", MONEY, nullable=False),
        sa.Column("tick_size", MONEY, nullable=False),
        sa.Column("tick_value", MONEY, nullable=False),
        sa.Column("currency", sa.String(8), nullable=False, server_default="USD"),
        sa.Column("first_trade_date", sa.Date(), nullable=True),
        sa.Column("last_trade_date", sa.Date(), nullable=True),
        sa.Column("provider_symbols", JSON, nullable=False, server_default=sa.text("'{}'::jsonb")),
    )
    _create(
        "v2_contract_mappings",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("source_contract_id", sa.String(96), sa.ForeignKey("v2_futures_contracts.contract_id"), nullable=False),
        sa.Column("target_contract_id", sa.String(96), sa.ForeignKey("v2_futures_contracts.contract_id"), nullable=False),
        sa.Column("expiration", sa.Date(), nullable=False),
        sa.UniqueConstraint("source_contract_id", "target_contract_id", name="uq_v2_contract_mapping"),
    )
    _create(
        "v2_broker_connections",
        sa.Column("connection_id", sa.String(96), primary_key=True),
        sa.Column("broker", sa.String(32), nullable=False),
        sa.Column("environment", sa.String(32), nullable=False),
        sa.Column("credential_ref", sa.String(256), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    _create(
        "v2_broker_accounts",
        sa.Column("account_id", sa.String(96), primary_key=True),
        sa.Column("connection_id", sa.String(96), sa.ForeignKey("v2_broker_connections.connection_id"), nullable=False),
        sa.Column("broker_account_ref", sa.String(256), nullable=False),
        sa.Column("display_name", sa.String(128), nullable=False, server_default=""),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    _create(
        "v2_risk_profiles",
        sa.Column("profile_id", sa.String(96), primary_key=True),
        sa.Column("account_id", sa.String(96), sa.ForeignKey("v2_broker_accounts.account_id"), nullable=True),
        sa.Column("sizing_mode", sa.String(40), nullable=False),
        sa.Column("sizing_value", MONEY, nullable=False),
        sa.Column("max_risk_per_trade", MONEY, nullable=False),
        sa.Column("max_contracts", sa.Integer(), nullable=False),
        sa.Column("max_concurrent_open_risk", MONEY, nullable=False),
        sa.Column("daily_realized_loss_ceiling", MONEY, nullable=False),
        sa.Column("daily_total_loss_ceiling", MONEY, nullable=False),
        sa.Column("margin_headroom_fraction", MONEY, nullable=False),
        sa.Column("limits_json", JSON, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("global_kill_switch", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("account_kill_switch", sa.Boolean(), nullable=False, server_default=sa.true()),
    )
    _create(
        "v2_session_schedules",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("contract_id", sa.String(96), sa.ForeignKey("v2_futures_contracts.contract_id"), nullable=False),
        sa.Column("trade_date", sa.Date(), nullable=False),
        sa.Column("session_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("session_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("maintenance_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("maintenance_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("source", sa.String(64), nullable=False),
        sa.UniqueConstraint("contract_id", "trade_date", name="uq_v2_schedule_contract_date"),
    )
    _create(
        "v2_opportunity_snapshots",
        sa.Column("opportunity_id", sa.String(96), primary_key=True),
        sa.Column("contract_id", sa.String(96), sa.ForeignKey("v2_futures_contracts.contract_id"), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("feature_cutoff_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("trade_date", sa.Date(), nullable=False),
        sa.Column("rule_version", sa.String(64), nullable=False),
        sa.Column("feature_schema_version", sa.String(64), nullable=False),
        sa.Column("market_data_lineage", JSON, nullable=False),
    )
    _create(
        "v2_trade_plan_versions",
        sa.Column("plan_id", sa.String(96), primary_key=True),
        sa.Column("trade_id", sa.String(96), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("contract_id", sa.String(96), sa.ForeignKey("v2_futures_contracts.contract_id"), nullable=False),
        sa.Column("direction", sa.String(8), nullable=False),
        sa.Column("planned_entry", MONEY, nullable=True),
        sa.Column("original_stop", MONEY, nullable=True),
        sa.Column("planned_targets", JSON, nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("planned_quantity", sa.Integer(), nullable=True),
        sa.Column("planned_risk_dollars", MONEY, nullable=True),
        sa.Column("source", sa.String(32), nullable=False),
        sa.UniqueConstraint("trade_id", "version", name="uq_v2_trade_plan_version"),
    )
    _create(
        "v2_master_orders",
        sa.Column("master_order_id", sa.String(96), primary_key=True),
        sa.Column("account_id", sa.String(96), sa.ForeignKey("v2_broker_accounts.account_id"), nullable=False),
        sa.Column("contract_id", sa.String(96), sa.ForeignKey("v2_futures_contracts.contract_id"), nullable=False),
        sa.Column("plan_id", sa.String(96), sa.ForeignKey("v2_trade_plan_versions.plan_id"), nullable=True),
        sa.Column("parent_order_id", sa.String(96), nullable=True),
        sa.Column("side", sa.String(8), nullable=False),
        sa.Column("quantity", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("latest_revision", sa.Integer(), nullable=False),
    )
    _create(
        "v2_order_events",
        sa.Column("event_id", sa.String(96), primary_key=True),
        sa.Column("master_order_id", sa.String(96), sa.ForeignKey("v2_master_orders.master_order_id"), nullable=True),
        sa.Column("follower_order_id", sa.String(96), nullable=True),
        sa.Column("broker", sa.String(32), nullable=False),
        sa.Column("account_id", sa.String(96), nullable=False),
        sa.Column("broker_order_id", sa.String(128), nullable=False),
        sa.Column("contract_id", sa.String(96), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("side", sa.String(8), nullable=False),
        sa.Column("quantity", sa.Integer(), nullable=False),
        sa.Column("filled_quantity", sa.Integer(), nullable=False),
        sa.Column("average_fill_price", MONEY, nullable=True),
        sa.Column("event_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("raw_reference", sa.String(256), nullable=True),
        sa.UniqueConstraint("broker", "account_id", "broker_order_id", "revision", name="uq_v2_order_event_revision"),
    )
    _create(
        "v2_master_executions",
        sa.Column("execution_id", sa.String(96), primary_key=True),
        sa.Column("master_order_id", sa.String(96), sa.ForeignKey("v2_master_orders.master_order_id"), nullable=False),
        sa.Column("account_id", sa.String(96), nullable=False),
        sa.Column("contract_id", sa.String(96), nullable=False),
        sa.Column("side", sa.String(8), nullable=False),
        sa.Column("quantity", sa.Integer(), nullable=False),
        sa.Column("price", MONEY, nullable=False),
        sa.Column("fee", MONEY, nullable=False),
        sa.Column("executed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
    )
    _create(
        "v2_trade_management_events",
        sa.Column("management_event_id", sa.String(96), primary_key=True),
        sa.Column("trade_id", sa.String(96), nullable=False),
        sa.Column("event_type", sa.String(32), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("quantity", sa.Integer(), nullable=True),
        sa.Column("previous_price", MONEY, nullable=True),
        sa.Column("new_price", MONEY, nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
    )
    _create(
        "v2_master_trades",
        sa.Column("trade_id", sa.String(96), primary_key=True),
        sa.Column("account_id", sa.String(96), sa.ForeignKey("v2_broker_accounts.account_id"), nullable=False),
        sa.Column("contract_id", sa.String(96), sa.ForeignKey("v2_futures_contracts.contract_id"), nullable=False),
        sa.Column("direction", sa.String(8), nullable=False),
        sa.Column("original_plan_id", sa.String(96), sa.ForeignKey("v2_trade_plan_versions.plan_id"), nullable=True),
        sa.Column("original_stop", MONEY, nullable=True),
        sa.Column("original_risk_dollars", MONEY, nullable=True),
        sa.Column("entry_vwap", MONEY, nullable=True),
        sa.Column("exit_vwap", MONEY, nullable=True),
        sa.Column("initial_quantity", sa.Integer(), nullable=False),
        sa.Column("maximum_quantity", sa.Integer(), nullable=False),
        sa.Column("opened_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("realized_gross_pnl", MONEY, nullable=True),
        sa.Column("realized_net_pnl", MONEY, nullable=True),
        sa.Column("realized_r", MONEY, nullable=True),
        sa.Column("mfe_r", MONEY, nullable=True),
        sa.Column("mae_r", MONEY, nullable=True),
    )
    _create(
        "v2_copy_intents",
        sa.Column("copy_intent_id", sa.String(96), primary_key=True),
        sa.Column("master_order_id", sa.String(96), sa.ForeignKey("v2_master_orders.master_order_id"), nullable=False),
        sa.Column("follower_account_id", sa.String(96), sa.ForeignKey("v2_broker_accounts.account_id"), nullable=False),
        sa.Column("source_contract_id", sa.String(96), nullable=False),
        sa.Column("target_contract_id", sa.String(96), nullable=False),
        sa.Column("intended_quantity", sa.Integer(), nullable=False),
        sa.Column("planned_risk_dollars", MONEY, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    _create(
        "v2_follower_orders",
        sa.Column("follower_order_id", sa.String(96), primary_key=True),
        sa.Column("copy_intent_id", sa.String(96), sa.ForeignKey("v2_copy_intents.copy_intent_id"), nullable=False),
        sa.Column("account_id", sa.String(96), nullable=False),
        sa.Column("contract_id", sa.String(96), nullable=False),
        sa.Column("side", sa.String(8), nullable=False),
        sa.Column("quantity", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("latest_revision", sa.Integer(), nullable=False),
    )
    _create(
        "v2_order_links",
        sa.Column("link_id", sa.String(96), primary_key=True),
        sa.Column("parent_order_id", sa.String(96), nullable=False),
        sa.Column("child_order_id", sa.String(96), nullable=False),
        sa.Column("link_type", sa.String(32), nullable=False),
        sa.Column("oco_group_id", sa.String(96), nullable=True),
    )
    _create(
        "v2_follower_executions",
        sa.Column("execution_id", sa.String(96), primary_key=True),
        sa.Column("follower_order_id", sa.String(96), sa.ForeignKey("v2_follower_orders.follower_order_id"), nullable=False),
        sa.Column("copy_intent_id", sa.String(96), nullable=False),
        sa.Column("account_id", sa.String(96), nullable=False),
        sa.Column("contract_id", sa.String(96), nullable=False),
        sa.Column("quantity", sa.Integer(), nullable=False),
        sa.Column("price", MONEY, nullable=False),
        sa.Column("fee", MONEY, nullable=False),
        sa.Column("executed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
    )
    _create(
        "v2_follower_trades",
        sa.Column("follower_trade_id", sa.String(96), primary_key=True),
        sa.Column("master_trade_id", sa.String(96), sa.ForeignKey("v2_master_trades.trade_id"), nullable=False),
        sa.Column("account_id", sa.String(96), nullable=False),
        sa.Column("contract_id", sa.String(96), nullable=False),
        sa.Column("entry_vwap", MONEY, nullable=True),
        sa.Column("exit_vwap", MONEY, nullable=True),
        sa.Column("realized_net_pnl", MONEY, nullable=True),
        sa.Column("realized_r", MONEY, nullable=True),
        sa.Column("execution_quality", JSON, nullable=False, server_default=sa.text("'{}'::jsonb")),
    )
    _create(
        "v2_position_snapshots",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("account_id", sa.String(96), nullable=False),
        sa.Column("contract_id", sa.String(96), nullable=False),
        sa.Column("captured_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("quantity", sa.Integer(), nullable=False),
        sa.Column("average_price", MONEY, nullable=True),
        sa.Column("mark_price", MONEY, nullable=True),
        sa.Column("unrealized_pnl", MONEY, nullable=False),
    )
    _create(
        "v2_account_risk_snapshots",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("account_id", sa.String(96), nullable=False),
        sa.Column("captured_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("equity", MONEY, nullable=False),
        sa.Column("available_margin", MONEY, nullable=False),
        sa.Column("open_risk", MONEY, nullable=False),
        sa.Column("realized_pnl_trade_date", MONEY, nullable=False),
        sa.Column("total_pnl_trade_date", MONEY, nullable=False),
        sa.Column("drawdown_trade_date", MONEY, nullable=False),
        sa.Column("global_kill_switch", sa.Boolean(), nullable=False),
        sa.Column("account_kill_switch", sa.Boolean(), nullable=False),
    )
    _create(
        "v2_reconciliation_incidents",
        sa.Column("incident_id", sa.String(96), primary_key=True),
        sa.Column("account_id", sa.String(96), nullable=False),
        sa.Column("detected_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("incident_type", sa.String(64), nullable=False),
        sa.Column("severity", sa.String(16), nullable=False),
        sa.Column("expected", JSON, nullable=False),
        sa.Column("actual", JSON, nullable=False),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
    )
    _create(
        "v2_model_artifacts",
        sa.Column("artifact_id", sa.String(96), primary_key=True),
        sa.Column("task", sa.String(64), nullable=False),
        sa.Column("feature_schema_version", sa.String(64), nullable=False),
        sa.Column("dataset_version", sa.String(64), nullable=False),
        sa.Column("artifact_namespace", sa.String(128), nullable=False),
        sa.Column("training_metrics", JSON, nullable=False),
        sa.Column("promotion_status", sa.String(32), nullable=False),
        sa.Column("source_lineage", JSON, nullable=False),
        sa.Column("asset_class", sa.String(16), nullable=False),
        sa.Column("product_codes", JSON, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    _create(
        "v2_execution_leases",
        sa.Column("lease_name", sa.String(96), primary_key=True),
        sa.Column("owner_id", sa.String(96), nullable=False),
        sa.Column("fence", sa.BigInteger(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    _index("ix_v2_futures_contracts_product_code", "v2_futures_contracts", ["product_code"])
    _index("ix_v2_futures_contracts_expiration", "v2_futures_contracts", ["expiration"])
    _index("ix_v2_trade_plan_versions_trade_id", "v2_trade_plan_versions", ["trade_id"])
    _index("ix_v2_trade_management_events_trade_id", "v2_trade_management_events", ["trade_id"])


def downgrade() -> None:
    # Downgrade only removes V2 objects; no legacy table is touched.
    for table in reversed(
        [
            "v2_futures_contracts",
            "v2_contract_mappings",
            "v2_broker_connections",
            "v2_broker_accounts",
            "v2_risk_profiles",
            "v2_session_schedules",
            "v2_opportunity_snapshots",
            "v2_trade_plan_versions",
            "v2_master_orders",
            "v2_order_events",
            "v2_master_executions",
            "v2_trade_management_events",
            "v2_master_trades",
            "v2_copy_intents",
            "v2_follower_orders",
            "v2_order_links",
            "v2_follower_executions",
            "v2_follower_trades",
            "v2_position_snapshots",
            "v2_account_risk_snapshots",
            "v2_reconciliation_incidents",
            "v2_model_artifacts",
            "v2_execution_leases",
        ]
    ):
        if table in sa.inspect(op.get_bind()).get_table_names():
            op.drop_table(table)
