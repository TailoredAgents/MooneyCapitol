"""Add append-only Rithmic read-capture persistence.

Revision ID: 0011_rithmic_read_capture
Revises: 0010_conner_nq_scout
Create Date: 2026-10-07
"""

from __future__ import annotations

from alembic import context, op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0011_rithmic_read_capture"
down_revision = "0010_conner_nq_scout"
branch_labels = None
depends_on = None

MONEY = sa.Numeric(24, 10)
JSON = postgresql.JSONB()

APPEND_ONLY_TABLES = (
    "v2_rithmic_broker_events",
    "v2_rithmic_account_observations",
    "v2_rithmic_order_observations",
    "v2_rithmic_execution_observations",
    "v2_rithmic_bracket_observations",
    "v2_rithmic_reference_observations",
    "v2_rithmic_pnl_observations",
    "v2_rithmic_rms_observations",
)


def _offline_mode() -> bool:
    try:
        return context.is_offline_mode()
    except NameError:
        return False


def _create(name: str, *columns, **kwargs) -> None:
    if _offline_mode() or name not in sa.inspect(op.get_bind()).get_table_names():
        op.create_table(name, *columns, **kwargs)


def _index(name: str, table: str, columns: list[str]) -> None:
    if _offline_mode():
        op.create_index(name, table, columns)
        return
    inspector = sa.inspect(op.get_bind())
    if table in inspector.get_table_names() and name not in {
        item["name"] for item in inspector.get_indexes(table)
    }:
        op.create_index(name, table, columns)


def _install_append_only_triggers() -> None:
    """Protect captured broker history from UPDATE and DELETE at the DB boundary."""

    op.execute(
        sa.text(
            """
            CREATE OR REPLACE FUNCTION v2_reject_rithmic_history_mutation()
            RETURNS trigger AS $$
            BEGIN
                RAISE EXCEPTION 'Rithmic capture history is append-only';
            END;
            $$ LANGUAGE plpgsql;
            """
        )
    )
    for table in APPEND_ONLY_TABLES:
        trigger = f"trg_{table}_append_only"
        op.execute(sa.text(f'DROP TRIGGER IF EXISTS "{trigger}" ON "{table}"'))
        op.execute(
            sa.text(
                f'CREATE TRIGGER "{trigger}" BEFORE UPDATE OR DELETE ON "{table}" '
                "FOR EACH ROW EXECUTE FUNCTION v2_reject_rithmic_history_mutation()"
            )
        )


def upgrade() -> None:
    _create(
        "v2_rithmic_connection_generations",
        sa.Column("generation_id", sa.String(96), primary_key=True),
        sa.Column(
            "connection_id",
            sa.String(96),
            sa.ForeignKey("v2_broker_connections.connection_id"),
            nullable=False,
        ),
        sa.Column("generation_ordinal", sa.BigInteger(), nullable=False),
        sa.Column("plant", sa.String(16), nullable=False),
        sa.Column("system_name", sa.String(128), nullable=False),
        sa.Column("state", sa.String(32), nullable=False),
        sa.Column("reconnect_attempt", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("heartbeat_interval_ms", sa.BigInteger(), nullable=True),
        sa.Column("connected_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("authenticated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("reconciled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("disconnected_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_message_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("forced_logout", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("ready", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("disconnect_reason", sa.Text(), nullable=True),
        sa.Column(
            "state_details", JSON, nullable=False, server_default=sa.text("'{}'::jsonb")
        ),
        sa.UniqueConstraint(
            "connection_id", "plant", "generation_ordinal", name="uq_v2_rithmic_generation"
        ),
        sa.CheckConstraint(
            "plant IN ('ORDER', 'PNL', 'TICKER')",
            name="ck_v2_rithmic_generation_plant",
        ),
        sa.CheckConstraint(
            "generation_ordinal >= 0 AND reconnect_attempt >= 0",
            name="ck_v2_rithmic_generation_counts",
        ),
        sa.CheckConstraint(
            "heartbeat_interval_ms IS NULL OR heartbeat_interval_ms > 0",
            name="ck_v2_rithmic_generation_heartbeat",
        ),
    )
    _create(
        "v2_rithmic_replay_batches",
        sa.Column("replay_batch_id", sa.String(96), primary_key=True),
        sa.Column(
            "generation_id",
            sa.String(96),
            sa.ForeignKey("v2_rithmic_connection_generations.generation_id"),
            nullable=False,
        ),
        sa.Column(
            "generation_map", JSON, nullable=False, server_default=sa.text("'{}'::jsonb")
        ),
        sa.Column(
            "account_id",
            sa.String(96),
            sa.ForeignKey("v2_broker_accounts.account_id"),
            nullable=True,
        ),
        sa.Column("broker_account_id", sa.String(256), nullable=True),
        sa.Column("batch_kind", sa.String(48), nullable=False),
        sa.Column("source_kind", sa.String(24), nullable=False),
        sa.Column("request_key", sa.String(256), nullable=True),
        sa.Column("user_message", sa.String(256), nullable=True),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("requested_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "terminal_response_received",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
        sa.Column("event_count", sa.BigInteger(), nullable=False, server_default=sa.text("0")),
        sa.Column(
            "boundary_facts", JSON, nullable=False, server_default=sa.text("'{}'::jsonb")
        ),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.CheckConstraint("event_count >= 0", name="ck_v2_rithmic_replay_event_count"),
    )
    _create(
        "v2_rithmic_broker_events",
        sa.Column("event_id", sa.String(96), primary_key=True),
        sa.Column(
            "generation_id",
            sa.String(96),
            sa.ForeignKey("v2_rithmic_connection_generations.generation_id"),
            nullable=False,
        ),
        sa.Column(
            "replay_batch_id",
            sa.String(96),
            sa.ForeignKey("v2_rithmic_replay_batches.replay_batch_id"),
            nullable=True,
        ),
        sa.Column(
            "account_id",
            sa.String(96),
            sa.ForeignKey("v2_broker_accounts.account_id"),
            nullable=True,
        ),
        sa.Column("plant", sa.String(16), nullable=False),
        sa.Column("template_id", sa.Integer(), nullable=False),
        sa.Column("template_name", sa.String(128), nullable=False),
        sa.Column("source_kind", sa.String(24), nullable=False),
        sa.Column("local_ingest_sequence", sa.BigInteger(), nullable=False),
        sa.Column("request_key", sa.String(256), nullable=True),
        sa.Column("user_message", sa.String(256), nullable=True),
        sa.Column("fcm_id", sa.String(256), nullable=True),
        sa.Column("ib_id", sa.String(256), nullable=True),
        sa.Column("broker_account_id", sa.String(256), nullable=True),
        sa.Column("basket_id", sa.String(256), nullable=True),
        sa.Column("original_basket_id", sa.String(256), nullable=True),
        sa.Column(
            "linked_basket_ids", JSON, nullable=False, server_default=sa.text("'[]'::jsonb")
        ),
        sa.Column("exchange_order_id", sa.String(256), nullable=True),
        sa.Column("ticker_plant_exchange_order_id", sa.String(256), nullable=True),
        sa.Column("fill_id", sa.String(256), nullable=True),
        sa.Column("sequence_number", sa.String(256), nullable=True),
        sa.Column("original_sequence_number", sa.String(256), nullable=True),
        sa.Column("correlation_sequence_number", sa.String(256), nullable=True),
        sa.Column("source_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("server_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("exchange_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("payload_fingerprint", sa.String(128), nullable=False),
        sa.Column("deduplication_key", sa.String(256), nullable=True),
        sa.Column(
            "event_facts", JSON, nullable=False, server_default=sa.text("'{}'::jsonb")
        ),
        sa.Column(
            "unknown_fields", JSON, nullable=False, server_default=sa.text("'{}'::jsonb")
        ),
        sa.UniqueConstraint(
            "generation_id", "local_ingest_sequence", name="uq_v2_rithmic_event_ingest"
        ),
        sa.UniqueConstraint("deduplication_key", name="uq_v2_rithmic_event_dedup"),
        sa.CheckConstraint(
            "local_ingest_sequence >= 0", name="ck_v2_rithmic_event_ingest_sequence"
        ),
        sa.CheckConstraint(
            "payload_fingerprint <> ''", name="ck_v2_rithmic_event_fingerprint"
        ),
    )
    _create(
        "v2_rithmic_account_observations",
        sa.Column("observation_id", sa.String(96), primary_key=True),
        sa.Column(
            "event_id",
            sa.String(96),
            sa.ForeignKey("v2_rithmic_broker_events.event_id"),
            nullable=True,
        ),
        sa.Column(
            "generation_id",
            sa.String(96),
            sa.ForeignKey("v2_rithmic_connection_generations.generation_id"),
            nullable=False,
        ),
        sa.Column(
            "account_id",
            sa.String(96),
            sa.ForeignKey("v2_broker_accounts.account_id"),
            nullable=True,
        ),
        sa.Column("fcm_id", sa.String(256), nullable=False),
        sa.Column("ib_id", sa.String(256), nullable=False),
        sa.Column("broker_account_id", sa.String(256), nullable=False),
        sa.Column("account_name", sa.String(256), nullable=True),
        sa.Column("currency", sa.String(16), nullable=True),
        sa.Column("access_type", sa.String(32), nullable=True),
        sa.Column("account_status", sa.String(64), nullable=True),
        sa.Column("user_id", sa.String(256), nullable=True),
        sa.Column("user_type", sa.String(64), nullable=True),
        sa.Column("user_status", sa.String(64), nullable=True),
        sa.Column("order_copy_status", sa.String(64), nullable=True),
        sa.Column("country_code", sa.String(16), nullable=True),
        sa.Column("state_code", sa.String(32), nullable=True),
        sa.Column("max_order_sessions", sa.Integer(), nullable=True),
        sa.Column("max_ticker_sessions", sa.Integer(), nullable=True),
        sa.Column("allowlisted", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "account_facts", JSON, nullable=False, server_default=sa.text("'{}'::jsonb")
        ),
        sa.UniqueConstraint("event_id", name="uq_v2_rithmic_account_event"),
        sa.CheckConstraint(
            "max_order_sessions IS NULL OR max_order_sessions >= 0",
            name="ck_v2_rithmic_account_order_sessions",
        ),
        sa.CheckConstraint(
            "max_ticker_sessions IS NULL OR max_ticker_sessions >= 0",
            name="ck_v2_rithmic_account_ticker_sessions",
        ),
    )
    _create(
        "v2_rithmic_order_observations",
        sa.Column("order_observation_id", sa.String(96), primary_key=True),
        sa.Column(
            "event_id",
            sa.String(96),
            sa.ForeignKey("v2_rithmic_broker_events.event_id"),
            nullable=False,
        ),
        sa.Column(
            "account_id",
            sa.String(96),
            sa.ForeignKey("v2_broker_accounts.account_id"),
            nullable=True,
        ),
        sa.Column("fcm_id", sa.String(256), nullable=True),
        sa.Column("ib_id", sa.String(256), nullable=True),
        sa.Column("broker_account_id", sa.String(256), nullable=False),
        sa.Column("basket_id", sa.String(256), nullable=False),
        sa.Column("original_basket_id", sa.String(256), nullable=True),
        sa.Column(
            "linked_basket_ids", JSON, nullable=False, server_default=sa.text("'[]'::jsonb")
        ),
        sa.Column("exchange_order_id", sa.String(256), nullable=True),
        sa.Column("ticker_plant_exchange_order_id", sa.String(256), nullable=True),
        sa.Column("symbol", sa.String(128), nullable=True),
        sa.Column("exchange", sa.String(64), nullable=True),
        sa.Column(
            "normalized_state",
            sa.String(48),
            nullable=False,
            server_default=sa.text("'UNKNOWN'"),
        ),
        sa.Column("broker_status", sa.String(128), nullable=True),
        sa.Column("notification_type", sa.String(128), nullable=True),
        sa.Column("completion_reason", sa.String(256), nullable=True),
        sa.Column("report_type", sa.String(128), nullable=True),
        sa.Column(
            "command_outcome",
            sa.String(48),
            nullable=False,
            server_default=sa.text("'NOT_APPLICABLE'"),
        ),
        sa.Column("side", sa.String(32), nullable=True),
        sa.Column("order_type", sa.String(64), nullable=True),
        sa.Column("duration", sa.String(64), nullable=True),
        sa.Column("quantity", sa.BigInteger(), nullable=True),
        sa.Column("fill_size", sa.BigInteger(), nullable=True),
        sa.Column("total_fill_size", sa.BigInteger(), nullable=True),
        sa.Column("total_unfilled_size", sa.BigInteger(), nullable=True),
        sa.Column("limit_price", MONEY, nullable=True),
        sa.Column("trigger_price", MONEY, nullable=True),
        sa.Column("fill_price", MONEY, nullable=True),
        sa.Column("average_fill_price", MONEY, nullable=True),
        sa.Column("fill_id", sa.String(256), nullable=True),
        sa.Column("sequence_number", sa.String(256), nullable=True),
        sa.Column("original_sequence_number", sa.String(256), nullable=True),
        sa.Column("correlation_sequence_number", sa.String(256), nullable=True),
        sa.Column("user_id", sa.String(256), nullable=True),
        sa.Column("application", sa.String(256), nullable=True),
        sa.Column("application_version", sa.String(128), nullable=True),
        sa.Column("originator_application", sa.String(256), nullable=True),
        sa.Column("originator_version", sa.String(128), nullable=True),
        sa.Column("window_name", sa.String(256), nullable=True),
        sa.Column("originator_window_name", sa.String(256), nullable=True),
        sa.Column("manual_or_auto", sa.String(32), nullable=True),
        sa.Column("user_tag", sa.String(256), nullable=True),
        sa.Column("mooney_owned", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("unknown_state", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("terminal", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("source_kind", sa.String(24), nullable=False),
        sa.Column("broker_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("server_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("exchange_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "order_facts", JSON, nullable=False, server_default=sa.text("'{}'::jsonb")
        ),
        sa.UniqueConstraint("event_id", name="uq_v2_rithmic_order_event"),
    )
    _create(
        "v2_rithmic_execution_observations",
        sa.Column("execution_observation_id", sa.String(96), primary_key=True),
        sa.Column(
            "event_id",
            sa.String(96),
            sa.ForeignKey("v2_rithmic_broker_events.event_id"),
            nullable=False,
        ),
        sa.Column(
            "account_id",
            sa.String(96),
            sa.ForeignKey("v2_broker_accounts.account_id"),
            nullable=True,
        ),
        sa.Column("broker_account_id", sa.String(256), nullable=False),
        sa.Column("basket_id", sa.String(256), nullable=False),
        sa.Column("fill_id", sa.String(256), nullable=False),
        sa.Column("execution_key", sa.String(512), nullable=False),
        sa.Column("exchange_order_id", sa.String(256), nullable=True),
        sa.Column("ticker_plant_exchange_order_id", sa.String(256), nullable=True),
        sa.Column("execution_kind", sa.String(32), nullable=False),
        sa.Column(
            "corrects_execution_id",
            sa.String(96),
            sa.ForeignKey("v2_rithmic_execution_observations.execution_observation_id"),
            nullable=True,
        ),
        sa.Column("side", sa.String(32), nullable=True),
        sa.Column("quantity", sa.BigInteger(), nullable=True),
        sa.Column("effective_quantity_delta", sa.BigInteger(), nullable=True),
        sa.Column("price", MONEY, nullable=True),
        sa.Column("commission", MONEY, nullable=True),
        sa.Column("sequence_number", sa.String(256), nullable=True),
        sa.Column("source_kind", sa.String(24), nullable=False),
        sa.Column("executed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("server_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "execution_facts", JSON, nullable=False, server_default=sa.text("'{}'::jsonb")
        ),
        sa.UniqueConstraint("event_id", name="uq_v2_rithmic_execution_event"),
        sa.UniqueConstraint("execution_key", name="uq_v2_rithmic_execution_key"),
    )
    _create(
        "v2_rithmic_bracket_observations",
        sa.Column("bracket_observation_id", sa.String(96), primary_key=True),
        sa.Column(
            "event_id",
            sa.String(96),
            sa.ForeignKey("v2_rithmic_broker_events.event_id"),
            nullable=False,
        ),
        sa.Column(
            "account_id",
            sa.String(96),
            sa.ForeignKey("v2_broker_accounts.account_id"),
            nullable=True,
        ),
        sa.Column("broker_account_id", sa.String(256), nullable=False),
        sa.Column("parent_basket_id", sa.String(256), nullable=False),
        sa.Column(
            "linked_basket_ids", JSON, nullable=False, server_default=sa.text("'[]'::jsonb")
        ),
        sa.Column("bracket_type", sa.String(64), nullable=True),
        sa.Column("operation_type", sa.String(64), nullable=True),
        sa.Column("status", sa.String(64), nullable=True),
        sa.Column("target_total_quantity", sa.BigInteger(), nullable=True),
        sa.Column("target_released_quantity", sa.BigInteger(), nullable=True),
        sa.Column("stop_total_quantity", sa.BigInteger(), nullable=True),
        sa.Column("stop_released_quantity", sa.BigInteger(), nullable=True),
        sa.Column(
            "target_tiers", JSON, nullable=False, server_default=sa.text("'[]'::jsonb")
        ),
        sa.Column("stop_tiers", JSON, nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column(
            "trailing_facts", JSON, nullable=False, server_default=sa.text("'{}'::jsonb")
        ),
        sa.Column("source_kind", sa.String(24), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "bracket_facts", JSON, nullable=False, server_default=sa.text("'{}'::jsonb")
        ),
        sa.UniqueConstraint(
            "event_id", "parent_basket_id", name="uq_v2_rithmic_bracket_event_parent"
        ),
    )
    _create(
        "v2_rithmic_reference_observations",
        sa.Column("reference_observation_id", sa.String(96), primary_key=True),
        sa.Column(
            "event_id",
            sa.String(96),
            sa.ForeignKey("v2_rithmic_broker_events.event_id"),
            nullable=False,
        ),
        sa.Column(
            "generation_id",
            sa.String(96),
            sa.ForeignKey("v2_rithmic_connection_generations.generation_id"),
            nullable=False,
        ),
        sa.Column(
            "account_id",
            sa.String(96),
            sa.ForeignKey("v2_broker_accounts.account_id"),
            nullable=True,
        ),
        sa.Column("broker_account_id", sa.String(256), nullable=True),
        sa.Column("observation_kind", sa.String(32), nullable=False),
        sa.Column("symbol", sa.String(128), nullable=True),
        sa.Column("exchange", sa.String(64), nullable=True),
        sa.Column("exchange_symbol", sa.String(128), nullable=True),
        sa.Column("symbol_name", sa.String(256), nullable=True),
        sa.Column("trading_symbol", sa.String(128), nullable=True),
        sa.Column("trading_exchange", sa.String(64), nullable=True),
        sa.Column("product_code", sa.String(64), nullable=True),
        sa.Column("instrument_type", sa.String(64), nullable=True),
        sa.Column("underlying_symbol", sa.String(128), nullable=True),
        sa.Column("expiration_date", sa.Date(), nullable=True),
        sa.Column("currency", sa.String(16), nullable=True),
        sa.Column("tick_size_type", sa.String(64), nullable=True),
        sa.Column("price_display_format", sa.String(64), nullable=True),
        sa.Column("is_tradable", sa.Boolean(), nullable=True),
        sa.Column("minimum_quoted_price_change", MONEY, nullable=True),
        sa.Column("minimum_feed_price_change", MONEY, nullable=True),
        sa.Column("single_point_value", MONEY, nullable=True),
        sa.Column("quote_to_feed_price_factor", MONEY, nullable=True),
        sa.Column("feed_to_quote_price_factor", MONEY, nullable=True),
        sa.Column("tick_table_first_price", MONEY, nullable=True),
        sa.Column("tick_table_last_price", MONEY, nullable=True),
        sa.Column("tick_table_first_price_operator", sa.String(32), nullable=True),
        sa.Column("tick_table_last_price_operator", sa.String(32), nullable=True),
        sa.Column("presence_bits", sa.BigInteger(), nullable=True),
        sa.Column("source_kind", sa.String(24), nullable=False),
        sa.Column("source_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "reference_facts", JSON, nullable=False, server_default=sa.text("'{}'::jsonb")
        ),
        sa.UniqueConstraint("event_id", name="uq_v2_rithmic_reference_event"),
    )
    _create(
        "v2_rithmic_pnl_observations",
        sa.Column("pnl_observation_id", sa.String(96), primary_key=True),
        sa.Column(
            "event_id",
            sa.String(96),
            sa.ForeignKey("v2_rithmic_broker_events.event_id"),
            nullable=False,
        ),
        sa.Column(
            "account_id",
            sa.String(96),
            sa.ForeignKey("v2_broker_accounts.account_id"),
            nullable=True,
        ),
        sa.Column("broker_account_id", sa.String(256), nullable=False),
        sa.Column("scope", sa.String(24), nullable=False),
        sa.Column("symbol", sa.String(128), nullable=True),
        sa.Column("exchange", sa.String(64), nullable=True),
        sa.Column("currency", sa.String(16), nullable=True),
        sa.Column("trade_date", sa.Date(), nullable=True),
        sa.Column("net_quantity", sa.BigInteger(), nullable=True),
        sa.Column("long_quantity", sa.BigInteger(), nullable=True),
        sa.Column("short_quantity", sa.BigInteger(), nullable=True),
        sa.Column("open_quantity", sa.BigInteger(), nullable=True),
        sa.Column("closed_quantity", sa.BigInteger(), nullable=True),
        sa.Column("working_buy_quantity", sa.BigInteger(), nullable=True),
        sa.Column("working_sell_quantity", sa.BigInteger(), nullable=True),
        sa.Column("average_open_fill_price", MONEY, nullable=True),
        sa.Column("open_position_pnl", MONEY, nullable=True),
        sa.Column("closed_position_pnl", MONEY, nullable=True),
        sa.Column("day_open_pnl", MONEY, nullable=True),
        sa.Column("day_closed_pnl", MONEY, nullable=True),
        sa.Column("day_total_pnl", MONEY, nullable=True),
        sa.Column("day_open_pnl_offset", MONEY, nullable=True),
        sa.Column("day_closed_pnl_offset", MONEY, nullable=True),
        sa.Column("account_balance", MONEY, nullable=True),
        sa.Column("cash_on_hand", MONEY, nullable=True),
        sa.Column("margin_balance", MONEY, nullable=True),
        sa.Column("available_buying_power", MONEY, nullable=True),
        sa.Column("used_buying_power", MONEY, nullable=True),
        sa.Column("reserved_buying_power", MONEY, nullable=True),
        sa.Column("excess_buy_margin", MONEY, nullable=True),
        sa.Column("excess_sell_margin", MONEY, nullable=True),
        sa.Column("commission", MONEY, nullable=True),
        sa.Column("source_kind", sa.String(24), nullable=False),
        sa.Column("is_snapshot", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column(
            "freshness",
            sa.String(24),
            nullable=False,
            server_default=sa.text("'UNKNOWN'"),
        ),
        sa.Column("source_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("pnl_facts", JSON, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.UniqueConstraint("event_id", name="uq_v2_rithmic_pnl_event"),
        sa.CheckConstraint(
            "scope IN ('ACCOUNT', 'INSTRUMENT', 'UNKNOWN')", name="ck_v2_rithmic_pnl_scope"
        ),
    )
    _create(
        "v2_rithmic_rms_observations",
        sa.Column("rms_observation_id", sa.String(96), primary_key=True),
        sa.Column(
            "event_id",
            sa.String(96),
            sa.ForeignKey("v2_rithmic_broker_events.event_id"),
            nullable=False,
        ),
        sa.Column(
            "account_id",
            sa.String(96),
            sa.ForeignKey("v2_broker_accounts.account_id"),
            nullable=True,
        ),
        sa.Column("broker_account_id", sa.String(256), nullable=False),
        sa.Column("scope", sa.String(24), nullable=False),
        sa.Column("product_code", sa.String(64), nullable=True),
        sa.Column("currency", sa.String(16), nullable=True),
        sa.Column("status", sa.String(64), nullable=True),
        sa.Column("algorithm", sa.String(128), nullable=True),
        sa.Column("loss_limit", MONEY, nullable=True),
        sa.Column("minimum_account_balance", MONEY, nullable=True),
        sa.Column("minimum_margin_balance", MONEY, nullable=True),
        sa.Column("account_balance", MONEY, nullable=True),
        sa.Column("current_auto_liquidate_threshold", MONEY, nullable=True),
        sa.Column("peak_account_balance", MONEY, nullable=True),
        sa.Column("peak_account_balance_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("auto_liquidate", sa.Boolean(), nullable=True),
        sa.Column("auto_liquidate_criteria", sa.String(128), nullable=True),
        sa.Column("disable_on_auto_liquidate", sa.Boolean(), nullable=True),
        sa.Column("max_order_quantity", sa.BigInteger(), nullable=True),
        sa.Column("buy_limit", sa.BigInteger(), nullable=True),
        sa.Column("sell_limit", sa.BigInteger(), nullable=True),
        sa.Column("buy_margin_rate", MONEY, nullable=True),
        sa.Column("sell_margin_rate", MONEY, nullable=True),
        sa.Column("commission_rate", MONEY, nullable=True),
        sa.Column("source_kind", sa.String(24), nullable=False),
        sa.Column("source_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("rms_facts", JSON, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.UniqueConstraint("event_id", name="uq_v2_rithmic_rms_event"),
        sa.CheckConstraint(
            "scope IN ('ACCOUNT', 'PRODUCT', 'UNKNOWN')", name="ck_v2_rithmic_rms_scope"
        ),
    )
    _create(
        "v2_rithmic_reconciliation_checkpoints",
        sa.Column("checkpoint_id", sa.String(96), primary_key=True),
        sa.Column(
            "generation_id",
            sa.String(96),
            sa.ForeignKey("v2_rithmic_connection_generations.generation_id"),
            nullable=False,
        ),
        sa.Column(
            "account_id",
            sa.String(96),
            sa.ForeignKey("v2_broker_accounts.account_id"),
            nullable=True,
        ),
        sa.Column("broker_account_id", sa.String(256), nullable=True),
        sa.Column("plant", sa.String(16), nullable=False),
        sa.Column("phase", sa.String(48), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column(
            "live_subscription_active", sa.Boolean(), nullable=False, server_default=sa.false()
        ),
        sa.Column("live_buffer_started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("snapshot_requested_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("snapshot_completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("buffered_events_applied_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("highest_ingest_sequence", sa.BigInteger(), nullable=True),
        sa.Column("order_cursor", sa.String(256), nullable=True),
        sa.Column("execution_cursor", sa.String(256), nullable=True),
        sa.Column("fill_cursor", sa.String(256), nullable=True),
        sa.Column("pnl_cursor", sa.String(256), nullable=True),
        sa.Column(
            "discrepancy_count", sa.BigInteger(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column("ready", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column(
            "checkpoint_facts", JSON, nullable=False, server_default=sa.text("'{}'::jsonb")
        ),
        sa.CheckConstraint(
            "highest_ingest_sequence IS NULL OR highest_ingest_sequence >= 0",
            name="ck_v2_rithmic_checkpoint_ingest",
        ),
        sa.CheckConstraint(
            "discrepancy_count >= 0", name="ck_v2_rithmic_checkpoint_discrepancy"
        ),
        sa.CheckConstraint(
            "NOT ready OR (status = 'COMPLETE' AND discrepancy_count = 0)",
            name="ck_v2_rithmic_checkpoint_ready",
        ),
    )

    _index(
        "ix_v2_rithmic_generation_connection_plant",
        "v2_rithmic_connection_generations",
        ["connection_id", "plant", "connected_at"],
    )
    _index(
        "ix_v2_rithmic_replay_generation_account",
        "v2_rithmic_replay_batches",
        ["generation_id", "account_id", "requested_at"],
    )
    _index(
        "ix_v2_rithmic_event_account_received",
        "v2_rithmic_broker_events",
        ["account_id", "received_at"],
    )
    _index(
        "ix_v2_rithmic_event_basket_received",
        "v2_rithmic_broker_events",
        ["basket_id", "received_at"],
    )
    _index(
        "ix_v2_rithmic_event_fill",
        "v2_rithmic_broker_events",
        ["fill_id", "received_at"],
    )
    _index(
        "ix_v2_rithmic_account_generation_observed",
        "v2_rithmic_account_observations",
        ["generation_id", "broker_account_id", "observed_at"],
    )
    _index(
        "ix_v2_rithmic_order_account_basket",
        "v2_rithmic_order_observations",
        ["broker_account_id", "basket_id", "received_at"],
    )
    _index(
        "ix_v2_rithmic_order_fill",
        "v2_rithmic_order_observations",
        ["fill_id", "received_at"],
    )
    _index(
        "ix_v2_rithmic_execution_account_fill",
        "v2_rithmic_execution_observations",
        ["broker_account_id", "fill_id"],
    )
    _index(
        "ix_v2_rithmic_execution_basket_time",
        "v2_rithmic_execution_observations",
        ["basket_id", "received_at"],
    )
    _index(
        "ix_v2_rithmic_bracket_account_parent",
        "v2_rithmic_bracket_observations",
        ["broker_account_id", "parent_basket_id", "received_at"],
    )
    _index(
        "ix_v2_rithmic_reference_symbol_exchange",
        "v2_rithmic_reference_observations",
        ["symbol", "exchange", "received_at"],
    )
    _index(
        "ix_v2_rithmic_reference_product_expiration",
        "v2_rithmic_reference_observations",
        ["product_code", "expiration_date", "received_at"],
    )
    _index(
        "ix_v2_rithmic_pnl_account_received",
        "v2_rithmic_pnl_observations",
        ["broker_account_id", "received_at"],
    )
    _index(
        "ix_v2_rithmic_pnl_instrument_received",
        "v2_rithmic_pnl_observations",
        ["symbol", "exchange", "received_at"],
    )
    _index(
        "ix_v2_rithmic_rms_account_product",
        "v2_rithmic_rms_observations",
        ["broker_account_id", "product_code", "received_at"],
    )
    _index(
        "ix_v2_rithmic_checkpoint_generation_account",
        "v2_rithmic_reconciliation_checkpoints",
        ["generation_id", "account_id", "recorded_at"],
    )
    _install_append_only_triggers()


def downgrade() -> None:
    offline = _offline_mode()
    existing = set() if offline else set(sa.inspect(op.get_bind()).get_table_names())
    for table in reversed(
        [
            "v2_rithmic_connection_generations",
            "v2_rithmic_replay_batches",
            "v2_rithmic_broker_events",
            "v2_rithmic_account_observations",
            "v2_rithmic_order_observations",
            "v2_rithmic_execution_observations",
            "v2_rithmic_bracket_observations",
            "v2_rithmic_reference_observations",
            "v2_rithmic_pnl_observations",
            "v2_rithmic_rms_observations",
            "v2_rithmic_reconciliation_checkpoints",
        ]
    ):
        if offline or table in existing:
            op.drop_table(table)
    op.execute(sa.text("DROP FUNCTION IF EXISTS v2_reject_rithmic_history_mutation()"))
