"""Add durable V2 copy lifecycle intents and commands.

Revision ID: 0013_v2_copy_lifecycle
Revises: 0012_v2_copy_control
Create Date: 2026-10-15
"""

from __future__ import annotations

from alembic import context, op
import sqlalchemy as sa


revision = "0013_v2_copy_lifecycle"
down_revision = "0012_v2_copy_control"
branch_labels = None
depends_on = None

MONEY = sa.Numeric(24, 10)


def _offline_mode() -> bool:
    try:
        return context.is_offline_mode()
    except (AttributeError, NameError, RuntimeError):
        return False


def _create(name: str, *columns: object, **kwargs: object) -> None:
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


def _install_append_only_trigger() -> None:
    op.execute(
        sa.text(
            """
            CREATE OR REPLACE FUNCTION v2_reject_copy_lifecycle_event_mutation()
            RETURNS trigger AS $$
            BEGIN
                RAISE EXCEPTION 'V2 copy lifecycle event history is append-only';
            END;
            $$ LANGUAGE plpgsql;
            """
        )
    )
    op.execute(
        sa.text(
            "DROP TRIGGER IF EXISTS trg_v2_copy_lifecycle_events_append_only "
            "ON v2_copy_lifecycle_events"
        )
    )
    op.execute(
        sa.text(
            "CREATE TRIGGER trg_v2_copy_lifecycle_events_append_only "
            "BEFORE UPDATE OR DELETE ON v2_copy_lifecycle_events FOR EACH ROW "
            "EXECUTE FUNCTION v2_reject_copy_lifecycle_event_mutation()"
        )
    )


def upgrade() -> None:
    _create(
        "v2_copy_lifecycle_intents",
        sa.Column("intent_id", sa.String(96), primary_key=True),
        sa.Column(
            "profile_id",
            sa.String(96),
            sa.ForeignKey("v2_copy_profiles.profile_id"),
            nullable=False,
        ),
        sa.Column("source_event_id", sa.String(96), nullable=False),
        sa.Column("source_fingerprint", sa.String(64), nullable=False),
        sa.Column("master_order_id", sa.String(96), nullable=False),
        sa.Column("master_revision", sa.Integer(), nullable=False),
        sa.Column("master_event_kind", sa.String(32), nullable=False),
        sa.Column("follower_connection_id", sa.String(96), nullable=False),
        sa.Column("directive", sa.String(32), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("allowed", sa.Boolean(), nullable=False),
        sa.Column("reason", sa.String(96), nullable=False),
        sa.Column("contract_id", sa.String(96), nullable=False),
        sa.Column("symbol", sa.String(64), nullable=False),
        sa.Column("exchange", sa.String(32), nullable=False),
        sa.Column("current_signed_quantity", sa.Integer(), nullable=False),
        sa.Column("evaluated_target_signed_quantity", sa.Integer(), nullable=True),
        sa.Column("achievable_target_signed_quantity", sa.Integer(), nullable=False),
        sa.Column("delta_signed_quantity", sa.Integer(), nullable=False),
        sa.Column("desired_quantity", MONEY, nullable=False),
        sa.Column("desired_risk_dollars", MONEY, nullable=False),
        sa.Column("achievable_risk_dollars", MONEY, nullable=False),
        sa.Column("desired_risk_fraction", MONEY, nullable=False),
        sa.Column("achievable_risk_fraction", MONEY, nullable=False),
        sa.Column("desired_exposure_fraction", MONEY, nullable=False),
        sa.Column("achievable_exposure_fraction", MONEY, nullable=False),
        sa.Column("risk_parity_error", MONEY, nullable=False),
        sa.Column("exposure_parity_error", MONEY, nullable=False),
        sa.Column("parity_error", MONEY, nullable=False),
        sa.Column("over_target", sa.Boolean(), nullable=False),
        sa.Column("target_margin_required", MONEY, nullable=False),
        sa.Column(
            "revision", sa.Integer(), nullable=False, server_default=sa.text("1")
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "profile_id",
            "source_event_id",
            "follower_connection_id",
            name="uq_v2_copy_lifecycle_source_follower",
        ),
        sa.CheckConstraint(
            "master_revision >= 1", name="ck_v2_copy_lifecycle_master_rev"
        ),
        sa.CheckConstraint("revision >= 1", name="ck_v2_copy_lifecycle_revision"),
        sa.CheckConstraint(
            "master_event_kind IN "
            "('NEW', 'REPLACE', 'CANCEL', 'PARTIAL_FILL', 'FILL', "
            "'REJECT', 'POSITION_TARGET')",
            name="ck_v2_copy_lifecycle_master_kind",
        ),
        sa.CheckConstraint(
            "directive IN ('SYNC_POSITION', 'REPLACE_WORKING', "
            "'CANCEL_WORKING', 'SUBMIT_BRACKET', 'SUBMIT_OCO')",
            name="ck_v2_copy_lifecycle_directive",
        ),
        sa.CheckConstraint(
            "status IN ('PLANNED', 'BLOCKED', 'DISPATCHING', 'DISPATCHED', "
            "'ACKNOWLEDGED', 'PARTIALLY_FILLED', 'FILLED', 'CANCELLED', "
            "'REJECTED', 'FAILED', 'DIVERGED')",
            name="ck_v2_copy_lifecycle_status",
        ),
        sa.CheckConstraint(
            "follower_connection_id IN "
            "('follower-1', 'follower-2', 'follower-3', 'follower-4')",
            name="ck_v2_copy_lifecycle_follower",
        ),
        sa.CheckConstraint(
            "desired_quantity >= 0 AND desired_risk_dollars >= 0 "
            "AND achievable_risk_dollars >= 0 "
            "AND desired_risk_fraction >= 0 "
            "AND achievable_risk_fraction >= 0 "
            "AND desired_exposure_fraction >= 0 "
            "AND achievable_exposure_fraction >= 0 "
            "AND risk_parity_error >= 0 "
            "AND exposure_parity_error >= 0 "
            "AND parity_error >= 0 AND target_margin_required >= 0",
            name="ck_v2_copy_lifecycle_nonnegative",
        ),
    )
    _create(
        "v2_copy_lifecycle_commands",
        sa.Column("command_id", sa.String(96), primary_key=True),
        sa.Column("idempotency_key", sa.String(96), nullable=False, unique=True),
        sa.Column(
            "intent_id",
            sa.String(96),
            sa.ForeignKey("v2_copy_lifecycle_intents.intent_id"),
            nullable=False,
        ),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("action", sa.String(32), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("contract_id", sa.String(96), nullable=False),
        sa.Column("symbol", sa.String(64), nullable=False),
        sa.Column("exchange", sa.String(32), nullable=False),
        sa.Column("side", sa.String(8), nullable=True),
        sa.Column("quantity", sa.Integer(), nullable=False),
        sa.Column("order_type", sa.String(24), nullable=False),
        sa.Column("duration", sa.String(8), nullable=False),
        sa.Column("limit_price", MONEY, nullable=True),
        sa.Column("trigger_price", MONEY, nullable=True),
        sa.Column("stop_ticks", sa.Integer(), nullable=True),
        sa.Column("target_ticks", sa.Integer(), nullable=True),
        sa.Column("bracket_operation", sa.String(16), nullable=True),
        sa.Column("existing_broker_order_id", sa.String(256), nullable=True),
        sa.Column("broker_order_id", sa.String(256), nullable=True),
        sa.Column(
            "broker_order_ids_json",
            sa.Text(),
            nullable=False,
            server_default=sa.text("'[]'"),
        ),
        sa.Column(
            "depends_on_command_id",
            sa.String(96),
            sa.ForeignKey("v2_copy_lifecycle_commands.command_id"),
            nullable=True,
        ),
        sa.Column(
            "cumulative_filled_quantity",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "attempt_count", sa.Integer(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column(
            "revision", sa.Integer(), nullable=False, server_default=sa.text("1")
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "intent_id", "ordinal", name="uq_v2_copy_command_ordinal"
        ),
        sa.CheckConstraint("ordinal >= 1", name="ck_v2_copy_command_ordinal"),
        sa.CheckConstraint(
            "action IN ('SUBMIT', 'MODIFY', 'CANCEL', 'BRACKET', 'OCO', "
            "'EXIT_POSITION')",
            name="ck_v2_copy_command_action",
        ),
        sa.CheckConstraint(
            "status IN ('PLANNED', 'BLOCKED', 'DISPATCHING', 'DISPATCHED', "
            "'ACKNOWLEDGED', 'PARTIALLY_FILLED', 'FILLED', 'CANCELLED', "
            "'REJECTED', 'FAILED', 'DIVERGED')",
            name="ck_v2_copy_command_status",
        ),
        sa.CheckConstraint(
            "side IS NULL OR side IN ('BUY', 'SELL')",
            name="ck_v2_copy_command_side",
        ),
        sa.CheckConstraint(
            "order_type IN ('MARKET', 'LIMIT', 'STOP_MARKET', 'STOP_LIMIT')",
            name="ck_v2_copy_command_order_type",
        ),
        sa.CheckConstraint(
            "duration IN ('DAY', 'GTC', 'IOC', 'FOK')",
            name="ck_v2_copy_command_duration",
        ),
        sa.CheckConstraint(
            "bracket_operation IS NULL OR bracket_operation IN "
            "('AFOCCA', 'FOCCA', 'CCA', 'FCA')",
            name="ck_v2_copy_command_bracket_operation",
        ),
        sa.CheckConstraint(
            "quantity >= 0 AND cumulative_filled_quantity >= 0 "
            "AND cumulative_filled_quantity <= quantity",
            name="ck_v2_copy_command_quantities",
        ),
        sa.CheckConstraint(
            "attempt_count >= 0 AND revision >= 1",
            name="ck_v2_copy_command_revision",
        ),
    )
    _create(
        "v2_copy_lifecycle_events",
        sa.Column("lifecycle_event_id", sa.String(96), primary_key=True),
        sa.Column("source_receipt_id", sa.String(96), nullable=True, unique=True),
        sa.Column(
            "intent_id",
            sa.String(96),
            sa.ForeignKey("v2_copy_lifecycle_intents.intent_id"),
            nullable=False,
        ),
        sa.Column(
            "command_id",
            sa.String(96),
            sa.ForeignKey("v2_copy_lifecycle_commands.command_id"),
            nullable=True,
        ),
        sa.Column("event_kind", sa.String(32), nullable=False),
        sa.Column("previous_status", sa.String(32), nullable=True),
        sa.Column("resulting_status", sa.String(32), nullable=False),
        sa.Column("command_revision", sa.Integer(), nullable=True),
        sa.Column("broker_order_id", sa.String(256), nullable=True),
        sa.Column(
            "cumulative_filled_quantity",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("fill_price", MONEY, nullable=True),
        sa.Column("rejection_code", sa.String(128), nullable=True),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "event_kind IN ('INTENT_CREATED', 'DISPATCH_STARTED', 'DISPATCHED', "
            "'ACKNOWLEDGED', 'PARTIAL_FILL', 'FILL', 'CANCELLED', "
            "'REJECTED', 'TRANSPORT_FAILURE', 'DIVERGED')",
            name="ck_v2_copy_event_kind",
        ),
        sa.CheckConstraint(
            "resulting_status IN ('PLANNED', 'BLOCKED', 'DISPATCHING', "
            "'DISPATCHED', 'ACKNOWLEDGED', 'PARTIALLY_FILLED', 'FILLED', "
            "'CANCELLED', 'REJECTED', 'FAILED', 'DIVERGED')",
            name="ck_v2_copy_event_resulting_status",
        ),
        sa.CheckConstraint(
            "command_revision IS NULL OR command_revision >= 1",
            name="ck_v2_copy_event_revision",
        ),
        sa.CheckConstraint(
            "cumulative_filled_quantity >= 0", name="ck_v2_copy_event_fill"
        ),
    )
    _index(
        "ix_v2_copy_lifecycle_follower_status",
        "v2_copy_lifecycle_intents",
        ["follower_connection_id", "status", "updated_at"],
    )
    _index(
        "ix_v2_copy_lifecycle_master",
        "v2_copy_lifecycle_intents",
        ["master_order_id", "master_revision"],
    )
    _index(
        "ix_v2_copy_command_intent_status",
        "v2_copy_lifecycle_commands",
        ["intent_id", "status", "ordinal"],
    )
    _index(
        "ix_v2_copy_event_intent_time",
        "v2_copy_lifecycle_events",
        ["intent_id", "recorded_at"],
    )
    _index(
        "ix_v2_copy_event_command_time",
        "v2_copy_lifecycle_events",
        ["command_id", "recorded_at"],
    )
    _install_append_only_trigger()


def downgrade() -> None:
    offline = _offline_mode()
    existing = set() if offline else set(sa.inspect(op.get_bind()).get_table_names())
    for table in (
        "v2_copy_lifecycle_events",
        "v2_copy_lifecycle_commands",
        "v2_copy_lifecycle_intents",
    ):
        if offline or table in existing:
            op.drop_table(table)
    op.execute(
        sa.text("DROP FUNCTION IF EXISTS v2_reject_copy_lifecycle_event_mutation()")
    )
