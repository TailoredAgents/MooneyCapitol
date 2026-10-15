"""Add fenced copy delivery, reconciliation, and latency persistence.

Revision ID: 0014_v2_execution_recovery
Revises: 0013_v2_copy_lifecycle
Create Date: 2026-10-15
"""

from __future__ import annotations

from alembic import context, op
import sqlalchemy as sa


revision = "0014_v2_execution_recovery"
down_revision = "0013_v2_copy_lifecycle"
branch_labels = None
depends_on = None


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


def _install_append_only_triggers() -> None:
    for table, function, message in (
        (
            "v2_copy_delivery_events",
            "v2_reject_copy_delivery_event_mutation",
            "V2 copy delivery history is append-only",
        ),
        (
            "v2_copy_reconciliation_findings",
            "v2_reject_copy_reconciliation_finding_mutation",
            "V2 copy reconciliation findings are append-only",
        ),
    ):
        op.execute(
            sa.text(
                f"""
                CREATE OR REPLACE FUNCTION {function}()
                RETURNS trigger AS $$
                BEGIN
                    RAISE EXCEPTION '{message}';
                END;
                $$ LANGUAGE plpgsql;
                """
            )
        )
        op.execute(
            sa.text(
                f"DROP TRIGGER IF EXISTS trg_{table}_append_only ON {table}"
            )
        )
        op.execute(
            sa.text(
                f"CREATE TRIGGER trg_{table}_append_only "
                f"BEFORE UPDATE OR DELETE ON {table} FOR EACH ROW "
                f"EXECUTE FUNCTION {function}()"
            )
        )


def _backfill_existing_commands() -> None:
    op.execute(
        sa.text(
            """
            INSERT INTO v2_copy_outbox (
                outbox_id, command_id, intent_id, follower_connection_id,
                status, claim_owner_id, claim_fence, claim_expires_at,
                attempt_count, last_error_code, enqueued_at, claimed_at,
                submission_started_at, dispatch_response_at, confirmed_at,
                updated_at
            )
            SELECT
                'ob_' || c.command_id,
                c.command_id,
                c.intent_id,
                i.follower_connection_id,
                CASE
                    WHEN c.status = 'PLANNED' THEN 'PENDING'
                    WHEN c.status = 'DISPATCHING' THEN 'UNCERTAIN'
                    WHEN c.status = 'FAILED' THEN 'FAILED'
                    WHEN c.status = 'DIVERGED' THEN 'DIVERGED'
                    WHEN c.status IN ('ACKNOWLEDGED', 'PARTIALLY_FILLED',
                                      'FILLED', 'CANCELLED', 'REJECTED')
                        THEN 'CONFIRMED'
                    ELSE 'SENT'
                END,
                NULL, NULL, NULL,
                c.attempt_count,
                CASE WHEN c.status = 'DISPATCHING'
                     THEN 'migration_requires_reconciliation' ELSE NULL END,
                c.created_at, NULL, NULL,
                CASE WHEN c.status = 'DISPATCHED' THEN c.updated_at ELSE NULL END,
                CASE WHEN c.status IN ('ACKNOWLEDGED', 'PARTIALLY_FILLED',
                                       'FILLED', 'CANCELLED', 'REJECTED')
                     THEN c.updated_at ELSE NULL END,
                c.updated_at
            FROM v2_copy_lifecycle_commands c
            JOIN v2_copy_lifecycle_intents i ON i.intent_id = c.intent_id
            ON CONFLICT (command_id) DO NOTHING
            """
        )
    )
    op.execute(
        sa.text(
            """
            INSERT INTO v2_copy_delivery_events (
                delivery_event_id, outbox_id, command_id, event_kind,
                previous_status, resulting_status, owner_id, fence,
                error_code, occurred_at, recorded_at
            )
            SELECT
                'de_' || c.command_id,
                'ob_' || c.command_id,
                c.command_id,
                'ENQUEUED', NULL, o.status, NULL, NULL,
                o.last_error_code, c.created_at, c.created_at
            FROM v2_copy_lifecycle_commands c
            JOIN v2_copy_outbox o ON o.command_id = c.command_id
            ON CONFLICT (delivery_event_id) DO NOTHING
            """
        )
    )
    op.execute(
        sa.text(
            """
            INSERT INTO v2_copy_latency (
                command_id, intent_id, follower_connection_id,
                event_occurred_at, event_received_at, decision_at, enqueued_at,
                submission_started_at, dispatch_response_at, acknowledged_at,
                first_fill_at, completed_at, updated_at
            )
            SELECT
                c.command_id, c.intent_id, i.follower_connection_id,
                COALESCE(e.occurred_at, i.created_at),
                COALESCE(e.occurred_at, i.created_at),
                i.created_at, c.created_at, NULL,
                CASE WHEN c.status = 'DISPATCHED' THEN c.updated_at ELSE NULL END,
                NULL, NULL,
                CASE WHEN c.status IN ('FILLED', 'CANCELLED', 'REJECTED',
                                       'FAILED', 'DIVERGED')
                     THEN c.updated_at ELSE NULL END,
                c.updated_at
            FROM v2_copy_lifecycle_commands c
            JOIN v2_copy_lifecycle_intents i ON i.intent_id = c.intent_id
            LEFT JOIN v2_copy_lifecycle_events e
              ON e.intent_id = i.intent_id AND e.event_kind = 'INTENT_CREATED'
            ON CONFLICT (command_id) DO NOTHING
            """
        )
    )


def upgrade() -> None:
    _create(
        "v2_copy_outbox",
        sa.Column("outbox_id", sa.String(96), primary_key=True),
        sa.Column(
            "command_id",
            sa.String(96),
            sa.ForeignKey("v2_copy_lifecycle_commands.command_id"),
            nullable=False,
            unique=True,
        ),
        sa.Column(
            "intent_id",
            sa.String(96),
            sa.ForeignKey("v2_copy_lifecycle_intents.intent_id"),
            nullable=False,
        ),
        sa.Column("follower_connection_id", sa.String(96), nullable=False),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("claim_owner_id", sa.String(96), nullable=True),
        sa.Column("claim_fence", sa.BigInteger(), nullable=True),
        sa.Column("claim_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "attempt_count", sa.Integer(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column("last_error_code", sa.String(128), nullable=True),
        sa.Column("enqueued_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "submission_started_at", sa.DateTime(timezone=True), nullable=True
        ),
        sa.Column("dispatch_response_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("confirmed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "status IN ('PENDING', 'CLAIMED', 'SENT', 'CONFIRMED', "
            "'UNCERTAIN', 'DIVERGED', 'FAILED')",
            name="ck_v2_copy_outbox_status",
        ),
        sa.CheckConstraint(
            "follower_connection_id IN "
            "('follower-1', 'follower-2', 'follower-3', 'follower-4')",
            name="ck_v2_copy_outbox_follower",
        ),
        sa.CheckConstraint(
            "attempt_count >= 0", name="ck_v2_copy_outbox_attempt_count"
        ),
        sa.CheckConstraint(
            "(claim_owner_id IS NULL AND claim_fence IS NULL "
            "AND claim_expires_at IS NULL) OR "
            "(claim_owner_id IS NOT NULL AND claim_fence IS NOT NULL "
            "AND claim_expires_at IS NOT NULL)",
            name="ck_v2_copy_outbox_claim_identity",
        ),
    )
    _create(
        "v2_copy_delivery_events",
        sa.Column("delivery_event_id", sa.String(96), primary_key=True),
        sa.Column(
            "outbox_id",
            sa.String(96),
            sa.ForeignKey("v2_copy_outbox.outbox_id"),
            nullable=False,
        ),
        sa.Column(
            "command_id",
            sa.String(96),
            sa.ForeignKey("v2_copy_lifecycle_commands.command_id"),
            nullable=False,
        ),
        sa.Column("event_kind", sa.String(32), nullable=False),
        sa.Column("previous_status", sa.String(24), nullable=True),
        sa.Column("resulting_status", sa.String(24), nullable=False),
        sa.Column("owner_id", sa.String(96), nullable=True),
        sa.Column("fence", sa.BigInteger(), nullable=True),
        sa.Column("error_code", sa.String(128), nullable=True),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "event_kind IN ('ENQUEUED', 'CLAIMED', 'SUBMISSION_STARTED', "
            "'SENT', 'CONFIRMED', 'UNCERTAIN', 'DIVERGED', 'FAILED')",
            name="ck_v2_copy_delivery_event_kind",
        ),
        sa.CheckConstraint(
            "resulting_status IN ('PENDING', 'CLAIMED', 'SENT', 'CONFIRMED', "
            "'UNCERTAIN', 'DIVERGED', 'FAILED')",
            name="ck_v2_copy_delivery_event_status",
        ),
        sa.CheckConstraint(
            "fence IS NULL OR fence >= 1", name="ck_v2_copy_delivery_event_fence"
        ),
    )
    _create(
        "v2_copy_reconciliation_runs",
        sa.Column("reconciliation_run_id", sa.String(96), primary_key=True),
        sa.Column(
            "profile_id",
            sa.String(96),
            sa.ForeignKey("v2_copy_profiles.profile_id"),
            nullable=False,
        ),
        sa.Column("follower_connection_id", sa.String(96), nullable=False),
        sa.Column("snapshot_id", sa.String(96), nullable=False, unique=True),
        sa.Column("connection_generation", sa.String(96), nullable=False),
        sa.Column("lease_owner_id", sa.String(96), nullable=False),
        sa.Column("lease_fence", sa.BigInteger(), nullable=False),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column(
            "latest_intent_id",
            sa.String(96),
            sa.ForeignKey("v2_copy_lifecycle_intents.intent_id"),
            nullable=True,
        ),
        sa.Column("expected_summary_json", sa.Text(), nullable=False),
        sa.Column("actual_summary_json", sa.Text(), nullable=False),
        sa.Column(
            "adopted_receipt_count",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "finding_count",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("snapshot_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "follower_connection_id IN "
            "('follower-1', 'follower-2', 'follower-3', 'follower-4')",
            name="ck_v2_copy_reconciliation_follower",
        ),
        sa.CheckConstraint(
            "status IN ('CLEAN', 'ADOPTED', 'LOCKED', 'INCOMPLETE')",
            name="ck_v2_copy_reconciliation_status",
        ),
        sa.CheckConstraint(
            "lease_fence >= 1 AND adopted_receipt_count >= 0 "
            "AND finding_count >= 0",
            name="ck_v2_copy_reconciliation_counts",
        ),
    )
    _create(
        "v2_copy_reconciliation_findings",
        sa.Column("finding_id", sa.String(96), primary_key=True),
        sa.Column(
            "reconciliation_run_id",
            sa.String(96),
            sa.ForeignKey("v2_copy_reconciliation_runs.reconciliation_run_id"),
            nullable=False,
        ),
        sa.Column(
            "command_id",
            sa.String(96),
            sa.ForeignKey("v2_copy_lifecycle_commands.command_id"),
            nullable=True,
        ),
        sa.Column("finding_kind", sa.String(64), nullable=False),
        sa.Column("severity", sa.String(16), nullable=False),
        sa.Column("expected_json", sa.Text(), nullable=False),
        sa.Column("actual_json", sa.Text(), nullable=False),
        sa.Column("detected_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "severity IN ('WARNING', 'BLOCKING')",
            name="ck_v2_copy_reconciliation_finding_severity",
        ),
    )
    _create(
        "v2_copy_latency",
        sa.Column(
            "command_id",
            sa.String(96),
            sa.ForeignKey("v2_copy_lifecycle_commands.command_id"),
            primary_key=True,
        ),
        sa.Column(
            "intent_id",
            sa.String(96),
            sa.ForeignKey("v2_copy_lifecycle_intents.intent_id"),
            nullable=False,
        ),
        sa.Column("follower_connection_id", sa.String(96), nullable=False),
        sa.Column("event_occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("event_received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("decision_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("enqueued_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "submission_started_at", sa.DateTime(timezone=True), nullable=True
        ),
        sa.Column("dispatch_response_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("acknowledged_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("first_fill_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "follower_connection_id IN "
            "('follower-1', 'follower-2', 'follower-3', 'follower-4')",
            name="ck_v2_copy_latency_follower",
        ),
    )
    _index(
        "ix_v2_copy_outbox_follower_status",
        "v2_copy_outbox",
        ["follower_connection_id", "status", "enqueued_at"],
    )
    _index(
        "ix_v2_copy_delivery_event_outbox_time",
        "v2_copy_delivery_events",
        ["outbox_id", "recorded_at"],
    )
    _index(
        "ix_v2_copy_reconciliation_follower_time",
        "v2_copy_reconciliation_runs",
        ["follower_connection_id", "completed_at"],
    )
    _index(
        "ix_v2_copy_reconciliation_finding_run",
        "v2_copy_reconciliation_findings",
        ["reconciliation_run_id", "detected_at"],
    )
    _index(
        "ix_v2_copy_latency_follower_event",
        "v2_copy_latency",
        ["follower_connection_id", "event_occurred_at"],
    )
    _backfill_existing_commands()
    _install_append_only_triggers()


def downgrade() -> None:
    offline = _offline_mode()
    existing = set() if offline else set(sa.inspect(op.get_bind()).get_table_names())
    for table in (
        "v2_copy_latency",
        "v2_copy_reconciliation_findings",
        "v2_copy_reconciliation_runs",
        "v2_copy_delivery_events",
        "v2_copy_outbox",
    ):
        if offline or table in existing:
            op.drop_table(table)
    op.execute(
        sa.text("DROP FUNCTION IF EXISTS v2_reject_copy_delivery_event_mutation()")
    )
    op.execute(
        sa.text(
            "DROP FUNCTION IF EXISTS "
            "v2_reject_copy_reconciliation_finding_mutation()"
        )
    )
