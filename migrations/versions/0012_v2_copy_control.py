"""Add the durable V2 copy-control plane.

Revision ID: 0012_v2_copy_control
Revises: 0011_rithmic_read_capture
Create Date: 2026-10-14
"""

from __future__ import annotations

from alembic import context, op
import sqlalchemy as sa


revision = "0012_v2_copy_control"
down_revision = "0011_rithmic_read_capture"
branch_labels = None
depends_on = None

_REASON_CODES_SQL = (
    "'operator_requested', 'simulation', 'scheduled_pause', "
    "'connection_issue', 'risk_review', 'reconciliation_issue', "
    "'emergency', 'recovery', 'system_confirmed_flat'"
)


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


def _seed_default_profile() -> None:
    op.execute(
        sa.text(
            """
            INSERT INTO v2_copy_profiles (
                profile_id, name, master_connection_id, global_paused,
                revision, created_at, updated_at
            ) VALUES (
                'mooney-five-account', 'Mooney five-account copier', 'master',
                TRUE, 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
            ) ON CONFLICT (profile_id) DO NOTHING
            """
        )
    )
    for follower_number in range(1, 5):
        op.execute(
            sa.text(
                """
                INSERT INTO v2_copy_follower_controls (
                    profile_id, connection_id, follower_number, state,
                    revision, updated_at
                ) VALUES (
                    'mooney-five-account', :connection_id, :follower_number,
                    'DISABLED', 1, CURRENT_TIMESTAMP
                ) ON CONFLICT (profile_id, connection_id) DO NOTHING
                """
            ).bindparams(
                connection_id=f"follower-{follower_number}",
                follower_number=follower_number,
            )
        )


def _install_audit_trigger() -> None:
    op.execute(
        sa.text(
            """
            CREATE OR REPLACE FUNCTION v2_reject_copy_control_audit_mutation()
            RETURNS trigger AS $$
            BEGIN
                RAISE EXCEPTION 'V2 copy-control audit history is append-only';
            END;
            $$ LANGUAGE plpgsql;
            """
        )
    )
    op.execute(
        sa.text(
            "DROP TRIGGER IF EXISTS "
            "trg_v2_copy_control_audit_append_only ON v2_copy_control_audit"
        )
    )
    op.execute(
        sa.text(
            "CREATE TRIGGER trg_v2_copy_control_audit_append_only "
            "BEFORE UPDATE OR DELETE ON v2_copy_control_audit FOR EACH ROW "
            "EXECUTE FUNCTION v2_reject_copy_control_audit_mutation()"
        )
    )


def upgrade() -> None:
    _create(
        "v2_copy_profiles",
        sa.Column("profile_id", sa.String(96), primary_key=True),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("master_connection_id", sa.String(96), nullable=False),
        sa.Column(
            "global_paused",
            sa.Boolean(),
            nullable=False,
            server_default=sa.true(),
        ),
        sa.Column(
            "revision",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("1"),
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("revision >= 1", name="ck_v2_copy_profile_revision"),
        sa.CheckConstraint(
            "profile_id = 'mooney-five-account'",
            name="ck_v2_copy_profile_identity",
        ),
        sa.CheckConstraint(
            "master_connection_id = 'master'",
            name="ck_v2_copy_profile_master",
        ),
    )
    _create(
        "v2_copy_follower_controls",
        sa.Column(
            "profile_id",
            sa.String(96),
            sa.ForeignKey("v2_copy_profiles.profile_id"),
            primary_key=True,
        ),
        sa.Column("connection_id", sa.String(96), primary_key=True),
        sa.Column("follower_number", sa.Integer(), nullable=False),
        sa.Column("state", sa.String(32), nullable=False),
        sa.Column(
            "revision",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("1"),
        ),
        sa.Column("reason_code", sa.String(64), nullable=True),
        sa.Column("flatten_requested_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("locked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "profile_id",
            "follower_number",
            name="uq_v2_copy_follower_number",
        ),
        sa.CheckConstraint(
            "follower_number BETWEEN 1 AND 4",
            name="ck_v2_copy_follower_number",
        ),
        sa.CheckConstraint(
            "(connection_id = 'follower-1' AND follower_number = 1) OR "
            "(connection_id = 'follower-2' AND follower_number = 2) OR "
            "(connection_id = 'follower-3' AND follower_number = 3) OR "
            "(connection_id = 'follower-4' AND follower_number = 4)",
            name="ck_v2_copy_follower_identity",
        ),
        sa.CheckConstraint(
            "state IN ('DISABLED', 'ENABLED', 'PAUSED', "
            "'FLATTEN_REQUESTED', 'LOCKED')",
            name="ck_v2_copy_follower_state",
        ),
        sa.CheckConstraint("revision >= 1", name="ck_v2_copy_follower_revision"),
        sa.CheckConstraint(
            "reason_code IS NULL OR reason_code IN (" + _REASON_CODES_SQL + ")",
            name="ck_v2_copy_follower_reason",
        ),
    )
    _create(
        "v2_copy_control_audit",
        sa.Column("audit_id", sa.String(96), primary_key=True),
        sa.Column(
            "profile_id",
            sa.String(96),
            sa.ForeignKey("v2_copy_profiles.profile_id"),
            nullable=False,
        ),
        sa.Column("connection_id", sa.String(96), nullable=True),
        sa.Column("action", sa.String(64), nullable=False),
        sa.Column("actor", sa.String(128), nullable=False),
        sa.Column("auth_method", sa.String(32), nullable=False),
        sa.Column("expected_revision", sa.Integer(), nullable=False),
        sa.Column("previous_revision", sa.Integer(), nullable=False),
        sa.Column("resulting_revision", sa.Integer(), nullable=False),
        sa.Column("previous_state", sa.String(32), nullable=False),
        sa.Column("resulting_state", sa.String(32), nullable=False),
        sa.Column("reason_code", sa.String(64), nullable=True),
        sa.Column("request_id", sa.String(96), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("request_id", name="uq_v2_copy_control_audit_request"),
        sa.CheckConstraint(
            "connection_id IS NULL OR connection_id IN "
            "('follower-1', 'follower-2', 'follower-3', 'follower-4')",
            name="ck_v2_copy_control_audit_connection",
        ),
        sa.CheckConstraint(
            "action IN ('ENABLE', 'PAUSE', 'REQUEST_FLATTEN_AND_DISABLE', "
            "'LOCK', 'UNLOCK_TO_PAUSED', 'CONFIRM_FLAT_AND_DISABLE', "
            "'SET_GLOBAL_PAUSE', 'CLEAR_GLOBAL_PAUSE')",
            name="ck_v2_copy_control_audit_action",
        ),
        sa.CheckConstraint(
            "auth_method IN ('api_token', 'basic', 'dashboard_session', "
            "'local_unconfigured', 'system')",
            name="ck_v2_copy_control_audit_auth",
        ),
        sa.CheckConstraint(
            "reason_code IS NULL OR reason_code IN (" + _REASON_CODES_SQL + ")",
            name="ck_v2_copy_control_audit_reason",
        ),
        sa.CheckConstraint(
            "expected_revision >= 1 AND previous_revision >= 1 "
            "AND resulting_revision > previous_revision",
            name="ck_v2_copy_control_audit_revisions",
        ),
    )
    _index(
        "ix_v2_copy_follower_profile_state",
        "v2_copy_follower_controls",
        ["profile_id", "state"],
    )
    _index(
        "ix_v2_copy_control_audit_profile_time",
        "v2_copy_control_audit",
        ["profile_id", "occurred_at"],
    )
    _index(
        "ix_v2_copy_control_audit_connection_time",
        "v2_copy_control_audit",
        ["connection_id", "occurred_at"],
    )
    _seed_default_profile()
    _install_audit_trigger()


def downgrade() -> None:
    offline = _offline_mode()
    existing = set() if offline else set(sa.inspect(op.get_bind()).get_table_names())
    for table in (
        "v2_copy_control_audit",
        "v2_copy_follower_controls",
        "v2_copy_profiles",
    ):
        if offline or table in existing:
            op.drop_table(table)
    op.execute(sa.text("DROP FUNCTION IF EXISTS v2_reject_copy_control_audit_mutation()"))
