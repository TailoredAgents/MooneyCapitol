"""Add shadow trader decisions.

Revision ID: 0007_shadow_decisions
Revises: 0006_fill_setup_match_metadata
Create Date: 2026-05-24
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0007_shadow_decisions"
down_revision = "0006_fill_setup_match_metadata"
branch_labels = None
depends_on = None


def _has_table(name: str) -> bool:
    return name in sa.inspect(op.get_bind()).get_table_names()


def _has_index(table_name: str, index_name: str) -> bool:
    if not _has_table(table_name):
        return False
    return index_name in {idx["name"] for idx in sa.inspect(op.get_bind()).get_indexes(table_name)}


def upgrade() -> None:
    if not _has_table("shadow_decisions"):
        op.create_table(
            "shadow_decisions",
            sa.Column("id", sa.BigInteger(), primary_key=True),
            sa.Column("alert_id", sa.BigInteger(), sa.ForeignKey("alerts.id"), nullable=True, unique=True),
            sa.Column("setup_id", sa.BigInteger(), sa.ForeignKey("setups.id"), nullable=True),
            sa.Column("symbol", sa.String(length=16), nullable=False),
            sa.Column("direction", sa.String(length=8), nullable=True),
            sa.Column("alert_type", sa.String(length=16), nullable=False, server_default="trigger"),
            sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("model_version", sa.String(length=64), nullable=False, server_default="shadow-v1"),
            sa.Column("p2r", sa.Float(), nullable=True),
            sa.Column("rr", sa.Float(), nullable=True),
            sa.Column("entry_price", sa.Float(), nullable=True),
            sa.Column("stop_price", sa.Float(), nullable=True),
            sa.Column("target_price", sa.Float(), nullable=True),
            sa.Column("suggested_size_pct", sa.Float(), nullable=True),
            sa.Column("would_take", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("decision", sa.String(length=16), nullable=False, server_default="skip"),
            sa.Column("confidence", sa.String(length=16), nullable=False, server_default="low"),
            sa.Column("reason", sa.Text(), nullable=True),
            sa.Column("reason_json", postgresql.JSONB(), nullable=True),
            sa.Column("payload_json", postgresql.JSONB(), nullable=True),
            sa.Column("status", sa.String(length=32), nullable=False, server_default="observed"),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )

    for index_name, columns in [
        ("ix_shadow_decisions_alert_id", ["alert_id"]),
        ("ix_shadow_decisions_setup_id", ["setup_id"]),
        ("ix_shadow_decisions_symbol", ["symbol"]),
        ("ix_shadow_decisions_observed_at", ["observed_at"]),
        ("ix_shadow_decisions_would_take", ["would_take"]),
        ("ix_shadow_decisions_decision", ["decision"]),
        ("ix_shadow_decisions_confidence", ["confidence"]),
        ("ix_shadow_decisions_status", ["status"]),
        ("ix_shadow_decisions_created_at", ["created_at"]),
        ("idx_shadow_decision_symbol_time", ["symbol", "observed_at"]),
        ("idx_shadow_decision_setup_time", ["setup_id", "observed_at"]),
    ]:
        if not _has_index("shadow_decisions", index_name):
            op.create_index(index_name, "shadow_decisions", columns)


def downgrade() -> None:
    if _has_table("shadow_decisions"):
        op.drop_table("shadow_decisions")
