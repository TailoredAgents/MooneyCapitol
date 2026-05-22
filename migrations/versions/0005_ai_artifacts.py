"""Add AI artifact storage.

Revision ID: 0005_ai_artifacts
Revises: 0004_pnl_monitoring
Create Date: 2026-05-22
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0005_ai_artifacts"
down_revision = "0004_pnl_monitoring"
branch_labels = None
depends_on = None


def _has_table(name: str) -> bool:
    return name in sa.inspect(op.get_bind()).get_table_names()


def _has_index(table_name: str, index_name: str) -> bool:
    return index_name in {idx["name"] for idx in sa.inspect(op.get_bind()).get_indexes(table_name)}


def upgrade() -> None:
    if not _has_table("ai_artifacts"):
        op.create_table(
            "ai_artifacts",
            sa.Column("id", sa.BigInteger(), primary_key=True),
            sa.Column("artifact_type", sa.String(length=64), nullable=False),
            sa.Column("source_type", sa.String(length=64), nullable=False),
            sa.Column("source_id", sa.String(length=128), nullable=True),
            sa.Column("symbol", sa.String(length=16), nullable=True),
            sa.Column("model", sa.String(length=64), nullable=False),
            sa.Column("prompt_version", sa.String(length=64), nullable=False),
            sa.Column("input_json", postgresql.JSONB(), nullable=True),
            sa.Column("output_json", postgresql.JSONB(), nullable=True),
            sa.Column("output_text", sa.Text(), nullable=True),
            sa.Column("status", sa.String(length=32), nullable=False, server_default="created"),
            sa.Column("error", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )

    for index_name, columns in [
        ("ix_ai_artifacts_artifact_type", ["artifact_type"]),
        ("ix_ai_artifacts_source_type", ["source_type"]),
        ("ix_ai_artifacts_source_id", ["source_id"]),
        ("ix_ai_artifacts_symbol", ["symbol"]),
        ("ix_ai_artifacts_status", ["status"]),
        ("ix_ai_artifacts_created_at", ["created_at"]),
        ("idx_ai_artifact_source", ["artifact_type", "source_type", "source_id"]),
        ("idx_ai_artifact_symbol_created", ["symbol", "created_at"]),
    ]:
        if not _has_index("ai_artifacts", index_name):
            op.create_index(index_name, "ai_artifacts", columns)


def downgrade() -> None:
    op.drop_table("ai_artifacts")
