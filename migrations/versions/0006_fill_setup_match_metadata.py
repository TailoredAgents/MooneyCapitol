"""Add fill setup match metadata.

Revision ID: 0006_fill_setup_match_metadata
Revises: 0005_ai_artifacts
Create Date: 2026-05-24
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0006_fill_setup_match_metadata"
down_revision = "0005_ai_artifacts"
branch_labels = None
depends_on = None


def _has_table(name: str) -> bool:
    return name in sa.inspect(op.get_bind()).get_table_names()


def _has_column(table_name: str, column_name: str) -> bool:
    if not _has_table(table_name):
        return False
    return column_name in {col["name"] for col in sa.inspect(op.get_bind()).get_columns(table_name)}


def _has_index(table_name: str, index_name: str) -> bool:
    if not _has_table(table_name):
        return False
    return index_name in {idx["name"] for idx in sa.inspect(op.get_bind()).get_indexes(table_name)}


def upgrade() -> None:
    if not _has_column("fills", "setup_match_score"):
        op.add_column("fills", sa.Column("setup_match_score", sa.Float(), nullable=True))
    if not _has_column("fills", "setup_match_confidence"):
        op.add_column("fills", sa.Column("setup_match_confidence", sa.String(length=16), nullable=True))
    if not _has_column("fills", "setup_match_reason"):
        op.add_column("fills", sa.Column("setup_match_reason", postgresql.JSONB(), nullable=True))
    if not _has_index("fills", "ix_fills_setup_match_confidence"):
        op.create_index("ix_fills_setup_match_confidence", "fills", ["setup_match_confidence"])


def downgrade() -> None:
    if _has_index("fills", "ix_fills_setup_match_confidence"):
        op.drop_index("ix_fills_setup_match_confidence", table_name="fills")
    if _has_column("fills", "setup_match_reason"):
        op.drop_column("fills", "setup_match_reason")
    if _has_column("fills", "setup_match_confidence"):
        op.drop_column("fills", "setup_match_confidence")
    if _has_column("fills", "setup_match_score"):
        op.drop_column("fills", "setup_match_score")
