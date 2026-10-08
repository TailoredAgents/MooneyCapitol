"""Add Conner NQ observation and shadow Scout persistence.

Revision ID: 0010_conner_nq_scout
Revises: 0009_v2_futures_foundation
Create Date: 2026-10-05
"""

from __future__ import annotations

from alembic import context, op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "0010_conner_nq_scout"
down_revision = "0009_v2_futures_foundation"
branch_labels = None
depends_on = None

VALUE = sa.Numeric(24, 10)
JSON = postgresql.JSONB()


def _offline_mode() -> bool:
    try:
        return context.is_offline_mode()
    except NameError:
        # Direct unit invocation has no Alembic EnvironmentContext proxy.
        return False


def _create(name: str, *columns, **kwargs) -> None:
    if _offline_mode() or name not in sa.inspect(op.get_bind()).get_table_names():
        op.create_table(name, *columns, **kwargs)


def _index(name: str, table: str, columns: list[str]) -> None:
    if _offline_mode():
        op.create_index(name, table, columns)
        return
    inspector = sa.inspect(op.get_bind())
    if table in inspector.get_table_names() and name not in {item["name"] for item in inspector.get_indexes(table)}:
        op.create_index(name, table, columns)


def upgrade() -> None:
    _create(
        "v2_observation_specs",
        sa.Column("observation_spec_id", sa.String(96), primary_key=True),
        sa.Column("strategy_namespace", sa.String(64), nullable=False),
        sa.Column("version", sa.String(64), nullable=False),
        sa.Column("feature_schema_version", sa.String(64), nullable=False),
        sa.Column("horizon_set_version", sa.String(64), nullable=False),
        sa.Column("synchronization_policy_version", sa.String(64), nullable=False),
        sa.Column("timeframes_json", JSON, nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("measurement_versions", JSON, nullable=False),
        sa.Column("required_capabilities", JSON, nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("strategy_window_candidates", JSON, nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("synchronization_tolerance_ms", sa.Integer(), nullable=False),
        sa.Column("parameters_json", JSON, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("strategy_namespace", "version", name="uq_v2_observation_spec_version"),
        sa.CheckConstraint("synchronization_tolerance_ms >= 0", name="ck_v2_observation_spec_sync"),
        sa.CheckConstraint(
            "jsonb_typeof(timeframes_json) = 'array' AND timeframes_json <> '[]'::jsonb "
            "AND jsonb_typeof(required_capabilities) = 'array' "
            "AND required_capabilities <> '[]'::jsonb "
            "AND jsonb_typeof(measurement_versions) = 'object' "
            "AND measurement_versions <> '{}'::jsonb "
            "AND jsonb_typeof(strategy_window_candidates) = 'array'",
            name="ck_v2_observation_spec_shapes",
        ),
    )
    _create(
        "v2_observation_horizons",
        sa.Column("horizon_id", sa.String(96), primary_key=True),
        sa.Column(
            "observation_spec_id",
            sa.String(96),
            sa.ForeignKey("v2_observation_specs.observation_spec_id"),
            nullable=False,
        ),
        sa.Column("name", sa.String(64), nullable=False),
        sa.Column("offset_seconds", sa.Integer(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("observation_spec_id", "name", name="uq_v2_observation_horizon_name"),
        sa.UniqueConstraint("observation_spec_id", "offset_seconds", name="uq_v2_observation_horizon_offset"),
        sa.CheckConstraint("offset_seconds >= 0", name="ck_v2_observation_horizon_offset"),
        sa.CheckConstraint("ordinal >= 0", name="ck_v2_observation_horizon_ordinal"),
    )
    _create(
        "v2_synchronized_market_states",
        sa.Column("market_state_id", sa.String(96), primary_key=True),
        sa.Column(
            "observation_spec_id",
            sa.String(96),
            sa.ForeignKey("v2_observation_specs.observation_spec_id"),
            nullable=False,
        ),
        sa.Column(
            "nq_contract_id",
            sa.String(96),
            sa.ForeignKey("v2_futures_contracts.contract_id"),
            nullable=False,
        ),
        sa.Column(
            "es_contract_id",
            sa.String(96),
            sa.ForeignKey("v2_futures_contracts.contract_id"),
            nullable=False,
        ),
        sa.Column("trade_date", sa.Date(), nullable=False),
        sa.Column("schema_name", sa.String(64), nullable=False),
        sa.Column("schema_version", sa.String(64), nullable=False),
        sa.Column("feature_cutoff_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("nq_source_max_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("es_source_max_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("nq_available_max_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("es_available_max_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("availability_mode", sa.String(32), nullable=False),
        sa.Column("captured_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("observation_purpose", sa.String(32), nullable=False),
        sa.Column("provider", sa.String(64), nullable=False),
        sa.Column("provider_dataset", sa.String(128), nullable=True),
        sa.Column("replay_reference", sa.String(256), nullable=True),
        sa.Column("synchronization_delta_ms", sa.Integer(), nullable=False),
        sa.Column("nq_state", JSON, nullable=False),
        sa.Column("es_state", JSON, nullable=False),
        sa.Column("source_lineage", JSON, nullable=False),
        sa.Column("data_quality", JSON, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("is_complete", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.UniqueConstraint(
            "observation_spec_id",
            "nq_contract_id",
            "es_contract_id",
            "feature_cutoff_at",
            name="uq_v2_market_state_cutoff",
        ),
        sa.CheckConstraint("nq_contract_id <> es_contract_id", name="ck_v2_market_state_contracts"),
        sa.CheckConstraint("synchronization_delta_ms >= 0", name="ck_v2_market_state_sync"),
        sa.CheckConstraint(
            "nq_source_max_at <= feature_cutoff_at AND es_source_max_at <= feature_cutoff_at",
            name="ck_v2_market_state_source_cutoff",
        ),
        sa.CheckConstraint(
            "nq_source_max_at <= nq_available_max_at "
            "AND es_source_max_at <= es_available_max_at",
            name="ck_v2_market_state_temporal_order",
        ),
        sa.CheckConstraint(
            "availability_mode IN ('POINT_IN_TIME', 'FINALIZED_HISTORICAL')",
            name="ck_v2_market_state_availability_mode",
        ),
        sa.CheckConstraint(
            "(availability_mode = 'POINT_IN_TIME' "
            "AND nq_available_max_at <= feature_cutoff_at "
            "AND es_available_max_at <= feature_cutoff_at) OR "
            "(availability_mode = 'FINALIZED_HISTORICAL' "
            "AND observation_purpose IN ('OUTCOME_RESEARCH', 'HISTORICAL_REPLAY'))",
            name="ck_v2_market_state_available_cutoff",
        ),
        sa.CheckConstraint(
            "feature_cutoff_at <= captured_at "
            "AND nq_available_max_at <= captured_at "
            "AND es_available_max_at <= captured_at",
            name="ck_v2_market_state_capture",
        ),
        sa.CheckConstraint(
            "observation_purpose IN ('LIVE_SHADOW', 'BEHAVIOR_TRAINING', "
            "'OUTCOME_RESEARCH', 'HISTORICAL_REPLAY')",
            name="ck_v2_market_state_purpose",
        ),
        sa.CheckConstraint("source_lineage <> '{}'::jsonb", name="ck_v2_market_state_lineage"),
    )
    _create(
        "v2_candidate_feature_definitions",
        sa.Column("feature_definition_id", sa.String(96), primary_key=True),
        sa.Column("strategy_namespace", sa.String(64), nullable=False),
        sa.Column("family", sa.String(64), nullable=False),
        sa.Column("name", sa.String(128), nullable=False),
        sa.Column("version", sa.String(64), nullable=False),
        sa.Column("value_type", sa.String(32), nullable=False),
        sa.Column("source_algorithm", sa.String(128), nullable=False),
        sa.Column("parameters_json", JSON, nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column(
            "research_status",
            sa.String(32),
            nullable=False,
            server_default="CANDIDATE",
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "strategy_namespace", "family", "name", "version", name="uq_v2_feature_definition_version"
        ),
        sa.CheckConstraint(
            "research_status IN ('CANDIDATE', 'EMPIRICALLY_SUPPORTED', 'RETIRED')",
            name="ck_v2_feature_definition_research_status",
        ),
    )
    _create(
        "v2_candidate_feature_values",
        sa.Column("measurement_id", sa.String(96), primary_key=True),
        sa.Column(
            "market_state_id",
            sa.String(96),
            sa.ForeignKey("v2_synchronized_market_states.market_state_id"),
            nullable=False,
        ),
        sa.Column(
            "feature_definition_id",
            sa.String(96),
            sa.ForeignKey("v2_candidate_feature_definitions.feature_definition_id"),
            nullable=False,
        ),
        sa.Column("candidate_key", sa.String(96), nullable=False, server_default="default"),
        sa.Column("instrument_role", sa.String(16), nullable=False),
        sa.Column("feature_cutoff_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("source_window_start_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("source_window_end_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("max_input_event_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("max_input_available_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("definition_available_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("computed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("availability_mode", sa.String(32), nullable=False),
        sa.Column("value_json", JSON, nullable=True),
        sa.Column("units", sa.String(32), nullable=True),
        sa.Column("parameters_json", JSON, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("source_event_ids", JSON, nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("is_missing", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("missing_reason", sa.Text(), nullable=True),
        sa.Column("source_lineage", JSON, nullable=False),
        sa.Column("data_quality", JSON, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.UniqueConstraint(
            "market_state_id",
            "feature_definition_id",
            "candidate_key",
            "instrument_role",
            name="uq_v2_candidate_feature_value",
        ),
        sa.CheckConstraint("instrument_role IN ('NQ', 'ES', 'CROSS')", name="ck_v2_feature_value_role"),
        sa.CheckConstraint(
            "max_input_event_at <= feature_cutoff_at "
            "AND definition_available_at <= feature_cutoff_at "
            "AND computed_at >= max_input_event_at "
            "AND computed_at >= definition_available_at "
            "AND (max_input_available_at IS NULL OR computed_at >= max_input_available_at)",
            name="ck_v2_feature_value_cutoff",
        ),
        sa.CheckConstraint(
            "availability_mode IN ('POINT_IN_TIME', 'FINALIZED_HISTORICAL')",
            name="ck_v2_feature_value_availability_mode",
        ),
        sa.CheckConstraint(
            "availability_mode <> 'POINT_IN_TIME' OR "
            "(max_input_available_at IS NOT NULL AND max_input_available_at <= feature_cutoff_at)",
            name="ck_v2_feature_value_available_at",
        ),
        sa.CheckConstraint(
            "(source_window_start_at IS NULL AND source_window_end_at IS NULL) OR "
            "(source_window_start_at IS NOT NULL AND source_window_end_at IS NOT NULL "
            "AND source_window_start_at <= source_window_end_at "
            "AND source_window_end_at <= feature_cutoff_at)",
            name="ck_v2_feature_value_window",
        ),
        sa.CheckConstraint(
            "(is_missing AND value_json IS NULL AND missing_reason IS NOT NULL) OR "
            "(NOT is_missing AND value_json IS NOT NULL)",
            name="ck_v2_feature_value_missing",
        ),
        sa.CheckConstraint("source_lineage <> '{}'::jsonb", name="ck_v2_feature_value_lineage"),
    )
    _create(
        "v2_scout_opportunities",
        sa.Column(
            "opportunity_id",
            sa.String(96),
            sa.ForeignKey("v2_opportunity_snapshots.opportunity_id"),
            primary_key=True,
        ),
        sa.Column(
            "observation_spec_id",
            sa.String(96),
            sa.ForeignKey("v2_observation_specs.observation_spec_id"),
            nullable=False,
        ),
        sa.Column(
            "market_state_id",
            sa.String(96),
            sa.ForeignKey("v2_synchronized_market_states.market_state_id"),
            nullable=False,
        ),
        sa.Column(
            "nq_contract_id",
            sa.String(96),
            sa.ForeignKey("v2_futures_contracts.contract_id"),
            nullable=False,
        ),
        sa.Column(
            "es_contract_id",
            sa.String(96),
            sa.ForeignKey("v2_futures_contracts.contract_id"),
            nullable=False,
        ),
        sa.Column("strategy_namespace", sa.String(64), nullable=False),
        sa.Column("candidate_series_id", sa.String(96), nullable=False),
        sa.Column("candidate_key", sa.String(96), nullable=False),
        sa.Column("candidate_generator", sa.String(128), nullable=False),
        sa.Column("candidate_generator_version", sa.String(64), nullable=False),
        sa.Column("generator_parameters", JSON, nullable=False),
        sa.Column("first_observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("latest_observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("candidate_direction", sa.String(8), nullable=True),
        sa.Column("seen_by_conner", sa.String(24), nullable=False, server_default="UNKNOWN"),
        sa.Column("consideration_status", sa.String(32), nullable=False, server_default="UNKNOWN"),
        sa.Column("behavior_label_state", sa.String(32), nullable=False, server_default="UNLABELED"),
        sa.Column(
            "linked_master_trade_id",
            sa.String(96),
            sa.ForeignKey("v2_master_trades.trade_id"),
            nullable=True,
        ),
        sa.Column("evidence_json", JSON, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("source_lineage", JSON, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "strategy_namespace",
            "market_state_id",
            "candidate_key",
            "candidate_generator_version",
            name="uq_v2_scout_opportunity_candidate",
        ),
        sa.CheckConstraint("nq_contract_id <> es_contract_id", name="ck_v2_scout_opportunity_contracts"),
        sa.CheckConstraint(
            "first_observed_at <= latest_observed_at", name="ck_v2_scout_opportunity_observed_at"
        ),
        sa.CheckConstraint(
            "candidate_direction IS NULL OR candidate_direction IN ('LONG', 'SHORT')",
            name="ck_v2_scout_opportunity_direction",
        ),
        sa.CheckConstraint(
            "seen_by_conner IN ('UNKNOWN', 'CONFIRMED_SEEN', 'CONFIRMED_NOT_SEEN')",
            name="ck_v2_scout_opportunity_seen",
        ),
        sa.CheckConstraint(
            "consideration_status IN ('UNKNOWN', 'PRESENTED', 'CONFIRMED_CONSIDERED')",
            name="ck_v2_scout_opportunity_consideration",
        ),
        sa.CheckConstraint(
            "behavior_label_state IN ('UNLABELED', 'POSITIVE_TRADE', 'EXPLICIT_PASS')",
            name="ck_v2_scout_opportunity_behavior_state",
        ),
        sa.CheckConstraint(
            "(behavior_label_state = 'UNLABELED' AND linked_master_trade_id IS NULL) OR "
            "(behavior_label_state = 'POSITIVE_TRADE' AND linked_master_trade_id IS NOT NULL "
            "AND seen_by_conner = 'CONFIRMED_SEEN' "
            "AND consideration_status = 'CONFIRMED_CONSIDERED' "
            "AND evidence_json <> '{}'::jsonb) OR "
            "(behavior_label_state = 'EXPLICIT_PASS' AND linked_master_trade_id IS NULL "
            "AND seen_by_conner = 'CONFIRMED_SEEN' "
            "AND consideration_status = 'CONFIRMED_CONSIDERED' "
            "AND evidence_json <> '{}'::jsonb)",
            name="ck_v2_scout_opportunity_behavior_evidence",
        ),
        sa.CheckConstraint("source_lineage <> '{}'::jsonb", name="ck_v2_scout_opportunity_lineage"),
    )
    _create(
        "v2_opportunity_behavior_labels",
        sa.Column("label_id", sa.String(96), primary_key=True),
        sa.Column(
            "opportunity_id",
            sa.String(96),
            sa.ForeignKey("v2_scout_opportunities.opportunity_id"),
            nullable=False,
        ),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("label_kind", sa.String(32), nullable=False),
        sa.Column(
            "master_trade_id",
            sa.String(96),
            sa.ForeignKey("v2_master_trades.trade_id"),
            nullable=True,
        ),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("evidence_source", sa.String(64), nullable=False),
        sa.Column("evidence_json", JSON, nullable=False),
        sa.Column(
            "supersedes_label_id",
            sa.String(96),
            sa.ForeignKey("v2_opportunity_behavior_labels.label_id"),
            nullable=True,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("opportunity_id", "revision", name="uq_v2_opportunity_behavior_revision"),
        sa.CheckConstraint("revision >= 1", name="ck_v2_opportunity_behavior_revision"),
        sa.CheckConstraint(
            "label_kind IN ('POSITIVE_TRADE', 'EXPLICIT_PASS')",
            name="ck_v2_opportunity_behavior_kind",
        ),
        sa.CheckConstraint(
            "(label_kind = 'POSITIVE_TRADE' AND master_trade_id IS NOT NULL "
            "AND evidence_json <> '{}'::jsonb) OR "
            "(label_kind = 'EXPLICIT_PASS' AND master_trade_id IS NULL "
            "AND evidence_json <> '{}'::jsonb)",
            name="ck_v2_opportunity_behavior_evidence",
        ),
        sa.CheckConstraint(
            "supersedes_label_id IS NULL OR supersedes_label_id <> label_id",
            name="ck_v2_opportunity_behavior_supersedes",
        ),
    )
    _create(
        "v2_trade_leadup_snapshots",
        sa.Column("trade_leadup_id", sa.String(96), primary_key=True),
        sa.Column("subject_id", sa.String(96), nullable=False),
        sa.Column("subject_kind", sa.String(32), nullable=False),
        sa.Column(
            "master_trade_id",
            sa.String(96),
            sa.ForeignKey("v2_master_trades.trade_id"),
            nullable=True,
        ),
        sa.Column(
            "horizon_id",
            sa.String(96),
            sa.ForeignKey("v2_observation_horizons.horizon_id"),
            nullable=False,
        ),
        sa.Column("horizon_set_version", sa.String(64), nullable=False),
        sa.Column(
            "market_state_id",
            sa.String(96),
            sa.ForeignKey("v2_synchronized_market_states.market_state_id"),
            nullable=False,
        ),
        sa.Column(
            "opportunity_id",
            sa.String(96),
            sa.ForeignKey("v2_scout_opportunities.opportunity_id"),
            nullable=True,
        ),
        sa.Column("decision_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("target_cutoff_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("actual_cutoff_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("origin", sa.String(32), nullable=False),
        sa.Column("captured_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("lead_time_ms", sa.BigInteger(), nullable=True),
        sa.Column("alignment_error_ms", sa.Integer(), nullable=False),
        sa.Column("source_lineage", JSON, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "subject_kind",
            "subject_id",
            "horizon_set_version",
            "horizon_id",
            name="uq_v2_trade_leadup_horizon",
        ),
        sa.CheckConstraint(
            "subject_kind IN ('MASTER_TRADE', 'SCOUT_CANDIDATE')",
            name="ck_v2_trade_leadup_subject_kind",
        ),
        sa.CheckConstraint(
            "(subject_kind = 'MASTER_TRADE' AND master_trade_id = subject_id "
            "AND opportunity_id IS NULL) OR "
            "(subject_kind = 'SCOUT_CANDIDATE' AND opportunity_id = subject_id "
            "AND master_trade_id IS NULL)",
            name="ck_v2_trade_leadup_subject",
        ),
        sa.CheckConstraint(
            "actual_cutoff_at <= target_cutoff_at AND target_cutoff_at <= decision_at",
            name="ck_v2_trade_leadup_cutoff",
        ),
        sa.CheckConstraint(
            "origin IN ('LIVE', 'BLIND_REPLAY', 'RETROSPECTIVE_TRADE_LEADUP')",
            name="ck_v2_trade_leadup_origin",
        ),
        sa.CheckConstraint(
            "(origin = 'RETROSPECTIVE_TRADE_LEADUP' AND lead_time_ms IS NULL) OR "
            "(origin IN ('LIVE', 'BLIND_REPLAY') AND lead_time_ms IS NOT NULL AND lead_time_ms >= 0)",
            name="ck_v2_trade_leadup_lead_time",
        ),
        sa.CheckConstraint("alignment_error_ms >= 0", name="ck_v2_trade_leadup_alignment"),
        sa.CheckConstraint("source_lineage <> '{}'::jsonb", name="ck_v2_trade_leadup_lineage"),
    )
    _create(
        "v2_learning_examples",
        sa.Column("example_id", sa.String(96), primary_key=True),
        sa.Column("strategy_namespace", sa.String(64), nullable=True),
        sa.Column("task", sa.String(64), nullable=False),
        sa.Column(
            "market_state_id",
            sa.String(96),
            sa.ForeignKey("v2_synchronized_market_states.market_state_id"),
            nullable=True,
        ),
        sa.Column(
            "opportunity_id",
            sa.String(96),
            sa.ForeignKey("v2_scout_opportunities.opportunity_id"),
            nullable=True,
        ),
        sa.Column(
            "behavior_label_id",
            sa.String(96),
            sa.ForeignKey("v2_opportunity_behavior_labels.label_id"),
            nullable=True,
        ),
        sa.Column(
            "master_trade_id",
            sa.String(96),
            sa.ForeignKey("v2_master_trades.trade_id"),
            nullable=True,
        ),
        sa.Column(
            "follower_trade_id",
            sa.String(96),
            sa.ForeignKey("v2_follower_trades.follower_trade_id"),
            nullable=True,
        ),
        sa.Column("trade_date", sa.Date(), nullable=False),
        sa.Column("feature_cutoff_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("label_state", sa.String(32), nullable=False),
        sa.Column("seen_by_conner", sa.String(24), nullable=False, server_default="UNKNOWN"),
        sa.Column("consideration_status", sa.String(32), nullable=False, server_default="UNKNOWN"),
        sa.Column("target_name", sa.String(128), nullable=True),
        sa.Column("target_numeric", VALUE, nullable=True),
        sa.Column("target_boolean", sa.Boolean(), nullable=True),
        sa.Column("target_json", JSON, nullable=True),
        sa.Column("target_observed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ranking_group_id", sa.String(96), nullable=True),
        sa.Column("relevance_grade", VALUE, nullable=True),
        sa.Column("sampling_policy_version", sa.String(64), nullable=False),
        sa.Column("feature_schema_version", sa.String(64), nullable=False),
        sa.Column("dataset_version", sa.String(64), nullable=False),
        sa.Column("source_kind", sa.String(32), nullable=False),
        sa.Column("source_lineage", JSON, nullable=False),
        sa.Column("eligible", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("exclusion_reason", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "task IN ('behavior_imitation', 'setup_outcome_quality', "
            "'conviction_behavior', 'copier_execution_quality')",
            name="ck_v2_learning_example_task",
        ),
        sa.CheckConstraint(
            "label_state IN ('UNLABELED', 'POSITIVE_TRADE', 'EXPLICIT_PASS', 'OBSERVED_TARGET')",
            name="ck_v2_learning_example_label_state",
        ),
        sa.CheckConstraint(
            "seen_by_conner IN ('UNKNOWN', 'CONFIRMED_SEEN', 'CONFIRMED_NOT_SEEN')",
            name="ck_v2_learning_example_seen",
        ),
        sa.CheckConstraint(
            "consideration_status IN ('UNKNOWN', 'PRESENTED', 'CONFIRMED_CONSIDERED')",
            name="ck_v2_learning_example_consideration",
        ),
        sa.CheckConstraint(
            "source_kind IN ('FUTURE_LIVE_MASTER', 'BLIND_POINT_IN_TIME_REPLAY', "
            "'FINALIZED_HISTORICAL_MARKET', 'EXPLICIT_TRADER_FEEDBACK', 'FOLLOWER_EXECUTION')",
            name="ck_v2_learning_example_source_kind",
        ),
        sa.CheckConstraint(
            "(label_state = 'UNLABELED' AND target_name IS NULL AND target_observed_at IS NULL "
            "AND num_nonnulls(target_numeric, target_boolean, target_json) = 0) OR "
            "(label_state <> 'UNLABELED' AND target_name IS NOT NULL "
            "AND target_observed_at IS NOT NULL AND target_observed_at >= feature_cutoff_at "
            "AND num_nonnulls(target_numeric, target_boolean, target_json) = 1)",
            name="ck_v2_learning_example_target",
        ),
        sa.CheckConstraint(
            "label_state <> 'POSITIVE_TRADE' OR "
            "(task = 'behavior_imitation' AND master_trade_id IS NOT NULL "
            "AND behavior_label_id IS NOT NULL AND seen_by_conner = 'CONFIRMED_SEEN' "
            "AND consideration_status = 'CONFIRMED_CONSIDERED')",
            name="ck_v2_learning_example_positive",
        ),
        sa.CheckConstraint(
            "label_state <> 'EXPLICIT_PASS' OR "
            "(task = 'behavior_imitation' AND master_trade_id IS NULL "
            "AND behavior_label_id IS NOT NULL AND seen_by_conner = 'CONFIRMED_SEEN' "
            "AND consideration_status = 'CONFIRMED_CONSIDERED')",
            name="ck_v2_learning_example_rejection",
        ),
        sa.CheckConstraint(
            "label_state <> 'OBSERVED_TARGET' OR task <> 'behavior_imitation'",
            name="ck_v2_learning_example_observed_target",
        ),
        sa.CheckConstraint(
            "label_state <> 'UNLABELED' OR behavior_label_id IS NULL",
            name="ck_v2_learning_example_unlabeled",
        ),
        sa.CheckConstraint(
            "(task = 'copier_execution_quality' AND follower_trade_id IS NOT NULL "
            "AND source_kind = 'FOLLOWER_EXECUTION' "
            "AND strategy_namespace IS NULL AND market_state_id IS NULL "
            "AND opportunity_id IS NULL AND behavior_label_id IS NULL AND master_trade_id IS NULL) OR "
            "(task <> 'copier_execution_quality' AND follower_trade_id IS NULL "
            "AND source_kind <> 'FOLLOWER_EXECUTION' "
            "AND strategy_namespace IS NOT NULL AND market_state_id IS NOT NULL)",
            name="ck_v2_learning_example_task_boundary",
        ),
        sa.CheckConstraint(
            "task <> 'conviction_behavior' OR source_kind = 'FUTURE_LIVE_MASTER'",
            name="ck_v2_learning_example_conviction_source",
        ),
        sa.CheckConstraint(
            "task <> 'behavior_imitation' OR source_kind <> 'FINALIZED_HISTORICAL_MARKET'",
            name="ck_v2_learning_example_behavior_source",
        ),
        sa.CheckConstraint(
            "eligible OR exclusion_reason IS NOT NULL", name="ck_v2_learning_example_exclusion"
        ),
        sa.CheckConstraint("source_lineage <> '{}'::jsonb", name="ck_v2_learning_example_lineage"),
        sa.CheckConstraint(
            "relevance_grade IS NULL OR relevance_grade >= 0", name="ck_v2_learning_example_relevance"
        ),
        sa.CheckConstraint(
            "relevance_grade IS NULL OR ranking_group_id IS NOT NULL",
            name="ck_v2_learning_example_ranking_group",
        ),
    )
    _create(
        "v2_scout_runs",
        sa.Column("run_id", sa.String(96), primary_key=True),
        sa.Column("strategy_namespace", sa.String(64), nullable=False),
        sa.Column("query_group_id", sa.String(96), nullable=False),
        sa.Column("trade_date", sa.Date(), nullable=False),
        sa.Column("prediction_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("feature_cutoff_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "nq_contract_id",
            sa.String(96),
            sa.ForeignKey("v2_futures_contracts.contract_id"),
            nullable=False,
        ),
        sa.Column(
            "es_contract_id",
            sa.String(96),
            sa.ForeignKey("v2_futures_contracts.contract_id"),
            nullable=False,
        ),
        sa.Column("origin", sa.String(32), nullable=False),
        sa.Column("observation_spec_version", sa.String(64), nullable=False),
        sa.Column("candidate_set_version", sa.String(64), nullable=False),
        sa.Column("feature_schema_version", sa.String(64), nullable=False),
        sa.Column("ranking_policy_version", sa.String(64), nullable=False),
        sa.Column(
            "behavior_artifact_id",
            sa.String(96),
            sa.ForeignKey("v2_model_artifacts.artifact_id"),
            nullable=True,
        ),
        sa.Column(
            "outcome_artifact_id",
            sa.String(96),
            sa.ForeignKey("v2_model_artifacts.artifact_id"),
            nullable=True,
        ),
        sa.Column(
            "conviction_artifact_id",
            sa.String(96),
            sa.ForeignKey("v2_model_artifacts.artifact_id"),
            nullable=True,
        ),
        sa.Column("model_versions", JSON, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("data_quality", JSON, nullable=False),
        sa.Column("source_lineage", JSON, nullable=False),
        sa.Column("execution_eligible", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("shadow_only", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("feature_cutoff_at <= prediction_at", name="ck_v2_scout_run_cutoff"),
        sa.CheckConstraint("nq_contract_id <> es_contract_id", name="ck_v2_scout_run_contracts"),
        sa.CheckConstraint(
            "origin IN ('LIVE', 'BLIND_REPLAY', 'RETROSPECTIVE_TRADE_LEADUP')",
            name="ck_v2_scout_run_origin",
        ),
        sa.CheckConstraint("execution_eligible = false", name="ck_v2_scout_run_execution"),
        sa.CheckConstraint("shadow_only = true", name="ck_v2_scout_run_shadow_only"),
        sa.CheckConstraint("source_lineage <> '{}'::jsonb", name="ck_v2_scout_run_lineage"),
    )
    _create(
        "v2_scout_rankings",
        sa.Column("ranking_id", sa.String(96), primary_key=True),
        sa.Column("run_id", sa.String(96), sa.ForeignKey("v2_scout_runs.run_id"), nullable=False),
        sa.Column(
            "opportunity_id",
            sa.String(96),
            sa.ForeignKey("v2_scout_opportunities.opportunity_id"),
            nullable=False,
        ),
        sa.Column(
            "market_state_id",
            sa.String(96),
            sa.ForeignKey("v2_synchronized_market_states.market_state_id"),
            nullable=False,
        ),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("predicted_direction", sa.String(8), nullable=True),
        sa.Column("behavior_score", VALUE, nullable=True),
        sa.Column("outcome_prediction", JSON, nullable=True),
        sa.Column("conviction_prediction", JSON, nullable=True),
        sa.Column("model_confidence", VALUE, nullable=True),
        sa.Column("data_quality_score", VALUE, nullable=True),
        sa.Column("ranking_score", VALUE, nullable=True),
        sa.Column("ranking_components", JSON, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("run_id", "rank", name="uq_v2_scout_ranking_rank"),
        sa.UniqueConstraint("run_id", "opportunity_id", name="uq_v2_scout_ranking_opportunity"),
        sa.CheckConstraint("rank >= 1", name="ck_v2_scout_ranking_rank"),
        sa.CheckConstraint(
            "predicted_direction IS NULL OR predicted_direction IN ('LONG', 'SHORT')",
            name="ck_v2_scout_ranking_direction",
        ),
        sa.CheckConstraint(
            "behavior_score IS NULL OR behavior_score BETWEEN 0 AND 1",
            name="ck_v2_scout_ranking_behavior",
        ),
        sa.CheckConstraint(
            "model_confidence IS NULL OR model_confidence BETWEEN 0 AND 1",
            name="ck_v2_scout_ranking_confidence",
        ),
        sa.CheckConstraint(
            "data_quality_score IS NULL OR data_quality_score BETWEEN 0 AND 1",
            name="ck_v2_scout_ranking_data_quality",
        ),
        sa.CheckConstraint(
            "ranking_score IS NULL OR ranking_score BETWEEN 0 AND 1",
            name="ck_v2_scout_ranking_display_score",
        ),
    )
    _create(
        "v2_shadow_evaluations",
        sa.Column("shadow_evaluation_id", sa.String(96), primary_key=True),
        sa.Column(
            "ranking_id", sa.String(96), sa.ForeignKey("v2_scout_rankings.ranking_id"), nullable=False
        ),
        sa.Column("evaluation_version", sa.String(64), nullable=False),
        sa.Column("match_status", sa.String(32), nullable=False),
        sa.Column("origin", sa.String(32), nullable=False),
        sa.Column("prediction_cutoff_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("behavior_label_state", sa.String(32), nullable=False, server_default="UNLABELED"),
        sa.Column("evidence_json", JSON, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column(
            "matched_master_trade_id",
            sa.String(96),
            sa.ForeignKey("v2_master_trades.trade_id"),
            nullable=True,
        ),
        sa.Column("conner_entry_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("actual_direction", sa.String(8), nullable=True),
        sa.Column("evaluated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("lead_time_ms", sa.BigInteger(), nullable=True),
        sa.Column("direction_match", sa.Boolean(), nullable=True),
        sa.Column("realized_r", VALUE, nullable=True),
        sa.Column("mfe_r", VALUE, nullable=True),
        sa.Column("mae_r", VALUE, nullable=True),
        sa.Column("outcome_observed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("behavior_metrics", JSON, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("outcome_metrics", JSON, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("data_quality_failures", JSON, nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("ranking_id", "evaluation_version", name="uq_v2_shadow_evaluation_version"),
        sa.CheckConstraint(
            "match_status IN ('PENDING', 'MATCHED_TRADE', 'NO_MATCH_OBSERVED_WINDOW', "
            "'EXPLICIT_REJECTION', 'DATA_INVALID')",
            name="ck_v2_shadow_evaluation_status",
        ),
        sa.CheckConstraint(
            "origin IN ('LIVE', 'BLIND_REPLAY', 'RETROSPECTIVE_TRADE_LEADUP')",
            name="ck_v2_shadow_evaluation_origin",
        ),
        sa.CheckConstraint(
            "behavior_label_state IN ('UNLABELED', 'POSITIVE_TRADE', 'EXPLICIT_PASS')",
            name="ck_v2_shadow_evaluation_behavior",
        ),
        sa.CheckConstraint(
            "actual_direction IS NULL OR actual_direction IN ('LONG', 'SHORT')",
            name="ck_v2_shadow_evaluation_direction",
        ),
        sa.CheckConstraint(
            "(match_status = 'MATCHED_TRADE' AND matched_master_trade_id IS NOT NULL "
            "AND conner_entry_at IS NOT NULL AND actual_direction IS NOT NULL "
            "AND behavior_label_state = 'POSITIVE_TRADE') OR "
            "(match_status = 'EXPLICIT_REJECTION' AND matched_master_trade_id IS NULL "
            "AND conner_entry_at IS NULL AND behavior_label_state = 'EXPLICIT_PASS' "
            "AND actual_direction IS NULL AND lead_time_ms IS NULL "
            "AND direction_match IS NULL AND outcome_observed_at IS NULL "
            "AND evidence_json <> '{}'::jsonb) OR "
            "(match_status IN ('PENDING', 'NO_MATCH_OBSERVED_WINDOW', 'DATA_INVALID') "
            "AND behavior_label_state = 'UNLABELED' AND matched_master_trade_id IS NULL "
            "AND conner_entry_at IS NULL AND actual_direction IS NULL "
            "AND lead_time_ms IS NULL AND direction_match IS NULL "
            "AND outcome_observed_at IS NULL)",
            name="ck_v2_shadow_evaluation_match",
        ),
        sa.CheckConstraint(
            "(origin = 'RETROSPECTIVE_TRADE_LEADUP' AND lead_time_ms IS NULL) OR "
            "(origin IN ('LIVE', 'BLIND_REPLAY') "
            "AND (match_status <> 'MATCHED_TRADE' OR lead_time_ms IS NOT NULL))",
            name="ck_v2_shadow_evaluation_origin_lead_time",
        ),
        sa.CheckConstraint(
            "lead_time_ms IS NULL OR lead_time_ms >= 0", name="ck_v2_shadow_evaluation_lead_time"
        ),
        sa.CheckConstraint(
            "outcome_observed_at IS NULL OR "
            "(conner_entry_at IS NOT NULL AND outcome_observed_at >= conner_entry_at)",
            name="ck_v2_shadow_evaluation_outcome_time",
        ),
        sa.CheckConstraint(
            "matched_master_trade_id IS NOT NULL OR "
            "(realized_r IS NULL AND mfe_r IS NULL AND mae_r IS NULL)",
            name="ck_v2_shadow_evaluation_outcomes",
        ),
    )
    _create(
        "v2_scout_evaluation_runs",
        sa.Column("evaluation_run_id", sa.String(96), primary_key=True),
        sa.Column("strategy_namespace", sa.String(64), nullable=False),
        sa.Column("task_namespace", sa.String(128), nullable=False),
        sa.Column("evaluation_version", sa.String(64), nullable=False),
        sa.Column("evaluation_mode", sa.String(32), nullable=False),
        sa.Column("dataset_version", sa.String(64), nullable=False),
        sa.Column("metric_version", sa.String(64), nullable=False),
        sa.Column("training_end_trade_date", sa.Date(), nullable=True),
        sa.Column("evaluation_start_trade_date", sa.Date(), nullable=False),
        sa.Column("evaluation_end_trade_date", sa.Date(), nullable=False),
        sa.Column("model_versions", JSON, nullable=False),
        sa.Column("behavior_metrics", JSON, nullable=False),
        sa.Column("outcome_metrics", JSON, nullable=False),
        sa.Column("usefulness_metrics", JSON, nullable=False),
        sa.Column("data_quality_metrics", JSON, nullable=False),
        sa.Column("source_lineage", JSON, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "evaluation_mode IN ('SHADOW_FORWARD', 'WALK_FORWARD')",
            name="ck_v2_scout_evaluation_mode",
        ),
        sa.CheckConstraint(
            "task_namespace LIKE 'v2.futures.%'", name="ck_v2_scout_evaluation_task_namespace"
        ),
        sa.CheckConstraint(
            "evaluation_start_trade_date <= evaluation_end_trade_date",
            name="ck_v2_scout_evaluation_dates",
        ),
        sa.CheckConstraint(
            "training_end_trade_date IS NULL OR training_end_trade_date < evaluation_start_trade_date",
            name="ck_v2_scout_evaluation_chronology",
        ),
        sa.CheckConstraint(
            "evaluation_mode <> 'WALK_FORWARD' OR training_end_trade_date IS NOT NULL",
            name="ck_v2_scout_evaluation_walk_forward",
        ),
        sa.CheckConstraint("source_lineage <> '{}'::jsonb", name="ck_v2_scout_evaluation_lineage"),
    )

    _index(
        "ix_v2_market_states_trade_cutoff",
        "v2_synchronized_market_states",
        ["trade_date", "feature_cutoff_at"],
    )
    _index(
        "ix_v2_feature_values_state_definition",
        "v2_candidate_feature_values",
        ["market_state_id", "feature_definition_id"],
    )
    _index(
        "ix_v2_scout_opportunities_namespace_series",
        "v2_scout_opportunities",
        ["strategy_namespace", "candidate_series_id"],
    )
    _index(
        "ix_v2_trade_leadup_trade_cutoff",
        "v2_trade_leadup_snapshots",
        ["subject_kind", "subject_id", "actual_cutoff_at"],
    )
    _index(
        "ix_v2_learning_examples_task_date_label",
        "v2_learning_examples",
        ["task", "trade_date", "label_state"],
    )
    _index(
        "ix_v2_scout_runs_date_cutoff",
        "v2_scout_runs",
        ["trade_date", "feature_cutoff_at"],
    )
    _index(
        "ix_v2_shadow_evaluations_master_trade",
        "v2_shadow_evaluations",
        ["matched_master_trade_id"],
    )
    _index(
        "ix_v2_scout_evaluations_dates",
        "v2_scout_evaluation_runs",
        ["evaluation_start_trade_date", "evaluation_end_trade_date"],
    )


def downgrade() -> None:
    # Remove only the additive observation/Scout objects introduced here.
    offline = _offline_mode()
    existing = set() if offline else set(sa.inspect(op.get_bind()).get_table_names())
    for table in reversed(
        [
            "v2_observation_specs",
            "v2_observation_horizons",
            "v2_synchronized_market_states",
            "v2_candidate_feature_definitions",
            "v2_candidate_feature_values",
            "v2_scout_opportunities",
            "v2_opportunity_behavior_labels",
            "v2_trade_leadup_snapshots",
            "v2_learning_examples",
            "v2_scout_runs",
            "v2_scout_rankings",
            "v2_shadow_evaluations",
            "v2_scout_evaluation_runs",
        ]
    ):
        if offline or table in existing:
            op.drop_table(table)
