from __future__ import annotations

import importlib.util
from pathlib import Path

from sqlalchemy import CheckConstraint, MetaData, Numeric, Table

from app.db.models import Base
from app.v2.intelligence.datasets import BehaviorLabelState, DatasetSourceKind, ObservationOrigin, SeenByConner
from app.v2.intelligence.evaluation import EvaluationMode, ShadowMatchStatus
from app.v2.intelligence.observation import ObservationPurpose
from app.v2.intelligence.specification import ResearchStatus
from app.v2.market_data import AvailabilityMode


MIGRATION_PATH = Path("migrations/versions/0010_conner_nq_scout.py")
CREATION_ORDER = (
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
)
EXPECTED = set(CREATION_ORDER)


def _load_migration():
    spec = importlib.util.spec_from_file_location("v2_migration_0010", MIGRATION_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _check_names(table_name: str) -> set[str]:
    return {
        constraint.name
        for constraint in Base.metadata.tables[table_name].constraints
        if isinstance(constraint, CheckConstraint)
    }


def _check_sql(table_name: str) -> str:
    return "\n".join(
        str(constraint.sqltext)
        for constraint in Base.metadata.tables[table_name].constraints
        if isinstance(constraint, CheckConstraint)
    )


def _foreign_key_target(table_name: str, column_name: str) -> str:
    foreign_keys = list(Base.metadata.tables[table_name].c[column_name].foreign_keys)
    assert len(foreign_keys) == 1
    return foreign_keys[0].target_fullname


def test_conner_nq_schema_is_additive_and_registered():
    assert EXPECTED <= set(Base.metadata.tables)
    # Both legacy V1 and the original V2 foundation remain registered.
    assert "symbols" in Base.metadata.tables
    assert "copy_orders" in Base.metadata.tables
    assert "v2_futures_contracts" in Base.metadata.tables
    assert "v2_execution_leases" in Base.metadata.tables


def test_synchronized_state_and_opportunity_use_exact_existing_entities():
    assert _foreign_key_target("v2_synchronized_market_states", "nq_contract_id") == (
        "v2_futures_contracts.contract_id"
    )
    assert _foreign_key_target("v2_synchronized_market_states", "es_contract_id") == (
        "v2_futures_contracts.contract_id"
    )
    assert _foreign_key_target("v2_scout_opportunities", "opportunity_id") == (
        "v2_opportunity_snapshots.opportunity_id"
    )
    assert _foreign_key_target("v2_trade_leadup_snapshots", "master_trade_id") == (
        "v2_master_trades.trade_id"
    )
    assert _foreign_key_target("v2_learning_examples", "follower_trade_id") == (
        "v2_follower_trades.follower_trade_id"
    )
    spec_columns = set(Base.metadata.tables["v2_observation_specs"].c.keys())
    assert {
        "horizon_set_version",
        "synchronization_policy_version",
        "measurement_versions",
        "strategy_window_candidates",
    } <= spec_columns
    assert "session_anchor_version" not in spec_columns


def test_point_in_time_and_missing_value_constraints_are_declared():
    market_checks = _check_names("v2_synchronized_market_states")
    assert {
        "ck_v2_market_state_contracts",
        "ck_v2_market_state_sync",
        "ck_v2_market_state_source_cutoff",
        "ck_v2_market_state_temporal_order",
        "ck_v2_market_state_availability_mode",
        "ck_v2_market_state_available_cutoff",
        "ck_v2_market_state_capture",
        "ck_v2_market_state_purpose",
    } <= market_checks
    assert "nq_source_max_at <= feature_cutoff_at" in _check_sql("v2_synchronized_market_states")
    assert "es_source_max_at <= feature_cutoff_at" in _check_sql("v2_synchronized_market_states")
    market_sql = _check_sql("v2_synchronized_market_states")
    assert "nq_available_max_at <= feature_cutoff_at" in market_sql
    assert "es_available_max_at <= feature_cutoff_at" in market_sql
    assert "FINALIZED_HISTORICAL" in market_sql
    assert "OUTCOME_RESEARCH" in market_sql and "HISTORICAL_REPLAY" in market_sql

    value_checks = _check_names("v2_candidate_feature_values")
    assert {
        "ck_v2_feature_value_role",
        "ck_v2_feature_value_cutoff",
        "ck_v2_feature_value_availability_mode",
        "ck_v2_feature_value_available_at",
        "ck_v2_feature_value_window",
        "ck_v2_feature_value_missing",
    } <= value_checks
    value_sql = _check_sql("v2_candidate_feature_values")
    assert "max_input_event_at <= feature_cutoff_at" in value_sql
    assert "max_input_available_at <= feature_cutoff_at" in value_sql
    assert "definition_available_at <= feature_cutoff_at" in value_sql
    assert "missing_reason IS NOT NULL" in value_sql


def test_behavior_storage_has_positive_unlabeled_semantics_not_automatic_negatives():
    opportunity = Base.metadata.tables["v2_scout_opportunities"]
    assert opportunity.c.seen_by_conner.default.arg == "UNKNOWN"
    opportunity_sql = _check_sql("v2_scout_opportunities")
    assert "CONFIRMED_SEEN" in opportunity_sql
    assert "CONFIRMED_NOT_SEEN" in opportunity_sql
    assert "POSITIVE_TRADE" in opportunity_sql
    assert "EXPLICIT_PASS" in opportunity_sql

    label_sql = _check_sql("v2_opportunity_behavior_labels")
    assert "POSITIVE_TRADE" in label_sql
    assert "EXPLICIT_PASS" in label_sql
    assert "master_trade_id IS NOT NULL" in label_sql
    assert "evidence_json <> '{}'::jsonb" in label_sql
    for invented_negative in ("not_taken", "automatic_negative", "conner_passed"):
        assert invented_negative not in label_sql.lower()

    learning_sql = _check_sql("v2_learning_examples")
    assert "label_state = 'UNLABELED'" in learning_sql
    assert "num_nonnulls(target_numeric, target_boolean, target_json) = 0" in learning_sql
    assert "label_state <> 'UNLABELED' OR behavior_label_id IS NULL" in learning_sql
    assert "synthetic" not in learning_sql.lower()
    assert "follower_trade_id IS NOT NULL" in learning_sql
    assert "market_state_id IS NULL" in learning_sql
    assert "FINALIZED_HISTORICAL_MARKET" in learning_sql
    assert "FUTURE_LIVE_MASTER" in learning_sql


def test_learning_and_ranking_values_preserve_task_separation():
    learning = Base.metadata.tables["v2_learning_examples"]
    ranking = Base.metadata.tables["v2_scout_rankings"]
    shadow = Base.metadata.tables["v2_shadow_evaluations"]

    assert isinstance(learning.c.target_numeric.type, Numeric)
    assert isinstance(learning.c.relevance_grade.type, Numeric)
    assert isinstance(ranking.c.behavior_score.type, Numeric)
    assert isinstance(shadow.c.realized_r.type, Numeric)

    assert {
        "behavior_score",
        "outcome_prediction",
        "conviction_prediction",
        "model_confidence",
        "data_quality_score",
        "ranking_score",
    } <= set(ranking.c.keys())
    assert not ({"quantity", "risk_dollars", "order_id", "submission_enabled"} & set(ranking.c.keys()))
    assert "copier_execution_quality" in _check_sql("v2_learning_examples")
    assert "ck_v2_learning_example_task_boundary" in _check_names("v2_learning_examples")


def test_database_vocabularies_match_intelligence_contract_enums():
    checks = {
        "v2_synchronized_market_states": ObservationPurpose,
        "v2_candidate_feature_definitions": ResearchStatus,
        "v2_candidate_feature_values": AvailabilityMode,
        "v2_scout_opportunities": SeenByConner,
        "v2_opportunity_behavior_labels": BehaviorLabelState,
        "v2_learning_examples": DatasetSourceKind,
        "v2_scout_runs": ObservationOrigin,
        "v2_shadow_evaluations": ShadowMatchStatus,
        "v2_scout_evaluation_runs": EvaluationMode,
    }
    for table_name, enum_type in checks.items():
        sql = _check_sql(table_name)
        for enum_value in enum_type:
            # UNLABELED belongs in the opportunity/learning state constraints,
            # not the append-only table of observed behavior evidence.
            if table_name == "v2_opportunity_behavior_labels" and enum_value is BehaviorLabelState.UNLABELED:
                continue
            assert enum_value.value in sql, (table_name, enum_value.value)


def test_shadow_and_walk_forward_constraints_are_declared():
    shadow_sql = _check_sql("v2_shadow_evaluations")
    assert "NO_MATCH_OBSERVED_WINDOW" in shadow_sql
    assert "EXPLICIT_REJECTION" in shadow_sql
    assert "EXPLICIT_PASS" in shadow_sql
    assert "lead_time_ms >= 0" in shadow_sql
    assert "matched_master_trade_id IS NOT NULL" in shadow_sql
    assert "match_status IN ('PENDING', 'NO_MATCH_OBSERVED_WINDOW', 'DATA_INVALID')" in shadow_sql
    assert "direction_match IS NULL" in shadow_sql
    assert "RETROSPECTIVE_TRADE_LEADUP" in shadow_sql

    evaluation_sql = _check_sql("v2_scout_evaluation_runs")
    assert "WALK_FORWARD" in evaluation_sql
    assert "training_end_trade_date < evaluation_start_trade_date" in evaluation_sql
    assert "evaluation_start_trade_date <= evaluation_end_trade_date" in evaluation_sql


def test_query_indexes_are_present_in_orm_metadata():
    expected_indexes = {
        "v2_synchronized_market_states": {"ix_v2_market_states_trade_cutoff"},
        "v2_candidate_feature_values": {"ix_v2_feature_values_state_definition"},
        "v2_scout_opportunities": {"ix_v2_scout_opportunities_namespace_series"},
        "v2_trade_leadup_snapshots": {"ix_v2_trade_leadup_trade_cutoff"},
        "v2_learning_examples": {"ix_v2_learning_examples_task_date_label"},
        "v2_scout_runs": {"ix_v2_scout_runs_date_cutoff"},
        "v2_shadow_evaluations": {"ix_v2_shadow_evaluations_master_trade"},
        "v2_scout_evaluation_runs": {"ix_v2_scout_evaluations_dates"},
    }
    for table_name, names in expected_indexes.items():
        assert names <= {index.name for index in Base.metadata.tables[table_name].indexes}


def test_migration_is_single_head_additive_and_does_not_edit_0009_or_v1():
    migration = MIGRATION_PATH.read_text(encoding="utf-8")
    assert 'revision = "0010_conner_nq_scout"' in migration
    assert 'down_revision = "0009_v2_futures_foundation"' in migration
    assert len("0010_conner_nq_scout") <= 32
    for protected in (
        "symbols",
        "copy_orders",
        "v2_futures_contracts",
        "v2_master_trades",
        "v2_model_artifacts",
        "v2_execution_leases",
    ):
        assert f'op.drop_table("{protected}")' not in migration


def test_migration_upgrade_declares_exact_tables_and_indexes(monkeypatch):
    module = _load_migration()
    declared: list[str] = []
    indexes: list[tuple[str, str, tuple[str, ...]]] = []
    monkeypatch.setattr(module, "_create", lambda name, *columns, **kwargs: declared.append(name))
    monkeypatch.setattr(
        module,
        "_index",
        lambda name, table, columns: indexes.append((name, table, tuple(columns))),
    )

    module.upgrade()

    assert tuple(declared) == CREATION_ORDER
    assert len(indexes) == 8
    assert {item[0] for item in indexes} == {
        "ix_v2_market_states_trade_cutoff",
        "ix_v2_feature_values_state_definition",
        "ix_v2_scout_opportunities_namespace_series",
        "ix_v2_trade_leadup_trade_cutoff",
        "ix_v2_learning_examples_task_date_label",
        "ix_v2_scout_runs_date_cutoff",
        "ix_v2_shadow_evaluations_master_trade",
        "ix_v2_scout_evaluations_dates",
    }


def test_migration_columns_constraints_and_foreign_keys_match_orm(monkeypatch):
    module = _load_migration()
    declared: dict[str, Table] = {}

    def capture(name, *items, **kwargs):
        declared[name] = Table(name, MetaData(), *items, **kwargs)

    monkeypatch.setattr(module, "_create", capture)
    monkeypatch.setattr(module, "_index", lambda name, table, columns: None)
    module.upgrade()

    for table_name in CREATION_ORDER:
        orm_table = Base.metadata.tables[table_name]
        migration_table = declared[table_name]
        assert set(migration_table.c.keys()) == set(orm_table.c.keys()), table_name
        for column_name in orm_table.c.keys():
            orm_column = orm_table.c[column_name]
            migration_column = migration_table.c[column_name]
            assert type(migration_column.type) is type(orm_column.type), (table_name, column_name)
            assert migration_column.nullable == orm_column.nullable, (table_name, column_name)
            assert migration_column.primary_key == orm_column.primary_key, (table_name, column_name)
            assert {key.target_fullname for key in migration_column.foreign_keys} == {
                key.target_fullname for key in orm_column.foreign_keys
            }, (table_name, column_name)
        assert {item.name for item in migration_table.constraints if item.name} == {
            item.name for item in orm_table.constraints if item.name
        }, table_name


def test_migration_downgrade_removes_only_0010_tables_in_reverse_order(monkeypatch):
    module = _load_migration()
    dropped: list[str] = []

    class Inspector:
        @staticmethod
        def get_table_names():
            return list(CREATION_ORDER) + ["symbols", "v2_futures_contracts"]

    monkeypatch.setattr(module.op, "get_bind", lambda: object())
    monkeypatch.setattr(module.sa, "inspect", lambda bind: Inspector())
    monkeypatch.setattr(module.op, "drop_table", lambda name: dropped.append(name))

    module.downgrade()

    assert tuple(dropped) == tuple(reversed(CREATION_ORDER))
    assert "symbols" not in dropped
    assert "v2_futures_contracts" not in dropped
