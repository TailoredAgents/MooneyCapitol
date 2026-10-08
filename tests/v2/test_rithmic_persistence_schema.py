from __future__ import annotations

import importlib.util
from pathlib import Path

from sqlalchemy import BigInteger, CheckConstraint, MetaData, String, Table

from app.db.models import Base


MIGRATION_PATH = Path("migrations/versions/0011_rithmic_read_capture.py")
CREATION_ORDER = (
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
)
EXPECTED = set(CREATION_ORDER)


def _load_migration():
    spec = importlib.util.spec_from_file_location("v2_migration_0011", MIGRATION_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _check_sql(table_name: str) -> str:
    return "\n".join(
        str(constraint.sqltext)
        for constraint in Base.metadata.tables[table_name].constraints
        if isinstance(constraint, CheckConstraint)
    )


def test_rithmic_capture_schema_is_additive_and_registered():
    assert EXPECTED <= set(Base.metadata.tables)
    assert "v2_broker_connections" in Base.metadata.tables
    assert "v2_broker_accounts" in Base.metadata.tables
    assert "symbols" in Base.metadata.tables
    assert "copy_orders" in Base.metadata.tables


def test_official_64_bit_quantities_use_big_integer():
    quantity_columns = (
        ("v2_rithmic_order_observations", "quantity"),
        ("v2_rithmic_order_observations", "fill_size"),
        ("v2_rithmic_order_observations", "total_fill_size"),
        ("v2_rithmic_order_observations", "total_unfilled_size"),
        ("v2_rithmic_execution_observations", "quantity"),
        ("v2_rithmic_execution_observations", "effective_quantity_delta"),
        ("v2_rithmic_bracket_observations", "target_total_quantity"),
        ("v2_rithmic_bracket_observations", "target_released_quantity"),
        ("v2_rithmic_bracket_observations", "stop_total_quantity"),
        ("v2_rithmic_bracket_observations", "stop_released_quantity"),
        ("v2_rithmic_pnl_observations", "net_quantity"),
        ("v2_rithmic_pnl_observations", "working_buy_quantity"),
        ("v2_rithmic_pnl_observations", "working_sell_quantity"),
        ("v2_rithmic_rms_observations", "max_order_quantity"),
        ("v2_rithmic_rms_observations", "buy_limit"),
        ("v2_rithmic_rms_observations", "sell_limit"),
    )
    for table_name, column_name in quantity_columns:
        column = Base.metadata.tables[table_name].c[column_name]
        assert isinstance(column.type, BigInteger), f"{table_name}.{column_name} must be 64-bit"


def test_native_sequences_and_unknown_order_state_are_preserved():
    event = Base.metadata.tables["v2_rithmic_broker_events"]
    order = Base.metadata.tables["v2_rithmic_order_observations"]

    for column_name in (
        "sequence_number",
        "original_sequence_number",
        "correlation_sequence_number",
    ):
        assert isinstance(event.c[column_name].type, String)
        assert isinstance(order.c[column_name].type, String)

    assert "unknown_state" in order.c
    assert "command_outcome" in order.c
    assert "completion_reason" in order.c
    assert "normalized_state" in order.c
    assert "normalized_state IN" not in _check_sql("v2_rithmic_order_observations")


def test_capture_schema_retains_broker_identity_provenance_and_corrections():
    event = Base.metadata.tables["v2_rithmic_broker_events"]
    for column_name in (
        "generation_id",
        "replay_batch_id",
        "source_kind",
        "payload_fingerprint",
        "deduplication_key",
        "basket_id",
        "original_basket_id",
        "linked_basket_ids",
        "exchange_order_id",
        "ticker_plant_exchange_order_id",
        "fill_id",
    ):
        assert column_name in event.c

    execution = Base.metadata.tables["v2_rithmic_execution_observations"]
    assert "execution_kind" in execution.c
    assert "corrects_execution_id" in execution.c
    assert (
        next(iter(execution.c.corrects_execution_id.foreign_keys)).target_fullname
        == "v2_rithmic_execution_observations.execution_observation_id"
    )

    checkpoint = Base.metadata.tables["v2_rithmic_reconciliation_checkpoints"]
    for column_name in (
        "live_subscription_active",
        "live_buffer_started_at",
        "snapshot_requested_at",
        "snapshot_completed_at",
        "buffered_events_applied_at",
        "discrepancy_count",
        "ready",
    ):
        assert column_name in checkpoint.c
    assert "NOT ready" in _check_sql("v2_rithmic_reconciliation_checkpoints")

    replay = Base.metadata.tables["v2_rithmic_replay_batches"]
    assert "generation_map" in replay.c
    account = Base.metadata.tables["v2_rithmic_account_observations"]
    for column_name in ("user_id", "user_status", "country_code", "state_code"):
        assert column_name in account.c


def test_pnl_rms_and_bracket_provenance_are_explicit():
    pnl = Base.metadata.tables["v2_rithmic_pnl_observations"]
    for column_name in (
        "scope",
        "net_quantity",
        "average_open_fill_price",
        "day_open_pnl",
        "day_closed_pnl",
        "day_total_pnl",
        "account_balance",
        "available_buying_power",
        "source_kind",
        "is_snapshot",
        "freshness",
        "source_at",
    ):
        assert column_name in pnl.c

    rms = Base.metadata.tables["v2_rithmic_rms_observations"]
    for column_name in (
        "loss_limit",
        "current_auto_liquidate_threshold",
        "auto_liquidate_criteria",
        "max_order_quantity",
        "buy_margin_rate",
        "sell_margin_rate",
        "commission_rate",
    ):
        assert column_name in rms.c

    bracket = Base.metadata.tables["v2_rithmic_bracket_observations"]
    for column_name in (
        "parent_basket_id",
        "linked_basket_ids",
        "bracket_type",
        "operation_type",
        "target_tiers",
        "stop_tiers",
        "trailing_facts",
    ):
        assert column_name in bracket.c

    reference = Base.metadata.tables["v2_rithmic_reference_observations"]
    for column_name in (
        "generation_id",
        "event_id",
        "observation_kind",
        "symbol",
        "exchange",
        "exchange_symbol",
        "trading_symbol",
        "trading_exchange",
        "product_code",
        "expiration_date",
        "tick_size_type",
        "minimum_feed_price_change",
        "single_point_value",
        "is_tradable",
        "source_kind",
        "reference_facts",
    ):
        assert column_name in reference.c


def test_migration_is_single_head_additive_and_installs_append_only_guards(monkeypatch):
    module = _load_migration()
    assert module.down_revision == "0010_conner_nq_scout"

    declared: list[str] = []
    indexes: list[str] = []
    trigger_calls: list[bool] = []
    monkeypatch.setattr(module, "_create", lambda name, *columns, **kwargs: declared.append(name))
    monkeypatch.setattr(
        module,
        "_index",
        lambda name, table, columns: indexes.append(name),
    )
    monkeypatch.setattr(
        module,
        "_install_append_only_triggers",
        lambda: trigger_calls.append(True),
    )
    module.upgrade()

    assert tuple(declared) == CREATION_ORDER
    assert indexes
    assert trigger_calls == [True]
    assert set(module.APPEND_ONLY_TABLES) == {
        table for table in EXPECTED if table.endswith("_observations")
    } | {"v2_rithmic_broker_events"}

    migration = MIGRATION_PATH.read_text(encoding="utf-8")
    assert "BEFORE UPDATE OR DELETE" in migration
    assert "v2_reject_rithmic_history_mutation" in migration
    assert "op.add_column" not in migration
    assert 'op.drop_table("symbols")' not in migration
    assert 'op.drop_table("copy_orders")' not in migration


def test_migration_columns_constraints_and_foreign_keys_match_orm(monkeypatch):
    module = _load_migration()
    declared: dict[str, Table] = {}

    def capture(name, *items, **kwargs):
        declared[name] = Table(name, MetaData(), *items, **kwargs)

    monkeypatch.setattr(module, "_create", capture)
    monkeypatch.setattr(module, "_index", lambda name, table, columns: None)
    monkeypatch.setattr(module, "_install_append_only_triggers", lambda: None)
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
            assert migration_column.primary_key == orm_column.primary_key, (
                table_name,
                column_name,
            )
            assert {key.target_fullname for key in migration_column.foreign_keys} == {
                key.target_fullname for key in orm_column.foreign_keys
            }, (table_name, column_name)
        assert {item.name for item in migration_table.constraints if item.name} == {
            item.name for item in orm_table.constraints if item.name
        }, table_name


def test_migration_downgrade_removes_only_0011_tables_in_reverse_order(monkeypatch):
    module = _load_migration()
    dropped: list[str] = []
    statements: list[str] = []

    class Inspector:
        @staticmethod
        def get_table_names():
            return list(CREATION_ORDER) + ["symbols", "v2_futures_contracts"]

    monkeypatch.setattr(module.op, "get_bind", lambda: object())
    monkeypatch.setattr(module.sa, "inspect", lambda bind: Inspector())
    monkeypatch.setattr(module.op, "drop_table", lambda name: dropped.append(name))
    monkeypatch.setattr(module.op, "execute", lambda statement: statements.append(str(statement)))

    module.downgrade()

    assert tuple(dropped) == tuple(reversed(CREATION_ORDER))
    assert "symbols" not in dropped
    assert "v2_futures_contracts" not in dropped
    assert statements == ["DROP FUNCTION IF EXISTS v2_reject_rithmic_history_mutation()"]
