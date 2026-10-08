from pathlib import Path
import importlib.util

from sqlalchemy import Integer, Numeric

from app.db.models import Base


EXPECTED = {
    "v2_futures_contracts",
    "v2_contract_mappings",
    "v2_broker_connections",
    "v2_broker_accounts",
    "v2_risk_profiles",
    "v2_session_schedules",
    "v2_opportunity_snapshots",
    "v2_trade_plan_versions",
    "v2_master_orders",
    "v2_order_events",
    "v2_master_executions",
    "v2_trade_management_events",
    "v2_master_trades",
    "v2_copy_intents",
    "v2_follower_orders",
    "v2_order_links",
    "v2_follower_executions",
    "v2_follower_trades",
    "v2_position_snapshots",
    "v2_account_risk_snapshots",
    "v2_reconciliation_incidents",
    "v2_model_artifacts",
    "v2_execution_leases",
}


def test_v2_schema_is_additive_and_registered():
    assert EXPECTED <= set(Base.metadata.tables)
    assert "symbols" in Base.metadata.tables
    assert "copy_orders" in Base.metadata.tables


def test_contract_quantities_are_integer_and_money_is_numeric():
    for table_name, quantity_name in (
        ("v2_master_orders", "quantity"),
        ("v2_master_executions", "quantity"),
        ("v2_copy_intents", "intended_quantity"),
        ("v2_follower_orders", "quantity"),
        ("v2_follower_executions", "quantity"),
        ("v2_position_snapshots", "quantity"),
    ):
        assert isinstance(Base.metadata.tables[table_name].c[quantity_name].type, Integer)
    assert isinstance(Base.metadata.tables["v2_futures_contracts"].c.tick_value.type, Numeric)
    assert isinstance(Base.metadata.tables["v2_master_trades"].c.realized_r.type, Numeric)


def test_migration_is_single_head_additive_and_never_targets_v1_tables_for_drop():
    migration = Path("migrations/versions/0009_v2_futures_foundation.py").read_text(encoding="utf-8")
    assert 'down_revision = "0008_paper_trades"' in migration
    assert "op.create_table" in migration
    assert '"v2_' in migration
    assert 'op.drop_table("symbols")' not in migration
    assert 'op.drop_table("copy_orders")' not in migration


def test_migration_upgrade_declares_every_v2_table(monkeypatch):
    path = Path("migrations/versions/0009_v2_futures_foundation.py")
    spec = importlib.util.spec_from_file_location("v2_migration_0009", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    declared = []
    monkeypatch.setattr(module, "_create", lambda name, *columns, **kwargs: declared.append(name))
    monkeypatch.setattr(module, "_index", lambda name, table, columns: None)
    module.upgrade()
    assert set(declared) == EXPECTED
    assert len(declared) == len(EXPECTED)
