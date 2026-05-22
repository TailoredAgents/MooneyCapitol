from datetime import datetime, timezone
from types import SimpleNamespace

from app.api.routes.copier import (
    CopierSettingsUpdate,
    CopyTargetUpdate,
    copier_readiness,
    disable_kill_switch,
    recent_audit_events,
    recent_copier_errors,
    recent_copy_orders,
    recent_copier_trades,
    recent_master_executions,
    recent_reconciliations,
    update_copier_settings,
    update_copier_target,
    _serialize_copier_trade,
)
from app.core.config import CopierConfig, CopyTargetAccountConfig
from app.core.config_store import CONFIG


class FakeScalarResult:
    def __init__(self, rows):
        self.rows = rows

    def all(self):
        return self.rows

    def first(self):
        return self.rows[0] if self.rows else None


class FakeResult:
    def __init__(self, rows):
        self.rows = rows

    def all(self):
        return self.rows

    def scalars(self):
        return FakeScalarResult(self.rows)

    def scalar_one_or_none(self):
        if not self.rows:
            return None
        first = self.rows[0]
        if isinstance(first, tuple):
            return first[0]
        return first


class FakeSession:
    def __init__(self, *results):
        self.results = list(results)

    def execute(self, stmt):
        return FakeResult(self.results.pop(0))


def _dt():
    return datetime(2026, 5, 16, 14, 30, tzinfo=timezone.utc)


def test_recent_master_executions_serializes_rows():
    row = SimpleNamespace(
        id=1,
        broker="webull",
        account_ref="master",
        broker_execution_id="exec-1",
        broker_order_id="order-1",
        symbol="AAPL",
        side="BUY",
        qty=3.0,
        price=12.5,
        asset_class="equity",
        executed_at=_dt(),
        received_at=_dt(),
    )

    payload = recent_master_executions(limit=50, session=FakeSession([row]))

    assert payload["count"] == 1
    assert payload["items"][0]["broker_execution_id"] == "exec-1"
    assert payload["items"][0]["executed_at"] == "2026-05-16T14:30:00+00:00"


def test_recent_copy_orders_includes_target_name():
    order = SimpleNamespace(
        id=2,
        master_execution_id=1,
        target_account_id=10,
        broker="webull",
        client_order_id="mc123",
        broker_order_id="child-1",
        symbol="AAPL",
        side="BUY",
        qty=3.0,
        order_type="market",
        time_in_force="day",
        status="submitted",
        submitted_at=_dt(),
        accepted_at=None,
        filled_at=None,
        filled_qty=None,
        avg_fill_price=None,
        reject_reason=None,
        latency_ms=11.2,
    )

    payload = recent_copy_orders(limit=50, session=FakeSession([(order, "personal")]))

    assert payload["count"] == 1
    assert payload["items"][0]["target"] == "personal"
    assert payload["items"][0]["client_order_id"] == "mc123"
    assert payload["items"][0]["latency_ms"] == 11.2


def test_recent_copier_trades_joins_master_and_copy_order():
    master = SimpleNamespace(
        id=1,
        broker_execution_id="exec-1",
        broker_order_id="order-1",
        account_ref="master",
        symbol="AAPL",
        side="BUY",
        qty=10.0,
        price=25.0,
        executed_at=_dt(),
        received_at=_dt(),
    )
    order = SimpleNamespace(
        id=2,
        client_order_id="mc123",
        broker_order_id="child-1",
        qty=4.0,
        status="filled",
        submitted_at=_dt(),
        accepted_at=_dt(),
        filled_at=_dt(),
        filled_qty=4.0,
        avg_fill_price=25.05,
        latency_ms=82.4,
        reject_reason=None,
    )

    payload = recent_copier_trades(limit=50, session=FakeSession([(master, order, "personal")]))

    item = payload["items"][0]
    assert item["symbol"] == "AAPL"
    assert item["target"] == "personal"
    assert item["copy_latency_ms"] == 82.4
    assert item["master_notional"] == 250.0
    assert item["copy_notional"] == 100.0
    assert item["copy_filled_notional"] == 100.2
    assert round(item["copy_slippage_bps"], 2) == 20.0
    assert item["ai_journal"] is None


def test_serialize_copier_trade_includes_ai_journal_text():
    master = SimpleNamespace(
        id=1,
        broker_execution_id="exec-1",
        broker_order_id="order-1",
        account_ref="master",
        symbol="AAPL",
        side="BUY",
        qty=10.0,
        price=25.0,
        executed_at=_dt(),
        received_at=_dt(),
    )
    order = SimpleNamespace(
        id=2,
        client_order_id="mc123",
        broker_order_id="child-1",
        qty=4.0,
        status="filled",
        submitted_at=_dt(),
        accepted_at=_dt(),
        filled_at=_dt(),
        filled_qty=4.0,
        avg_fill_price=25.05,
        latency_ms=82.4,
        reject_reason=None,
    )

    payload = _serialize_copier_trade(master, order, "personal", "- Copied under 300 ms.")

    assert payload["ai_journal"] == "- Copied under 300 ms."


def test_recent_audit_events_serializes_payload():
    event = SimpleNamespace(
        id=3,
        event_type="copy_blocked",
        actor="operator",
        target_account_id=10,
        message="blocked",
        created_at=_dt(),
        payload={"reason": "global_kill_switch"},
    )

    payload = recent_audit_events(limit=50, session=FakeSession([event]))

    assert payload["count"] == 1
    assert payload["items"][0]["event_type"] == "copy_blocked"
    assert payload["items"][0]["payload"]["reason"] == "global_kill_switch"


def test_recent_copier_errors_returns_failed_orders_and_audit_events():
    order = SimpleNamespace(
        id=4,
        master_execution_id=1,
        target_account_id=10,
        broker="webull",
        client_order_id="mc456",
        broker_order_id=None,
        symbol="AAPL",
        side="BUY",
        qty=3.0,
        order_type="market",
        time_in_force="day",
        status="submit_failed",
        submitted_at=_dt(),
        accepted_at=None,
        filled_at=None,
        filled_qty=None,
        avg_fill_price=None,
        reject_reason="broker unavailable",
        latency_ms=30.0,
    )
    event = SimpleNamespace(
        id=5,
        event_type="copy_submit_failed",
        actor=None,
        target_account_id=10,
        message="failed",
        created_at=_dt(),
        payload={"error": "broker unavailable"},
    )

    payload = recent_copier_errors(limit=50, session=FakeSession([(order, "personal")], [event]))

    assert payload["copy_orders"][0]["status"] == "submit_failed"
    assert payload["copy_orders"][0]["reject_reason"] == "broker unavailable"
    assert payload["audit_events"][0]["event_type"] == "copy_submit_failed"


def test_recent_reconciliations_serializes_rows():
    row = SimpleNamespace(
        id=6,
        target_account_id=10,
        symbol="AAPL",
        severity="error",
        status="open",
        message="Copied order rejected",
        detected_at=_dt(),
        resolved_at=None,
        raw_context={"client_order_id": "mc789"},
    )

    payload = recent_reconciliations(limit=50, session=FakeSession([row]))

    assert payload["count"] == 1
    assert payload["items"][0]["severity"] == "error"
    assert payload["items"][0]["raw_context"]["client_order_id"] == "mc789"


def test_update_copier_settings_updates_master_equity_and_audits(monkeypatch):
    original = CONFIG.copier.model_dump()
    audits = []
    monkeypatch.setattr("app.api.routes.copier.refresh_config", lambda: True)
    monkeypatch.setattr("app.api.routes.copier.persist_config", lambda: None)
    monkeypatch.setattr("app.api.routes.copier.record_audit_event", lambda **kwargs: audits.append(kwargs))
    try:
        CONFIG.copier = CopierConfig(enabled=False, mode="test")

        payload = update_copier_settings(
            CopierSettingsUpdate(enabled=True, master_equity=30_000),
            x_actor="tester",
            x_confirm="ENABLE_COPIER",
        )

        assert payload["enabled"] is True
        assert CONFIG.copier.master_equity == 30_000
        assert audits[0]["event_type"] == "copier_settings_updated"
    finally:
        CONFIG.copier = CopierConfig.model_validate(original)


def test_update_copier_target_sets_percent_equity_controls(monkeypatch):
    original = CONFIG.copier.model_dump()
    audits = []
    monkeypatch.setattr("app.api.routes.copier.refresh_config", lambda: True)
    monkeypatch.setattr("app.api.routes.copier.persist_config", lambda: None)
    monkeypatch.setattr("app.api.routes.copier.record_audit_event", lambda **kwargs: audits.append(kwargs))
    try:
        CONFIG.copier = CopierConfig(
            enabled=True,
            mode="test",
            master_equity=30_000,
            targets=[
                CopyTargetAccountConfig(
                    name="personal",
                    endpoint_env="WEBULL_PERSONAL_API_ENDPOINT",
                    api_key_env="WEBULL_PERSONAL_APP_KEY",
                    api_secret_env="WEBULL_PERSONAL_APP_SECRET",
                    account_id_env="WEBULL_PERSONAL_ACCOUNT_ID",
                    sizing_mode="disabled",
                )
            ],
        )

        payload = update_copier_target(
            "personal",
            CopyTargetUpdate(
                enabled=True,
                equity=10_000,
                sizing_mode="percent_equity",
                min_notional=25,
                max_notional_per_trade=750,
                max_position_pct=2.0,
                max_daily_notional=2_500,
                max_daily_trades=20,
                allowlist=["aapl", " AAPL ", "tsla"],
            ),
            x_actor="tester",
            x_confirm="ENABLE_TARGET:personal",
        )

        assert payload["enabled"] is True
        assert payload["equity"] == 10_000
        assert payload["sizing_mode"] == "percent_equity"
        assert payload["min_notional"] == 25
        assert payload["max_position_pct"] == 2.0
        assert payload["max_daily_notional"] == 2_500
        assert payload["max_daily_trades"] == 20
        assert payload["allowlist"] == ["AAPL", "TSLA"]
        assert audits[0]["event_type"] == "copy_target_updated"
    finally:
        CONFIG.copier = CopierConfig.model_validate(original)


def test_update_copier_target_rejects_enable_without_global_copier(monkeypatch):
    original = CONFIG.copier.model_dump()
    monkeypatch.setattr("app.api.routes.copier.refresh_config", lambda: True)
    try:
        CONFIG.copier = CopierConfig(
            enabled=False,
            master_equity=30_000,
            targets=[
                CopyTargetAccountConfig(
                    name="personal",
                    endpoint_env="WEBULL_PERSONAL_API_ENDPOINT",
                    api_key_env="WEBULL_PERSONAL_APP_KEY",
                    api_secret_env="WEBULL_PERSONAL_APP_SECRET",
                    account_id_env="WEBULL_PERSONAL_ACCOUNT_ID",
                    equity=10_000,
                    sizing_mode="percent_equity",
                )
            ],
        )

        import pytest

        with pytest.raises(Exception) as exc:
            update_copier_target("personal", CopyTargetUpdate(enabled=True))

        assert "Confirmation required" in exc.value.detail
        assert CONFIG.copier.targets[0].enabled is False
    finally:
        CONFIG.copier = CopierConfig.model_validate(original)


def test_update_copier_settings_rejects_enable_without_confirmation(monkeypatch):
    original = CONFIG.copier.model_dump()
    monkeypatch.setattr("app.api.routes.copier.refresh_config", lambda: True)
    try:
        CONFIG.copier = CopierConfig(enabled=False, mode="test")

        import pytest

        with pytest.raises(Exception) as exc:
            update_copier_settings(CopierSettingsUpdate(enabled=True))

        assert exc.value.detail == "Confirmation required: ENABLE_COPIER"
        assert CONFIG.copier.enabled is False
    finally:
        CONFIG.copier = CopierConfig.model_validate(original)


def test_disable_kill_switch_requires_confirmation(monkeypatch):
    original = CONFIG.copier.model_dump()
    monkeypatch.setattr("app.api.routes.copier.refresh_config", lambda: True)
    try:
        CONFIG.copier = CopierConfig(mode="test", global_kill_switch=True)

        import pytest

        with pytest.raises(Exception) as exc:
            disable_kill_switch()

        assert exc.value.detail == "Confirmation required: DISABLE_KILL_SWITCH"
        assert CONFIG.copier.global_kill_switch is True
    finally:
        CONFIG.copier = CopierConfig.model_validate(original)


def test_disable_kill_switch_accepts_confirmation(monkeypatch):
    original = CONFIG.copier.model_dump()
    audits = []
    monkeypatch.setattr("app.api.routes.copier.refresh_config", lambda: True)
    monkeypatch.setattr("app.api.routes.copier.persist_config", lambda: None)
    monkeypatch.setattr("app.api.routes.copier.record_audit_event", lambda **kwargs: audits.append(kwargs))
    try:
        CONFIG.copier = CopierConfig(mode="test", global_kill_switch=True)

        payload = disable_kill_switch(x_actor="tester", x_confirm="DISABLE_KILL_SWITCH")

        assert payload["global_kill_switch"] is False
        assert audits[0]["event_type"] == "kill_switch_disabled"
    finally:
        CONFIG.copier = CopierConfig.model_validate(original)


def test_copier_readiness_reports_blockers_when_missing_config(monkeypatch):
    original = CONFIG.copier.model_dump()
    monkeypatch.setattr("app.api.routes.copier.refresh_config", lambda: True)
    try:
        CONFIG.copier = CopierConfig(enabled=False, mode="test", master_equity=None, targets=[])

        payload = copier_readiness(session=FakeSession([], [], []))

        assert payload["ready"] is False
        keys = {item["key"] for item in payload["blockers"]}
        assert "copier_enabled" in keys
        assert "master_account" in keys
        assert "master_credentials" in keys
        assert "enabled_targets" in keys
    finally:
        CONFIG.copier = CopierConfig.model_validate(original)


def test_copier_readiness_passes_with_configured_test_target(monkeypatch):
    original = CONFIG.copier.model_dump()
    monkeypatch.setattr("app.api.routes.copier.refresh_config", lambda: True)
    monkeypatch.setenv("WEBULL_MASTER_ACCOUNT_ID", "master")
    monkeypatch.setenv("WEBULL_MASTER_API_ENDPOINT", "endpoint")
    monkeypatch.setenv("WEBULL_MASTER_EVENTS_ENDPOINT", "events-endpoint")
    monkeypatch.setenv("WEBULL_MASTER_APP_KEY", "key")
    monkeypatch.setenv("WEBULL_MASTER_APP_SECRET", "secret")
    monkeypatch.setenv("WEBULL_PERSONAL_ACCOUNT_ID", "copy")
    monkeypatch.setenv("WEBULL_PERSONAL_API_ENDPOINT", "endpoint")
    monkeypatch.setenv("WEBULL_PERSONAL_APP_KEY", "key")
    monkeypatch.setenv("WEBULL_PERSONAL_APP_SECRET", "secret")
    try:
        CONFIG.copier = CopierConfig(
            enabled=True,
            mode="test",
            global_kill_switch=True,
            master_equity=30_000,
            targets=[
                CopyTargetAccountConfig(
                    name="personal",
                    endpoint_env="WEBULL_PERSONAL_API_ENDPOINT",
                    api_key_env="WEBULL_PERSONAL_APP_KEY",
                    api_secret_env="WEBULL_PERSONAL_APP_SECRET",
                    account_id_env="WEBULL_PERSONAL_ACCOUNT_ID",
                    enabled=True,
                    equity=10_000,
                    sizing_mode="percent_equity",
                )
            ],
        )

        payload = copier_readiness(session=FakeSession([], [], [], [], [], []))

        assert payload["ready"] is True
        assert payload["ready_to_disable_kill_switch"] is True
        assert payload["blockers"] == []
        assert "target:personal:risk_caps" not in {check["key"] for check in payload["checks"]}
    finally:
        CONFIG.copier = CopierConfig.model_validate(original)
