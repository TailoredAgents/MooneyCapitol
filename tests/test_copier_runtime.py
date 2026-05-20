from datetime import datetime, timezone

from app.copier.engine import CopyResult, CopyTarget
from app.copier.runtime import WebullCopierRuntime
from app.copier.risk import RiskPolicy
from app.copier.sizing import SizingPolicy
from app.core.config import CopierConfig


class FakeClient:
    def place_equity_order(self, account_id, order):
        return {"order_id": "child-1"}


class FakeOrchestrator:
    def __init__(self):
        self.copied = []
        self.recorded = []
        self.planned = []

    def copy_execution(self, master, targets):
        self.copied.append((master, targets))
        return [
            CopyResult(
                target="personal",
                allowed=True,
                submitted=True,
                reason=None,
                client_order_id="mc123",
                quantity=1,
                submitted_at=datetime(2026, 5, 16, 14, 30, tzinfo=timezone.utc),
                latency_ms=12.5,
            )
        ]

    def record_master_only(self, master):
        self.recorded.append(master)
        return True

    def plan_execution(self, master, targets):
        self.planned.append((master, targets))
        return [
            CopyResult(
                target="personal",
                allowed=True,
                submitted=False,
                reason="read_only",
                client_order_id="mc123",
                quantity=1,
            )
        ]


def _target():
    return CopyTarget(
        name="personal",
        account_id="copy-acct",
        client=FakeClient(),
        sizing=SizingPolicy(mode="fixed_quantity", value=1),
        risk=RiskPolicy(enabled=True, global_kill_switch=False),
    )


def _payload(status="FILLED"):
    return {
        "account_id": "master",
        "order_id": "order-1",
        "symbol": "AAPL",
        "side": "BUY",
        "status": status,
        "filled_qty": "3",
        "avg_fill_price": "12.50",
        "filled_at": "2026-05-16T14:30:00+00:00",
    }


def _official_trade_event(scene_type="FINAL_FILLED"):
    return {
        "id": "trade-event-1",
        "event_type": "TRADE",
        "position": "cursor-1",
        "timestamp": "2026-05-16T14:30:00+00:00",
        "payload": {
            "account_id": "master",
            "order_id": "order-1",
            "client_order_id": ["master-client-1"],
            "order_status": "FILLED",
            "symbol": "AAPL",
            "qty": "3",
            "filled_qty": "3",
            "filled_price": "12.50",
            "filled_time": "1778941800000",
            "side": "BUY",
            "category": "US_STOCK",
            "scene_type": scene_type,
            "biz_type": "TRADE",
        },
    }


def test_runtime_copies_filled_webull_payload(monkeypatch):
    monkeypatch.setattr("app.copier.runtime.refresh_config", lambda: True)
    monkeypatch.setattr("app.copier.runtime.set_copier_status", lambda **updates: updates)
    orchestrator = FakeOrchestrator()
    runtime = WebullCopierRuntime(
        orchestrator=orchestrator,
        target_builder=lambda config: [_target()],
        config_provider=lambda: CopierConfig(enabled=True, global_kill_switch=False),
    )

    result = runtime.handle_webull_event("topic", "event", _payload(), None)

    assert result.processed == 1
    assert result.submitted == 1
    assert orchestrator.copied[0][0].execution_id == "order-1"
    assert orchestrator.copied[0][1][0].name == "personal"


def test_runtime_copies_official_webull_trade_event_envelope(monkeypatch):
    monkeypatch.setattr("app.copier.runtime.refresh_config", lambda: True)
    monkeypatch.setattr("app.copier.runtime.set_copier_status", lambda **updates: updates)
    orchestrator = FakeOrchestrator()
    runtime = WebullCopierRuntime(
        orchestrator=orchestrator,
        target_builder=lambda config: [_target()],
        config_provider=lambda: CopierConfig(enabled=True, global_kill_switch=False),
    )

    result = runtime.handle_webull_event(_official_trade_event())

    assert result.processed == 1
    assert result.submitted == 1
    master = orchestrator.copied[0][0]
    assert master.execution_id == "order-1"
    assert master.client_order_id == "master-client-1"
    assert master.symbol == "AAPL"


def test_runtime_ignores_official_webull_non_fill_event(monkeypatch):
    monkeypatch.setattr("app.copier.runtime.refresh_config", lambda: True)
    orchestrator = FakeOrchestrator()
    runtime = WebullCopierRuntime(
        orchestrator=orchestrator,
        target_builder=lambda config: [_target()],
        config_provider=lambda: CopierConfig(enabled=True, global_kill_switch=False),
    )

    result = runtime.handle_webull_event(_official_trade_event(scene_type="CANCEL_SUCCESS"))

    assert result.processed == 0
    assert result.ignored == 1
    assert orchestrator.copied == []


def test_runtime_reuses_target_cache_for_same_config(monkeypatch):
    monkeypatch.setattr("app.copier.runtime.refresh_config", lambda: True)
    monkeypatch.setattr("app.copier.runtime.set_copier_status", lambda **updates: updates)
    calls = []
    orchestrator = FakeOrchestrator()

    def build_targets(config):
        calls.append(config)
        return [_target()]

    runtime = WebullCopierRuntime(
        orchestrator=orchestrator,
        target_builder=build_targets,
        config_provider=lambda: CopierConfig(enabled=True, global_kill_switch=False),
    )

    runtime.handle_webull_event(_payload())
    runtime.handle_webull_event({**_payload(), "order_id": "order-2"})

    assert len(calls) == 1


def test_runtime_ignores_non_fill_payload(monkeypatch):
    monkeypatch.setattr("app.copier.runtime.refresh_config", lambda: True)
    orchestrator = FakeOrchestrator()
    runtime = WebullCopierRuntime(
        orchestrator=orchestrator,
        target_builder=lambda config: [_target()],
        config_provider=lambda: CopierConfig(enabled=True),
    )

    result = runtime.handle_webull_event(_payload(status="NEW"))

    assert result.processed == 0
    assert result.ignored == 1
    assert orchestrator.copied == []


def test_runtime_read_only_plans_without_submitting(monkeypatch):
    monkeypatch.setattr("app.copier.runtime.refresh_config", lambda: True)
    monkeypatch.setattr("app.copier.runtime.set_copier_status", lambda **updates: updates)
    orchestrator = FakeOrchestrator()
    runtime = WebullCopierRuntime(
        orchestrator=orchestrator,
        target_builder=lambda config: [_target()],
        config_provider=lambda: CopierConfig(enabled=True, mode="read_only"),
    )

    result = runtime.handle_webull_event(_payload())

    assert result.processed == 1
    assert result.submitted == 0
    assert len(orchestrator.planned) == 1
    assert orchestrator.planned[0][1][0].name == "personal"
    assert orchestrator.recorded == []
    assert orchestrator.copied == []
