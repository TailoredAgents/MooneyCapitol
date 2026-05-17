from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from app.copier.engine import CopyTarget
from app.copier.reconciliation import (
    CopyPositionReconciler,
    CopierStartupRecovery,
    ReconciliationResult,
    apply_order_detail,
    mark_stale_open_orders,
    normalize_order_detail,
    normalize_positions,
    record_order_mismatches,
)
from app.copier.risk import RiskPolicy
from app.copier.sizing import SizingPolicy
from app.core.config import CopierConfig


class FakeSession:
    def __init__(self):
        self.added = []

    def add(self, row):
        self.added.append(row)


class FakeResult:
    def __init__(self, rows):
        self.rows = rows

    def all(self):
        return self.rows

    def scalars(self):
        return self


class FakePositionClient:
    def __init__(self, positions):
        self.positions = positions

    def get_account_positions(self, account_id):
        return self.positions


def _order(status="submitted", qty=3.0):
    return SimpleNamespace(
        id=11,
        target_account_id=7,
        client_order_id="mc123",
        broker_order_id=None,
        symbol="AAPL",
        qty=qty,
        status=status,
        filled_qty=None,
        avg_fill_price=None,
        reject_reason=None,
        accepted_at=None,
        filled_at=None,
        raw_response_payload=None,
    )


def _target_row():
    return SimpleNamespace(id=7, name="personal")


def test_normalize_order_detail_maps_webull_filled_payload():
    detail = normalize_order_detail(
        {
            "data": {
                "orderId": "child-1",
                "orderStatus": "FILLED",
                "filledQuantity": "3",
                "averagePrice": "12.50",
                "updatedAt": "2026-05-16T14:31:00+00:00",
            }
        }
    )

    assert detail.status == "filled"
    assert detail.broker_order_id == "child-1"
    assert detail.filled_qty == 3.0
    assert detail.avg_fill_price == 12.5
    assert detail.filled_at == datetime(2026, 5, 16, 14, 31, tzinfo=timezone.utc)


def test_normalize_positions_maps_webull_payload_variants():
    positions = normalize_positions(
        {
            "data": {
                "positions": [
                    {
                        "symbol": "aapl",
                        "positionQty": "3",
                        "avgPrice": "12.50",
                        "marketValue": "37.50",
                    }
                ]
            }
        }
    )

    assert positions[0].symbol == "AAPL"
    assert positions[0].qty == 3.0
    assert positions[0].avg_price == 12.5
    assert positions[0].market_value == 37.5


def test_apply_order_detail_updates_order_and_records_event():
    session = FakeSession()
    order = _order()
    detail = normalize_order_detail(
        {
            "order_id": "child-1",
            "status": "FILLED",
            "filled_qty": "3",
            "avg_fill_price": "12.50",
            "filled_at": "2026-05-16T14:31:00+00:00",
        }
    )

    changed = apply_order_detail(session, order, detail)

    assert changed is True
    assert order.status == "filled"
    assert order.broker_order_id == "child-1"
    assert order.filled_qty == 3.0
    assert order.avg_fill_price == 12.5
    assert session.added[0].event_type == "reconciled"
    assert session.added[0].status == "filled"


def test_record_order_mismatches_creates_reconciliation_for_reject():
    session = FakeSession()
    order = _order()
    detail = normalize_order_detail({"status": "REJECTED", "reject_reason": "insufficient buying power"})

    created = record_order_mismatches(session, order, 7, detail)

    assert created is True
    assert session.added[0].severity == "error"
    assert "rejected" in session.added[0].message
    assert session.added[0].raw_context["client_order_id"] == "mc123"


def test_record_order_mismatches_creates_reconciliation_for_short_fill():
    session = FakeSession()
    order = _order(qty=3.0)
    detail = normalize_order_detail({"status": "FILLED", "filled_qty": "2", "avg_fill_price": "12.50"})

    created = record_order_mismatches(session, order, 7, detail)

    assert created is True
    assert session.added[0].severity == "warning"
    assert "filled 2.0 of expected 3.0" in session.added[0].message


def test_mark_stale_open_orders_creates_single_reconciliation():
    order = _order(status="submitted")
    order.submitted_at = datetime(2026, 5, 16, 14, 0, tzinfo=timezone.utc)

    class Session(FakeSession):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def execute(self, stmt):
            self.calls += 1
            if self.calls == 1:
                return FakeResult([(order, _target_row())])
            return FakeResult([])

    session = Session()

    count = mark_stale_open_orders(
        session,
        stale_after=timedelta(minutes=10),
        now=datetime(2026, 5, 16, 14, 30, tzinfo=timezone.utc),
    )

    assert count == 1
    assert session.added[0].severity == "warning"
    assert "still submitted" in session.added[0].message
    assert session.added[0].raw_context["stale_after_seconds"] == 600.0


def test_startup_recovery_runs_reconcile_then_stale_scan(monkeypatch):
    calls = []

    class Reconciler:
        def reconcile_open_orders(self, limit):
            calls.append(("reconcile", limit))
            return ReconciliationResult(checked=2, updated=1, mismatches=0, errors=0)

    @contextmanager
    def session_scope():
        yield "session"

    def fake_stale(session, stale_after, limit, now=None):
        calls.append(("stale", session, limit, stale_after.total_seconds()))
        return 3

    monkeypatch.setattr("app.copier.reconciliation.mark_stale_open_orders", fake_stale)
    recovery = CopierStartupRecovery(
        reconciler=Reconciler(),
        session_scope=session_scope,
        stale_after_minutes=15,
    )

    result = recovery.recover(limit=25)

    assert result.checked == 2
    assert result.updated == 1
    assert result.stale_open_orders == 3
    assert calls == [("reconcile", 25), ("stale", "session", 25, 900.0)]


def test_position_reconciler_creates_mismatch_reconciliation(monkeypatch):
    class Session(FakeSession):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def execute(self, stmt):
            self.calls += 1
            if self.calls == 1:
                return FakeResult([("personal", "AAPL", "BUY", 3.0, 3.0, "filled")])
            if self.calls == 2:
                return FakeResult([SimpleNamespace(id=7, name="personal")])
            return FakeResult([])

    session = Session()

    @contextmanager
    def session_scope():
        yield session

    target = CopyTarget(
        name="personal",
        account_id="copy-acct",
        client=FakePositionClient([{"symbol": "AAPL", "quantity": "2"}]),
        sizing=SizingPolicy(mode="percent_equity"),
        risk=RiskPolicy(enabled=True, global_kill_switch=False),
    )
    monkeypatch.setattr("app.copier.reconciliation.refresh_config", lambda: True)
    reconciler = CopyPositionReconciler(
        session_scope=session_scope,
        config_provider=lambda: CopierConfig(enabled=True),
        target_builder=lambda config: [target],
    )

    result = reconciler.sync_positions()

    assert result.targets_checked == 1
    assert result.symbols_checked == 1
    assert result.mismatches == 1
    assert session.added[0].symbol == "AAPL"
    assert "position mismatch" in session.added[0].message
    assert session.added[0].raw_context["local_qty"] == 3.0
    assert session.added[0].raw_context["webull_qty"] == 2.0
