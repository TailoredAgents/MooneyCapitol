from datetime import datetime, timezone
from types import SimpleNamespace

from app.api.routes.launch import _latency_stats, _read_only_counts, _read_only_history, router


class FakeScalarResult:
    def __init__(self, rows):
        self.rows = rows

    def all(self):
        return self.rows


class FakeResult:
    def __init__(self, rows):
        self.rows = rows

    def scalar(self):
        return self.rows[0] if self.rows else None

    def scalars(self):
        return FakeScalarResult(self.rows)

    def all(self):
        return self.rows


class FakeSession:
    def __init__(self, *results):
        self.results = list(results)

    def execute(self, stmt):
        return FakeResult(self.results.pop(0))


def test_launch_readiness_route_is_registered():
    paths = {route.path for route in router.routes}

    assert "/launch/readiness" in paths


def test_launch_readiness_counts_read_only_decisions():
    counts = _read_only_counts(FakeSession([3], [2]))

    assert counts == {"would_copy": 3, "blocked": 2, "total": 5}


def test_launch_readiness_returns_read_only_history_rows():
    executed_at = datetime(2026, 5, 20, 14, 30, tzinfo=timezone.utc)
    master = SimpleNamespace(
        id=11,
        executed_at=executed_at,
        received_at=executed_at,
        symbol="AAPL",
        side="BUY",
        qty=10,
        price=125.5,
    )
    order = SimpleNamespace(
        id=21,
        client_order_id="mc-test-21",
        qty=2,
        status="would_copy",
        reject_reason="read_only",
        latency_ms=82.4,
    )

    history = _read_only_history(FakeSession([(master, order, "personal")]))

    assert history == [
        {
            "master_execution_id": 11,
            "copy_order_id": 21,
            "master_executed_at": executed_at.isoformat(),
            "master_received_at": executed_at.isoformat(),
            "symbol": "AAPL",
            "side": "BUY",
            "master_qty": 10,
            "master_price": 125.5,
            "master_notional": 1255.0,
            "target": "personal",
            "client_order_id": "mc-test-21",
            "copy_qty": 2,
            "copy_notional": 251.0,
            "copy_status": "would_copy",
            "copy_reject_reason": "read_only",
            "copy_latency_ms": 82.4,
        }
    ]


def test_launch_readiness_latency_stats_include_under_300ms_rate():
    stats = _latency_stats(FakeSession([100.0, 250.0, 450.0]))

    assert stats["count"] == 3
    assert stats["mean_ms"] == 266.67
    assert stats["max_ms"] == 450.0
    assert stats["under_300ms_rate"] == 0.6667
