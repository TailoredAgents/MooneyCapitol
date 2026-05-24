from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from app.api.routes import paper
from app.db.models import PaperTrade
from app.services import paper_trader


class FakeScalarResult:
    def __init__(self, rows):
        self.rows = rows

    def __iter__(self):
        return iter(self.rows)


class FakeResult:
    def __init__(self, rows):
        self.rows = rows

    def scalar_one_or_none(self):
        return self.rows[0] if self.rows else None

    def scalars(self):
        return FakeScalarResult(self.rows)


class FakeSession:
    def __init__(self, decision=None, existing=None, trades=None):
        self.decision = decision
        self.existing = existing
        self.trades = trades or []
        self.added = []

    def get(self, model, row_id):
        return self.decision

    def execute(self, stmt):
        if self.existing is not None:
            existing = self.existing
            self.existing = None
            return FakeResult([existing])
        return FakeResult(self.trades)

    def add(self, row):
        self.added.append(row)

    def flush(self):
        for idx, row in enumerate(self.added, start=1):
            row.id = idx


class FakeScope:
    def __init__(self, session):
        self.session = session

    def __enter__(self):
        return self.session

    def __exit__(self, exc_type, exc, tb):
        return False


def test_paper_trade_opens_from_would_take_shadow_decision(monkeypatch):
    decision = SimpleNamespace(
        id=7,
        would_take=True,
        entry_price=4.0,
        stop_price=3.8,
        target_price=4.6,
        suggested_size_pct=0.05,
        alert_id=11,
        setup_id=12,
        symbol="MNY",
        direction="long",
        p2r=0.82,
        rr=3.0,
        confidence="medium",
        reason="p2R and RR met thresholds",
    )
    session = FakeSession(decision=decision)
    monkeypatch.setenv("PAPER_TRADER_ACCOUNT_EQUITY", "10000")
    monkeypatch.setattr(paper_trader, "get_session", lambda: FakeScope(session))

    paper_id = paper_trader.maybe_open_paper_trade_from_shadow_decision(7)

    assert paper_id == 1
    trade = session.added[0]
    assert trade.symbol == "MNY"
    assert trade.notional == 500
    assert trade.qty == 125
    assert trade.status == "open"


def test_paper_trade_closes_at_stop_conservatively(monkeypatch):
    trade = PaperTrade(
        id=3,
        symbol="MNY",
        direction="long",
        status="open",
        opened_at=datetime(2026, 5, 24, 14, 0, tzinfo=timezone.utc),
        entry_price=4.0,
        stop_price=3.8,
        target_price=4.6,
        size_pct=0.05,
        account_equity=10000,
        notional=500,
        qty=125,
    )
    session = FakeSession(trades=[trade])
    monkeypatch.setattr(paper_trader, "get_session", lambda: FakeScope(session))

    updated = paper_trader.update_open_paper_trades("MNY", high=4.7, low=3.75, close=4.5)

    assert updated == 1
    assert trade.status == "closed"
    assert trade.exit_reason == "stop"
    assert trade.realized_pnl == -25
    assert trade.realized_r == -1


def test_paper_trades_route_is_registered():
    paths = {route.path for route in paper.router.routes}

    assert "/paper/trades" in paths


def test_paper_route_returns_service_payload(monkeypatch):
    monkeypatch.setattr(
        paper,
        "list_paper_trades",
        lambda limit: {"items": [], "count": 0, "summary": {"open": 0}, "limit": limit},
    )

    payload = paper.get_paper_trades(limit=17)

    assert payload["count"] == 0
    assert payload["limit"] == 17
