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
    monkeypatch.setenv("AI_LAB_STARTING_EQUITY", "100000")
    monkeypatch.setattr(paper_trader, "get_session", lambda: FakeScope(session))

    paper_id = paper_trader.maybe_open_paper_trade_from_shadow_decision(7)

    assert paper_id == 1
    trade = session.added[0]
    assert trade.symbol == "MNY"
    assert trade.notional == 5000
    assert trade.qty == 1250
    assert trade.status == "open"


def test_paper_trade_keeps_legacy_equity_alias(monkeypatch):
    monkeypatch.delenv("AI_LAB_STARTING_EQUITY", raising=False)
    decision = SimpleNamespace(
        id=8,
        would_take=True,
        entry_price=5.0,
        stop_price=4.8,
        target_price=5.6,
        suggested_size_pct=None,
        alert_id=21,
        setup_id=22,
        symbol="LAB",
        direction="long",
        p2r=0.82,
        rr=3.0,
        confidence="medium",
        reason="legacy config",
    )
    session = FakeSession(decision=decision)
    monkeypatch.setenv("PAPER_TRADER_ACCOUNT_EQUITY", "10000")
    monkeypatch.setattr(paper_trader, "get_session", lambda: FakeScope(session))

    paper_id = paper_trader.maybe_open_paper_trade_from_shadow_decision(8)

    assert paper_id == 1
    assert session.added[0].notional == 500


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


def test_paper_promotion_readiness_passes_when_rules_are_met(monkeypatch):
    base = datetime(2026, 5, 24, 14, 0, tzinfo=timezone.utc)
    trades = []
    for idx, pnl in enumerate([120, 90, -40, 110]):
        trades.append(
            PaperTrade(
                id=idx + 10,
                symbol=f"T{idx}",
                direction="long",
                status="closed",
                opened_at=base,
                closed_at=base.replace(day=24 + (idx % 2)),
                entry_price=4.0,
                stop_price=3.8,
                target_price=4.6,
                size_pct=0.05,
                account_equity=10000,
                notional=500,
                qty=125,
                realized_pnl=pnl,
                realized_r=1.0 if pnl > 0 else -1.0,
            )
        )
    session = FakeSession(trades=trades)
    monkeypatch.setenv("PAPER_PROMOTION_MIN_CLOSED_TRADES", "4")
    monkeypatch.setenv("PAPER_PROMOTION_MIN_TRADING_DAYS", "2")
    monkeypatch.setenv("PAPER_PROMOTION_MIN_WIN_RATE", "0.50")
    monkeypatch.setenv("PAPER_PROMOTION_MIN_AVG_R", "0.20")
    monkeypatch.setenv("PAPER_PROMOTION_MAX_DRAWDOWN_PCT", "0.05")

    readiness = paper_trader.paper_promotion_readiness_from_session(session)

    assert readiness["ready"] is True
    assert readiness["summary"]["closed"] == 4
    assert readiness["summary"]["win_rate"] == 0.75
    assert readiness["summary"]["max_drawdown_pct"] == 0.004


def test_ai_lab_summary_tracks_portfolio_value_cash_and_period_pnl(monkeypatch):
    now = datetime.now(timezone.utc)
    trades = [
        PaperTrade(
            id=41,
            symbol="WIN",
            direction="long",
            status="closed",
            opened_at=now,
            closed_at=now,
            entry_price=10.0,
            size_pct=0.05,
            account_equity=100000,
            notional=5000,
            qty=500,
            realized_pnl=600,
            realized_r=1.2,
        ),
        PaperTrade(
            id=42,
            symbol="OPEN",
            direction="long",
            status="open",
            opened_at=now,
            entry_price=20.0,
            size_pct=0.05,
            account_equity=100000,
            notional=5000,
            qty=250,
            unrealized_pnl=250,
            last_price=21.0,
        ),
    ]
    monkeypatch.setenv("AI_LAB_STARTING_EQUITY", "100000")

    summary = paper_trader._paper_summary_from_rows(trades)

    assert summary["starting_equity"] == 100000
    assert summary["account_value"] == 100850
    assert summary["cash_balance"] == 95600
    assert summary["open_exposure"] == 5000
    assert summary["total_pnl"] == 850
    assert summary["today_pnl"] == 850
    assert summary["weekly_pnl"] == 850
    assert summary["total_return_pct"] == 0.0085
    assert len(summary["positions"]) == 1
    position = summary["positions"][0]
    assert position["symbol"] == "OPEN"
    assert position["market_value"] == 5250
    assert position["unrealized_pnl_pct"] == 0.05
    assert position["seconds_open"] is not None


def test_paper_promotion_readiness_blocks_too_few_trades(monkeypatch):
    trade = PaperTrade(
        id=31,
        symbol="MNY",
        direction="long",
        status="closed",
        opened_at=datetime(2026, 5, 24, 14, 0, tzinfo=timezone.utc),
        closed_at=datetime(2026, 5, 24, 14, 10, tzinfo=timezone.utc),
        entry_price=4.0,
        size_pct=0.05,
        account_equity=10000,
        notional=500,
        qty=125,
        realized_pnl=100,
        realized_r=1.0,
    )
    session = FakeSession(trades=[trade])
    monkeypatch.setenv("PAPER_PROMOTION_MIN_CLOSED_TRADES", "2")
    monkeypatch.setenv("PAPER_PROMOTION_MIN_TRADING_DAYS", "1")
    monkeypatch.setenv("PAPER_PROMOTION_MIN_WIN_RATE", "0.50")
    monkeypatch.setenv("PAPER_PROMOTION_MIN_AVG_R", "0.20")
    monkeypatch.setenv("PAPER_PROMOTION_MAX_DRAWDOWN_PCT", "0.10")

    readiness = paper_trader.paper_promotion_readiness_from_session(session)

    assert readiness["ready"] is False
    assert readiness["checks"][0]["key"] == "min_closed_trades"
    assert readiness["checks"][0]["ok"] is False


def test_paper_trades_route_is_registered():
    paths = {route.path for route in paper.router.routes}

    assert "/paper/trades" in paths
    assert "/paper/readiness" in paths


def test_paper_route_returns_service_payload(monkeypatch):
    monkeypatch.setattr(
        paper,
        "list_paper_trades",
        lambda limit: {"items": [], "count": 0, "summary": {"open": 0}, "limit": limit},
    )

    payload = paper.get_paper_trades(limit=17)

    assert payload["count"] == 0
    assert payload["limit"] == 17


def test_paper_readiness_route_returns_service_payload(monkeypatch):
    monkeypatch.setattr(
        paper,
        "paper_promotion_readiness",
        lambda: {"ready": False, "checks": [], "summary": {}},
    )

    payload = paper.get_paper_readiness()

    assert payload["ready"] is False
