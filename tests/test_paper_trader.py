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


class FakeSessionMulti:
    """FakeSession that pops successive result lists for each execute call."""

    def __init__(self, decision=None, results=None):
        self.decision = decision
        self._results = list(results or [])
        self.added = []

    def get(self, model, row_id):
        return self.decision

    def execute(self, stmt):
        rows = self._results.pop(0) if self._results else []
        return FakeResult(rows)

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
            shadow_decision_id=99,
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
    matched = paper_trader._paper_summary_from_rows(
        trades,
        shadow_map={99: SimpleNamespace(setup_id=123)},
        taken_map={123: True},
    )
    assert matched["ai_vs_connor"]["matched_connor"] == 1
    assert matched["ai_vs_connor"]["match_rate"] == 0.5
    assert matched["recent_closed"][0]["symbol"] == "WIN"
    assert matched["recent_closed"][0]["master_taken"] is True
    assert matched["best_trades"][0]["symbol"] == "WIN"


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


def test_period_breakdown_includes_all_slices():
    now = datetime.now(timezone.utc)
    trades = [
        PaperTrade(
            id=50,
            symbol="AAPL",
            direction="long",
            status="closed",
            opened_at=now,
            closed_at=now,
            entry_price=10.0,
            size_pct=0.05,
            account_equity=100000,
            notional=5000,
            qty=500,
            realized_pnl=200,
            realized_r=1.0,
        ),
        PaperTrade(
            id=51,
            symbol="TSLA",
            direction="long",
            status="closed",
            opened_at=now,
            closed_at=now,
            entry_price=20.0,
            size_pct=0.05,
            account_equity=100000,
            notional=5000,
            qty=250,
            realized_pnl=-100,
            realized_r=-1.0,
        ),
    ]

    breakdown = paper_trader._period_breakdown(trades)

    assert set(breakdown.keys()) == {"today", "this_week", "last_30_days", "all_time"}
    all_time = breakdown["all_time"]
    assert all_time["trades"] == 2
    assert all_time["wins"] == 1
    assert all_time["losses"] == 1
    assert all_time["win_rate"] == 0.5
    assert all_time["realized_pnl"] == 100
    # trades happened now so they should appear in every period
    for period in breakdown.values():
        assert period["trades"] == 2


def test_symbol_stats_ranks_by_pnl():
    now = datetime.now(timezone.utc)
    trades = [
        PaperTrade(
            id=60, symbol="AAPL", direction="long", status="closed",
            opened_at=now, closed_at=now, entry_price=10.0,
            size_pct=0.05, account_equity=100000, notional=5000, qty=500,
            realized_pnl=500, realized_r=2.0,
        ),
        PaperTrade(
            id=61, symbol="TSLA", direction="long", status="closed",
            opened_at=now, closed_at=now, entry_price=20.0,
            size_pct=0.05, account_equity=100000, notional=5000, qty=250,
            realized_pnl=-200, realized_r=-1.0,
        ),
        PaperTrade(
            id=62, symbol="AAPL", direction="long", status="closed",
            opened_at=now, closed_at=now, entry_price=11.0,
            size_pct=0.05, account_equity=100000, notional=5000, qty=454,
            realized_pnl=300, realized_r=1.5,
        ),
    ]

    stats = paper_trader._symbol_stats(trades)

    assert stats[0]["symbol"] == "AAPL"
    assert stats[0]["trades"] == 2
    assert stats[0]["wins"] == 2
    assert stats[0]["realized_pnl"] == 800
    assert stats[1]["symbol"] == "TSLA"
    assert stats[1]["realized_pnl"] == -200


def test_exit_reason_stats_groups_by_reason():
    now = datetime.now(timezone.utc)
    trades = [
        PaperTrade(
            id=70, symbol="X", direction="long", status="closed",
            opened_at=now, closed_at=now, entry_price=5.0,
            size_pct=0.05, account_equity=10000, notional=500, qty=100,
            realized_pnl=100, realized_r=1.0, exit_reason="target",
        ),
        PaperTrade(
            id=71, symbol="Y", direction="long", status="closed",
            opened_at=now, closed_at=now, entry_price=5.0,
            size_pct=0.05, account_equity=10000, notional=500, qty=100,
            realized_pnl=-50, realized_r=-1.0, exit_reason="stop",
        ),
        PaperTrade(
            id=72, symbol="Z", direction="long", status="closed",
            opened_at=now, closed_at=now, entry_price=5.0,
            size_pct=0.05, account_equity=10000, notional=500, qty=100,
            realized_pnl=80, realized_r=0.8, exit_reason="target",
        ),
    ]

    stats = paper_trader._exit_reason_stats(trades)

    assert "target" in stats
    assert "stop" in stats
    assert stats["target"]["trades"] == 2
    assert stats["target"]["wins"] == 2
    assert stats["target"]["win_rate"] == 1.0
    assert stats["stop"]["trades"] == 1
    assert stats["stop"]["wins"] == 0
    assert stats["stop"]["win_rate"] == 0.0


def test_summary_includes_period_breakdown_and_symbol_stats(monkeypatch):
    monkeypatch.setenv("AI_LAB_STARTING_EQUITY", "100000")
    now = datetime.now(timezone.utc)
    trade = PaperTrade(
        id=80, symbol="MNY", direction="long", status="closed",
        opened_at=now, closed_at=now, entry_price=4.0,
        size_pct=0.05, account_equity=100000, notional=5000, qty=1250,
        realized_pnl=250, realized_r=1.0, exit_reason="target",
    )

    summary = paper_trader._paper_summary_from_rows([trade])

    assert "period_breakdown" in summary
    assert "symbol_stats" in summary
    assert "exit_reason_stats" in summary
    assert summary["period_breakdown"]["all_time"]["trades"] == 1
    assert summary["symbol_stats"][0]["symbol"] == "MNY"
    assert "target" in summary["exit_reason_stats"]


# ── Phase 6: entry guards and smarter exits ──────────────────────────────────


def _make_decision(symbol="MNY", p2r=0.80, rr=3.0):
    return SimpleNamespace(
        id=7,
        would_take=True,
        entry_price=4.0,
        stop_price=3.8,
        target_price=4.6,
        suggested_size_pct=None,
        alert_id=11,
        setup_id=12,
        symbol=symbol,
        direction="long",
        p2r=p2r,
        rr=rr,
        confidence="medium",
        reason="p2R and RR met thresholds",
    )


def _open_trade(symbol="OTHER", trade_id=99):
    return PaperTrade(
        id=trade_id,
        symbol=symbol,
        direction="long",
        status="open",
        opened_at=datetime.now(timezone.utc),
        entry_price=5.0,
        size_pct=0.05,
        account_equity=100000,
        notional=5000,
        qty=1000,
    )


def test_duplicate_symbol_blocks_entry(monkeypatch):
    decision = _make_decision(symbol="MNY")
    open_in_mny = _open_trade(symbol="MNY")
    session = FakeSessionMulti(
        decision=decision,
        results=[
            [],              # existing shadow_decision_id check → none
            [open_in_mny],  # open trades → MNY already open
        ],
    )
    monkeypatch.setenv("AI_LAB_ENABLED", "1")
    monkeypatch.setenv("AI_LAB_NO_DUPLICATE_SYMBOLS", "1")
    monkeypatch.setattr(paper_trader, "get_session", lambda: FakeScope(session))

    result = paper_trader.maybe_open_paper_trade_from_shadow_decision(7)

    assert result is None
    assert session.added == []


def test_duplicate_symbol_allows_different_symbol(monkeypatch):
    decision = _make_decision(symbol="MNY")
    open_in_other = _open_trade(symbol="TSLA")
    session = FakeSessionMulti(
        decision=decision,
        results=[
            [],              # no existing by shadow_decision_id
            [open_in_other], # open trades — different symbol, should not block
        ],
    )
    monkeypatch.setenv("AI_LAB_ENABLED", "1")
    monkeypatch.setenv("AI_LAB_STARTING_EQUITY", "100000")
    monkeypatch.setenv("AI_LAB_NO_DUPLICATE_SYMBOLS", "1")
    monkeypatch.setenv("AI_LAB_MAX_OPEN_POSITIONS", "5")
    monkeypatch.setattr(paper_trader, "get_session", lambda: FakeScope(session))

    result = paper_trader.maybe_open_paper_trade_from_shadow_decision(7)

    assert result == 1
    assert len(session.added) == 1


def test_max_open_positions_blocks_entry(monkeypatch):
    decision = _make_decision(symbol="NEW")
    full_book = [_open_trade(symbol=f"T{i}", trade_id=100 + i) for i in range(5)]
    session = FakeSessionMulti(
        decision=decision,
        results=[
            [],         # no existing by shadow_decision_id
            full_book,  # 5 open trades → at limit
        ],
    )
    monkeypatch.setenv("AI_LAB_ENABLED", "1")
    monkeypatch.setenv("AI_LAB_MAX_OPEN_POSITIONS", "5")
    monkeypatch.setattr(paper_trader, "get_session", lambda: FakeScope(session))

    result = paper_trader.maybe_open_paper_trade_from_shadow_decision(7)

    assert result is None
    assert session.added == []


def test_high_confidence_entry_uses_larger_size(monkeypatch):
    decision = _make_decision(symbol="HI", p2r=0.90)
    session = FakeSessionMulti(
        decision=decision,
        results=[[], []],  # no existing, no open trades
    )
    monkeypatch.setenv("AI_LAB_ENABLED", "1")
    monkeypatch.setenv("AI_LAB_STARTING_EQUITY", "100000")
    monkeypatch.setenv("AI_LAB_SIZE_PCT", "0.05")
    monkeypatch.setenv("AI_LAB_HIGH_CONF_SIZE_MULT", "1.5")
    monkeypatch.setenv("AI_LAB_MAX_OPEN_POSITIONS", "5")
    monkeypatch.setattr(paper_trader, "get_session", lambda: FakeScope(session))

    paper_trader.maybe_open_paper_trade_from_shadow_decision(7)

    trade = session.added[0]
    assert trade.size_pct == 0.075        # 5% * 1.5
    assert trade.notional == 7500.0       # 100000 * 7.5%


def test_medium_confidence_entry_uses_base_size(monkeypatch):
    decision = _make_decision(symbol="MED", p2r=0.75)
    session = FakeSessionMulti(
        decision=decision,
        results=[[], []],
    )
    monkeypatch.setenv("AI_LAB_ENABLED", "1")
    monkeypatch.setenv("AI_LAB_STARTING_EQUITY", "100000")
    monkeypatch.setenv("AI_LAB_SIZE_PCT", "0.05")
    monkeypatch.setenv("AI_LAB_HIGH_CONF_SIZE_MULT", "1.5")
    monkeypatch.setenv("AI_LAB_MAX_OPEN_POSITIONS", "5")
    monkeypatch.setattr(paper_trader, "get_session", lambda: FakeScope(session))

    paper_trader.maybe_open_paper_trade_from_shadow_decision(7)

    trade = session.added[0]
    assert trade.size_pct == 0.05         # no multiplier for medium confidence
    assert trade.notional == 5000.0


def test_timeout_closes_trade_at_close_price(monkeypatch):
    now = datetime.now(timezone.utc)
    from datetime import timedelta
    opened = now - timedelta(minutes=400)  # well past 390-minute default
    trade = PaperTrade(
        id=90,
        symbol="TMO",
        direction="long",
        status="open",
        opened_at=opened,
        entry_price=10.0,
        size_pct=0.05,
        account_equity=100000,
        notional=5000,
        qty=500,
    )
    session = FakeSession(trades=[trade])
    monkeypatch.setenv("AI_LAB_MAX_HOLD_MINUTES", "390")
    monkeypatch.setattr(paper_trader, "get_session", lambda: FakeScope(session))

    paper_trader.update_open_paper_trades("TMO", high=10.5, low=9.8, close=10.2, mark_at=now)

    assert trade.status == "closed"
    assert trade.exit_reason == "timeout"
    assert trade.exit_price == 10.2


def test_timeout_does_not_fire_before_threshold(monkeypatch):
    now = datetime.now(timezone.utc)
    from datetime import timedelta
    opened = now - timedelta(minutes=100)  # short hold, under threshold
    trade = PaperTrade(
        id=91,
        symbol="TMO",
        direction="long",
        status="open",
        opened_at=opened,
        entry_price=10.0,
        size_pct=0.05,
        account_equity=100000,
        notional=5000,
        qty=500,
        stop_price=9.5,
        target_price=11.5,
    )
    session = FakeSession(trades=[trade])
    monkeypatch.setenv("AI_LAB_MAX_HOLD_MINUTES", "390")
    monkeypatch.setattr(paper_trader, "get_session", lambda: FakeScope(session))

    paper_trader.update_open_paper_trades("TMO", high=10.5, low=9.8, close=10.2, mark_at=now)

    assert trade.status == "open"


def test_timeout_disabled_when_set_to_zero(monkeypatch):
    now = datetime.now(timezone.utc)
    from datetime import timedelta
    opened = now - timedelta(minutes=9999)
    trade = PaperTrade(
        id=92,
        symbol="TMO",
        direction="long",
        status="open",
        opened_at=opened,
        entry_price=10.0,
        size_pct=0.05,
        account_equity=100000,
        notional=5000,
        qty=500,
        stop_price=9.5,
        target_price=11.5,
    )
    session = FakeSession(trades=[trade])
    monkeypatch.setenv("AI_LAB_MAX_HOLD_MINUTES", "0")
    monkeypatch.setattr(paper_trader, "get_session", lambda: FakeScope(session))

    paper_trader.update_open_paper_trades("TMO", high=10.5, low=9.8, close=10.2, mark_at=now)

    assert trade.status == "open"
