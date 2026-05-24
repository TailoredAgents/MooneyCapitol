from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select

from app.db.models import PaperTrade, ShadowDecision
from app.db.session import get_session
from app.observability.logging import get_logger


logger = get_logger("paper_trader")
MODEL_VERSION = "paper-v1"


def _enabled() -> bool:
    return os.getenv("PAPER_TRADER_ENABLED", "1").lower() in {"1", "true", "yes", "on"}


def _account_equity() -> float:
    return max(0.0, float(os.getenv("PAPER_TRADER_ACCOUNT_EQUITY", "10000")))


def _risk_per_share(direction: str, entry: float, stop: float | None) -> float | None:
    if stop is None:
        return None
    risk = entry - stop if direction == "long" else stop - entry
    return risk if risk > 0 else None


def _pnl(direction: str, qty: float, entry: float, price: float) -> float:
    return (price - entry) * qty if direction == "long" else (entry - price) * qty


def maybe_open_paper_trade_from_shadow_decision(shadow_decision_id: int | None) -> int | None:
    if not _enabled() or shadow_decision_id is None:
        return None
    with get_session() as session:
        decision = session.get(ShadowDecision, shadow_decision_id)
        if not decision or not decision.would_take:
            return None
        existing = session.execute(
            select(PaperTrade).where(PaperTrade.shadow_decision_id == decision.id)
        ).scalar_one_or_none()
        if existing:
            return int(existing.id)
        if not decision.entry_price or decision.entry_price <= 0:
            return None

        equity = _account_equity()
        size_pct = float(decision.suggested_size_pct or float(os.getenv("PAPER_TRADER_SIZE_PCT", "0.05")))
        notional = equity * max(0.0, size_pct)
        qty = notional / decision.entry_price if decision.entry_price > 0 else 0.0
        if qty <= 0:
            return None

        now = datetime.now(timezone.utc)
        trade = PaperTrade(
            shadow_decision_id=decision.id,
            alert_id=decision.alert_id,
            setup_id=decision.setup_id,
            symbol=decision.symbol,
            direction=decision.direction or "long",
            status="open",
            opened_at=now,
            entry_price=decision.entry_price,
            stop_price=decision.stop_price,
            target_price=decision.target_price,
            size_pct=size_pct,
            account_equity=equity,
            notional=notional,
            qty=qty,
            last_price=decision.entry_price,
            last_mark_at=now,
            model_version=MODEL_VERSION,
            reason=decision.reason,
            raw_context={
                "shadow_decision_id": decision.id,
                "p2r": decision.p2r,
                "rr": decision.rr,
                "confidence": decision.confidence,
            },
        )
        session.add(trade)
        session.flush()
        logger.info("paper_trade.opened", paper_trade_id=trade.id, symbol=trade.symbol, direction=trade.direction)
        return int(trade.id)


def update_open_paper_trades(symbol: str, *, high: float, low: float, close: float, mark_at: datetime | None = None) -> int:
    if not _enabled():
        return 0
    mark_at = mark_at or datetime.now(timezone.utc)
    symbol_upper = symbol.upper()
    updated = 0
    with get_session() as session:
        trades = list(
            session.execute(
                select(PaperTrade).where(PaperTrade.symbol == symbol_upper, PaperTrade.status == "open")
            ).scalars()
        )
        for trade in trades:
            exit_price = None
            exit_reason = None
            direction = str(trade.direction or "long").lower()
            if direction == "long":
                # Conservative same-candle assumption: if stop and target both touch,
                # count the stop first.
                if trade.stop_price is not None and low <= trade.stop_price:
                    exit_price = trade.stop_price
                    exit_reason = "stop"
                elif trade.target_price is not None and high >= trade.target_price:
                    exit_price = trade.target_price
                    exit_reason = "target"
            else:
                if trade.stop_price is not None and high >= trade.stop_price:
                    exit_price = trade.stop_price
                    exit_reason = "stop"
                elif trade.target_price is not None and low <= trade.target_price:
                    exit_price = trade.target_price
                    exit_reason = "target"

            trade.last_price = close
            trade.last_mark_at = mark_at
            trade.unrealized_pnl = round(_pnl(direction, trade.qty, trade.entry_price, close), 4)
            if exit_price is not None:
                risk = _risk_per_share(direction, trade.entry_price, trade.stop_price)
                realized_pnl = _pnl(direction, trade.qty, trade.entry_price, exit_price)
                trade.status = "closed"
                trade.closed_at = mark_at
                trade.exit_price = exit_price
                trade.exit_reason = exit_reason
                trade.realized_pnl = round(realized_pnl, 4)
                trade.realized_r = round((realized_pnl / (risk * trade.qty)), 4) if risk and trade.qty else None
                trade.unrealized_pnl = 0.0
            updated += 1
        return updated


def list_paper_trades(limit: int = 100) -> dict[str, Any]:
    limit = max(1, min(int(limit), 250))
    with get_session() as session:
        trades = list(
            session.execute(select(PaperTrade).order_by(PaperTrade.opened_at.desc()).limit(limit)).scalars()
        )
        return {"items": [_serialize(row) for row in trades], "count": len(trades), "summary": paper_summary_from_session(session)}


def paper_summary_from_session(session) -> dict[str, Any]:
    rows = list(session.execute(select(PaperTrade)).scalars())
    closed = [row for row in rows if row.status == "closed"]
    wins = [row for row in closed if (row.realized_pnl or 0.0) > 0]
    total_realized = sum(float(row.realized_pnl or 0.0) for row in closed)
    open_unrealized = sum(float(row.unrealized_pnl or 0.0) for row in rows if row.status == "open")
    avg_r = sum(float(row.realized_r or 0.0) for row in closed) / len(closed) if closed else None
    return {
        "total": len(rows),
        "open": sum(1 for row in rows if row.status == "open"),
        "closed": len(closed),
        "wins": len(wins),
        "win_rate": round(len(wins) / len(closed), 4) if closed else None,
        "realized_pnl": round(total_realized, 2),
        "open_unrealized_pnl": round(open_unrealized, 2),
        "avg_r": round(avg_r, 4) if avg_r is not None else None,
    }


def _serialize(row: PaperTrade) -> dict[str, Any]:
    return {
        "id": row.id,
        "shadow_decision_id": row.shadow_decision_id,
        "alert_id": row.alert_id,
        "setup_id": row.setup_id,
        "symbol": row.symbol,
        "direction": row.direction,
        "status": row.status,
        "opened_at": row.opened_at.isoformat() if row.opened_at else None,
        "closed_at": row.closed_at.isoformat() if row.closed_at else None,
        "entry_price": row.entry_price,
        "stop_price": row.stop_price,
        "target_price": row.target_price,
        "exit_price": row.exit_price,
        "exit_reason": row.exit_reason,
        "size_pct": row.size_pct,
        "account_equity": row.account_equity,
        "notional": row.notional,
        "qty": row.qty,
        "unrealized_pnl": row.unrealized_pnl,
        "realized_pnl": row.realized_pnl,
        "realized_r": row.realized_r,
        "last_price": row.last_price,
        "last_mark_at": row.last_mark_at.isoformat() if row.last_mark_at else None,
        "reason": row.reason,
    }
