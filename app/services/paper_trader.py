from __future__ import annotations

import os
from datetime import datetime, time, timedelta, timezone
from typing import Any

from sqlalchemy import select

from app.db.models import Fill, PaperTrade, ShadowDecision
from app.db.session import get_session
from app.observability.logging import get_logger


logger = get_logger("paper_trader")
MODEL_VERSION = "paper-v1"


def _enabled() -> bool:
    value = os.getenv("AI_LAB_ENABLED", os.getenv("PAPER_TRADER_ENABLED", "1"))
    return value.lower() in {"1", "true", "yes", "on"}


def _account_equity() -> float:
    value = os.getenv("AI_LAB_STARTING_EQUITY", os.getenv("PAPER_TRADER_ACCOUNT_EQUITY", "100000"))
    return max(0.0, float(value))


def _max_open_positions() -> int:
    return max(1, int(os.getenv("AI_LAB_MAX_OPEN_POSITIONS", "5")))


def _max_hold_minutes() -> int:
    return max(0, int(os.getenv("AI_LAB_MAX_HOLD_MINUTES", "390")))


def _no_duplicate_symbols() -> bool:
    return os.getenv("AI_LAB_NO_DUPLICATE_SYMBOLS", "1").lower() in {"1", "true", "yes", "on"}


def _confidence_size_mult(p2r: float | None) -> float:
    if p2r is not None and p2r >= 0.85:
        return float(os.getenv("AI_LAB_HIGH_CONF_SIZE_MULT", "1.5"))
    return 1.0


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

        # Load open positions once for entry guards
        open_trades = list(
            session.execute(select(PaperTrade).where(PaperTrade.status == "open")).scalars()
        )

        # Block duplicate symbol — one open trade per symbol at a time
        if _no_duplicate_symbols():
            sym = str(decision.symbol or "").upper()
            if any(str(t.symbol or "").upper() == sym for t in open_trades):
                logger.info("paper_trade.skipped.duplicate_symbol", symbol=decision.symbol)
                return None

        # Block entry when at the max concurrent open position limit
        if len(open_trades) >= _max_open_positions():
            logger.info("paper_trade.skipped.max_positions", open=len(open_trades), max=_max_open_positions())
            return None

        equity = _account_equity()
        base_size = float(decision.suggested_size_pct or float(os.getenv("AI_LAB_SIZE_PCT", os.getenv("PAPER_TRADER_SIZE_PCT", "0.05"))))
        size_pct = round(base_size * _confidence_size_mult(decision.p2r), 6)
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
        max_hold = _max_hold_minutes()
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

            # Time-decay exit: close at current price after max hold time
            if exit_price is None and max_hold > 0 and trade.opened_at is not None:
                minutes_open = (mark_at - _as_utc(trade.opened_at)).total_seconds() / 60
                if minutes_open >= max_hold:
                    exit_price = close
                    exit_reason = "timeout"

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


def paper_promotion_readiness() -> dict[str, Any]:
    with get_session() as session:
        return paper_promotion_readiness_from_session(session)


def paper_promotion_readiness_from_session(session) -> dict[str, Any]:
    rows = list(session.execute(select(PaperTrade)).scalars())
    closed = sorted(
        [row for row in rows if row.status == "closed"],
        key=lambda row: row.closed_at or row.opened_at or datetime.min.replace(tzinfo=timezone.utc),
    )
    summary = _paper_summary_from_rows(rows)
    min_closed = int(os.getenv("PAPER_PROMOTION_MIN_CLOSED_TRADES", "50"))
    min_days = int(os.getenv("PAPER_PROMOTION_MIN_TRADING_DAYS", "3"))
    min_win_rate = float(os.getenv("PAPER_PROMOTION_MIN_WIN_RATE", "0.55"))
    min_avg_r = float(os.getenv("PAPER_PROMOTION_MIN_AVG_R", "0.20"))
    max_drawdown_pct_limit = float(os.getenv("PAPER_PROMOTION_MAX_DRAWDOWN_PCT", "0.10"))

    trading_days = {
        (row.closed_at or row.opened_at).date().isoformat()
        for row in closed
        if (row.closed_at or row.opened_at) is not None
    }
    drawdown = _max_realized_drawdown(closed)
    checks = [
        {
            "key": "min_closed_trades",
            "ok": len(closed) >= min_closed,
            "label": f"At least {min_closed} closed AI paper trades",
            "actual": len(closed),
            "required": min_closed,
        },
        {
            "key": "min_trading_days",
            "ok": len(trading_days) >= min_days,
            "label": f"At least {min_days} trading days covered",
            "actual": len(trading_days),
            "required": min_days,
        },
        {
            "key": "min_win_rate",
            "ok": summary["win_rate"] is not None and summary["win_rate"] >= min_win_rate,
            "label": f"Win rate is at least {min_win_rate:.0%}",
            "actual": summary["win_rate"],
            "required": min_win_rate,
        },
        {
            "key": "min_avg_r",
            "ok": summary["avg_r"] is not None and summary["avg_r"] >= min_avg_r,
            "label": f"Average R is at least {min_avg_r:.2f}",
            "actual": summary["avg_r"],
            "required": min_avg_r,
        },
        {
            "key": "max_drawdown",
            "ok": drawdown["max_drawdown_pct"] is not None and drawdown["max_drawdown_pct"] <= max_drawdown_pct_limit,
            "label": f"Realized drawdown stays under {max_drawdown_pct_limit:.0%}",
            "actual": drawdown["max_drawdown_pct"],
            "required": max_drawdown_pct_limit,
        },
    ]
    ready = bool(checks) and all(check["ok"] for check in checks)
    return {
        "ready": ready,
        "status": "passed" if ready else "observing",
        "message": "AI paper trader has met promotion rules." if ready else "AI paper trader is still being observed.",
        "checks": checks,
        "summary": summary | drawdown | {"trading_days": len(trading_days)},
    }


def paper_summary_from_session(session) -> dict[str, Any]:
    rows = list(session.execute(select(PaperTrade)).scalars())
    shadow_map = _shadow_decision_map(session, rows)
    taken_map = _master_taken_map(session, shadow_map.values())
    return _paper_summary_from_rows(rows, shadow_map=shadow_map, taken_map=taken_map)


def _paper_summary_from_rows(
    rows: list[PaperTrade],
    *,
    shadow_map: dict[int, ShadowDecision] | None = None,
    taken_map: dict[int, bool] | None = None,
) -> dict[str, Any]:
    shadow_map = shadow_map or {}
    taken_map = taken_map or {}
    closed = [row for row in rows if row.status == "closed"]
    open_rows = [row for row in rows if row.status == "open"]
    wins = [row for row in closed if (row.realized_pnl or 0.0) > 0]
    losses = [row for row in closed if (row.realized_pnl or 0.0) < 0]
    matched_master = [row for row in rows if _master_taken_for_trade(row, shadow_map, taken_map)]
    starting_equity = _account_equity()
    total_realized = sum(float(row.realized_pnl or 0.0) for row in closed)
    open_unrealized = sum(float(row.unrealized_pnl or 0.0) for row in open_rows)
    open_notional = sum(float(row.notional or 0.0) for row in open_rows)
    total_pnl = total_realized + open_unrealized
    account_value = starting_equity + total_pnl
    cash_balance = starting_equity + total_realized - open_notional
    now = datetime.now(timezone.utc)
    today_start = datetime.combine(now.date(), time.min, tzinfo=timezone.utc)
    week_start = today_start - timedelta(days=today_start.weekday())
    today_realized = _realized_since(closed, today_start)
    week_realized = _realized_since(closed, week_start)
    drawdown = _max_realized_drawdown(closed, starting_equity=starting_equity)
    avg_r = sum(float(row.realized_r or 0.0) for row in closed) / len(closed) if closed else None
    return {
        "starting_equity": round(starting_equity, 2),
        "account_value": round(account_value, 2),
        "cash_balance": round(cash_balance, 2),
        "open_exposure": round(open_notional, 2),
        "open_exposure_pct": round(open_notional / account_value, 6) if account_value > 0 else None,
        "total_pnl": round(total_pnl, 2),
        "total_return_pct": round(total_pnl / starting_equity, 6) if starting_equity > 0 else None,
        "today_pnl": round(today_realized + open_unrealized, 2),
        "today_realized_pnl": round(today_realized, 2),
        "weekly_pnl": round(week_realized + open_unrealized, 2),
        "weekly_realized_pnl": round(week_realized, 2),
        "max_drawdown": drawdown["max_drawdown"] or 0.0,
        "max_drawdown_pct": drawdown["max_drawdown_pct"] or 0.0,
        "total": len(rows),
        "open": len(open_rows),
        "positions": [_position_from_trade(row, now=now) for row in sorted(open_rows, key=lambda item: item.opened_at or now, reverse=True)],
        "recent_closed": [
            _performance_trade(row, shadow_map, taken_map)
            for row in sorted(closed, key=lambda item: item.closed_at or item.opened_at or datetime.min.replace(tzinfo=timezone.utc), reverse=True)[:10]
        ],
        "best_trades": [
            _performance_trade(row, shadow_map, taken_map)
            for row in sorted(closed, key=lambda item: float(item.realized_pnl or 0.0), reverse=True)[:5]
        ],
        "worst_trades": [
            _performance_trade(row, shadow_map, taken_map)
            for row in sorted(closed, key=lambda item: float(item.realized_pnl or 0.0))[:5]
        ],
        "ai_vs_connor": {
            "lab_trades": len(rows),
            "closed_lab_trades": len(closed),
            "matched_connor": len(matched_master),
            "match_rate": round(len(matched_master) / len(rows), 4) if rows else None,
            "connor_also_took_winners": sum(1 for row in matched_master if float(row.realized_pnl or 0.0) > 0),
            "connor_also_took_losers": sum(1 for row in matched_master if float(row.realized_pnl or 0.0) < 0),
        },
        "period_breakdown": _period_breakdown(rows),
        "symbol_stats": _symbol_stats(rows),
        "exit_reason_stats": _exit_reason_stats(rows),
        "closed": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(closed), 4) if closed else None,
        "realized_pnl": round(total_realized, 2),
        "open_unrealized_pnl": round(open_unrealized, 2),
        "avg_r": round(avg_r, 4) if avg_r is not None else None,
    }


def _avg_hold_seconds(rows: list[PaperTrade]) -> int | None:
    holds = [
        int((_as_utc(row.closed_at) - _as_utc(row.opened_at)).total_seconds())
        for row in rows
        if row.opened_at and row.closed_at
    ]
    return int(sum(holds) / len(holds)) if holds else None


def _period_stats(rows: list[PaperTrade], *, start: datetime | None = None) -> dict[str, Any]:
    closed = [row for row in rows if row.status == "closed"]
    if start is not None:
        closed = [row for row in closed if row.closed_at is not None and _as_utc(row.closed_at) >= start]
    wins = [row for row in closed if (row.realized_pnl or 0.0) > 0]
    losses = [row for row in closed if (row.realized_pnl or 0.0) < 0]
    realized_pnl = sum(float(row.realized_pnl or 0.0) for row in closed)
    avg_r_val = sum(float(row.realized_r or 0.0) for row in closed) / len(closed) if closed else None
    return {
        "trades": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(closed), 4) if closed else None,
        "realized_pnl": round(realized_pnl, 2),
        "avg_r": round(avg_r_val, 4) if avg_r_val is not None else None,
        "avg_hold_seconds": _avg_hold_seconds(closed),
    }


def _period_breakdown(rows: list[PaperTrade]) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    today_start = datetime.combine(now.date(), time.min, tzinfo=timezone.utc)
    week_start = today_start - timedelta(days=today_start.weekday())
    last_30_start = today_start - timedelta(days=30)
    return {
        "today": _period_stats(rows, start=today_start),
        "this_week": _period_stats(rows, start=week_start),
        "last_30_days": _period_stats(rows, start=last_30_start),
        "all_time": _period_stats(rows),
    }


def _symbol_stats(rows: list[PaperTrade], *, top_n: int = 10) -> list[dict[str, Any]]:
    closed = [row for row in rows if row.status == "closed"]
    by_symbol: dict[str, list[PaperTrade]] = {}
    for row in closed:
        sym = str(row.symbol or "").upper()
        by_symbol.setdefault(sym, []).append(row)
    result = []
    for sym, sym_rows in by_symbol.items():
        wins = [r for r in sym_rows if (r.realized_pnl or 0.0) > 0]
        realized_pnl = sum(float(r.realized_pnl or 0.0) for r in sym_rows)
        avg_r_val = sum(float(r.realized_r or 0.0) for r in sym_rows) / len(sym_rows)
        result.append({
            "symbol": sym,
            "trades": len(sym_rows),
            "wins": len(wins),
            "win_rate": round(len(wins) / len(sym_rows), 4),
            "realized_pnl": round(realized_pnl, 2),
            "avg_r": round(avg_r_val, 4),
        })
    result.sort(key=lambda item: item["realized_pnl"], reverse=True)
    return result[:top_n]


def _exit_reason_stats(rows: list[PaperTrade]) -> dict[str, Any]:
    closed = [row for row in rows if row.status == "closed"]
    by_reason: dict[str, list[float]] = {}
    for row in closed:
        reason = str(row.exit_reason or "other").lower()
        by_reason.setdefault(reason, []).append(float(row.realized_pnl or 0.0))
    result = {}
    for reason, pnls in by_reason.items():
        wins = sum(1 for p in pnls if p > 0)
        result[reason] = {
            "trades": len(pnls),
            "wins": wins,
            "win_rate": round(wins / len(pnls), 4) if pnls else None,
            "realized_pnl": round(sum(pnls), 2),
        }
    return result


def _shadow_decision_map(session, rows: list[PaperTrade]) -> dict[int, ShadowDecision]:
    ids = [int(row.shadow_decision_id) for row in rows if row.shadow_decision_id is not None]
    if not ids:
        return {}
    decisions = list(session.execute(select(ShadowDecision).where(ShadowDecision.id.in_(ids))).scalars())
    return {int(row.id): row for row in decisions if row.id is not None}


def _master_taken_map(session, decisions) -> dict[int, bool]:
    setup_ids = [int(row.setup_id) for row in decisions if row.setup_id is not None]
    if not setup_ids:
        return {}
    rows = session.execute(
        select(Fill.setup_id).where(Fill.setup_id.in_(setup_ids)).group_by(Fill.setup_id)
    ).all()
    return {int(setup_id): True for setup_id, in rows if setup_id is not None}


def _master_taken_for_trade(row: PaperTrade, shadow_map: dict[int, ShadowDecision], taken_map: dict[int, bool]) -> bool:
    if row.shadow_decision_id is None:
        return False
    decision = shadow_map.get(int(row.shadow_decision_id))
    return bool(decision and decision.setup_id is not None and taken_map.get(int(decision.setup_id), False))


def _realized_since(closed: list[PaperTrade], start: datetime) -> float:
    return sum(
        float(row.realized_pnl or 0.0)
        for row in closed
        if row.closed_at is not None and _as_utc(row.closed_at) >= start
    )


def _max_realized_drawdown(closed: list[PaperTrade], *, starting_equity: float | None = None) -> dict[str, float | None]:
    if not closed:
        return {"max_drawdown": None, "max_drawdown_pct": None}
    equity = float(starting_equity if starting_equity is not None else closed[0].account_equity or _account_equity() or 0.0)
    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for row in sorted(closed, key=lambda item: _as_utc(item.closed_at or item.opened_at or datetime.min.replace(tzinfo=timezone.utc))):
        cumulative += float(row.realized_pnl or 0.0)
        peak = max(peak, cumulative)
        max_drawdown = max(max_drawdown, peak - cumulative)
    return {
        "max_drawdown": round(max_drawdown, 2),
        "max_drawdown_pct": round(max_drawdown / equity, 6) if equity > 0 else None,
    }


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _position_from_trade(row: PaperTrade, *, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    opened_at = _as_utc(row.opened_at) if row.opened_at else None
    seconds_open = int((now - opened_at).total_seconds()) if opened_at else None
    last_price = row.last_price if row.last_price is not None else row.entry_price
    market_value = float(row.qty or 0.0) * float(last_price or 0.0)
    return {
        "id": row.id,
        "symbol": row.symbol,
        "direction": row.direction,
        "opened_at": row.opened_at.isoformat() if row.opened_at else None,
        "seconds_open": seconds_open,
        "qty": row.qty,
        "entry_price": row.entry_price,
        "last_price": last_price,
        "market_value": round(market_value, 2),
        "notional": row.notional,
        "stop_price": row.stop_price,
        "target_price": row.target_price,
        "unrealized_pnl": row.unrealized_pnl,
        "unrealized_pnl_pct": round(float(row.unrealized_pnl or 0.0) / float(row.notional or 0.0), 6) if row.notional else None,
        "size_pct": row.size_pct,
        "reason": row.reason,
    }


def _performance_trade(row: PaperTrade, shadow_map: dict[int, ShadowDecision], taken_map: dict[int, bool]) -> dict[str, Any]:
    return _serialize(row) | {
        "master_taken": _master_taken_for_trade(row, shadow_map, taken_map),
        "pnl": row.realized_pnl if row.status == "closed" else row.unrealized_pnl,
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
