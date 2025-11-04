from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

from sqlalchemy import select

from app.db.models import Fill, Setup
from app.db.session import get_session
from app.services.watchlist import get_or_create_symbol


def _parse_timestamp(value: str) -> datetime:
    if value.endswith("Z"):
        value = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def ingest_fills(fills: Iterable[dict[str, Any]]) -> int:
    inserted = 0
    with get_session() as session:
        for item in fills:
            ext_id = item.get("ext_trade_id")
            if ext_id:
                existing = session.execute(select(Fill).where(Fill.ext_trade_id == ext_id)).scalar_one_or_none()
                if existing:
                    continue
            ts = item.get("ts")
            ts_dt = _parse_timestamp(ts) if isinstance(ts, str) else ts
            symbol = item.get("symbol")
            side = item.get("side", "BUY")
            qty = int(item.get("qty", 0))
            price = float(item.get("price", 0.0))
            fee = float(item.get("fee", 0.0)) if item.get("fee") is not None else None
            account = item.get("account", "demo")
            setup_id = item.get("setup_id")

            if setup_id is None and symbol:
                # Attempt to link to latest setup for the symbol within 30 min window
                symbol_row = get_or_create_symbol(session, symbol)
                stmt = (
                    select(Setup)
                    .where(Setup.symbol_id == symbol_row.id, Setup.detected_ts <= ts_dt)
                    .order_by(Setup.detected_ts.desc())
                    .limit(1)
                )
                setup = session.execute(stmt).scalar_one_or_none()
                if setup and abs((ts_dt - setup.detected_ts).total_seconds()) <= 1800:
                    setup_id = setup.id

            fill = Fill(
                ext_trade_id=ext_id,
                account=account,
                symbol=symbol,
                ts=ts_dt,
                side=side,
                qty=qty,
                price=price,
                fee=fee,
                setup_id=setup_id,
            )
            session.add(fill)
            inserted += 1
    return inserted


class LedgerService:
    def __init__(self) -> None:
        self.demo_mode = os.getenv("TRADES_DEMO", "0").lower() in {"1", "true", "yes"}
        self._demo_loaded: set[date] = set()
        self._summary_cache: dict[date, dict[str, Any]] = {}

    def invalidate(self, trade_date: date | None = None) -> None:
        if trade_date:
            self._summary_cache.pop(trade_date, None)
        else:
            self._summary_cache.clear()

    def sync(self, trade_date: date) -> int:
        if self.demo_mode:
            if trade_date in self._demo_loaded:
                return 0
            sample_path = Path("samples/demo_fills.json")
            if not sample_path.exists():
                return 0
            data = json.loads(sample_path.read_text())
            fills = data.get("fills", [])
            inserted = ingest_fills(fills)
            if inserted:
                self._demo_loaded.add(trade_date)
                self.invalidate(trade_date)
            return inserted
        # TODO: integrate live TradeStation client when credentials available
        return 0

    def _fetch_setups(self, setup_ids: Iterable[int]) -> Dict[int, Setup]:
        ids = [sid for sid in set(setup_ids) if sid]
        if not ids:
            return {}
        with get_session() as session:
            stmt = select(Setup).where(Setup.id.in_(ids))
            rows = session.execute(stmt).scalars().all()
            return {row.id: row for row in rows}

    def _collect_fills(self, trade_date: date) -> List[Fill]:
        start = datetime.combine(trade_date, time(0), tzinfo=timezone.utc)
        end = start + timedelta(days=1)
        with get_session() as session:
            stmt = select(Fill).where(Fill.ts >= start, Fill.ts < end).order_by(Fill.ts.asc())
            return list(session.execute(stmt).scalars())

    def daily_summary(self, trade_date: date) -> dict[str, Any]:
        if trade_date in self._summary_cache:
            return self._summary_cache[trade_date]

        fills = self._collect_fills(trade_date)
        if not fills:
            summary = {
                "date": trade_date.isoformat(),
                "total_trades": 0,
                "wins": 0,
                "win_rate": 0.0,
                "avg_r": 0.0,
                "expectancy": 0.0,
                "net_pnl": 0.0,
                "by_setup": [],
                "by_hour": {},
            }
            self._summary_cache[trade_date] = summary
            return summary

        grouped: dict[int | str, dict[str, Any]] = {}
        for fill in fills:
            key = fill.setup_id if fill.setup_id is not None else f"{fill.symbol}:{fill.id}"
            record = grouped.setdefault(
                key,
                {
                    "symbol": fill.symbol,
                    "setup_id": fill.setup_id,
                    "buy_value": 0.0,
                    "sell_value": 0.0,
                    "fees": 0.0,
                    "qty": 0,
                    "first_ts": fill.ts,
                    "fills": [],
                },
            )
            side = (fill.side or "").upper()
            value = fill.price * fill.qty
            if side.startswith("S"):
                record["sell_value"] += value
            else:
                record["buy_value"] += value
            record["fees"] += fill.fee or 0.0
            record["qty"] += fill.qty
            record["fills"].append(fill)
            if fill.ts < record["first_ts"]:
                record["first_ts"] = fill.ts

        setup_ids = [r["setup_id"] for r in grouped.values() if r["setup_id"]]
        setups = self._fetch_setups(setup_ids)

        summaries = []
        wins = 0
        total_r = 0.0
        counted_r = 0
        total_pnl = 0.0
        by_hour: dict[str, float] = defaultdict(float)

        for key, record in grouped.items():
            pnl = record["sell_value"] - record["buy_value"] - record["fees"]
            setup = setups.get(record["setup_id"])
            risk_per_share = None
            direction = None
            if setup and setup.entry_price and setup.invalidation:
                risk_per_share = abs(setup.entry_price - setup.invalidation)
                direction = setup.direction
            qty = record["qty"] or 1
            realized_r = None
            if risk_per_share and risk_per_share > 0:
                realized_r = pnl / (risk_per_share * qty)
                total_r += realized_r
                counted_r += 1
            if realized_r is not None and realized_r > 0:
                wins += 1
            total_pnl += pnl
            hour_key = record["first_ts"].astimezone(timezone.utc).strftime("%H")
            by_hour[hour_key] += pnl
            summaries.append(
                {
                    "key": key,
                    "setup_id": record["setup_id"],
                    "symbol": record["symbol"],
                    "pnl": round(pnl, 2),
                    "realized_r": round(realized_r, 2) if realized_r is not None else None,
                    "direction": direction,
                }
            )

        summaries.sort(key=lambda row: row.get("pnl", 0), reverse=True)

        total_trades = len(summaries)
        win_rate = (wins / total_trades * 100.0) if total_trades else 0.0
        avg_r = (total_r / counted_r) if counted_r else 0.0
        expectancy = avg_r

        # persist realized metrics back to setups for learning
        with get_session() as session:
            for row in summaries:
                sid = row.get("setup_id")
                if not sid:
                    continue
                setup = session.get(Setup, sid)
                if not setup:
                    continue
                payload = setup.payload_json or {}
                payload.setdefault("features", {})
                payload["realized_r"] = row.get("realized_r")
                payload["pnl"] = row.get("pnl")
                setup.payload_json = payload
                session.flush()

        summary = {
            "date": trade_date.isoformat(),
            "total_trades": total_trades,
            "wins": wins,
            "win_rate": round(win_rate, 2),
            "avg_r": round(avg_r, 2),
            "expectancy": round(expectancy, 2),
            "net_pnl": round(total_pnl, 2),
            "by_setup": summaries,
            "by_hour": {k: round(v, 2) for k, v in by_hour.items()},
        }
        self._summary_cache[trade_date] = summary
        return summary

    def summary_csv(self, trade_date: date) -> str:
        import csv
        import io

        summary = self.daily_summary(trade_date)
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["setup_id", "symbol", "direction", "pnl", "realized_r"])
        for row in summary["by_setup"]:
            writer.writerow([
                row.get("setup_id"),
                row.get("symbol"),
                row.get("direction"),
                row.get("pnl"),
                row.get("realized_r"),
            ])
        return buffer.getvalue()


_LEDGER: LedgerService | None = None


def get_ledger_service() -> LedgerService:
    global _LEDGER
    if _LEDGER is None:
        _LEDGER = LedgerService()
    return _LEDGER
