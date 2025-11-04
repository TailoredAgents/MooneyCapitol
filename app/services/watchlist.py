from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, List

from sqlalchemy import select, delete

from app.db.models import WatchlistEntry, Symbol


@dataclass
class WatchlistItem:
    ticker: str
    price: float
    gap_pct: float
    direction: str
    premkt_volume: int
    rank: int | None = None

    def as_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "price": self.price,
            "gap_pct": self.gap_pct,
            "direction": self.direction,
            "premkt_volume": self.premkt_volume,
            "rank": self.rank,
        }


def upsert_watchlist_entries(session, trade_date: date, entries: Iterable[WatchlistItem], source: str = "top100") -> None:
    entries = list(entries)
    session.execute(
        delete(WatchlistEntry).where(WatchlistEntry.trade_date == trade_date, WatchlistEntry.source == source)
    )
    for idx, item in enumerate(entries, start=1):
        session.add(
            WatchlistEntry(
                trade_date=trade_date,
                ticker=item.ticker.upper(),
                rank=item.rank or idx,
                gap_pct=item.gap_pct,
                direction=item.direction,
                premkt_volume=item.premkt_volume,
                price=item.price,
                source=source,
            )
        )


def get_watchlist_entries(session, trade_date: date, source: str = "top100") -> List[WatchlistEntry]:
    stmt = (
        select(WatchlistEntry)
        .where(WatchlistEntry.trade_date == trade_date, WatchlistEntry.source == source)
        .order_by(WatchlistEntry.rank.asc())
    )
    return list(session.execute(stmt).scalars())


def get_or_create_symbol(session, ticker: str, exchange: str | None = None) -> Symbol:
    stmt = select(Symbol).where(Symbol.ticker == ticker)
    symbol = session.execute(stmt).scalar_one_or_none()
    if symbol:
        return symbol
    symbol = Symbol(ticker=ticker, exchange=exchange)
    session.add(symbol)
    session.flush()
    return symbol
