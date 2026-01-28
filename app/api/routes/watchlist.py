from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.config_store import CONFIG, persist_config, refresh_config
from app.db.session import get_session
from app.services.watchlist import get_watchlist_entries
from app.utils.time import now_et
from app.services.kv_store import StateStoreError


router = APIRouter(prefix="", tags=["watchlist"])
def db_session():
    with get_session() as session:
        yield session


@router.get("/watchlist/today")
def watchlist_today(session: Session = Depends(db_session)):
    try:
        refresh_config()
    except StateStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    trade_date: date = now_et().date()
    entries = get_watchlist_entries(session, trade_date, source="top100")
    if not entries:
        entries = get_watchlist_entries(session, trade_date, source="longlist")
    items = [
        {
            "rank": entry.rank,
            "ticker": entry.ticker,
            "gap_pct": entry.gap_pct,
            "direction": entry.direction,
            "premkt_volume": entry.premkt_volume,
            "price": entry.price,
        }
        for entry in entries
    ]
    return {"date": trade_date.isoformat(), "count": len(items), "items": items}


@router.post("/watch/pin")
def watch_pin(ticker: str):
    try:
        refresh_config()
    except StateStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    symbol = ticker.upper()
    if symbol in CONFIG.universe.blocklist:
        raise HTTPException(status_code=400, detail="Ticker is blocklisted")
    if symbol not in CONFIG.universe.pinned:
        CONFIG.universe.pinned.append(symbol)
        try:
            persist_config()
        except StateStoreError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"ok": True, "pinned": sorted(CONFIG.universe.pinned)}


@router.post("/watch/block")
def watch_block(ticker: str):
    try:
        refresh_config()
    except StateStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    symbol = ticker.upper()
    if symbol in CONFIG.universe.pinned:
        CONFIG.universe.pinned.remove(symbol)
    if symbol not in CONFIG.universe.blocklist:
        CONFIG.universe.blocklist.append(symbol)
    try:
        persist_config()
    except StateStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"ok": True, "blocklist": sorted(CONFIG.universe.blocklist)}
