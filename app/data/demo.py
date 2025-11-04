from __future__ import annotations

from datetime import datetime, timedelta
from random import Random
from typing import Any


_rng = Random(42)
_DEMO_BASE_PRICE = 4.0


def _gen_gap(idx: int) -> float:
    sign = 1 if idx % 2 == 0 else -1
    magnitude = 0.05 + (idx % 10) * 0.003
    return sign * magnitude


def get_demo_watchlist(limit: int = 100) -> list[dict[str, Any]]:
    items = []
    for i in range(limit):
        ticker = f"D{i:03d}"
        gap = _gen_gap(i)
        price = _DEMO_BASE_PRICE + (i % 20) * 0.12
        premkt = 20000 + (i % 15) * 1500
        items.append(
            {
                "ticker": ticker,
                "price": round(price, 2),
                "gap_pct": gap,
                "direction": "up" if gap >= 0 else "down",
                "premkt_volume": premkt,
            }
        )
    items.sort(key=lambda x: abs(x["gap_pct"]), reverse=True)
    return items[:limit]


def get_demo_prev_close(ticker: str) -> float:
    idx = int(ticker[1:]) if ticker[1:].isdigit() else 0
    return round(_DEMO_BASE_PRICE + (idx % 30) * 0.1, 2)


def get_demo_prev_volume(ticker: str) -> int:
    idx = int(ticker[1:]) if ticker[1:].isdigit() else 0
    return 150000 + (idx % 40) * 5000


def get_demo_premarket_ref_price(ticker: str) -> float:
    prev = get_demo_prev_close(ticker)
    gap = _gen_gap(int(ticker[1:]) if ticker[1:].isdigit() else 0)
    return round(prev * (1 + gap), 2)


def get_demo_candles(ticker: str, timespan: str, limit: int = 60) -> list[dict[str, Any]]:
    now = datetime.utcnow()
    candles = []
    base = get_demo_prev_close(ticker)
    increment = 0.01 if timespan == "minute" else 0.02
    for i in range(limit):
        price = base + increment * (i - limit / 2)
        o = price
        c = price + _rng.uniform(-0.02, 0.02)
        h = max(o, c) + _rng.uniform(0.0, 0.03)
        l = min(o, c) - _rng.uniform(0.0, 0.03)
        v = 10000 + i * 250
        ts = now - timedelta(minutes=limit - i)
        candles.append({"t": int(ts.timestamp() * 1000), "o": o, "h": h, "l": l, "c": c, "v": v})
    return candles[-limit:]

