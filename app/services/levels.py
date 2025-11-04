from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass
class LevelContext:
    swing_high: float | None
    swing_low: float | None
    recent_highs: list[float]
    recent_lows: list[float]
    gap_high: float | None
    gap_low: float | None


def _recent_extrema(candles: Iterable[dict], lookback: int = 20) -> tuple[list[float], list[float]]:
    highs: list[float] = []
    lows: list[float] = []
    for idx, candle in enumerate(list(candles)[-lookback:]):
        highs.append(float(candle.get("h", 0.0)))
        lows.append(float(candle.get("l", 0.0)))
    highs = sorted(set(round(h, 4) for h in highs), reverse=True)
    lows = sorted(set(round(l, 4) for l in lows))
    return highs, lows


def find_htf_levels(candles_5m: Iterable[dict], candles_15m: Iterable[dict]) -> dict:
    """Derive simple higher-timeframe levels from 5m/15m context.

    Strategy: take the nearest swing highs/lows from the last 20 bars on both
    5m and 15m, deduplicate, and return ordered lists for target lookup.
    """

    levels_up: list[float] = []
    levels_down: list[float] = []

    for candles in (candles_5m, candles_15m):
        highs, lows = _recent_extrema(candles, lookback=40)
        levels_up.extend(highs[:8])
        levels_down.extend(lows[:8])

    levels_up = sorted({round(level, 4) for level in levels_up})
    levels_down = sorted({round(level, 4) for level in levels_down})

    swing_high = levels_up[-1] if levels_up else None
    swing_low = levels_down[0] if levels_down else None

    return {
        "swing_high": swing_high,
        "swing_low": swing_low,
        "levels_up": levels_up,
        "levels_down": levels_down,
    }


def find_gap_edges(prev_close: float | None, today_open_candles: Iterable[dict]) -> dict:
    candles_list = list(today_open_candles)
    if not candles_list:
        return {"gap_high": None, "gap_low": None}
    highs = [float(c.get("h", 0.0)) for c in candles_list[:6]]
    lows = [float(c.get("l", 0.0)) for c in candles_list[:6]]
    gap_high = max(highs) if highs else None
    gap_low = min(lows) if lows else None
    return {
        "gap_high": gap_high,
        "gap_low": gap_low,
        "prev_close": prev_close,
    }


def nearest_target(direction: str, price: float, levels: dict) -> Optional[float]:
    candidates: list[float] = []

    if direction == "long":
        for lvl in levels.get("levels_up", []):
            if lvl > price:
                candidates.append(lvl)
        gap_high = levels.get("gap_high")
        if gap_high and gap_high > price:
            candidates.append(gap_high)
        prev_close = levels.get("prev_close")
        if prev_close and prev_close > price:
            candidates.append(prev_close)
        return min(candidates) if candidates else None

    if direction == "short":
        for lvl in levels.get("levels_down", []):
            if lvl < price:
                candidates.append(lvl)
        gap_low = levels.get("gap_low")
        if gap_low and gap_low < price:
            candidates.append(gap_low)
        prev_close = levels.get("prev_close")
        if prev_close and prev_close < price:
            candidates.append(prev_close)
        return max(candidates) if candidates else None

    return None

