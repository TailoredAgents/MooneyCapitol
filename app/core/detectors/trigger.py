from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class Candle:
    ts: int
    o: float
    h: float
    l: float
    c: float
    v: int


@dataclass
class BreakRetest:
    break_ts: int
    retest_ts: int
    direction: str
    entry_price: float
    invalidation: float
    volume_burst_rvol: float
    spread_cents: float


def check_break_and_retest(
    candles: Iterable[Candle],
    box_hi: float,
    box_lo: float,
    close_outside_ticks: int,
    burst_rvol: float,
    retest_max_bars: int,
) -> BreakRetest | None:
    candles = list(candles)
    if len(candles) < 10:
        return None

    tick_size = 0.01 * close_outside_ticks

    def _avg_volume(idx: int, lookback: int = 5) -> float:
        start = max(0, idx - lookback)
        window = candles[start:idx]
        if not window:
            return 0.0
        return sum(c.v for c in window) / len(window)

    for idx in range(len(candles) - 1, 4, -1):
        candle = candles[idx]
        avg_vol = _avg_volume(idx)
        if avg_vol <= 0:
            continue
        ratio = candle.v / avg_vol
        if ratio < burst_rvol:
            continue

        spread_cents = (candle.h - candle.l) * 100

        if candle.c >= box_hi + tick_size:
            direction = "long"
            entry = box_hi
            stop = box_lo
            # seek retest in following bars
            for j in range(idx + 1, min(len(candles), idx + 1 + retest_max_bars)):
                retest = candles[j]
                if retest.l <= box_hi + tick_size and retest.h >= box_hi - tick_size:
                    return BreakRetest(
                        break_ts=candle.ts,
                        retest_ts=retest.ts,
                        direction=direction,
                        entry_price=entry,
                        invalidation=stop,
                        volume_burst_rvol=ratio,
                        spread_cents=spread_cents,
                    )

        if candle.c <= box_lo - tick_size:
            direction = "short"
            entry = box_lo
            stop = box_hi
            for j in range(idx + 1, min(len(candles), idx + 1 + retest_max_bars)):
                retest = candles[j]
                if retest.h >= box_lo - tick_size and retest.l <= box_lo + tick_size:
                    return BreakRetest(
                        break_ts=candle.ts,
                        retest_ts=retest.ts,
                        direction=direction,
                        entry_price=entry,
                        invalidation=stop,
                        volume_burst_rvol=ratio,
                        spread_cents=spread_cents,
                    )

    return None
