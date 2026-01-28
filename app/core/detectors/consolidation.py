from __future__ import annotations

from dataclasses import dataclass
from statistics import median, pstdev
from typing import Iterable


def _calc_atr(candles: list[Candle], period: int = 14) -> float:
    if len(candles) < period + 1:
        return 0.0
    trs = []
    for idx in range(1, len(candles)):
        curr = candles[idx]
        prev = candles[idx - 1]
        tr = max(curr.h - curr.l, abs(curr.h - prev.c), abs(curr.l - prev.c))
        trs.append(tr)
    if len(trs) < period:
        return 0.0
    return sum(trs[-period:]) / period


def _calc_rvol(candles: list[Candle], lookback: int = 20) -> float:
    if len(candles) < lookback + 1:
        return 0.0
    last_volume = candles[-1].v
    avg = sum(c.v for c in candles[-lookback - 1 : -1]) / lookback
    return last_volume / avg if avg else 0.0


def _compression_ok(closes: list[float]) -> bool:
    if len(closes) < 25:
        return False
    recent = closes[-5:]
    recent_stdev = pstdev(recent)
    rolling = [pstdev(closes[i : i + 5]) for i in range(len(closes) - 25, len(closes) - 5)]
    baseline = median(rolling) if rolling else 0.0
    return baseline > 0 and recent_stdev <= baseline


def _height_cap(cap_rules: list[str], mid: float, atr14: float) -> float:
    caps = []
    for rule in cap_rules:
        rule = rule.strip()
        if rule.startswith("$"):
            caps.append(float(rule.replace("$", "")))
        elif "pct" in rule:
            pct = float(rule.replace("pct_mid", "").replace("pct", "")) / 100.0
            caps.append(pct * mid)
        elif "ATR14_1m" in rule:
            factor = float(rule.split("*")[0]) if "*" in rule else 1.0
            caps.append(factor * atr14)
    return min([c for c in caps if c > 0] or [0.0])


@dataclass
class Candle:
    ts: int
    o: float
    h: float
    l: float
    c: float
    v: int


@dataclass
class Box:
    tf: str
    start_ts: int
    end_ts: int
    hi: float
    lo: float
    bars: int
    height: float
    quality_score: float
    rvol: float
    spread_cents: float


def find_consolidation_box(
    candles_1m: Iterable[Candle],
    candles_2m: Iterable[Candle],
    cfg,
    latest_spread: float | None = None,
) -> Box | None:
    candles_1m = list(candles_1m)
    if len(candles_1m) < cfg.max_bars:
        return None

    atr14 = _calc_atr(candles_1m[-(cfg.max_bars + 14) :])
    closes = [c.c for c in candles_1m]
    # `latest_spread` is expected to be bid/ask spread in dollars (not candle range).
    # If unavailable, do not filter on spread here; runtime gating can enforce it when L2/BBO is present.
    spread_cents = 0.0
    if latest_spread is not None:
        spread_cents = float(latest_spread) * 100.0
        if spread_cents > 1.01:
            return None

    best_box: Box | None = None
    best_score = -1.0
    for bars in range(cfg.min_bars, cfg.max_bars + 1):
        window = candles_1m[-bars:]
        hi = max(c.h for c in window)
        lo = min(c.l for c in window)
        height = hi - lo
        mid = (hi + lo) / 2
        cap = _height_cap(cfg.box_height_caps, mid, atr14)
        if cap and height > cap:
            continue
        rvol = _calc_rvol(window)
        if rvol < cfg.rvol_min_inside:
            continue
        if cfg.compression_check and not _compression_ok([c.c for c in window]):
            continue
        quality_score = max(0.0, 1.0 - (height / (cap or height + 1e-6))) + rvol
        start_ts = window[0].ts
        end_ts = window[-1].ts
        if quality_score > best_score:
            best_box = Box(
                tf="1m",
                start_ts=start_ts,
                end_ts=end_ts,
                hi=hi,
                lo=lo,
                bars=bars,
                height=height,
                quality_score=quality_score,
                rvol=rvol,
                spread_cents=spread_cents,
            )
            best_score = quality_score
    return best_box
