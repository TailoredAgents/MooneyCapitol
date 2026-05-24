from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable


CONFIDENCE_RANK = {
    "none": 0,
    "weak": 1,
    "likely": 2,
    "exact": 3,
    "direct": 4,
}


@dataclass(frozen=True)
class SetupMatch:
    setup: Any | None
    confidence: str
    score: float
    reason: dict[str, Any]

    @property
    def matched(self) -> bool:
        return self.setup is not None and self.confidence != "none"

    def meets(self, minimum: str) -> bool:
        return CONFIDENCE_RANK.get(self.confidence, 0) >= CONFIDENCE_RANK.get(minimum, 0)


def normalize_entry_side(side: str | None) -> str | None:
    if not side:
        return None
    value = str(side).strip().upper()
    if value in {"BUY", "B", "BOT", "BTO", "COVER", "BUY_TO_COVER"}:
        return "long"
    if value in {"SELL_SHORT", "SSHORT", "SHORT", "SELL_TO_OPEN", "STO"}:
        return "short"
    if value in {"SELL", "S", "SLD", "STC", "SELL_TO_CLOSE"}:
        return "exit_or_short"
    return None


def setup_symbol(setup: Any) -> str | None:
    payload = getattr(setup, "payload_json", None) or {}
    symbol = payload.get("symbol") or getattr(setup, "symbol", None)
    return str(symbol).upper() if symbol else None


def setup_entry_price(setup: Any) -> float | None:
    payload = getattr(setup, "payload_json", None) or {}
    value = getattr(setup, "entry_price", None) or payload.get("entry_price") or payload.get("entry")
    try:
        return float(value) if value not in (None, "", "n/a") else None
    except (TypeError, ValueError):
        return None


def direction_matches(fill_side: str | None, setup_direction: str | None) -> bool:
    entry_side = normalize_entry_side(fill_side)
    direction = str(setup_direction or "").strip().lower()
    if direction == "long":
        return entry_side == "long"
    if direction == "short":
        return entry_side == "short"
    return entry_side in {"long", "short"}


def price_distance_bps(fill_price: float | None, entry_price: float | None) -> float | None:
    if fill_price is None or entry_price is None or entry_price <= 0:
        return None
    return abs(float(fill_price) - float(entry_price)) / float(entry_price) * 10_000.0


def _time_delta_seconds(fill_ts: datetime | None, setup_ts: datetime | None) -> float | None:
    if not fill_ts or not setup_ts:
        return None
    return (fill_ts - setup_ts).total_seconds()


def score_setup_match(
    *,
    symbol: str | None,
    side: str | None,
    ts: datetime | None,
    price: float | None,
    setup: Any,
    window_seconds: int,
    pre_alert_seconds: int,
) -> SetupMatch:
    expected_symbol = setup_symbol(setup)
    actual_symbol = str(symbol).upper() if symbol else None
    delta_s = _time_delta_seconds(ts, getattr(setup, "detected_ts", None))
    direction_ok = direction_matches(side, getattr(setup, "direction", None))
    entry_price = setup_entry_price(setup)
    price_bps = price_distance_bps(price, entry_price)

    reason: dict[str, Any] = {
        "symbol": actual_symbol,
        "setup_symbol": expected_symbol,
        "direction": getattr(setup, "direction", None),
        "side": side,
        "seconds_after_alert": delta_s,
        "entry_price": entry_price,
        "fill_price": price,
        "price_distance_bps": price_bps,
    }

    if not actual_symbol or actual_symbol != expected_symbol:
        return SetupMatch(None, "none", 0.0, reason | {"reject": "symbol_mismatch"})
    if delta_s is None:
        return SetupMatch(None, "none", 0.0, reason | {"reject": "missing_timestamp"})
    if delta_s < -pre_alert_seconds:
        return SetupMatch(None, "none", 0.0, reason | {"reject": "before_alert"})
    if delta_s > window_seconds:
        return SetupMatch(None, "none", 0.0, reason | {"reject": "outside_window"})
    if not direction_ok:
        return SetupMatch(None, "none", 0.0, reason | {"reject": "direction_mismatch"})

    seconds_after = max(delta_s, 0.0)
    time_score = max(0.0, 1.0 - (seconds_after / max(window_seconds, 1)))
    if price_bps is None:
        price_score = 0.65
    elif price_bps <= 100:
        price_score = 1.0
    elif price_bps <= 300:
        price_score = 0.78
    elif price_bps <= 800:
        price_score = 0.45
    else:
        price_score = 0.15

    score = round((0.45 * time_score) + (0.35 * price_score) + 0.20, 6)
    if score >= 0.82 and (price_bps is None or price_bps <= 100):
        confidence = "exact"
    elif score >= 0.62 and (price_bps is None or price_bps <= 300):
        confidence = "likely"
    elif score >= 0.40 and (price_bps is None or price_bps <= 800):
        confidence = "weak"
    else:
        confidence = "none"

    return SetupMatch(
        setup=setup if confidence != "none" else None,
        confidence=confidence,
        score=score if confidence != "none" else 0.0,
        reason=reason | {
            "time_score": round(time_score, 6),
            "price_score": round(price_score, 6),
            "match_score": score,
            "confidence": confidence,
        },
    )


def best_setup_match(
    *,
    symbol: str | None,
    side: str | None,
    ts: datetime | None,
    price: float | None,
    setups: Iterable[Any],
    window_seconds: int,
    pre_alert_seconds: int = 60,
) -> SetupMatch:
    best = SetupMatch(None, "none", 0.0, {"reject": "no_candidates"})
    for setup in setups:
        candidate = score_setup_match(
            symbol=symbol,
            side=side,
            ts=ts,
            price=price,
            setup=setup,
            window_seconds=window_seconds,
            pre_alert_seconds=pre_alert_seconds,
        )
        if candidate.score > best.score or best.reason.get("reject") == "no_candidates":
            best = candidate
    return best
