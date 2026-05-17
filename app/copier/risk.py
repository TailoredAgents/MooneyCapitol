from __future__ import annotations

from dataclasses import dataclass, field

from app.copier.models import MasterExecutionEvent


@dataclass(frozen=True)
class RiskPolicy:
    enabled: bool = True
    global_kill_switch: bool = True
    max_notional_per_trade: float = 0.0
    max_position_pct: float = 0.0
    max_daily_notional: float = 0.0
    max_daily_trades: int = 0
    max_orders_per_minute: int = 0
    shorting_enabled: bool = False
    allowlist: set[str] = field(default_factory=set)
    blocklist: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class RiskUsage:
    daily_notional: float = 0.0
    daily_orders: int = 0
    minute_orders: int = 0


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    reason: str | None = None


def validate_copy(
    master: MasterExecutionEvent,
    quantity: float,
    policy: RiskPolicy,
    target_equity: float | None = None,
    usage: RiskUsage | None = None,
    target_position_qty: float = 0.0,
) -> RiskDecision:
    if policy.global_kill_switch:
        return RiskDecision(False, "global_kill_switch")
    if not policy.enabled:
        return RiskDecision(False, "target_disabled")
    if quantity <= 0:
        return RiskDecision(False, "zero_quantity")
    symbol = master.symbol.upper()
    if symbol in {item.upper() for item in policy.blocklist}:
        return RiskDecision(False, "symbol_blocked")
    if policy.allowlist and symbol not in {item.upper() for item in policy.allowlist}:
        return RiskDecision(False, "symbol_not_allowed")
    if master.side == "SELL" and not policy.shorting_enabled:
        if quantity > max(target_position_qty, 0.0) + 0.000001:
            return RiskDecision(False, "short_copy_disabled")
    notional = quantity * master.price
    if policy.max_notional_per_trade and notional > policy.max_notional_per_trade:
        return RiskDecision(False, "max_notional_exceeded")
    if policy.max_position_pct and target_equity and target_equity > 0:
        if notional > target_equity * policy.max_position_pct:
            return RiskDecision(False, "max_position_pct_exceeded")
    usage = usage or RiskUsage()
    if policy.max_daily_notional and usage.daily_notional + notional > policy.max_daily_notional:
        return RiskDecision(False, "max_daily_notional_exceeded")
    if policy.max_daily_trades and usage.daily_orders + 1 > policy.max_daily_trades:
        return RiskDecision(False, "max_daily_trades_exceeded")
    if policy.max_orders_per_minute and usage.minute_orders + 1 > policy.max_orders_per_minute:
        return RiskDecision(False, "max_orders_per_minute_exceeded")
    return RiskDecision(True)
