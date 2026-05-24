from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.copier.models import MasterExecutionEvent
from app.observability.logging import get_logger


logger = get_logger("copier.sizing")

SizingMode = Literal["disabled", "fixed_quantity", "fixed_multiplier", "percent_equity", "equity_ratio"]

# Warn when whole-share truncation silently discards more than this fraction of intended notional
_TRUNCATION_WARN_THRESHOLD = 0.05


@dataclass(frozen=True)
class SizingPolicy:
    mode: SizingMode = "disabled"
    value: float = 0.0
    min_quantity: float = 1.0
    min_notional: float = 0.0
    whole_shares: bool = True


def size_child_order(
    master: MasterExecutionEvent,
    policy: SizingPolicy,
    master_equity: float | None = None,
    target_equity: float | None = None,
) -> float:
    if policy.mode == "disabled":
        return 0.0
    if policy.mode == "fixed_quantity":
        qty = policy.value
    elif policy.mode == "fixed_multiplier":
        qty = master.quantity * policy.value
    elif policy.mode in {"percent_equity", "equity_ratio"}:
        if not master_equity or not target_equity or master_equity <= 0:
            return 0.0
        master_notional = master.quantity * master.price
        master_account_pct = master_notional / master_equity
        target_notional = target_equity * master_account_pct
        qty = target_notional / master.price
    else:
        return 0.0

    if policy.whole_shares:
        raw_qty = qty
        qty = int(qty)
        if raw_qty > 0 and qty == 0:
            # Entire position rounded away — order will be blocked below
            logger.warning(
                "copier.sizing.truncated_to_zero",
                symbol=master.symbol,
                raw_qty=round(raw_qty, 4),
                price=master.price,
                mode=policy.mode,
                master_equity=master_equity,
                target_equity=target_equity,
            )
        elif raw_qty > 0:
            lost_pct = (raw_qty - qty) / raw_qty
            if lost_pct > _TRUNCATION_WARN_THRESHOLD:
                logger.warning(
                    "copier.sizing.truncation_drift",
                    symbol=master.symbol,
                    raw_qty=round(raw_qty, 4),
                    truncated_qty=qty,
                    lost_pct=round(lost_pct, 4),
                    price=master.price,
                    mode=policy.mode,
                )

    if policy.min_notional and qty * master.price < policy.min_notional:
        return 0.0
    if qty < policy.min_quantity:
        return 0.0
    return float(qty)


def master_account_trade_pct(master: MasterExecutionEvent, master_equity: float | None) -> float | None:
    if not master_equity or master_equity <= 0:
        return None
    return (master.quantity * master.price) / master_equity
