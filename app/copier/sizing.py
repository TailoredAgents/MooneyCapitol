from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.copier.models import MasterExecutionEvent


SizingMode = Literal["disabled", "fixed_quantity", "fixed_multiplier", "percent_equity", "equity_ratio"]


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
        qty = int(qty)
    if policy.min_notional and qty * master.price < policy.min_notional:
        return 0.0
    if qty < policy.min_quantity:
        return 0.0
    return float(qty)


def master_account_trade_pct(master: MasterExecutionEvent, master_equity: float | None) -> float | None:
    if not master_equity or master_equity <= 0:
        return None
    return (master.quantity * master.price) / master_equity
