from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP
from enum import Enum
from typing import Mapping, Sequence

from app.v2.domain.models import FuturesContract, ZERO
from app.v2.math import RoundingPolicy


class SizingMode(str, Enum):
    FIXED_DOLLAR = "FIXED_DOLLAR"
    PERCENT_EQUITY = "PERCENT_EQUITY"
    MASTER_RISK_MULTIPLIER = "MASTER_RISK_MULTIPLIER"
    FIXED_CONTRACTS = "FIXED_CONTRACTS"
    DISABLED = "DISABLED"


class InstrumentSelection(str, Enum):
    EXACT = "EXACT"
    AUTO_NQ_MNQ = "AUTO_NQ_MNQ"


@dataclass(frozen=True)
class RiskLimits:
    sizing_mode: SizingMode = SizingMode.DISABLED
    sizing_value: Decimal = ZERO
    max_risk_per_trade: Decimal = ZERO
    max_contracts: int = 0
    max_concurrent_open_risk: Decimal = ZERO
    per_product_max_contracts: Mapping[str, int] = field(default_factory=dict)
    daily_realized_loss_ceiling: Decimal = ZERO
    daily_total_loss_ceiling: Decimal = ZERO
    margin_headroom_fraction: Decimal = Decimal("0.20")
    rounding: RoundingPolicy = RoundingPolicy.FLOOR
    instrument_selection: InstrumentSelection = InstrumentSelection.EXACT
    cost_buffer_per_contract: Decimal = ZERO

    def __post_init__(self) -> None:
        decimal_values = (
            self.sizing_value,
            self.max_risk_per_trade,
            self.max_concurrent_open_risk,
            self.daily_realized_loss_ceiling,
            self.daily_total_loss_ceiling,
            self.margin_headroom_fraction,
            self.cost_buffer_per_contract,
        )
        if any(value < ZERO for value in decimal_values):
            raise ValueError("risk limits cannot be negative")
        if not ZERO <= self.margin_headroom_fraction < Decimal("1"):
            raise ValueError("margin_headroom_fraction must be in [0, 1)")
        if self.max_contracts < 0 or any(value < 0 for value in self.per_product_max_contracts.values()):
            raise ValueError("contract limits cannot be negative")


@dataclass(frozen=True)
class RiskState:
    equity: Decimal
    available_margin: Decimal
    current_open_risk: Decimal = ZERO
    daily_realized_pnl: Decimal = ZERO
    daily_total_pnl: Decimal = ZERO
    daily_drawdown: Decimal = ZERO
    global_kill_switch: bool = True
    account_kill_switch: bool = True

    def __post_init__(self) -> None:
        if self.equity < ZERO or self.available_margin < ZERO or self.current_open_risk < ZERO:
            raise ValueError("equity, available margin, and open risk cannot be negative")
        if self.daily_drawdown < ZERO:
            raise ValueError("daily_drawdown is a non-negative peak-to-current distance")


@dataclass(frozen=True)
class SizingCandidate:
    contract: FuturesContract
    risk_per_contract: Decimal
    initial_margin_per_contract: Decimal

    def __post_init__(self) -> None:
        if self.risk_per_contract <= ZERO:
            raise ValueError("risk_per_contract must be positive")
        if self.initial_margin_per_contract < ZERO:
            raise ValueError("initial_margin_per_contract cannot be negative")


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    reason: str
    contract_id: str | None = None
    contracts: int = 0
    economic_risk: Decimal = ZERO
    margin_required: Decimal = ZERO
    risk_budget: Decimal = ZERO


def evaluate_futures_risk(
    limits: RiskLimits,
    state: RiskState,
    candidates: Sequence[SizingCandidate],
    *,
    master_original_risk: Decimal | None = None,
) -> RiskDecision:
    """Size a futures position without broker side effects.

    Every operational ceiling is fail-closed: zero means no authorized capacity,
    not unlimited. Margin is an independent constraint and never substitutes for
    the economic loss at the stop.
    """
    blocked = _preflight(limits, state)
    if blocked:
        return RiskDecision(False, blocked)
    if not candidates:
        return RiskDecision(False, "no_contract_candidate")
    if limits.instrument_selection == InstrumentSelection.EXACT and len(candidates) != 1:
        return RiskDecision(False, "exact_selection_requires_one_contract")
    if limits.instrument_selection == InstrumentSelection.AUTO_NQ_MNQ:
        products = {item.contract.product_code.upper() for item in candidates}
        expiries = {item.contract.expiration for item in candidates}
        if not products.issubset({"NQ", "MNQ"}) or len(expiries) != 1:
            return RiskDecision(False, "auto_selection_requires_same_expiry_nq_mnq")

    budget = _requested_budget(limits, state, master_original_risk)
    if budget <= ZERO:
        return RiskDecision(False, "sizing_budget_not_positive")
    available_open_risk = limits.max_concurrent_open_risk - state.current_open_risk
    budget = min(budget, limits.max_risk_per_trade, available_open_risk)
    if budget <= ZERO:
        return RiskDecision(False, "concurrent_open_risk_exhausted")

    options: list[RiskDecision] = []
    for candidate in candidates:
        decision = _size_candidate(limits, state, candidate, budget)
        if decision.allowed:
            options.append(decision)
    if not options:
        return RiskDecision(False, "no_contract_fits_risk_and_margin", risk_budget=budget)
    # Highest budget utilization wins; larger point value resolves exact ties.
    return max(
        options,
        key=lambda item: (
            item.economic_risk,
            next(c.contract.specification.point_value for c in candidates if c.contract.contract_id == item.contract_id),
        ),
    )


def _preflight(limits: RiskLimits, state: RiskState) -> str | None:
    if limits.sizing_mode == SizingMode.DISABLED:
        return "sizing_disabled"
    if state.global_kill_switch:
        return "global_kill_switch"
    if state.account_kill_switch:
        return "account_kill_switch"
    if limits.max_risk_per_trade <= ZERO:
        return "max_risk_per_trade_not_configured"
    if limits.max_contracts <= 0:
        return "max_contracts_not_configured"
    if limits.max_concurrent_open_risk <= ZERO:
        return "max_concurrent_open_risk_not_configured"
    if limits.daily_realized_loss_ceiling <= ZERO:
        return "daily_realized_loss_ceiling_not_configured"
    if limits.daily_total_loss_ceiling <= ZERO:
        return "daily_total_loss_ceiling_not_configured"
    if state.daily_realized_pnl <= -limits.daily_realized_loss_ceiling:
        return "daily_realized_loss_ceiling_reached"
    if state.daily_total_pnl <= -limits.daily_total_loss_ceiling or state.daily_drawdown >= limits.daily_total_loss_ceiling:
        return "daily_total_loss_ceiling_reached"
    return None


def _requested_budget(limits: RiskLimits, state: RiskState, master_risk: Decimal | None) -> Decimal:
    if limits.sizing_mode == SizingMode.FIXED_DOLLAR:
        return limits.sizing_value
    if limits.sizing_mode == SizingMode.PERCENT_EQUITY:
        return state.equity * limits.sizing_value
    if limits.sizing_mode == SizingMode.MASTER_RISK_MULTIPLIER:
        return (master_risk or ZERO) * limits.sizing_value
    if limits.sizing_mode == SizingMode.FIXED_CONTRACTS:
        # The budget still remains capped below; quantity is applied per candidate.
        return limits.max_risk_per_trade
    return ZERO


def _size_candidate(
    limits: RiskLimits,
    state: RiskState,
    candidate: SizingCandidate,
    budget: Decimal,
) -> RiskDecision:
    unit_risk = candidate.risk_per_contract + limits.cost_buffer_per_contract
    if limits.sizing_mode == SizingMode.FIXED_CONTRACTS:
        quantity = _round(limits.sizing_value, limits.rounding)
    else:
        quantity = _round(budget / unit_risk, limits.rounding)
    product_limit = limits.per_product_max_contracts.get(candidate.contract.product_code.upper(), limits.max_contracts)
    quantity = min(quantity, limits.max_contracts, product_limit)
    usable_margin = state.available_margin * (Decimal("1") - limits.margin_headroom_fraction)
    if candidate.initial_margin_per_contract > ZERO:
        quantity = min(quantity, int((usable_margin / candidate.initial_margin_per_contract).to_integral_value(rounding=ROUND_FLOOR)))
    # Rounding policies are configurable, but the hard ceiling always wins.
    while quantity > 0 and unit_risk * Decimal(quantity) > budget:
        quantity -= 1
    if quantity <= 0:
        return RiskDecision(False, "candidate_does_not_fit", risk_budget=budget)
    economic_risk = unit_risk * Decimal(quantity)
    margin = candidate.initial_margin_per_contract * Decimal(quantity)
    return RiskDecision(
        True,
        "allowed",
        candidate.contract.contract_id,
        quantity,
        economic_risk,
        margin,
        budget,
    )


def _round(value: Decimal, policy: RoundingPolicy) -> int:
    modes = {
        RoundingPolicy.FLOOR: ROUND_FLOOR,
        RoundingPolicy.NEAREST: ROUND_HALF_UP,
        RoundingPolicy.CEILING: ROUND_CEILING,
    }
    return max(0, int(value.to_integral_value(rounding=modes[policy])))
