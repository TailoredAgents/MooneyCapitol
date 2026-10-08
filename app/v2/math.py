from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP
from enum import Enum
from typing import Iterable

from app.v2.domain.models import ContractSpecification, PositionSide, Side, ZERO


class RoundingPolicy(str, Enum):
    FLOOR = "FLOOR"
    NEAREST = "NEAREST"
    CEILING = "CEILING"


@dataclass(frozen=True)
class FillLot:
    price: Decimal
    quantity: int
    fee: Decimal = ZERO

    def __post_init__(self) -> None:
        if self.price <= ZERO or self.quantity <= 0 or self.fee < ZERO:
            raise ValueError("fill price/quantity must be positive and fee non-negative")


@dataclass(frozen=True)
class Excursion:
    points: Decimal
    ticks: Decimal
    dollars: Decimal
    r_multiple: Decimal | None


@dataclass(frozen=True)
class ReversalSplit:
    close_quantity: int
    open_quantity: int
    resulting_signed_quantity: int


class PositionTransitionType(str, Enum):
    OPEN = "OPEN"
    ADD = "ADD"
    PARTIAL_EXIT = "PARTIAL_EXIT"
    CLOSE = "CLOSE"
    REVERSAL = "REVERSAL"


@dataclass(frozen=True)
class PositionTransition:
    transition: PositionTransitionType
    previous_signed_quantity: int
    fill_quantity: int
    closed_quantity: int
    opened_quantity: int
    resulting_signed_quantity: int


def price_distance_to_ticks(start: Decimal, end: Decimal, specification: ContractSpecification) -> Decimal:
    return abs(end - start) / specification.tick_size


def ticks_to_dollars(ticks: Decimal, contracts: int, specification: ContractSpecification) -> Decimal:
    _contracts(contracts)
    return abs(ticks) * specification.tick_value * Decimal(contracts)


def contract_risk(
    entry: Decimal,
    stop: Decimal,
    specification: ContractSpecification,
    *,
    cost_buffer_per_contract: Decimal = ZERO,
) -> Decimal:
    if cost_buffer_per_contract < ZERO:
        raise ValueError("cost buffer cannot be negative")
    ticks = price_distance_to_ticks(entry, stop, specification)
    return ticks * specification.tick_value + cost_buffer_per_contract


def planned_risk_dollars(
    entry: Decimal,
    stop: Decimal | None,
    contracts: int,
    specification: ContractSpecification,
    *,
    cost_buffer_per_contract: Decimal = ZERO,
) -> Decimal | None:
    if stop is None:
        return None
    _contracts(contracts)
    return contract_risk(entry, stop, specification, cost_buffer_per_contract=cost_buffer_per_contract) * Decimal(contracts)


def weighted_vwap(fills: Iterable[FillLot]) -> Decimal | None:
    rows = tuple(fills)
    if not rows:
        return None
    quantity = sum(item.quantity for item in rows)
    if quantity <= 0:
        return None
    return sum(item.price * Decimal(item.quantity) for item in rows) / Decimal(quantity)


def weighted_entry(fills: Iterable[FillLot]) -> Decimal | None:
    return weighted_vwap(fills)


def weighted_exit(fills: Iterable[FillLot]) -> Decimal | None:
    return weighted_vwap(fills)


def realized_gross_pnl(
    direction: PositionSide,
    entries: Iterable[FillLot],
    exits: Iterable[FillLot],
    specification: ContractSpecification,
) -> Decimal:
    entry_rows = tuple(entries)
    exit_rows = tuple(exits)
    entry_qty = sum(item.quantity for item in entry_rows)
    exit_qty = sum(item.quantity for item in exit_rows)
    if not entry_rows or not exit_rows or exit_qty > entry_qty:
        raise ValueError("realized P&L requires exits not exceeding entered quantity")
    sign = Decimal("1") if direction == PositionSide.LONG else Decimal("-1")
    if direction not in {PositionSide.LONG, PositionSide.SHORT}:
        raise ValueError("realized P&L requires long or short direction")
    if exit_qty == entry_qty:
        # Preserve exact Decimal arithmetic for a fully closed set of lots;
        # dividing two repeating VWAPs first would introduce context noise.
        entry_notional = sum((item.price * Decimal(item.quantity) for item in entry_rows), ZERO)
        exit_notional = sum((item.price * Decimal(item.quantity) for item in exit_rows), ZERO)
        return (exit_notional - entry_notional) * sign * specification.point_value
    entry_price = weighted_vwap(entry_rows)
    exit_price = weighted_vwap(exit_rows)
    assert entry_price is not None and exit_price is not None
    return (exit_price - entry_price) * sign * specification.point_value * Decimal(exit_qty)


def realized_net_pnl(gross_pnl: Decimal, fills: Iterable[FillLot], *, extra_cost_buffer: Decimal = ZERO) -> Decimal:
    if extra_cost_buffer < ZERO:
        raise ValueError("extra cost buffer cannot be negative")
    return gross_pnl - sum((item.fee for item in fills), ZERO) - extra_cost_buffer


def realized_r(net_pnl: Decimal, original_risk_dollars: Decimal | None) -> Decimal | None:
    if original_risk_dollars is None:
        return None
    if original_risk_dollars <= ZERO:
        raise ValueError("original risk denominator must be positive")
    return net_pnl / original_risk_dollars


def mfe(
    direction: PositionSide,
    entry_price: Decimal,
    best_price: Decimal,
    contracts: int,
    specification: ContractSpecification,
    original_risk_dollars: Decimal | None,
) -> Excursion:
    points = _favorable_points(direction, entry_price, best_price)
    return _excursion(max(points, ZERO), contracts, specification, original_risk_dollars)


def mae(
    direction: PositionSide,
    entry_price: Decimal,
    worst_price: Decimal,
    contracts: int,
    specification: ContractSpecification,
    original_risk_dollars: Decimal | None,
) -> Excursion:
    points = -_favorable_points(direction, entry_price, worst_price)
    return _excursion(max(points, ZERO), contracts, specification, original_risk_dollars)


def split_reversal(current_signed_quantity: int, side: Side, fill_quantity: int) -> ReversalSplit:
    _contracts(fill_quantity, "fill_quantity")
    signed_fill = fill_quantity if side == Side.BUY else -fill_quantity
    if current_signed_quantity == 0 or current_signed_quantity * signed_fill > 0:
        return ReversalSplit(0, fill_quantity, current_signed_quantity + signed_fill)
    close_quantity = min(abs(current_signed_quantity), fill_quantity)
    open_quantity = fill_quantity - close_quantity
    return ReversalSplit(close_quantity, open_quantity, current_signed_quantity + signed_fill)


def apply_position_fill(current_signed_quantity: int, side: Side, fill_quantity: int) -> PositionTransition:
    """Classify an integer fill as open/add/scale-out/close/reversal."""
    split = split_reversal(current_signed_quantity, side, fill_quantity)
    signed_fill = fill_quantity if side == Side.BUY else -fill_quantity
    if current_signed_quantity == 0:
        kind = PositionTransitionType.OPEN
    elif current_signed_quantity * signed_fill > 0:
        kind = PositionTransitionType.ADD
    elif fill_quantity < abs(current_signed_quantity):
        kind = PositionTransitionType.PARTIAL_EXIT
    elif fill_quantity == abs(current_signed_quantity):
        kind = PositionTransitionType.CLOSE
    else:
        kind = PositionTransitionType.REVERSAL
    return PositionTransition(
        kind,
        current_signed_quantity,
        fill_quantity,
        split.close_quantity,
        split.open_quantity,
        split.resulting_signed_quantity,
    )


def equivalent_contracts(
    source_contracts: int,
    source_specification: ContractSpecification,
    target_specification: ContractSpecification,
    stop_distance_points: Decimal,
    *,
    rounding: RoundingPolicy = RoundingPolicy.FLOOR,
) -> int:
    _contracts(source_contracts, "source_contracts")
    if stop_distance_points <= ZERO:
        raise ValueError("stop distance must be positive")
    source_risk = stop_distance_points * source_specification.point_value * Decimal(source_contracts)
    target_risk = stop_distance_points * target_specification.point_value
    raw = source_risk / target_risk
    quantity = _round_contracts(raw, rounding)
    while quantity > 0 and target_risk * Decimal(quantity) > source_risk:
        quantity -= 1
    return quantity


def _favorable_points(direction: PositionSide, entry: Decimal, price: Decimal) -> Decimal:
    if direction == PositionSide.LONG:
        return price - entry
    if direction == PositionSide.SHORT:
        return entry - price
    raise ValueError("excursion requires long or short direction")


def _excursion(
    points: Decimal,
    contracts: int,
    specification: ContractSpecification,
    original_risk_dollars: Decimal | None,
) -> Excursion:
    _contracts(contracts)
    ticks = points / specification.tick_size
    dollars = points * specification.point_value * Decimal(contracts)
    return Excursion(points, ticks, dollars, realized_r(dollars, original_risk_dollars))


def _round_contracts(value: Decimal, policy: RoundingPolicy) -> int:
    modes = {
        RoundingPolicy.FLOOR: ROUND_FLOOR,
        RoundingPolicy.NEAREST: ROUND_HALF_UP,
        RoundingPolicy.CEILING: ROUND_CEILING,
    }
    return max(0, int(value.to_integral_value(rounding=modes[policy])))


def _contracts(value: int, label: str = "contracts") -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
