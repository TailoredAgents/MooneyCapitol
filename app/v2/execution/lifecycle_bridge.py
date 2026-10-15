from __future__ import annotations

from decimal import Decimal

from app.v2.domain.models import Side
from app.v2.execution.copier import CumulativeFollowerPlan
from app.v2.execution.lifecycle import (
    BracketOperation,
    CopyDirective,
    FollowerLifecyclePlan,
    MasterLifecycleEvent,
    OrderDuration,
    OrderType,
)


def lifecycle_plan_from_cumulative(
    *,
    profile_id: str,
    follower_connection_id: str,
    event: MasterLifecycleEvent,
    sizing: CumulativeFollowerPlan,
    contract_id: str,
    symbol: str,
    exchange: str,
    directive: CopyDirective = CopyDirective.SYNC_POSITION,
    order_type: OrderType = OrderType.MARKET,
    duration: OrderDuration = OrderDuration.DAY,
    limit_price: Decimal | None = None,
    trigger_price: Decimal | None = None,
    stop_ticks: int | None = None,
    target_ticks: int | None = None,
    bracket_operation: BracketOperation | None = None,
    requested_side: Side | None = None,
    requested_quantity: int | None = None,
    existing_broker_order_id: str | None = None,
) -> FollowerLifecyclePlan:
    """Carry an immutable cumulative-sizing result into durable orchestration."""

    evaluated_contract = sizing.evaluated_contract_id
    current_contract = sizing.current_contract_id
    relevant_contracts = {
        item for item in (evaluated_contract, current_contract) if item is not None
    }
    if relevant_contracts and contract_id not in relevant_contracts:
        raise ValueError("lifecycle contract must match the cumulative sizing result")
    return FollowerLifecyclePlan(
        profile_id=profile_id,
        follower_connection_id=follower_connection_id,
        event=event,
        directive=directive,
        contract_id=contract_id,
        symbol=symbol,
        exchange=exchange,
        current_signed_quantity=sizing.current_signed_quantity,
        evaluated_target_signed_quantity=sizing.evaluated_target_signed_quantity,
        achievable_target_signed_quantity=sizing.achievable_target_signed_quantity,
        delta_signed_quantity=sizing.delta_signed_quantity,
        allowed=sizing.allowed,
        reason=sizing.reason,
        order_type=order_type,
        duration=duration,
        limit_price=limit_price,
        trigger_price=trigger_price,
        stop_ticks=stop_ticks,
        target_ticks=target_ticks,
        bracket_operation=bracket_operation,
        requested_side=requested_side,
        requested_quantity=requested_quantity,
        existing_broker_order_id=existing_broker_order_id,
        desired_quantity=abs(sizing.desired_target_quantity),
        desired_risk_dollars=sizing.desired_risk_dollars,
        achievable_risk_dollars=sizing.achievable_risk_dollars,
        desired_risk_fraction=sizing.desired_risk_fraction,
        achievable_risk_fraction=sizing.achievable_risk_fraction,
        desired_exposure_fraction=sizing.desired_point_exposure_fraction,
        achievable_exposure_fraction=sizing.achievable_point_exposure_fraction,
        risk_parity_error=sizing.risk_parity_error,
        exposure_parity_error=sizing.exposure_parity_error,
        parity_error=sizing.parity_error,
        over_target=sizing.over_target,
        target_margin_required=sizing.target_margin_required,
    )
