from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Iterable

from app.v2.domain.models import Side, ZERO


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,95}$")


class MasterEventKind(str, Enum):
    NEW = "NEW"
    REPLACE = "REPLACE"
    CANCEL = "CANCEL"
    PARTIAL_FILL = "PARTIAL_FILL"
    FILL = "FILL"
    REJECT = "REJECT"
    POSITION_TARGET = "POSITION_TARGET"


class CopyDirective(str, Enum):
    SYNC_POSITION = "SYNC_POSITION"
    REPLACE_WORKING = "REPLACE_WORKING"
    CANCEL_WORKING = "CANCEL_WORKING"
    SUBMIT_BRACKET = "SUBMIT_BRACKET"
    SUBMIT_OCO = "SUBMIT_OCO"


class CopyAction(str, Enum):
    SUBMIT = "SUBMIT"
    MODIFY = "MODIFY"
    CANCEL = "CANCEL"
    BRACKET = "BRACKET"
    OCO = "OCO"
    EXIT_POSITION = "EXIT_POSITION"


class LifecycleStatus(str, Enum):
    PLANNED = "PLANNED"
    BLOCKED = "BLOCKED"
    DISPATCHING = "DISPATCHING"
    DISPATCHED = "DISPATCHED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"
    DIVERGED = "DIVERGED"


class ReceiptKind(str, Enum):
    ACKNOWLEDGED = "ACKNOWLEDGED"
    PARTIAL_FILL = "PARTIAL_FILL"
    FILL = "FILL"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    TRANSPORT_FAILURE = "TRANSPORT_FAILURE"
    DIVERGED = "DIVERGED"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    STOP_LIMIT = "STOP_LIMIT"


class OrderDuration(str, Enum):
    DAY = "DAY"
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"


class BracketOperation(str, Enum):
    AFOCCA = "AFOCCA"
    FOCCA = "FOCCA"
    CCA = "CCA"
    FCA = "FCA"


TERMINAL_STATUSES = frozenset(
    {
        LifecycleStatus.FILLED,
        LifecycleStatus.CANCELLED,
        LifecycleStatus.REJECTED,
        LifecycleStatus.FAILED,
        LifecycleStatus.BLOCKED,
        LifecycleStatus.DIVERGED,
    }
)


@dataclass(frozen=True)
class MasterLifecycleEvent:
    source_event_id: str
    master_order_id: str
    revision: int
    kind: MasterEventKind
    occurred_at: datetime
    received_at: datetime

    def __post_init__(self) -> None:
        _require_identifier(self.source_event_id, "source_event_id")
        _require_identifier(self.master_order_id, "master_order_id")
        if isinstance(self.revision, bool) or self.revision < 1:
            raise ValueError("master event revision must be positive")
        if not isinstance(self.kind, MasterEventKind):
            raise ValueError("master event kind is invalid")
        _require_aware(self.occurred_at, "occurred_at")
        _require_aware(self.received_at, "received_at")
        if self.received_at < self.occurred_at:
            raise ValueError("received_at cannot precede occurred_at")

    @property
    def fingerprint(self) -> str:
        return _digest(
            {
                "source_event_id": self.source_event_id,
                "master_order_id": self.master_order_id,
                "revision": self.revision,
                "kind": self.kind.value,
                "occurred_at": self.occurred_at.astimezone(timezone.utc).isoformat(),
            }
        )


@dataclass(frozen=True)
class FollowerLifecyclePlan:
    profile_id: str
    follower_connection_id: str
    event: MasterLifecycleEvent
    directive: CopyDirective
    contract_id: str
    symbol: str
    exchange: str
    current_signed_quantity: int
    evaluated_target_signed_quantity: int | None
    achievable_target_signed_quantity: int
    delta_signed_quantity: int
    allowed: bool
    reason: str
    order_type: OrderType = OrderType.MARKET
    duration: OrderDuration = OrderDuration.DAY
    limit_price: Decimal | None = None
    trigger_price: Decimal | None = None
    stop_ticks: int | None = None
    target_ticks: int | None = None
    bracket_operation: BracketOperation | None = None
    requested_side: Side | None = None
    requested_quantity: int | None = None
    existing_broker_order_id: str | None = field(default=None, repr=False)
    desired_quantity: Decimal = ZERO
    desired_risk_dollars: Decimal = ZERO
    achievable_risk_dollars: Decimal = ZERO
    desired_risk_fraction: Decimal = ZERO
    achievable_risk_fraction: Decimal = ZERO
    desired_exposure_fraction: Decimal = ZERO
    achievable_exposure_fraction: Decimal = ZERO
    risk_parity_error: Decimal = ZERO
    exposure_parity_error: Decimal = ZERO
    parity_error: Decimal = ZERO
    over_target: bool = False
    target_margin_required: Decimal = ZERO

    def __post_init__(self) -> None:
        _require_identifier(self.profile_id, "profile_id")
        _require_identifier(self.follower_connection_id, "follower_connection_id")
        _require_text(self.contract_id, "contract_id", 96)
        _require_text(self.symbol, "symbol", 64)
        _require_text(self.exchange, "exchange", 32)
        _require_text(self.reason, "reason", 96)
        if not isinstance(self.directive, CopyDirective):
            raise ValueError("copy directive is invalid")
        if not isinstance(self.order_type, OrderType):
            raise ValueError("order type is invalid")
        if not isinstance(self.duration, OrderDuration):
            raise ValueError("order duration is invalid")
        if self.requested_side is not None and not isinstance(
            self.requested_side, Side
        ):
            raise ValueError("requested side is invalid")
        if self.bracket_operation is not None and not isinstance(
            self.bracket_operation, BracketOperation
        ):
            raise ValueError("bracket operation is invalid")
        if not isinstance(self.allowed, bool) or not isinstance(self.over_target, bool):
            raise ValueError("lifecycle flags must be boolean")
        for name in (
            "current_signed_quantity",
            "achievable_target_signed_quantity",
            "delta_signed_quantity",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.evaluated_target_signed_quantity is not None and (
            isinstance(self.evaluated_target_signed_quantity, bool)
            or not isinstance(self.evaluated_target_signed_quantity, int)
        ):
            raise ValueError("evaluated_target_signed_quantity must be an integer")
        expected_delta = (
            self.achievable_target_signed_quantity - self.current_signed_quantity
        )
        if self.directive == CopyDirective.SYNC_POSITION and (
            self.delta_signed_quantity != expected_delta
        ):
            raise ValueError("position delta must equal achievable target minus current")
        for name in (
            "desired_quantity",
            "desired_risk_dollars",
            "achievable_risk_dollars",
            "desired_risk_fraction",
            "achievable_risk_fraction",
            "desired_exposure_fraction",
            "achievable_exposure_fraction",
            "risk_parity_error",
            "exposure_parity_error",
            "parity_error",
            "target_margin_required",
        ):
            _require_nonnegative_decimal(getattr(self, name), name)
        if self.limit_price is not None and (
            not isinstance(self.limit_price, Decimal)
            or not self.limit_price.is_finite()
            or self.limit_price <= ZERO
        ):
            raise ValueError("limit_price must be positive")
        if self.trigger_price is not None and (
            not isinstance(self.trigger_price, Decimal)
            or not self.trigger_price.is_finite()
            or self.trigger_price <= ZERO
        ):
            raise ValueError("trigger_price must be positive")
        if self.order_type in {OrderType.LIMIT, OrderType.STOP_LIMIT} and (
            self.limit_price is None
        ):
            raise ValueError("limit orders require limit_price")
        if self.order_type in {OrderType.STOP_MARKET, OrderType.STOP_LIMIT} and (
            self.trigger_price is None
        ):
            raise ValueError("stop orders require trigger_price")
        for name in ("stop_ticks", "target_ticks"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
        if self.requested_quantity is not None and (
            isinstance(self.requested_quantity, bool)
            or not isinstance(self.requested_quantity, int)
            or self.requested_quantity <= 0
        ):
            raise ValueError("requested_quantity must be a positive integer")
        if (self.requested_side is None) != (self.requested_quantity is None):
            raise ValueError("requested side and quantity must be supplied together")
        if self.directive in {
            CopyDirective.REPLACE_WORKING,
            CopyDirective.CANCEL_WORKING,
        } and not (self.existing_broker_order_id or "").strip():
            raise ValueError("replace/cancel requires an existing broker order ID")
        if (
            self.event.kind == MasterEventKind.REJECT
            and self.allowed
            and self.directive != CopyDirective.CANCEL_WORKING
        ):
            raise ValueError(
                "rejected master events may only cancel an existing follower order"
            )
        if self.directive == CopyDirective.SUBMIT_BRACKET and not (
            self.stop_ticks
        ):
            raise ValueError("bracket entries require a protective stop distance")
        if self.directive == CopyDirective.SUBMIT_BRACKET and (
            self.bracket_operation is None
        ):
            raise ValueError("bracket orders require an explicit operation policy")
        if self.directive == CopyDirective.SUBMIT_OCO and not (
            self.limit_price and self.trigger_price
        ):
            raise ValueError("OCO orders require target and stop prices")
        if (
            self.allowed
            and self.directive == CopyDirective.SYNC_POSITION
            and _increases_exposure(
                self.current_signed_quantity,
                self.achievable_target_signed_quantity,
            )
            and (self.stop_ticks is None or self.bracket_operation is None)
        ):
            raise ValueError(
                "position-increasing sync requires attached protective stop policy"
            )

    @property
    def intent_id(self) -> str:
        return deterministic_intent_id(
            self.profile_id,
            self.event.source_event_id,
            self.follower_connection_id,
        )


@dataclass(frozen=True)
class LifecycleCommand:
    command_id: str
    idempotency_key: str
    intent_id: str
    ordinal: int
    action: CopyAction
    contract_id: str
    symbol: str
    exchange: str
    side: Side | None
    quantity: int
    order_type: OrderType
    duration: OrderDuration
    limit_price: Decimal | None = None
    trigger_price: Decimal | None = None
    stop_ticks: int | None = None
    target_ticks: int | None = None
    bracket_operation: BracketOperation | None = None
    existing_broker_order_id: str | None = field(default=None, repr=False)
    depends_on_command_id: str | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.command_id, "command_id")
        _require_identifier(self.idempotency_key, "idempotency_key")
        _require_identifier(self.intent_id, "intent_id")
        if isinstance(self.ordinal, bool) or self.ordinal < 1:
            raise ValueError("command ordinal must be positive")
        if not isinstance(self.action, CopyAction):
            raise ValueError("copy action is invalid")
        if self.side is not None and not isinstance(self.side, Side):
            raise ValueError("command side is invalid")
        if not isinstance(self.order_type, OrderType):
            raise ValueError("order type is invalid")
        if not isinstance(self.duration, OrderDuration):
            raise ValueError("order duration is invalid")
        if self.bracket_operation is not None and not isinstance(
            self.bracket_operation, BracketOperation
        ):
            raise ValueError("bracket operation is invalid")
        if isinstance(self.quantity, bool) or self.quantity < 0:
            raise ValueError("command quantity cannot be negative")
        if self.action != CopyAction.CANCEL and (
            self.quantity == 0 or self.side is None
        ):
            raise ValueError("order commands require a positive quantity and side")
        if self.action in {CopyAction.MODIFY, CopyAction.CANCEL} and not (
            self.existing_broker_order_id or ""
        ).strip():
            raise ValueError("modify/cancel commands require a broker order ID")
        if self.limit_price is not None and (
            not isinstance(self.limit_price, Decimal)
            or not self.limit_price.is_finite()
            or self.limit_price <= ZERO
        ):
            raise ValueError("limit_price must be positive")
        if self.trigger_price is not None and (
            not isinstance(self.trigger_price, Decimal)
            or not self.trigger_price.is_finite()
            or self.trigger_price <= ZERO
        ):
            raise ValueError("trigger_price must be positive")
        if self.order_type in {OrderType.LIMIT, OrderType.STOP_LIMIT} and (
            self.limit_price is None
        ):
            raise ValueError("limit orders require limit_price")
        if self.order_type in {OrderType.STOP_MARKET, OrderType.STOP_LIMIT} and (
            self.trigger_price is None
        ):
            raise ValueError("stop orders require trigger_price")
        for name in ("stop_ticks", "target_ticks"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
        if self.action == CopyAction.BRACKET and (
            self.bracket_operation is None or self.stop_ticks is None
        ):
            raise ValueError(
                "bracket commands require protection and an operation policy"
            )


@dataclass(frozen=True)
class LifecycleReceipt:
    receipt_id: str
    command_id: str
    kind: ReceiptKind
    occurred_at: datetime
    broker_order_id: str | None = field(default=None, repr=False)
    cumulative_filled_quantity: int = 0
    fill_price: Decimal | None = None
    rejection_code: str | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.receipt_id, "receipt_id")
        _require_identifier(self.command_id, "command_id")
        _require_aware(self.occurred_at, "occurred_at")
        if not isinstance(self.kind, ReceiptKind):
            raise ValueError("receipt kind is invalid")
        if (
            isinstance(self.cumulative_filled_quantity, bool)
            or self.cumulative_filled_quantity < 0
        ):
            raise ValueError("cumulative filled quantity cannot be negative")
        if self.fill_price is not None and (
            not isinstance(self.fill_price, Decimal)
            or not self.fill_price.is_finite()
            or self.fill_price <= ZERO
        ):
            raise ValueError("fill price must be positive")
        if self.kind in {ReceiptKind.PARTIAL_FILL, ReceiptKind.FILL} and (
            self.cumulative_filled_quantity <= 0
        ):
            raise ValueError("fill receipts require a positive cumulative quantity")
        if self.kind == ReceiptKind.REJECTED and not (
            self.rejection_code or ""
        ).strip():
            raise ValueError("rejection receipts require a rejection code")


def compile_commands(plan: FollowerLifecyclePlan) -> tuple[LifecycleCommand, ...]:
    """Compile one durable intent into deterministic, ordered broker commands."""

    if not plan.allowed:
        return ()
    if plan.directive == CopyDirective.CANCEL_WORKING:
        return (_command(plan, 1, CopyAction.CANCEL, None, 0),)
    if plan.directive == CopyDirective.REPLACE_WORKING:
        side, quantity = _requested_order(plan)
        return (
            _command(
                plan,
                1,
                CopyAction.MODIFY,
                side,
                quantity,
            ),
        )
    if plan.directive == CopyDirective.SUBMIT_BRACKET:
        side, quantity = _requested_order(plan)
        return (
            _command(
                plan,
                1,
                CopyAction.BRACKET,
                side,
                quantity,
            ),
        )
    if plan.directive == CopyDirective.SUBMIT_OCO:
        side, quantity = _requested_order(plan)
        return (
            _command(
                plan,
                1,
                CopyAction.OCO,
                side,
                quantity,
            ),
        )

    current = plan.current_signed_quantity
    target = plan.achievable_target_signed_quantity
    delta = target - current
    if delta == 0:
        return ()
    if current and target and (current > 0) != (target > 0):
        close = _command(
            plan,
            1,
            CopyAction.EXIT_POSITION,
            Side.SELL if current > 0 else Side.BUY,
            abs(current),
        )
        opening = _command(
            plan,
            2,
            CopyAction.BRACKET,
            Side.BUY if target > 0 else Side.SELL,
            abs(target),
            depends_on=close.command_id,
        )
        return close, opening
    if target == 0:
        return (
            _command(
                plan,
                1,
                CopyAction.EXIT_POSITION,
                Side.SELL if current > 0 else Side.BUY,
                abs(current),
            ),
        )
    action = (
        CopyAction.BRACKET
        if _increases_exposure(current, target)
        else CopyAction.SUBMIT
    )
    return (
        _command(
            plan,
            1,
            action,
            _side_for_delta(delta),
            abs(delta),
        ),
    )


def transition_status(
    current: LifecycleStatus,
    receipt: ReceiptKind,
    *,
    previous_filled_quantity: int,
    new_filled_quantity: int,
    command_quantity: int,
) -> LifecycleStatus:
    if current in TERMINAL_STATUSES:
        raise ValueError("terminal lifecycle status cannot transition")
    if new_filled_quantity < previous_filled_quantity:
        raise ValueError("cumulative fill quantity cannot move backward")
    if command_quantity and new_filled_quantity > command_quantity:
        raise ValueError("cumulative fill quantity cannot exceed command quantity")
    target = {
        ReceiptKind.ACKNOWLEDGED: LifecycleStatus.ACKNOWLEDGED,
        ReceiptKind.PARTIAL_FILL: LifecycleStatus.PARTIALLY_FILLED,
        ReceiptKind.FILL: LifecycleStatus.FILLED,
        ReceiptKind.CANCELLED: LifecycleStatus.CANCELLED,
        ReceiptKind.REJECTED: LifecycleStatus.REJECTED,
        ReceiptKind.TRANSPORT_FAILURE: LifecycleStatus.FAILED,
        ReceiptKind.DIVERGED: LifecycleStatus.DIVERGED,
    }[receipt]
    rank = {
        LifecycleStatus.PLANNED: 0,
        LifecycleStatus.DISPATCHING: 1,
        LifecycleStatus.DISPATCHED: 2,
        LifecycleStatus.ACKNOWLEDGED: 3,
        LifecycleStatus.PARTIALLY_FILLED: 4,
        LifecycleStatus.FILLED: 5,
    }
    if target in rank and current in rank and rank[target] < rank[current]:
        raise ValueError("lifecycle status cannot move backward")
    if receipt == ReceiptKind.PARTIAL_FILL and (
        new_filled_quantity <= 0
        or (command_quantity and new_filled_quantity >= command_quantity)
    ):
        raise ValueError("partial fill quantity must be below command quantity")
    if receipt == ReceiptKind.FILL and command_quantity and (
        new_filled_quantity != command_quantity
    ):
        raise ValueError("fill receipt must equal command quantity")
    return target


def deterministic_intent_id(
    profile_id: str,
    source_event_id: str,
    follower_connection_id: str,
) -> str:
    return "ci_" + _stable_id(profile_id, source_event_id, follower_connection_id)


def deterministic_command_id(intent_id: str, ordinal: int) -> str:
    return "cc_" + _stable_id(intent_id, str(ordinal))


def plan_fingerprint(plan: FollowerLifecyclePlan) -> str:
    values = {
        "event": plan.event.fingerprint,
        "directive": plan.directive.value,
        "contract_id": plan.contract_id,
        "symbol": plan.symbol,
        "exchange": plan.exchange,
        "current": plan.current_signed_quantity,
        "evaluated": plan.evaluated_target_signed_quantity,
        "achievable": plan.achievable_target_signed_quantity,
        "delta": plan.delta_signed_quantity,
        "allowed": plan.allowed,
        "reason": plan.reason,
        "order_type": plan.order_type.value,
        "duration": plan.duration.value,
        "limit_price": _decimal_text(plan.limit_price),
        "trigger_price": _decimal_text(plan.trigger_price),
        "stop_ticks": plan.stop_ticks,
        "target_ticks": plan.target_ticks,
        "bracket_operation": (
            plan.bracket_operation.value if plan.bracket_operation else None
        ),
        "requested_side": plan.requested_side.value if plan.requested_side else None,
        "requested_quantity": plan.requested_quantity,
        "existing_broker_order_id": plan.existing_broker_order_id,
        "desired_quantity": _decimal_text(plan.desired_quantity),
        "desired_risk_dollars": _decimal_text(plan.desired_risk_dollars),
        "achievable_risk_dollars": _decimal_text(plan.achievable_risk_dollars),
        "desired_risk_fraction": _decimal_text(plan.desired_risk_fraction),
        "achievable_risk_fraction": _decimal_text(plan.achievable_risk_fraction),
        "desired_exposure_fraction": _decimal_text(
            plan.desired_exposure_fraction
        ),
        "achievable_exposure_fraction": _decimal_text(
            plan.achievable_exposure_fraction
        ),
        "risk_parity_error": _decimal_text(plan.risk_parity_error),
        "exposure_parity_error": _decimal_text(plan.exposure_parity_error),
        "parity_error": _decimal_text(plan.parity_error),
        "over_target": plan.over_target,
        "target_margin_required": _decimal_text(plan.target_margin_required),
    }
    return _digest(values)


def overall_intent_status(
    command_statuses: Iterable[LifecycleStatus],
    *,
    blocked: bool = False,
) -> LifecycleStatus:
    statuses = tuple(command_statuses)
    if blocked:
        return LifecycleStatus.BLOCKED
    if not statuses:
        return LifecycleStatus.FILLED
    for status in (
        LifecycleStatus.DIVERGED,
        LifecycleStatus.FAILED,
        LifecycleStatus.REJECTED,
        LifecycleStatus.PARTIALLY_FILLED,
        LifecycleStatus.DISPATCHING,
        LifecycleStatus.DISPATCHED,
        LifecycleStatus.ACKNOWLEDGED,
        LifecycleStatus.PLANNED,
        LifecycleStatus.CANCELLED,
    ):
        if status in statuses:
            return status
    return LifecycleStatus.FILLED


def _command(
    plan: FollowerLifecyclePlan,
    ordinal: int,
    action: CopyAction,
    side: Side | None,
    quantity: int,
    *,
    depends_on: str | None = None,
) -> LifecycleCommand:
    command_id = deterministic_command_id(plan.intent_id, ordinal)
    return LifecycleCommand(
        command_id=command_id,
        idempotency_key=command_id,
        intent_id=plan.intent_id,
        ordinal=ordinal,
        action=action,
        contract_id=plan.contract_id,
        symbol=plan.symbol,
        exchange=plan.exchange,
        side=side,
        quantity=quantity,
        order_type=plan.order_type,
        duration=plan.duration,
        limit_price=plan.limit_price,
        trigger_price=plan.trigger_price,
        stop_ticks=plan.stop_ticks,
        target_ticks=plan.target_ticks,
        bracket_operation=plan.bracket_operation,
        existing_broker_order_id=plan.existing_broker_order_id,
        depends_on_command_id=depends_on,
    )


def _requested_order(plan: FollowerLifecyclePlan) -> tuple[Side, int]:
    if plan.requested_side is not None and plan.requested_quantity is not None:
        return plan.requested_side, plan.requested_quantity
    if plan.delta_signed_quantity == 0:
        raise ValueError("submission directive requires a requested order or nonzero delta")
    return _side_for_delta(plan.delta_signed_quantity), abs(plan.delta_signed_quantity)


def _side_for_delta(delta: int) -> Side:
    if delta == 0:
        raise ValueError("zero quantity has no side")
    return Side.BUY if delta > 0 else Side.SELL


def _increases_exposure(current: int, target: int) -> bool:
    if target == 0:
        return False
    if current == 0:
        return True
    if (current > 0) != (target > 0):
        return True
    return abs(target) > abs(current)


def _stable_id(*parts: str) -> str:
    payload = "\x00".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:48]


def _digest(values: dict[str, object]) -> str:
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _decimal_text(value: Decimal | None) -> str | None:
    return None if value is None else format(value, "f")


def _require_identifier(value: str, name: str) -> None:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{name} is invalid")


def _require_text(value: str, name: str, maximum: int) -> None:
    if not value.strip() or len(value) > maximum or any(ord(char) < 32 for char in value):
        raise ValueError(f"{name} is invalid")


def _require_aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


def _require_nonnegative_decimal(value: Decimal, name: str) -> None:
    if not isinstance(value, Decimal) or not value.is_finite() or value < ZERO:
        raise ValueError(f"{name} must be a finite nonnegative Decimal")
