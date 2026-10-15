from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Iterable, Mapping

from app.v2.domain.broker_observation import ObservedOrderState
from app.v2.domain.models import Side
from app.v2.execution.lifecycle import (
    CopyAction,
    CopyDirective,
    LifecycleReceipt,
    LifecycleStatus,
    ReceiptKind,
)


class OutboxStatus(str, Enum):
    PENDING = "PENDING"
    CLAIMED = "CLAIMED"
    SENT = "SENT"
    CONFIRMED = "CONFIRMED"
    UNCERTAIN = "UNCERTAIN"
    DIVERGED = "DIVERGED"
    FAILED = "FAILED"


class ReconciliationStatus(str, Enum):
    CLEAN = "CLEAN"
    ADOPTED = "ADOPTED"
    LOCKED = "LOCKED"
    INCOMPLETE = "INCOMPLETE"


class FindingSeverity(str, Enum):
    WARNING = "WARNING"
    BLOCKING = "BLOCKING"


class LatencyStage(str, Enum):
    EVENT_TO_DECISION = "EVENT_TO_DECISION"
    EVENT_TO_SUBMISSION = "EVENT_TO_SUBMISSION"
    EVENT_RECEIVED_TO_DECISION = "EVENT_RECEIVED_TO_DECISION"
    EVENT_RECEIVED_TO_SUBMISSION = "EVENT_RECEIVED_TO_SUBMISSION"
    DECISION_TO_SUBMISSION = "DECISION_TO_SUBMISSION"
    SUBMISSION_TO_RESPONSE = "SUBMISSION_TO_RESPONSE"
    SUBMISSION_TO_ACKNOWLEDGEMENT = "SUBMISSION_TO_ACKNOWLEDGEMENT"
    SUBMISSION_TO_FIRST_FILL = "SUBMISSION_TO_FIRST_FILL"
    EVENT_TO_COMPLETION = "EVENT_TO_COMPLETION"


@dataclass(frozen=True)
class ExpectedReconciliationCommand:
    command_id: str
    idempotency_key: str
    intent_id: str
    action: CopyAction
    lifecycle_status: LifecycleStatus
    outbox_status: OutboxStatus
    contract_id: str
    symbol: str
    exchange: str
    side: Side | None
    quantity: int
    cumulative_filled_quantity: int
    broker_order_ids: tuple[str, ...] = field(default=(), repr=False)


@dataclass(frozen=True)
class FollowerReconciliationExpectation:
    profile_id: str
    follower_connection_id: str
    latest_intent_id: str | None
    latest_intent_status: LifecycleStatus | None
    latest_directive: CopyDirective | None
    contract_id: str | None
    symbol: str | None
    exchange: str | None
    starting_signed_quantity: int | None
    expected_signed_quantity: int | None
    commands: tuple[ExpectedReconciliationCommand, ...]


@dataclass(frozen=True)
class BrokerPositionState:
    contract_id: str
    symbol: str
    exchange: str
    signed_quantity: int

    def __post_init__(self) -> None:
        if not self.contract_id.strip() or not self.symbol.strip() or not self.exchange.strip():
            raise ValueError("broker position identity is required")
        if isinstance(self.signed_quantity, bool) or not isinstance(
            self.signed_quantity, int
        ):
            raise ValueError("broker signed quantity must be an integer")


@dataclass(frozen=True)
class BrokerOrderState:
    broker_order_id: str = field(repr=False)
    normalized_state: ObservedOrderState
    symbol: str
    exchange: str
    side: Side | None
    quantity: int | None
    cumulative_filled_quantity: int = 0
    local_command_id: str | None = None
    idempotency_key: str | None = None

    def __post_init__(self) -> None:
        if not self.broker_order_id.strip():
            raise ValueError("broker order ID is required")
        if not isinstance(self.normalized_state, ObservedOrderState):
            raise ValueError("broker order state is invalid")
        if self.side is not None and not isinstance(self.side, Side):
            raise ValueError("broker order side is invalid")
        if self.quantity is not None and (
            isinstance(self.quantity, bool) or self.quantity < 0
        ):
            raise ValueError("broker order quantity cannot be negative")
        if (
            isinstance(self.cumulative_filled_quantity, bool)
            or self.cumulative_filled_quantity < 0
        ):
            raise ValueError("broker cumulative fill cannot be negative")

    @property
    def active(self) -> bool:
        return self.normalized_state in {
            ObservedOrderState.PENDING,
            ObservedOrderState.WORKING,
            ObservedOrderState.PARTIALLY_FILLED,
            ObservedOrderState.UNKNOWN,
        }


@dataclass(frozen=True)
class BrokerExecutionState:
    execution_id: str
    broker_order_id: str = field(repr=False)
    quantity: int
    price: Decimal
    occurred_at: datetime
    local_command_id: str | None = None
    effective_quantity_delta: int | None = None

    def __post_init__(self) -> None:
        if not self.execution_id.strip() or not self.broker_order_id.strip():
            raise ValueError("execution and broker order IDs are required")
        if isinstance(self.quantity, bool) or self.quantity <= 0:
            raise ValueError("execution quantity must be positive")
        if self.effective_quantity_delta is not None and (
            isinstance(self.effective_quantity_delta, bool)
            or not isinstance(self.effective_quantity_delta, int)
        ):
            raise ValueError("effective execution delta must be an integer")
        if (
            not isinstance(self.price, Decimal)
            or not self.price.is_finite()
            or self.price <= 0
        ):
            raise ValueError("execution price must be positive")
        _require_aware(self.occurred_at, "execution occurred_at")


@dataclass(frozen=True)
class BrokerProtectionState:
    contract_id: str
    active_stop_quantity: int
    active_target_quantity: int = 0

    def __post_init__(self) -> None:
        if not self.contract_id.strip():
            raise ValueError("protection contract identity is required")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (
                self.active_stop_quantity,
                self.active_target_quantity,
            )
        ):
            raise ValueError("protection quantities cannot be negative")


@dataclass(frozen=True)
class FollowerBrokerSnapshot:
    snapshot_id: str
    follower_connection_id: str
    connection_generation: str
    captured_at: datetime
    positions_complete: bool
    orders_complete: bool
    executions_complete: bool
    protection_complete: bool
    positions: tuple[BrokerPositionState, ...] = ()
    orders: tuple[BrokerOrderState, ...] = ()
    executions: tuple[BrokerExecutionState, ...] = ()
    protections: tuple[BrokerProtectionState, ...] = ()

    def __post_init__(self) -> None:
        if not self.snapshot_id.strip() or not self.connection_generation.strip():
            raise ValueError("snapshot and connection generation IDs are required")
        if self.follower_connection_id not in {
            "follower-1",
            "follower-2",
            "follower-3",
            "follower-4",
        }:
            raise ValueError("unknown follower connection")
        if not all(
            isinstance(value, bool)
            for value in (
                self.positions_complete,
                self.orders_complete,
                self.executions_complete,
                self.protection_complete,
            )
        ):
            raise ValueError("snapshot completeness flags must be boolean")
        _require_unique(
            (item.contract_id for item in self.positions),
            "position contract",
        )
        _require_unique(
            (item.broker_order_id for item in self.orders),
            "broker order",
        )
        _require_unique(
            (item.execution_id for item in self.executions),
            "execution",
        )
        _require_unique(
            (item.contract_id for item in self.protections),
            "protection contract",
        )
        _require_aware(self.captured_at, "snapshot captured_at")

    @property
    def complete(self) -> bool:
        return all(
            (
                self.positions_complete,
                self.orders_complete,
                self.executions_complete,
                self.protection_complete,
            )
        )


@dataclass(frozen=True)
class ReconciliationFinding:
    finding_kind: str
    severity: FindingSeverity
    expected: Mapping[str, object]
    actual: Mapping[str, object]
    command_id: str | None = None


@dataclass(frozen=True)
class ReconciliationResult:
    status: ReconciliationStatus
    receipts: tuple[LifecycleReceipt, ...]
    findings: tuple[ReconciliationFinding, ...]
    expected_summary: Mapping[str, object]
    actual_summary: Mapping[str, object]

    @property
    def lock_required(self) -> bool:
        return self.status == ReconciliationStatus.LOCKED


@dataclass(frozen=True)
class LatencySummary:
    stage: LatencyStage
    sample_count: int
    p50_ms: Decimal | None
    p95_ms: Decimal | None
    p99_ms: Decimal | None


def evaluate_broker_snapshot(
    expected: FollowerReconciliationExpectation,
    snapshot: FollowerBrokerSnapshot,
) -> ReconciliationResult:
    """Resolve local uncertainty from a complete broker-authoritative snapshot."""

    if expected.follower_connection_id != snapshot.follower_connection_id:
        raise ValueError("reconciliation follower does not match the snapshot")
    expected_summary = _expected_summary(expected)
    actual_summary = _actual_summary(snapshot)
    if not snapshot.complete:
        missing = [
            name
            for name, complete in (
                ("positions", snapshot.positions_complete),
                ("orders", snapshot.orders_complete),
                ("executions", snapshot.executions_complete),
                ("protection", snapshot.protection_complete),
            )
            if not complete
        ]
        return ReconciliationResult(
            status=ReconciliationStatus.INCOMPLETE,
            receipts=(),
            findings=(
                ReconciliationFinding(
                    finding_kind="snapshot_incomplete",
                    severity=FindingSeverity.WARNING,
                    expected={"complete_sections": 4},
                    actual={"missing_sections": missing},
                ),
            ),
            expected_summary=expected_summary,
            actual_summary=actual_summary,
        )

    commands = {item.command_id: item for item in expected.commands}
    by_key = {item.idempotency_key: item for item in expected.commands}
    by_broker_id = {
        broker_id: item
        for item in expected.commands
        for broker_id in item.broker_order_ids
    }
    order_evidence: dict[str, list[BrokerOrderState]] = {}
    execution_evidence: dict[str, list[BrokerExecutionState]] = {}
    findings: list[ReconciliationFinding] = []

    for order in snapshot.orders:
        command = _match_order(order, commands, by_key, by_broker_id)
        if command is None:
            if order.active:
                findings.append(
                    ReconciliationFinding(
                        finding_kind="untracked_working_order",
                        severity=FindingSeverity.BLOCKING,
                        expected={"known_command": True},
                        actual={
                            "state": order.normalized_state.value,
                            "symbol": order.symbol,
                            "exchange": order.exchange,
                        },
                    )
                )
            continue
        order_evidence.setdefault(command.command_id, []).append(order)
        if order.normalized_state in {
            ObservedOrderState.UNKNOWN,
            ObservedOrderState.COMPLETED_UNKNOWN,
        }:
            findings.append(
                ReconciliationFinding(
                    finding_kind="unknown_broker_order_state",
                    severity=FindingSeverity.BLOCKING,
                    expected={"normalized_state": "known"},
                    actual={"normalized_state": order.normalized_state.value},
                    command_id=command.command_id,
                )
            )
        mismatch = _order_mismatch(command, order)
        if mismatch:
            findings.append(
                ReconciliationFinding(
                    finding_kind="command_order_mismatch",
                    severity=FindingSeverity.BLOCKING,
                    expected=mismatch[0],
                    actual=mismatch[1],
                    command_id=command.command_id,
                )
            )

    for execution in snapshot.executions:
        command = commands.get(execution.local_command_id or "") or by_broker_id.get(
            execution.broker_order_id
        )
        if command is None:
            findings.append(
                ReconciliationFinding(
                    finding_kind="untracked_execution",
                    severity=FindingSeverity.BLOCKING,
                    expected={"known_command": True},
                    actual={"quantity": execution.quantity},
                )
            )
            continue
        execution_evidence.setdefault(command.command_id, []).append(execution)

    receipts: list[LifecycleReceipt] = []
    projected_receipts: dict[str, LifecycleReceipt] = {}
    for command in expected.commands:
        if command.lifecycle_status == LifecycleStatus.FILLED:
            continue
        if command.lifecycle_status in {
            LifecycleStatus.CANCELLED,
            LifecycleStatus.REJECTED,
            LifecycleStatus.FAILED,
            LifecycleStatus.DIVERGED,
            LifecycleStatus.BLOCKED,
        }:
            continue
        executions = execution_evidence.get(command.command_id, [])
        orders = order_evidence.get(command.command_id, [])
        broker_filled = sum(_effective_quantity(item) for item in executions)
        if executions and broker_filled < command.cumulative_filled_quantity:
            findings.append(
                ReconciliationFinding(
                    finding_kind="execution_fill_regression",
                    severity=FindingSeverity.BLOCKING,
                    expected={
                        "minimum_cumulative_fill": command.cumulative_filled_quantity
                    },
                    actual={"broker_cumulative_fill": broker_filled},
                    command_id=command.command_id,
                )
            )
        if broker_filled > command.quantity:
            findings.append(
                ReconciliationFinding(
                    finding_kind="execution_overfill",
                    severity=FindingSeverity.BLOCKING,
                    expected={"maximum_cumulative_fill": command.quantity},
                    actual={"broker_cumulative_fill": broker_filled},
                    command_id=command.command_id,
                )
            )
        if (
            command.lifecycle_status == LifecycleStatus.PLANNED
            and command.outbox_status == OutboxStatus.PENDING
            and (orders or executions)
        ):
            findings.append(
                ReconciliationFinding(
                    finding_kind="broker_evidence_for_unclaimed_command",
                    severity=FindingSeverity.BLOCKING,
                    expected={"delivery_status": OutboxStatus.PENDING.value},
                    actual={"broker_evidence": "present"},
                    command_id=command.command_id,
                )
            )
            continue
        receipt = _receipt_from_evidence(snapshot, command, executions, orders)
        if receipt is not None:
            receipts.append(receipt)
            projected_receipts[command.command_id] = receipt
            if (
                receipt.kind
                in {ReceiptKind.ACKNOWLEDGED, ReceiptKind.PARTIAL_FILL}
                and not any(item.active for item in orders)
            ):
                findings.append(
                    ReconciliationFinding(
                        finding_kind="active_order_missing",
                        severity=FindingSeverity.BLOCKING,
                        expected={"working_order": True},
                        actual={"working_order": False},
                        command_id=command.command_id,
                    )
                )
            continue
        if not orders and not executions and (
            command.outbox_status
            in {
                OutboxStatus.CLAIMED,
                OutboxStatus.SENT,
                OutboxStatus.UNCERTAIN,
            }
            or command.lifecycle_status
            in {
                LifecycleStatus.DISPATCHING,
                LifecycleStatus.DISPATCHED,
                LifecycleStatus.ACKNOWLEDGED,
                LifecycleStatus.PARTIALLY_FILLED,
            }
        ):
            findings.append(
                ReconciliationFinding(
                    finding_kind="ambiguous_command_missing",
                    severity=FindingSeverity.BLOCKING,
                    expected={
                        "command_id": command.command_id,
                        "delivery_status": command.outbox_status.value,
                    },
                    actual={"broker_evidence": "absent"},
                    command_id=command.command_id,
                )
            )

    _compare_position(
        expected,
        snapshot,
        projected_receipts,
        findings,
    )
    if any(item.severity == FindingSeverity.BLOCKING for item in findings):
        status = ReconciliationStatus.LOCKED
    elif receipts:
        status = ReconciliationStatus.ADOPTED
    else:
        status = ReconciliationStatus.CLEAN
    return ReconciliationResult(
        status=status,
        receipts=tuple(receipts),
        findings=tuple(findings),
        expected_summary=expected_summary,
        actual_summary=actual_summary,
    )


def summarize_latency(
    stage: LatencyStage,
    values_ms: Iterable[Decimal],
) -> LatencySummary:
    values = sorted(Decimal(value) for value in values_ms)
    return LatencySummary(
        stage=stage,
        sample_count=len(values),
        p50_ms=_nearest_rank(values, Decimal("0.50")),
        p95_ms=_nearest_rank(values, Decimal("0.95")),
        p99_ms=_nearest_rank(values, Decimal("0.99")),
    )


def _match_order(
    order: BrokerOrderState,
    commands: Mapping[str, ExpectedReconciliationCommand],
    by_key: Mapping[str, ExpectedReconciliationCommand],
    by_broker_id: Mapping[str, ExpectedReconciliationCommand],
) -> ExpectedReconciliationCommand | None:
    return (
        commands.get(order.local_command_id or "")
        or by_key.get(order.idempotency_key or "")
        or by_broker_id.get(order.broker_order_id)
    )


def _order_mismatch(
    command: ExpectedReconciliationCommand,
    order: BrokerOrderState,
) -> tuple[dict[str, object], dict[str, object]] | None:
    expected = {
        "symbol": command.symbol,
        "exchange": command.exchange,
        "side": command.side.value if command.side else None,
        "quantity": command.quantity,
    }
    actual = {
        "symbol": order.symbol,
        "exchange": order.exchange,
        "side": order.side.value if order.side else None,
        "quantity": order.quantity,
    }
    comparable = {
        key: value
        for key, value in actual.items()
        if value is not None
    }
    if any(expected[key] != value for key, value in comparable.items()):
        return expected, actual
    return None


def _receipt_from_evidence(
    snapshot: FollowerBrokerSnapshot,
    command: ExpectedReconciliationCommand,
    executions: list[BrokerExecutionState],
    orders: list[BrokerOrderState],
) -> LifecycleReceipt | None:
    filled = max(
        [command.cumulative_filled_quantity]
        + [sum(_effective_quantity(item) for item in executions)]
        + [item.cumulative_filled_quantity for item in orders]
    )
    broker_id = (
        executions[-1].broker_order_id
        if executions
        else orders[-1].broker_order_id
        if orders
        else None
    )
    if command.quantity and filled >= command.quantity:
        kind = ReceiptKind.FILL
        filled = command.quantity
    elif filled > command.cumulative_filled_quantity:
        kind = ReceiptKind.PARTIAL_FILL
    elif orders:
        states = {item.normalized_state for item in orders}
        if ObservedOrderState.REJECTED in states:
            kind = ReceiptKind.REJECTED
        elif ObservedOrderState.CANCELLED in states:
            kind = ReceiptKind.CANCELLED
        elif ObservedOrderState.FILLED in states:
            kind = ReceiptKind.FILL
            filled = command.quantity
        elif states & {
            ObservedOrderState.PENDING,
            ObservedOrderState.WORKING,
            ObservedOrderState.PARTIALLY_FILLED,
        }:
            kind = ReceiptKind.ACKNOWLEDGED
        else:
            return None
    else:
        return None
    if kind == ReceiptKind.ACKNOWLEDGED and command.lifecycle_status in {
        LifecycleStatus.ACKNOWLEDGED,
        LifecycleStatus.PARTIALLY_FILLED,
    }:
        return None
    receipt_id = "reconcile:" + _digest(
        snapshot.snapshot_id,
        command.command_id,
        kind.value,
        str(filled),
    )
    fill_price = executions[-1].price if executions else None
    return LifecycleReceipt(
        receipt_id=receipt_id,
        command_id=command.command_id,
        kind=kind,
        occurred_at=snapshot.captured_at,
        broker_order_id=broker_id,
        cumulative_filled_quantity=filled,
        fill_price=fill_price if kind in {ReceiptKind.PARTIAL_FILL, ReceiptKind.FILL} else None,
        rejection_code="broker_snapshot_rejected" if kind == ReceiptKind.REJECTED else None,
    )


def _compare_position(
    expected: FollowerReconciliationExpectation,
    snapshot: FollowerBrokerSnapshot,
    projected_receipts: Mapping[str, LifecycleReceipt],
    findings: list[ReconciliationFinding],
) -> None:
    if (
        expected.contract_id is None
        or expected.starting_signed_quantity is None
        or expected.expected_signed_quantity is None
    ):
        unexpected = [item for item in snapshot.positions if item.signed_quantity]
        if unexpected:
            findings.append(
                ReconciliationFinding(
                    finding_kind="unexpected_position",
                    severity=FindingSeverity.BLOCKING,
                    expected={"signed_quantity": 0},
                    actual={"nonzero_position_count": len(unexpected)},
                )
            )
        return
    actual = sum(
        item.signed_quantity
        for item in snapshot.positions
        if item.contract_id == expected.contract_id
    )
    other = [
        item
        for item in snapshot.positions
        if item.contract_id != expected.contract_id and item.signed_quantity
    ]
    if other:
        findings.append(
            ReconciliationFinding(
                finding_kind="unexpected_contract_position",
                severity=FindingSeverity.BLOCKING,
                expected={"contract_id": expected.contract_id},
                actual={"nonzero_other_contracts": len(other)},
            )
        )
    latest_commands = [
        item for item in expected.commands if item.intent_id == expected.latest_intent_id
    ]
    projected_quantity = expected.starting_signed_quantity
    settled = True
    for command in latest_commands:
        receipt = projected_receipts.get(command.command_id)
        cumulative = max(
            command.cumulative_filled_quantity,
            receipt.cumulative_filled_quantity if receipt else 0,
        )
        if command.side == Side.BUY:
            projected_quantity += cumulative
        elif command.side == Side.SELL:
            projected_quantity -= cumulative
        receipt_terminal = receipt is not None and receipt.kind in {
            ReceiptKind.FILL,
            ReceiptKind.CANCELLED,
            ReceiptKind.REJECTED,
            ReceiptKind.DIVERGED,
        }
        if command.lifecycle_status not in {
            LifecycleStatus.FILLED,
            LifecycleStatus.CANCELLED,
            LifecycleStatus.REJECTED,
            LifecycleStatus.FAILED,
            LifecycleStatus.DIVERGED,
            LifecycleStatus.BLOCKED,
        } and not receipt_terminal:
            settled = False
    if (
        expected.latest_directive == CopyDirective.SYNC_POSITION
        and settled
        and actual != projected_quantity
    ):
        findings.append(
            ReconciliationFinding(
                finding_kind="position_quantity_mismatch",
                severity=FindingSeverity.BLOCKING,
                expected={
                    "contract_id": expected.contract_id,
                    "signed_quantity": projected_quantity,
                },
                actual={"signed_quantity": actual},
            )
        )
    if actual:
        protected = sum(
            item.active_stop_quantity
            for item in snapshot.protections
            if item.contract_id == expected.contract_id
        )
        if protected < abs(actual):
            findings.append(
                ReconciliationFinding(
                    finding_kind="protective_stop_shortfall",
                    severity=FindingSeverity.BLOCKING,
                    expected={"active_stop_quantity": abs(actual)},
                    actual={"active_stop_quantity": protected},
                )
            )


def _expected_summary(
    expected: FollowerReconciliationExpectation,
) -> dict[str, object]:
    return {
        "latest_intent_id": expected.latest_intent_id,
        "latest_intent_status": (
            expected.latest_intent_status.value
            if expected.latest_intent_status
            else None
        ),
        "contract_id": expected.contract_id,
        "starting_signed_quantity": expected.starting_signed_quantity,
        "expected_signed_quantity": expected.expected_signed_quantity,
        "command_count": len(expected.commands),
    }


def _actual_summary(snapshot: FollowerBrokerSnapshot) -> dict[str, object]:
    return {
        "complete": snapshot.complete,
        "position_count": len(snapshot.positions),
        "working_order_count": sum(item.active for item in snapshot.orders),
        "execution_count": len(snapshot.executions),
        "protection_count": len(snapshot.protections),
    }


def _nearest_rank(
    values: list[Decimal],
    percentile: Decimal,
) -> Decimal | None:
    if not values:
        return None
    rank = max(1, math.ceil(float(percentile * len(values))))
    return values[rank - 1]


def _effective_quantity(execution: BrokerExecutionState) -> int:
    return (
        execution.effective_quantity_delta
        if execution.effective_quantity_delta is not None
        else execution.quantity
    )


def _require_unique(values: Iterable[str], name: str) -> None:
    materialized = tuple(values)
    if len(materialized) != len(set(materialized)):
        raise ValueError(f"snapshot {name} identities must be unique")


def _digest(*parts: str) -> str:
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()[:48]


def _require_aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
