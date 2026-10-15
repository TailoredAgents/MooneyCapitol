from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Callable, Protocol

from sqlalchemy import desc, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.v2.db_models import (
    V2CopyDeliveryEvent,
    V2CopyLatency,
    V2CopyLifecycleCommand,
    V2CopyLifecycleEvent,
    V2CopyLifecycleIntent,
    V2CopyOutbox,
    V2CopyReconciliationFinding,
    V2CopyReconciliationRun,
    V2ExecutionLease,
)
from app.v2.domain.models import Side
from app.v2.execution.lifecycle import (
    BracketOperation,
    CopyAction,
    CopyDirective,
    FollowerLifecyclePlan,
    LifecycleCommand,
    LifecycleReceipt,
    LifecycleStatus,
    OrderDuration,
    OrderType,
    ReceiptKind,
    compile_commands,
    overall_intent_status,
    plan_fingerprint,
    transition_status,
)
from app.v2.execution.lease import LeaseToken
from app.v2.execution.recovery import (
    ExpectedReconciliationCommand,
    FollowerBrokerSnapshot,
    FollowerReconciliationExpectation,
    LatencyStage,
    LatencySummary,
    OutboxStatus,
    ReconciliationResult,
    summarize_latency,
)


class LifecycleStoreError(RuntimeError):
    pass


class LifecycleIntentNotFound(LifecycleStoreError):
    pass


class LifecycleCommandNotFound(LifecycleStoreError):
    pass


class DuplicateMasterEventConflict(LifecycleStoreError):
    pass


class StaleMasterEvent(LifecycleStoreError):
    pass


class CommandNotDispatchable(LifecycleStoreError):
    pass


class ExecutionFenceLost(LifecycleStoreError):
    pass


@dataclass(frozen=True)
class StoredCommand:
    command: LifecycleCommand
    status: LifecycleStatus
    broker_order_id: str | None = field(repr=False)
    broker_order_ids: tuple[str, ...] = field(repr=False)
    cumulative_filled_quantity: int
    attempt_count: int
    revision: int
    outbox_status: OutboxStatus | None = None


@dataclass(frozen=True)
class StoredIntent:
    intent_id: str
    source_event_id: str
    master_order_id: str
    master_revision: int
    follower_connection_id: str
    status: LifecycleStatus
    allowed: bool
    reason: str
    revision: int
    commands: tuple[StoredCommand, ...]
    created: bool = False

    @property
    def next_dispatchable(self) -> StoredCommand | None:
        by_id = {item.command.command_id: item for item in self.commands}
        for item in self.commands:
            if item.status != LifecycleStatus.PLANNED or item.outbox_status not in {
                None,
                OutboxStatus.PENDING,
            }:
                continue
            dependency = item.command.depends_on_command_id
            if dependency is None:
                return item
            prerequisite = by_id.get(dependency)
            if prerequisite and prerequisite.status == LifecycleStatus.FILLED:
                return item
        return None


class LifecycleStore(Protocol):
    def persist_intent(self, plan: FollowerLifecyclePlan) -> StoredIntent: ...

    def get_intent(self, intent_id: str) -> StoredIntent: ...

    def claim_dispatch(
        self,
        intent_id: str,
        *,
        token: LeaseToken,
        claim_ttl: timedelta,
    ) -> StoredCommand | None: ...

    def mark_dispatched(
        self,
        command_id: str,
        *,
        token: LeaseToken,
        broker_order_ids: tuple[str, ...],
    ) -> StoredIntent: ...

    def mark_uncertain(
        self,
        command_id: str,
        *,
        token: LeaseToken,
        error_code: str,
    ) -> StoredIntent: ...

    def record_receipt(self, receipt: LifecycleReceipt) -> StoredIntent: ...


class SqlAlchemyLifecycleStore:
    """Commits every lifecycle transition before returning it to the caller."""

    def __init__(
        self,
        session: Session,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.session = session
        self._clock = clock

    def persist_intent(self, plan: FollowerLifecyclePlan) -> StoredIntent:
        fingerprint = plan_fingerprint(plan)
        existing = self.session.get(V2CopyLifecycleIntent, plan.intent_id)
        if existing is not None:
            if existing.source_fingerprint != fingerprint:
                raise DuplicateMasterEventConflict(
                    "source event was already persisted with different content"
                )
            return self._snapshot(existing, created=False)

        latest = self.session.execute(
            select(V2CopyLifecycleIntent)
            .where(
                V2CopyLifecycleIntent.profile_id == plan.profile_id,
                V2CopyLifecycleIntent.master_order_id == plan.event.master_order_id,
                V2CopyLifecycleIntent.follower_connection_id
                == plan.follower_connection_id,
            )
            .order_by(desc(V2CopyLifecycleIntent.master_revision))
            .limit(1)
        ).scalar_one_or_none()
        if latest is not None:
            if latest.master_revision > plan.event.revision:
                raise StaleMasterEvent("master lifecycle revision is stale")
            if (
                latest.master_revision == plan.event.revision
                and latest.source_event_id != plan.event.source_event_id
            ):
                raise DuplicateMasterEventConflict(
                    "master lifecycle revision already belongs to another source event"
                )

        commands = compile_commands(plan)
        if not plan.allowed:
            status = LifecycleStatus.BLOCKED
        elif not commands:
            status = LifecycleStatus.FILLED
        else:
            status = LifecycleStatus.PLANNED
        now = self._now()
        row = V2CopyLifecycleIntent(
            intent_id=plan.intent_id,
            profile_id=plan.profile_id,
            source_event_id=plan.event.source_event_id,
            source_fingerprint=fingerprint,
            master_order_id=plan.event.master_order_id,
            master_revision=plan.event.revision,
            master_event_kind=plan.event.kind.value,
            follower_connection_id=plan.follower_connection_id,
            directive=plan.directive.value,
            status=status.value,
            allowed=plan.allowed,
            reason=plan.reason,
            contract_id=plan.contract_id,
            symbol=plan.symbol,
            exchange=plan.exchange,
            current_signed_quantity=plan.current_signed_quantity,
            evaluated_target_signed_quantity=plan.evaluated_target_signed_quantity,
            achievable_target_signed_quantity=plan.achievable_target_signed_quantity,
            delta_signed_quantity=plan.delta_signed_quantity,
            desired_quantity=plan.desired_quantity,
            desired_risk_dollars=plan.desired_risk_dollars,
            achievable_risk_dollars=plan.achievable_risk_dollars,
            desired_risk_fraction=plan.desired_risk_fraction,
            achievable_risk_fraction=plan.achievable_risk_fraction,
            desired_exposure_fraction=plan.desired_exposure_fraction,
            achievable_exposure_fraction=plan.achievable_exposure_fraction,
            risk_parity_error=plan.risk_parity_error,
            exposure_parity_error=plan.exposure_parity_error,
            parity_error=plan.parity_error,
            over_target=plan.over_target,
            target_margin_required=plan.target_margin_required,
            revision=1,
            created_at=now,
            updated_at=now,
        )
        self.session.add(row)
        for command in commands:
            self.session.add(_command_row(command, now))
            outbox_id = _outbox_id(command.command_id)
            self.session.add(
                V2CopyOutbox(
                    outbox_id=outbox_id,
                    command_id=command.command_id,
                    intent_id=command.intent_id,
                    follower_connection_id=plan.follower_connection_id,
                    status=OutboxStatus.PENDING.value,
                    attempt_count=0,
                    enqueued_at=now,
                    updated_at=now,
                )
            )
            self.session.add(
                V2CopyDeliveryEvent(
                    delivery_event_id=_delivery_event_id(
                        outbox_id, "ENQUEUED", "1"
                    ),
                    outbox_id=outbox_id,
                    command_id=command.command_id,
                    event_kind="ENQUEUED",
                    previous_status=None,
                    resulting_status=OutboxStatus.PENDING.value,
                    occurred_at=now,
                    recorded_at=now,
                )
            )
            self.session.add(
                V2CopyLatency(
                    command_id=command.command_id,
                    intent_id=command.intent_id,
                    follower_connection_id=plan.follower_connection_id,
                    event_occurred_at=plan.event.occurred_at,
                    event_received_at=plan.event.received_at,
                    decision_at=now,
                    enqueued_at=now,
                    updated_at=now,
                )
            )
        self.session.add(
            V2CopyLifecycleEvent(
                lifecycle_event_id=_event_id(plan.intent_id, "INTENT_CREATED"),
                intent_id=plan.intent_id,
                command_id=None,
                event_kind="INTENT_CREATED",
                previous_status=None,
                resulting_status=status.value,
                command_revision=None,
                cumulative_filled_quantity=0,
                occurred_at=plan.event.received_at,
                recorded_at=now,
            )
        )
        try:
            self.session.commit()
        except IntegrityError as exc:
            self.session.rollback()
            existing = self.session.get(V2CopyLifecycleIntent, plan.intent_id)
            if existing is not None and existing.source_fingerprint == fingerprint:
                return self._snapshot(existing, created=False)
            raise DuplicateMasterEventConflict(
                "copy intent conflicted with an existing lifecycle fact"
            ) from exc
        return self._snapshot(row, created=True)

    def get_intent(self, intent_id: str) -> StoredIntent:
        row = self.session.get(V2CopyLifecycleIntent, intent_id)
        if row is None:
            raise LifecycleIntentNotFound("copy lifecycle intent does not exist")
        return self._snapshot(row, created=False)

    def claim_dispatch(
        self,
        intent_id: str,
        *,
        token: LeaseToken,
        claim_ttl: timedelta,
    ) -> StoredCommand | None:
        if claim_ttl <= timedelta(0):
            raise ValueError("outbox claim TTL must be positive")
        now = self._now()
        self._assert_fence(token, now)
        intent = self._locked_intent(intent_id)
        snapshot = self._snapshot(intent, created=False)
        pending = snapshot.next_dispatchable
        if pending is None:
            self.session.rollback()
            return None
        command = self._locked_command(pending.command.command_id)
        outbox = self._locked_outbox(command.command_id)
        if OutboxStatus(outbox.status) != OutboxStatus.PENDING:
            self.session.rollback()
            return None
        if LifecycleStatus(command.status) != LifecycleStatus.PLANNED:
            self.session.rollback()
            raise CommandNotDispatchable("command is not in PLANNED state")
        if command.depends_on_command_id:
            dependency = self.session.get(
                V2CopyLifecycleCommand, command.depends_on_command_id
            )
            if dependency is None or LifecycleStatus(
                dependency.status
            ) != LifecycleStatus.FILLED:
                self.session.rollback()
                raise CommandNotDispatchable("command dependency has not filled")

        previous_command = command.status
        command.status = LifecycleStatus.DISPATCHING.value
        command.attempt_count += 1
        command.revision += 1
        command.updated_at = now
        previous_outbox = outbox.status
        outbox.status = OutboxStatus.CLAIMED.value
        outbox.claim_owner_id = token.owner_id
        outbox.claim_fence = token.fence
        outbox.claim_expires_at = min(
            now + claim_ttl,
            _aware_utc(token.expires_at),
        )
        outbox.claimed_at = now
        outbox.submission_started_at = now
        outbox.attempt_count += 1
        outbox.last_error_code = None
        outbox.updated_at = now
        self._update_intent_status(intent, now)
        self._add_command_event(
            command, "DISPATCH_STARTED", previous_command, now
        )
        self._add_delivery_event(
            outbox,
            "CLAIMED",
            previous_outbox,
            now,
            token=token,
        )
        latency = self.session.get(V2CopyLatency, command.command_id)
        if latency is not None:
            latency.submission_started_at = now
            latency.updated_at = now
        self._add_delivery_event(
            outbox,
            "SUBMISSION_STARTED",
            outbox.status,
            now,
            token=token,
        )
        self.session.commit()
        return self._stored_command(command.command_id)

    def mark_dispatched(
        self,
        command_id: str,
        *,
        token: LeaseToken,
        broker_order_ids: tuple[str, ...],
    ) -> StoredIntent:
        now = self._now()
        self._assert_fence(token, now)
        command = self._locked_command(command_id)
        if LifecycleStatus(command.status) != LifecycleStatus.DISPATCHING:
            self.session.rollback()
            raise CommandNotDispatchable("command was not durably marked DISPATCHING")
        outbox = self._locked_outbox(command_id)
        self._assert_claim(outbox, token, now)
        if outbox.submission_started_at is None:
            self.session.rollback()
            raise CommandNotDispatchable(
                "command was not durably marked as submission started"
            )
        previous = command.status
        command.status = LifecycleStatus.DISPATCHED.value
        cleaned_ids = tuple(item.strip() for item in broker_order_ids if item.strip())
        if cleaned_ids:
            command.broker_order_id = cleaned_ids[0]
            command.broker_order_ids_json = json.dumps(cleaned_ids)
        command.revision += 1
        command.updated_at = now
        intent = self._locked_intent(command.intent_id)
        self._update_intent_status(intent, now)
        self._add_command_event(command, "DISPATCHED", previous, now)
        previous_outbox = outbox.status
        outbox.status = OutboxStatus.SENT.value
        outbox.dispatch_response_at = now
        outbox.updated_at = now
        latency = self.session.get(V2CopyLatency, command_id)
        if latency is not None:
            latency.dispatch_response_at = now
            latency.updated_at = now
        self._add_delivery_event(
            outbox,
            "SENT",
            previous_outbox,
            now,
            token=token,
        )
        self.session.commit()
        return self._snapshot(intent, created=False)

    def mark_uncertain(
        self,
        command_id: str,
        *,
        token: LeaseToken,
        error_code: str,
    ) -> StoredIntent:
        now = self._now()
        self._assert_fence(token, now)
        command = self._locked_command(command_id)
        outbox = self._locked_outbox(command_id)
        self._assert_claim(outbox, token, now)
        previous = outbox.status
        outbox.status = OutboxStatus.UNCERTAIN.value
        outbox.last_error_code = _error_code(error_code)
        outbox.claim_owner_id = None
        outbox.claim_fence = None
        outbox.claim_expires_at = None
        outbox.updated_at = now
        self._add_delivery_event(
            outbox,
            "UNCERTAIN",
            previous,
            now,
            token=token,
            error_code=outbox.last_error_code,
        )
        self.session.commit()
        return self.get_intent(command.intent_id)

    def recover_abandoned_claims(
        self,
        *,
        token: LeaseToken,
        follower_connection_id: str | None = None,
    ) -> tuple[str, ...]:
        """Fence abandoned in-flight writes; they require broker reconciliation."""

        now = self._now()
        self._assert_fence(token, now)
        query = (
            select(V2CopyOutbox)
            .where(V2CopyOutbox.status == OutboxStatus.CLAIMED.value)
            .with_for_update()
        )
        if follower_connection_id is not None:
            query = query.where(
                V2CopyOutbox.follower_connection_id == follower_connection_id
            )
        rows = self.session.execute(query).scalars().all()
        recovered: list[str] = []
        for row in rows:
            expired = _aware_utc(row.claim_expires_at) <= now
            superseded = (
                row.claim_fence != token.fence
                or row.claim_owner_id != token.owner_id
            )
            if not expired and not superseded:
                continue
            previous = row.status
            previous_owner = row.claim_owner_id
            previous_fence = row.claim_fence
            row.status = OutboxStatus.UNCERTAIN.value
            row.last_error_code = "abandoned_dispatch_claim"
            row.claim_owner_id = None
            row.claim_fence = None
            row.claim_expires_at = None
            row.updated_at = now
            self._add_delivery_event(
                row,
                "UNCERTAIN",
                previous,
                now,
                owner_id=previous_owner,
                fence=previous_fence,
                error_code=row.last_error_code,
            )
            recovered.append(row.command_id)
        self.session.commit()
        return tuple(recovered)

    def record_receipt(self, receipt: LifecycleReceipt) -> StoredIntent:
        duplicate = self.session.execute(
            select(V2CopyLifecycleEvent).where(
                V2CopyLifecycleEvent.source_receipt_id == receipt.receipt_id
            )
        ).scalar_one_or_none()
        if duplicate is not None:
            conflicting = (
                duplicate.command_id != receipt.command_id
                or duplicate.event_kind != receipt.kind.value
                or duplicate.cumulative_filled_quantity
                != receipt.cumulative_filled_quantity
                or duplicate.fill_price != receipt.fill_price
                or duplicate.rejection_code != receipt.rejection_code
                or (
                    receipt.broker_order_id is not None
                    and duplicate.broker_order_id != receipt.broker_order_id
                )
            )
            if conflicting:
                self.session.rollback()
                raise DuplicateMasterEventConflict(
                    "receipt ID was already used for different lifecycle content"
                )
            return self.get_intent(duplicate.intent_id)

        command = self._locked_command(receipt.command_id)
        previous_status = LifecycleStatus(command.status)
        if (
            previous_status == LifecycleStatus.PLANNED
            and receipt.kind != ReceiptKind.DIVERGED
        ):
            self.session.rollback()
            raise CommandNotDispatchable(
                "broker receipt cannot advance an unclaimed command"
            )
        try:
            resulting = transition_status(
                previous_status,
                receipt.kind,
                previous_filled_quantity=command.cumulative_filled_quantity,
                new_filled_quantity=receipt.cumulative_filled_quantity,
                command_quantity=command.quantity,
            )
        except ValueError:
            self.session.rollback()
            raise
        now = self._now()
        command.status = resulting.value
        command.cumulative_filled_quantity = receipt.cumulative_filled_quantity
        if receipt.broker_order_id:
            command.broker_order_id = receipt.broker_order_id
            known_ids = json.loads(command.broker_order_ids_json or "[]")
            if receipt.broker_order_id not in known_ids:
                known_ids.append(receipt.broker_order_id)
                command.broker_order_ids_json = json.dumps(known_ids)
        command.revision += 1
        command.updated_at = now
        intent = self._locked_intent(command.intent_id)
        self._update_intent_status(intent, now)
        self.session.add(
            V2CopyLifecycleEvent(
                lifecycle_event_id=_event_id(command.command_id, receipt.receipt_id),
                source_receipt_id=receipt.receipt_id,
                intent_id=command.intent_id,
                command_id=command.command_id,
                event_kind=receipt.kind.value,
                previous_status=previous_status.value,
                resulting_status=resulting.value,
                command_revision=command.revision,
                broker_order_id=command.broker_order_id,
                cumulative_filled_quantity=receipt.cumulative_filled_quantity,
                fill_price=receipt.fill_price,
                rejection_code=receipt.rejection_code,
                occurred_at=receipt.occurred_at,
                recorded_at=now,
            )
        )
        outbox = self.session.execute(
            select(V2CopyOutbox)
            .where(V2CopyOutbox.command_id == command.command_id)
            .with_for_update()
        ).scalar_one_or_none()
        if outbox is not None:
            self._confirm_delivery(outbox, receipt, now)
        self._record_receipt_latency(command.command_id, receipt, now)
        self.session.commit()
        return self._snapshot(intent, created=False)

    def reconciliation_expectation(
        self,
        follower_connection_id: str,
    ) -> FollowerReconciliationExpectation:
        latest = self.session.execute(
            select(V2CopyLifecycleIntent)
            .join(
                V2CopyLifecycleEvent,
                V2CopyLifecycleEvent.intent_id
                == V2CopyLifecycleIntent.intent_id,
            )
            .where(
                V2CopyLifecycleIntent.follower_connection_id
                == follower_connection_id,
                V2CopyLifecycleEvent.event_kind == "INTENT_CREATED",
            )
            .order_by(
                desc(V2CopyLifecycleEvent.occurred_at),
                desc(V2CopyLifecycleIntent.master_revision),
                desc(V2CopyLifecycleIntent.intent_id),
            )
            .limit(1)
        ).scalar_one_or_none()
        intent_ids = self.session.execute(
            select(V2CopyLifecycleIntent.intent_id).where(
                V2CopyLifecycleIntent.follower_connection_id
                == follower_connection_id
            )
        ).scalars().all()
        command_rows = (
            self.session.execute(
                select(V2CopyLifecycleCommand)
                .where(V2CopyLifecycleCommand.intent_id.in_(intent_ids))
                .order_by(V2CopyLifecycleCommand.created_at)
            ).scalars().all()
            if intent_ids
            else []
        )
        outboxes = {
            row.command_id: row
            for row in (
                self.session.execute(
                    select(V2CopyOutbox).where(
                        V2CopyOutbox.command_id.in_(
                            [item.command_id for item in command_rows]
                        )
                    )
                ).scalars().all()
                if command_rows
                else []
            )
        }
        commands = tuple(
            _expected_command(row, outboxes.get(row.command_id))
            for row in command_rows
        )
        return FollowerReconciliationExpectation(
            profile_id=latest.profile_id if latest else "mooney-five-account",
            follower_connection_id=follower_connection_id,
            latest_intent_id=latest.intent_id if latest else None,
            latest_intent_status=(
                LifecycleStatus(latest.status) if latest else None
            ),
            latest_directive=(CopyDirective(latest.directive) if latest else None),
            contract_id=latest.contract_id if latest else None,
            symbol=latest.symbol if latest else None,
            exchange=latest.exchange if latest else None,
            starting_signed_quantity=(
                latest.current_signed_quantity if latest else None
            ),
            expected_signed_quantity=(
                latest.achievable_target_signed_quantity if latest else None
            ),
            commands=commands,
        )

    def persist_reconciliation(
        self,
        *,
        token: LeaseToken,
        snapshot: FollowerBrokerSnapshot,
        expectation: FollowerReconciliationExpectation,
        result: ReconciliationResult,
    ) -> str:
        now = self._now()
        self._assert_fence(token, now)
        run_id = "rr_" + _digest(
            snapshot.snapshot_id,
            snapshot.follower_connection_id,
        )
        existing = self.session.get(V2CopyReconciliationRun, run_id)
        if existing is not None:
            self.session.rollback()
            return run_id
        row = V2CopyReconciliationRun(
            reconciliation_run_id=run_id,
            profile_id=expectation.profile_id,
            follower_connection_id=snapshot.follower_connection_id,
            snapshot_id=snapshot.snapshot_id,
            connection_generation=snapshot.connection_generation,
            lease_owner_id=token.owner_id,
            lease_fence=token.fence,
            status=result.status.value,
            latest_intent_id=expectation.latest_intent_id,
            expected_summary_json=_json(result.expected_summary),
            actual_summary_json=_json(result.actual_summary),
            adopted_receipt_count=len(result.receipts),
            finding_count=len(result.findings),
            snapshot_at=snapshot.captured_at,
            completed_at=now,
        )
        self.session.add(row)
        for index, finding in enumerate(result.findings, start=1):
            self.session.add(
                V2CopyReconciliationFinding(
                    finding_id="rf_"
                    + _digest(run_id, str(index), finding.finding_kind),
                    reconciliation_run_id=run_id,
                    command_id=finding.command_id,
                    finding_kind=finding.finding_kind,
                    severity=finding.severity.value,
                    expected_json=_json(finding.expected),
                    actual_json=_json(finding.actual),
                    detected_at=now,
                )
            )
        self.session.commit()
        return run_id

    def latency_summary(
        self,
        stage: LatencyStage,
        *,
        follower_connection_id: str | None = None,
    ) -> LatencySummary:
        query = select(V2CopyLatency)
        if follower_connection_id is not None:
            query = query.where(
                V2CopyLatency.follower_connection_id == follower_connection_id
            )
        rows = self.session.execute(query).scalars().all()
        start_name, end_name = {
            LatencyStage.EVENT_TO_DECISION: (
                "event_occurred_at",
                "decision_at",
            ),
            LatencyStage.EVENT_TO_SUBMISSION: (
                "event_occurred_at",
                "submission_started_at",
            ),
            LatencyStage.EVENT_RECEIVED_TO_DECISION: (
                "event_received_at",
                "decision_at",
            ),
            LatencyStage.EVENT_RECEIVED_TO_SUBMISSION: (
                "event_received_at",
                "submission_started_at",
            ),
            LatencyStage.DECISION_TO_SUBMISSION: (
                "decision_at",
                "submission_started_at",
            ),
            LatencyStage.SUBMISSION_TO_RESPONSE: (
                "submission_started_at",
                "dispatch_response_at",
            ),
            LatencyStage.SUBMISSION_TO_ACKNOWLEDGEMENT: (
                "submission_started_at",
                "acknowledged_at",
            ),
            LatencyStage.SUBMISSION_TO_FIRST_FILL: (
                "submission_started_at",
                "first_fill_at",
            ),
            LatencyStage.EVENT_TO_COMPLETION: (
                "event_occurred_at",
                "completed_at",
            ),
        }[stage]
        values: list[Decimal] = []
        for row in rows:
            start = getattr(row, start_name)
            end = getattr(row, end_name)
            if start is None or end is None:
                continue
            elapsed = _aware_utc(end) - _aware_utc(start)
            if elapsed.total_seconds() >= 0:
                values.append(
                    Decimal(str(elapsed.total_seconds())) * Decimal("1000")
                )
        return summarize_latency(stage, values)

    def database_now(self) -> datetime:
        """Return the same durable clock used for leases and outbox transitions."""

        return self._now()

    def _locked_outbox(self, command_id: str) -> V2CopyOutbox:
        row = self.session.execute(
            select(V2CopyOutbox)
            .where(V2CopyOutbox.command_id == command_id)
            .with_for_update()
        ).scalar_one_or_none()
        if row is None:
            raise LifecycleStoreError("copy command has no durable outbox record")
        return row

    def _stored_command(self, command_id: str) -> StoredCommand:
        command = self.session.get(V2CopyLifecycleCommand, command_id)
        if command is None:
            raise LifecycleCommandNotFound("copy lifecycle command does not exist")
        outbox = self.session.execute(
            select(V2CopyOutbox).where(V2CopyOutbox.command_id == command_id)
        ).scalar_one_or_none()
        return _command_snapshot(command, outbox)

    def _assert_fence(self, token: LeaseToken, now: datetime) -> None:
        row = self.session.execute(
            select(V2ExecutionLease)
            .where(V2ExecutionLease.lease_name == token.name)
            .execution_options(populate_existing=True)
            .with_for_update()
        ).scalar_one_or_none()
        if (
            row is None
            or row.owner_id != token.owner_id
            or int(row.fence) != token.fence
            or _aware_utc(row.expires_at) <= now
        ):
            self.session.rollback()
            raise ExecutionFenceLost("execution lease fence is no longer current")

    def _assert_claim(
        self,
        outbox: V2CopyOutbox,
        token: LeaseToken,
        now: datetime,
    ) -> None:
        if (
            OutboxStatus(outbox.status) != OutboxStatus.CLAIMED
            or outbox.claim_owner_id != token.owner_id
            or outbox.claim_fence != token.fence
            or _aware_utc(outbox.claim_expires_at) <= now
        ):
            self.session.rollback()
            raise ExecutionFenceLost("outbox claim is no longer current")

    def _add_delivery_event(
        self,
        outbox: V2CopyOutbox,
        kind: str,
        previous_status: str | None,
        now: datetime,
        *,
        token: LeaseToken | None = None,
        owner_id: str | None = None,
        fence: int | None = None,
        error_code: str | None = None,
        identity: str | None = None,
    ) -> None:
        selected_owner = token.owner_id if token else owner_id
        selected_fence = token.fence if token else fence
        self.session.add(
            V2CopyDeliveryEvent(
                delivery_event_id=_delivery_event_id(
                    outbox.outbox_id,
                    kind,
                    str(outbox.attempt_count),
                    outbox.status,
                    identity or "",
                ),
                outbox_id=outbox.outbox_id,
                command_id=outbox.command_id,
                event_kind=kind,
                previous_status=previous_status,
                resulting_status=outbox.status,
                owner_id=selected_owner,
                fence=selected_fence,
                error_code=error_code,
                occurred_at=now,
                recorded_at=now,
            )
        )

    def _confirm_delivery(
        self,
        outbox: V2CopyOutbox,
        receipt: LifecycleReceipt,
        now: datetime,
    ) -> None:
        previous = outbox.status
        if receipt.kind == ReceiptKind.TRANSPORT_FAILURE:
            status = OutboxStatus.FAILED
        elif receipt.kind == ReceiptKind.DIVERGED:
            status = OutboxStatus.DIVERGED
        else:
            status = OutboxStatus.CONFIRMED
        outbox.status = status.value
        outbox.confirmed_at = now
        outbox.claim_owner_id = None
        outbox.claim_fence = None
        outbox.claim_expires_at = None
        outbox.updated_at = now
        self._add_delivery_event(
            outbox,
            status.value,
            previous,
            now,
            error_code=(
                receipt.rejection_code
                if receipt.kind
                in {ReceiptKind.REJECTED, ReceiptKind.TRANSPORT_FAILURE}
                else None
            ),
            identity=receipt.receipt_id,
        )

    def _record_receipt_latency(
        self,
        command_id: str,
        receipt: LifecycleReceipt,
        now: datetime,
    ) -> None:
        row = self.session.get(V2CopyLatency, command_id)
        if row is None:
            return
        occurred_at = _aware_utc(receipt.occurred_at)
        if receipt.kind == ReceiptKind.ACKNOWLEDGED and row.acknowledged_at is None:
            row.acknowledged_at = occurred_at
        if receipt.kind in {ReceiptKind.PARTIAL_FILL, ReceiptKind.FILL}:
            if row.first_fill_at is None:
                row.first_fill_at = occurred_at
        if receipt.kind in {
            ReceiptKind.FILL,
            ReceiptKind.CANCELLED,
            ReceiptKind.REJECTED,
            ReceiptKind.TRANSPORT_FAILURE,
            ReceiptKind.DIVERGED,
        }:
            row.completed_at = occurred_at
        row.updated_at = now

    def _now(self) -> datetime:
        value = self._clock() if self._clock is not None else self.session.scalar(
            select(func.now())
        )
        if value is None:  # pragma: no cover - database contract guard
            raise RuntimeError("database did not return its current time")
        return _aware_utc(value)

    def _locked_intent(self, intent_id: str) -> V2CopyLifecycleIntent:
        row = self.session.execute(
            select(V2CopyLifecycleIntent)
            .where(V2CopyLifecycleIntent.intent_id == intent_id)
            .with_for_update()
        ).scalar_one_or_none()
        if row is None:
            raise LifecycleIntentNotFound("copy lifecycle intent does not exist")
        return row

    def _locked_command(self, command_id: str) -> V2CopyLifecycleCommand:
        row = self.session.execute(
            select(V2CopyLifecycleCommand)
            .where(V2CopyLifecycleCommand.command_id == command_id)
            .with_for_update()
        ).scalar_one_or_none()
        if row is None:
            raise LifecycleCommandNotFound("copy lifecycle command does not exist")
        return row

    def _update_intent_status(
        self,
        intent: V2CopyLifecycleIntent,
        now: datetime,
    ) -> None:
        rows = self.session.execute(
            select(V2CopyLifecycleCommand).where(
                V2CopyLifecycleCommand.intent_id == intent.intent_id
            )
        ).scalars().all()
        intent.status = overall_intent_status(
            (LifecycleStatus(row.status) for row in rows),
            blocked=not intent.allowed,
        ).value
        intent.revision += 1
        intent.updated_at = now

    def _add_command_event(
        self,
        command: V2CopyLifecycleCommand,
        kind: str,
        previous: str,
        now: datetime,
    ) -> None:
        self.session.add(
            V2CopyLifecycleEvent(
                lifecycle_event_id=_event_id(
                    command.command_id, kind, str(command.revision)
                ),
                intent_id=command.intent_id,
                command_id=command.command_id,
                event_kind=kind,
                previous_status=previous,
                resulting_status=command.status,
                command_revision=command.revision,
                broker_order_id=command.broker_order_id,
                cumulative_filled_quantity=command.cumulative_filled_quantity,
                occurred_at=now,
                recorded_at=now,
            )
        )

    def _snapshot(
        self,
        row: V2CopyLifecycleIntent,
        *,
        created: bool,
    ) -> StoredIntent:
        commands = self.session.execute(
            select(V2CopyLifecycleCommand)
            .where(V2CopyLifecycleCommand.intent_id == row.intent_id)
            .order_by(V2CopyLifecycleCommand.ordinal)
        ).scalars().all()
        outboxes = {
            item.command_id: item
            for item in (
                self.session.execute(
                    select(V2CopyOutbox).where(
                        V2CopyOutbox.command_id.in_(
                            [command.command_id for command in commands]
                        )
                    )
                ).scalars().all()
                if commands
                else []
            )
        }
        return StoredIntent(
            intent_id=row.intent_id,
            source_event_id=row.source_event_id,
            master_order_id=row.master_order_id,
            master_revision=row.master_revision,
            follower_connection_id=row.follower_connection_id,
            status=LifecycleStatus(row.status),
            allowed=row.allowed,
            reason=row.reason,
            revision=row.revision,
            commands=tuple(
                _command_snapshot(command, outboxes.get(command.command_id))
                for command in commands
            ),
            created=created,
        )


def _command_row(
    command: LifecycleCommand,
    now: datetime,
) -> V2CopyLifecycleCommand:
    known_broker_ids = (
        (command.existing_broker_order_id,)
        if command.existing_broker_order_id
        else ()
    )
    return V2CopyLifecycleCommand(
        command_id=command.command_id,
        idempotency_key=command.idempotency_key,
        intent_id=command.intent_id,
        ordinal=command.ordinal,
        action=command.action.value,
        status=LifecycleStatus.PLANNED.value,
        contract_id=command.contract_id,
        symbol=command.symbol,
        exchange=command.exchange,
        side=command.side.value if command.side else None,
        quantity=command.quantity,
        order_type=command.order_type.value,
        duration=command.duration.value,
        limit_price=command.limit_price,
        trigger_price=command.trigger_price,
        stop_ticks=command.stop_ticks,
        target_ticks=command.target_ticks,
        bracket_operation=(
            command.bracket_operation.value if command.bracket_operation else None
        ),
        existing_broker_order_id=command.existing_broker_order_id,
        broker_order_id=(known_broker_ids[0] if known_broker_ids else None),
        broker_order_ids_json=json.dumps(known_broker_ids),
        depends_on_command_id=command.depends_on_command_id,
        cumulative_filled_quantity=0,
        attempt_count=0,
        revision=1,
        created_at=now,
        updated_at=now,
    )


def _command_snapshot(
    row: V2CopyLifecycleCommand,
    outbox: V2CopyOutbox | None = None,
) -> StoredCommand:
    try:
        decoded_ids = json.loads(row.broker_order_ids_json or "[]")
    except (TypeError, json.JSONDecodeError) as exc:
        raise LifecycleStoreError("stored broker order IDs are invalid") from exc
    if not isinstance(decoded_ids, list) or not all(
        isinstance(item, str) for item in decoded_ids
    ):
        raise LifecycleStoreError("stored broker order IDs are invalid")
    return StoredCommand(
        command=LifecycleCommand(
            command_id=row.command_id,
            idempotency_key=row.idempotency_key,
            intent_id=row.intent_id,
            ordinal=row.ordinal,
            action=CopyAction(row.action),
            contract_id=row.contract_id,
            symbol=row.symbol,
            exchange=row.exchange,
            side=Side(row.side) if row.side else None,
            quantity=row.quantity,
            order_type=OrderType(row.order_type),
            duration=OrderDuration(row.duration),
            limit_price=Decimal(row.limit_price) if row.limit_price is not None else None,
            trigger_price=(
                Decimal(row.trigger_price) if row.trigger_price is not None else None
            ),
            stop_ticks=row.stop_ticks,
            target_ticks=row.target_ticks,
            bracket_operation=(
                BracketOperation(row.bracket_operation)
                if row.bracket_operation
                else None
            ),
            existing_broker_order_id=row.existing_broker_order_id,
            depends_on_command_id=row.depends_on_command_id,
        ),
        status=LifecycleStatus(row.status),
        broker_order_id=row.broker_order_id,
        broker_order_ids=tuple(decoded_ids),
        cumulative_filled_quantity=row.cumulative_filled_quantity,
        attempt_count=row.attempt_count,
        revision=row.revision,
        outbox_status=(OutboxStatus(outbox.status) if outbox else None),
    )


def _event_id(*parts: str) -> str:
    digest = hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()[:48]
    return "le_" + digest


def _outbox_id(command_id: str) -> str:
    return "ob_" + _digest(command_id)


def _delivery_event_id(*parts: str) -> str:
    return "de_" + _digest(*parts)


def _digest(*parts: str) -> str:
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()[:48]


def _expected_command(
    row: V2CopyLifecycleCommand,
    outbox: V2CopyOutbox | None,
) -> ExpectedReconciliationCommand:
    snapshot = _command_snapshot(row, outbox)
    return ExpectedReconciliationCommand(
        command_id=row.command_id,
        idempotency_key=row.idempotency_key,
        intent_id=row.intent_id,
        action=CopyAction(row.action),
        lifecycle_status=LifecycleStatus(row.status),
        outbox_status=(
            OutboxStatus(outbox.status) if outbox else OutboxStatus.UNCERTAIN
        ),
        contract_id=row.contract_id,
        symbol=row.symbol,
        exchange=row.exchange,
        side=Side(row.side) if row.side else None,
        quantity=row.quantity,
        cumulative_filled_quantity=row.cumulative_filled_quantity,
        broker_order_ids=snapshot.broker_order_ids,
    )


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _error_code(value: str) -> str:
    cleaned = value.strip()
    if not cleaned or len(cleaned) > 128 or any(ord(char) < 32 for char in cleaned):
        raise ValueError("delivery error code is invalid")
    return cleaned


def _aware_utc(value: datetime | None) -> datetime:
    if value is None:
        raise ValueError("required timestamp is missing")
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
