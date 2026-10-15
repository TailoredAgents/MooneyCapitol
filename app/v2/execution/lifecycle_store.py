from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Protocol

from sqlalchemy import desc, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.v2.db_models import (
    V2CopyLifecycleCommand,
    V2CopyLifecycleEvent,
    V2CopyLifecycleIntent,
)
from app.v2.domain.models import Side
from app.v2.execution.lifecycle import (
    BracketOperation,
    CopyAction,
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


@dataclass(frozen=True)
class StoredCommand:
    command: LifecycleCommand
    status: LifecycleStatus
    broker_order_id: str | None = field(repr=False)
    broker_order_ids: tuple[str, ...] = field(repr=False)
    cumulative_filled_quantity: int
    attempt_count: int
    revision: int


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
            if item.status != LifecycleStatus.PLANNED:
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

    def mark_dispatching(self, command_id: str) -> StoredIntent: ...

    def mark_dispatched(
        self, command_id: str, *, broker_order_ids: tuple[str, ...]
    ) -> StoredIntent: ...

    def record_receipt(self, receipt: LifecycleReceipt) -> StoredIntent: ...


class SqlAlchemyLifecycleStore:
    """Commits every lifecycle transition before returning it to the caller."""

    def __init__(self, session: Session) -> None:
        self.session = session

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
        now = datetime.now(timezone.utc)
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

    def mark_dispatching(self, command_id: str) -> StoredIntent:
        command = self._locked_command(command_id)
        if LifecycleStatus(command.status) != LifecycleStatus.PLANNED:
            raise CommandNotDispatchable("command is not in PLANNED state")
        if command.depends_on_command_id:
            dependency = self.session.get(
                V2CopyLifecycleCommand, command.depends_on_command_id
            )
            if dependency is None or LifecycleStatus(
                dependency.status
            ) != LifecycleStatus.FILLED:
                raise CommandNotDispatchable("command dependency has not filled")
        previous = command.status
        now = datetime.now(timezone.utc)
        command.status = LifecycleStatus.DISPATCHING.value
        command.attempt_count += 1
        command.revision += 1
        command.updated_at = now
        intent = self._locked_intent(command.intent_id)
        self._update_intent_status(intent, now)
        self._add_command_event(command, "DISPATCH_STARTED", previous, now)
        self.session.commit()
        return self._snapshot(intent, created=False)

    def mark_dispatched(
        self,
        command_id: str,
        *,
        broker_order_ids: tuple[str, ...],
    ) -> StoredIntent:
        command = self._locked_command(command_id)
        if LifecycleStatus(command.status) != LifecycleStatus.DISPATCHING:
            raise CommandNotDispatchable("command was not durably marked DISPATCHING")
        previous = command.status
        now = datetime.now(timezone.utc)
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
        self.session.commit()
        return self._snapshot(intent, created=False)

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
                raise DuplicateMasterEventConflict(
                    "receipt ID was already used for different lifecycle content"
                )
            return self.get_intent(duplicate.intent_id)

        command = self._locked_command(receipt.command_id)
        previous_status = LifecycleStatus(command.status)
        resulting = transition_status(
            previous_status,
            receipt.kind,
            previous_filled_quantity=command.cumulative_filled_quantity,
            new_filled_quantity=receipt.cumulative_filled_quantity,
            command_quantity=command.quantity,
        )
        now = datetime.now(timezone.utc)
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
        self.session.commit()
        return self._snapshot(intent, created=False)

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
            commands=tuple(_command_snapshot(command) for command in commands),
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


def _command_snapshot(row: V2CopyLifecycleCommand) -> StoredCommand:
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
    )


def _event_id(*parts: str) -> str:
    digest = hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()[:48]
    return "le_" + digest
