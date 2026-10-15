from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from app.v2.execution.copy_control_store import SqlAlchemyCopyControlStore
from app.v2.execution.lifecycle import LifecycleReceipt, ReceiptKind, TERMINAL_STATUSES
from app.v2.execution.lifecycle_store import SqlAlchemyLifecycleStore
from app.v2.execution.lease import LeaseToken
from app.v2.execution.recovery import (
    FollowerBrokerSnapshot,
    ReconciliationResult,
    evaluate_broker_snapshot,
)


@dataclass(frozen=True)
class ReconciliationOutcome:
    reconciliation_run_id: str
    recovered_command_ids: tuple[str, ...]
    result: ReconciliationResult


class BrokerAuthoritativeReconciler:
    """Adopts explained broker facts and locks instead of guessing on divergence."""

    def __init__(
        self,
        lifecycle_store: SqlAlchemyLifecycleStore,
        control_store: SqlAlchemyCopyControlStore,
        *,
        max_snapshot_age: timedelta = timedelta(seconds=30),
    ) -> None:
        if lifecycle_store.session is not control_store.session:
            raise ValueError(
                "reconciliation and copy-control stores must share one transaction"
            )
        self.lifecycle_store = lifecycle_store
        self.control_store = control_store
        if max_snapshot_age <= timedelta(0):
            raise ValueError("maximum snapshot age must be positive")
        self.max_snapshot_age = max_snapshot_age

    def reconcile(
        self,
        *,
        token: LeaseToken,
        snapshot: FollowerBrokerSnapshot,
    ) -> ReconciliationOutcome:
        recovered = self.lifecycle_store.recover_abandoned_claims(
            token=token,
            follower_connection_id=snapshot.follower_connection_id,
        )
        now = self.lifecycle_store.database_now()
        captured_at = _aware_utc(snapshot.captured_at)
        if captured_at > now + timedelta(seconds=1):
            self.lifecycle_store.session.rollback()
            raise ValueError("broker reconciliation snapshot is from the future")
        if captured_at < now - self.max_snapshot_age:
            self.lifecycle_store.session.rollback()
            raise ValueError("broker reconciliation snapshot is stale")
        expectation = self.lifecycle_store.reconciliation_expectation(
            snapshot.follower_connection_id
        )
        result = evaluate_broker_snapshot(expectation, snapshot)

        for receipt in result.receipts:
            self.lifecycle_store.record_receipt(receipt)

        diverged_commands: set[str] = set()
        for finding in result.findings:
            if finding.command_id is None or finding.command_id in diverged_commands:
                continue
            current = self.lifecycle_store.get_intent(
                next(
                    item.intent_id
                    for item in expectation.commands
                    if item.command_id == finding.command_id
                )
            )
            command = next(
                item
                for item in current.commands
                if item.command.command_id == finding.command_id
            )
            if command.status in TERMINAL_STATUSES:
                continue
            self.lifecycle_store.record_receipt(
                LifecycleReceipt(
                    receipt_id="divergence:"
                    + _digest(snapshot.snapshot_id, finding.command_id),
                    command_id=finding.command_id,
                    kind=ReceiptKind.DIVERGED,
                    occurred_at=snapshot.captured_at,
                    cumulative_filled_quantity=command.cumulative_filled_quantity,
                )
            )
            diverged_commands.add(finding.command_id)

        if result.lock_required:
            self.control_store.lock_for_reconciliation(
                profile_id=expectation.profile_id,
                connection_id=snapshot.follower_connection_id,
                request_id="reconcile-lock-"
                + _digest(snapshot.snapshot_id, snapshot.follower_connection_id),
            )
        run_id = self.lifecycle_store.persist_reconciliation(
            token=token,
            snapshot=snapshot,
            expectation=expectation,
            result=result,
        )
        return ReconciliationOutcome(
            reconciliation_run_id=run_id,
            recovered_command_ids=recovered,
            result=result,
        )


def _digest(*parts: str) -> str:
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()[:48]


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
