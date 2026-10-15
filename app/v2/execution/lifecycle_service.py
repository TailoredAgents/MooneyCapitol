from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Mapping, Protocol, Sequence

from app.v2.execution.lifecycle import (
    FollowerLifecyclePlan,
    LifecycleCommand,
    LifecycleReceipt,
    ReceiptKind,
)
from app.v2.execution.lifecycle_store import LifecycleStore, StoredIntent


class LifecycleAdapter(Protocol):
    @property
    def connection_id(self) -> str: ...

    async def dispatch(self, command: LifecycleCommand): ...


@dataclass(frozen=True)
class FollowerDispatchOutcome:
    follower_connection_id: str
    intent_id: str
    transmitted: bool
    command_id: str | None
    broker_order_ids: tuple[str, ...] = field(default=(), repr=False)
    error_type: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error_type is None


class DurableLifecycleCopier:
    """Persists all intents before independently dispatching follower commands."""

    def __init__(
        self,
        store: LifecycleStore,
        adapters: Mapping[str, LifecycleAdapter],
    ) -> None:
        self.store = store
        self.adapters = dict(adapters)
        mismatched = [
            key
            for key, adapter in self.adapters.items()
            if key != adapter.connection_id
        ]
        if mismatched:
            raise ValueError("lifecycle adapter key does not match its follower connection")
        self._store_lock = asyncio.Lock()

    def persist_plans(
        self,
        plans: Sequence[FollowerLifecyclePlan],
    ) -> tuple[StoredIntent, ...]:
        if not plans:
            raise ValueError("at least one follower lifecycle plan is required")
        identities = {
            (plan.event.source_event_id, plan.follower_connection_id) for plan in plans
        }
        if len(identities) != len(plans):
            raise ValueError("a source event may have only one plan per follower")
        # Every transaction commits inside the store. No adapter call is made
        # until the complete input batch has durable intent records.
        return tuple(self.store.persist_intent(plan) for plan in plans)

    async def dispatch_persisted(
        self,
        intents: Sequence[StoredIntent],
    ) -> tuple[FollowerDispatchOutcome, ...]:
        tasks = [self._dispatch_one(item.intent_id) for item in intents]
        return tuple(await asyncio.gather(*tasks))

    async def persist_and_dispatch(
        self,
        plans: Sequence[FollowerLifecyclePlan],
    ) -> tuple[FollowerDispatchOutcome, ...]:
        intents = self.persist_plans(plans)
        return await self.dispatch_persisted(intents)

    async def resume_intent(self, intent_id: str) -> FollowerDispatchOutcome:
        return await self._dispatch_one(intent_id)

    async def record_receipt(self, receipt: LifecycleReceipt) -> StoredIntent:
        async with self._store_lock:
            return self.store.record_receipt(receipt)

    async def _dispatch_one(self, intent_id: str) -> FollowerDispatchOutcome:
        async with self._store_lock:
            intent = self.store.get_intent(intent_id)
            pending = intent.next_dispatchable
            if pending is None:
                return FollowerDispatchOutcome(
                    follower_connection_id=intent.follower_connection_id,
                    intent_id=intent.intent_id,
                    transmitted=False,
                    command_id=None,
                )
            adapter = self.adapters.get(intent.follower_connection_id)
            if adapter is None:
                return FollowerDispatchOutcome(
                    follower_connection_id=intent.follower_connection_id,
                    intent_id=intent.intent_id,
                    transmitted=False,
                    command_id=pending.command.command_id,
                    error_type="adapter_unavailable",
                )
            marked = self.store.mark_dispatching(pending.command.command_id)
            dispatching = next(
                item
                for item in marked.commands
                if item.command.command_id == pending.command.command_id
            )

        try:
            result = await adapter.dispatch(dispatching.command)
            transmitted = bool(result.transmitted)
            broker_order_ids = tuple(result.broker_order_ids)
            if not transmitted:
                raise RuntimeError("execution dispatcher requires a transmitting Test adapter")
        except Exception as exc:
            receipt_kind = (
                ReceiptKind.REJECTED
                if getattr(exc, "broker_rejection", False)
                else ReceiptKind.TRANSPORT_FAILURE
            )
            receipt = LifecycleReceipt(
                receipt_id=_failure_receipt_id(
                    dispatching.command.command_id,
                    dispatching.attempt_count,
                ),
                command_id=dispatching.command.command_id,
                kind=receipt_kind,
                occurred_at=datetime.now(timezone.utc),
                cumulative_filled_quantity=dispatching.cumulative_filled_quantity,
                rejection_code=type(exc).__name__,
            )
            async with self._store_lock:
                self.store.record_receipt(receipt)
            return FollowerDispatchOutcome(
                follower_connection_id=intent.follower_connection_id,
                intent_id=intent.intent_id,
                transmitted=False,
                command_id=dispatching.command.command_id,
                error_type=type(exc).__name__,
            )

        async with self._store_lock:
            self.store.mark_dispatched(
                dispatching.command.command_id,
                broker_order_ids=broker_order_ids,
            )
        return FollowerDispatchOutcome(
            follower_connection_id=intent.follower_connection_id,
            intent_id=intent.intent_id,
            transmitted=True,
            command_id=dispatching.command.command_id,
            broker_order_ids=broker_order_ids,
        )


def _failure_receipt_id(command_id: str, attempt: int) -> str:
    return f"failure:{command_id}:{attempt}"
