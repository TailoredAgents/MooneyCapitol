from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from uuid import uuid4

from app.v2.brokers.rithmic import RithmicAdapterPlaceholder
from app.v2.execution.config import ExecutionConfigSnapshot
from app.v2.execution.journal import JournalQueue
from app.v2.execution.lease import InMemoryFencedLease, LeaseToken


@dataclass(frozen=True)
class ServiceHealth:
    lifecycle: str
    live: bool
    ready: bool
    submission_enabled: bool
    lease_fence: int | None
    journal_depth: int
    blockers: tuple[str, ...]


class FuturesExecutionService:
    def __init__(
        self,
        config: ExecutionConfigSnapshot,
        *,
        adapter: RithmicAdapterPlaceholder | None = None,
        lease: InMemoryFencedLease | None = None,
        journal: JournalQueue | None = None,
    ) -> None:
        self.config = config
        self.adapter = adapter or RithmicAdapterPlaceholder()
        self.lease = lease or InMemoryFencedLease()
        self.journal = journal or JournalQueue()
        self.owner_id = str(uuid4())
        self._token: LeaseToken | None = None
        self._lifecycle = "created"

    async def start(self) -> ServiceHealth:
        self._lifecycle = "starting"
        self._token = self.lease.acquire(self.config.lease_name, self.owner_id, timedelta(seconds=30))
        self._lifecycle = "blocked" if self._token is None else "foundation_only"
        return self.health()

    async def stop(self) -> ServiceHealth:
        self._lifecycle = "stopping"
        await self.adapter.disconnect()
        if self._token:
            self.lease.release(self._token)
            self._token = None
        self._lifecycle = "stopped"
        return self.health()

    def health(self) -> ServiceHealth:
        blockers = (
            "broker submission is compile-time/config disabled",
            "official Rithmic transport is unresolved",
            "durable shared lease is not configured",
        )
        return ServiceHealth(
            lifecycle=self._lifecycle,
            live=self._lifecycle not in {"created", "stopped"},
            ready=False,
            submission_enabled=False,
            lease_fence=self._token.fence if self._token else None,
            journal_depth=self.journal.depth,
            blockers=blockers,
        )
