from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from uuid import uuid4

from app.v2.brokers.rithmic import RithmicAdapterPlaceholder
from app.v2.execution.config import ExecutionConfigSnapshot
from app.v2.execution.connection_supervisor import (
    IndependentAccountConnectionSupervisor,
    SimulatedConnectionFixture,
)
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
    connection_supervision_mode: str
    supervised_account_count: int
    ready_supervised_account_count: int
    master_connection_ready: bool
    ready_follower_connection_count: int
    blockers: tuple[str, ...]


class FuturesExecutionService:
    def __init__(
        self,
        config: ExecutionConfigSnapshot,
        *,
        adapter: RithmicAdapterPlaceholder | None = None,
        lease: InMemoryFencedLease | None = None,
        journal: JournalQueue | None = None,
        connection_supervisor: IndependentAccountConnectionSupervisor | None = None,
        simulation_fixture: SimulatedConnectionFixture | None = None,
    ) -> None:
        self.config = config
        self.adapter = adapter or RithmicAdapterPlaceholder()
        self.lease = lease or InMemoryFencedLease()
        self.journal = journal or JournalQueue()
        self.connection_supervisor = connection_supervisor
        self.simulation_fixture = simulation_fixture
        self.owner_id = str(uuid4())
        self._token: LeaseToken | None = None
        self._lifecycle = "created"

    async def start(self) -> ServiceHealth:
        self._lifecycle = "starting"
        self._token = self.lease.acquire(self.config.lease_name, self.owner_id, timedelta(seconds=30))
        if self._token is not None and self.connection_supervisor is not None:
            try:
                if self.simulation_fixture is None:
                    self.connection_supervisor.start()
                else:
                    self.connection_supervisor.start_simulation(self.simulation_fixture)
            except Exception:
                self.lease.release(self._token)
                self._token = None
                self._lifecycle = "blocked"
                raise
        self._lifecycle = "blocked" if self._token is None else "foundation_only"
        return self.health()

    async def stop(self) -> ServiceHealth:
        self._lifecycle = "stopping"
        if self.connection_supervisor is not None:
            self.connection_supervisor.stop()
        await self.adapter.disconnect()
        if self._token:
            self.lease.release(self._token)
            self._token = None
        self._lifecycle = "stopped"
        return self.health()

    def health(self) -> ServiceHealth:
        supervision = (
            self.connection_supervisor.health()
            if self.connection_supervisor is not None
            else None
        )
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
            connection_supervision_mode=(
                supervision.mode if supervision is not None else "OFF"
            ),
            supervised_account_count=(
                len(supervision.accounts) if supervision is not None else 0
            ),
            ready_supervised_account_count=(
                sum(account.ready for account in supervision.accounts)
                if supervision is not None
                else 0
            ),
            master_connection_ready=(
                supervision.master_ready if supervision is not None else False
            ),
            ready_follower_connection_count=(
                supervision.ready_follower_count if supervision is not None else 0
            ),
            blockers=blockers,
        )
