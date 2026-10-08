from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Mapping

from app.v2.capture.config import CaptureConfig
from app.v2.capture.contracts import (
    CaptureEvent,
    CaptureBufferOverflow,
    CaptureJournal,
    CapturePlant,
    CaptureSource,
    PlantHealth,
    ReadOnlyObserver,
    ReconciliationCheckpoint,
    RecoveryResult,
)


class CaptureLifecycle(str, Enum):
    CREATED = "created"
    BLOCKED = "blocked"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(frozen=True)
class PublicPlantHealth:
    plant: str
    connected: bool
    authenticated: bool
    reconciled: bool
    reconnecting: bool
    generation_present: bool
    last_message_at: datetime | None
    blocker: str | None


@dataclass(frozen=True)
class CaptureHealth:
    lifecycle: str
    live: bool
    ready: bool
    connectivity_enabled: bool
    submission_enabled: bool
    configured_account_count: int
    discovered_allowed_account_count: int
    reconciled_account_count: int
    buffered_event_count: int
    journal_depth: int
    plants: tuple[PublicPlantHealth, ...]
    blockers: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "lifecycle": self.lifecycle,
            "live": self.live,
            "ready": self.ready,
            "connectivity_enabled": self.connectivity_enabled,
            "submission_enabled": False,
            "configured_account_count": self.configured_account_count,
            "discovered_allowed_account_count": self.discovered_allowed_account_count,
            "reconciled_account_count": self.reconciled_account_count,
            "buffered_event_count": self.buffered_event_count,
            "journal_depth": self.journal_depth,
            "plants": [
                {
                    "plant": plant.plant,
                    "connected": plant.connected,
                    "authenticated": plant.authenticated,
                    "reconciled": plant.reconciled,
                    "reconnecting": plant.reconnecting,
                    "generation_present": plant.generation_present,
                    "last_message_at": plant.last_message_at.isoformat()
                    if plant.last_message_at
                    else None,
                    "blocker": plant.blocker,
                }
                for plant in self.plants
            ],
            "blockers": list(self.blockers),
        }


class RithmicCaptureService:
    """Read-only lifecycle and recovery coordinator.

    Connectivity is never attempted unless ``CaptureConfig`` passes every
    TEST-only preflight gate.  Live subscriptions for every allowed account are
    installed before the first snapshot/replay reconciliation request.
    """

    def __init__(
        self,
        config: CaptureConfig,
        *,
        observer: ReadOnlyObserver,
        journal: CaptureJournal,
        startup_blockers: tuple[str, ...] = (),
    ) -> None:
        self.config = config
        self.observer = observer
        self.journal = journal
        self._startup_blockers = startup_blockers
        self._runtime_blockers: set[str] = set()
        self._lifecycle = CaptureLifecycle.CREATED
        self._plant_health: dict[CapturePlant, PlantHealth] = {}
        self._generation_signature: tuple[tuple[str, str], ...] | None = None
        self._attempted_generation_signature: tuple[tuple[str, str], ...] | None = None
        self._discovered_allowed: set[str] = set()
        self._reconciled_accounts: set[str] = set()
        self._reconciling_accounts: set[str] = set()
        self._buffers: dict[str, list[CaptureEvent]] = {}
        self._buffer_event_ids: dict[str, set[str]] = {}
        self._overflowed_accounts: set[str] = set()
        self._reconcile_lock = asyncio.Lock()
        self._buffer_lock = asyncio.Lock()
        self._supervisor: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()
        self._observer_started = False
        self._observer_start_failures = 0
        self._reference_query_slots = asyncio.Semaphore(2)

    async def start(self) -> CaptureHealth:
        if self._lifecycle not in {CaptureLifecycle.CREATED, CaptureLifecycle.STOPPED}:
            return self.health()
        blockers = (*self.config.preflight_blockers, *self._startup_blockers)
        if blockers:
            self._lifecycle = CaptureLifecycle.BLOCKED
            return self.health()
        if not self.journal.durable:
            self._runtime_blockers.add("durable_journal_required")
            self._lifecycle = CaptureLifecycle.BLOCKED
            return self.health()

        self._lifecycle = CaptureLifecycle.STARTING
        self._stopping.clear()
        try:
            await asyncio.wait_for(
                self.observer.start(self.ingest),
                timeout=self.config.reconcile_timeout_seconds,
            )
        except Exception:
            self._runtime_blockers.add("observer_start_failed")
            self._observer_start_failures = 1
        else:
            self._observer_started = True
            self._observer_start_failures = 0
            self._runtime_blockers.discard("observer_start_failed")
            self._lifecycle = CaptureLifecycle.RUNNING
        self._supervisor = asyncio.create_task(self._supervise(), name="rithmic-capture-supervisor")
        return self.health()

    async def stop(self) -> CaptureHealth:
        if self._lifecycle == CaptureLifecycle.STOPPED:
            return self.health()
        self._lifecycle = CaptureLifecycle.STOPPING
        self._stopping.set()
        if self._supervisor is not None:
            self._supervisor.cancel()
            try:
                await self._supervisor
            except asyncio.CancelledError:
                pass
            self._supervisor = None
        try:
            await self.observer.stop()
        finally:
            self._plant_health.clear()
            self._reconciled_accounts.clear()
            self._reconciling_accounts.clear()
            self._buffers.clear()
            self._buffer_event_ids.clear()
            self._overflowed_accounts.clear()
            self._generation_signature = None
            self._attempted_generation_signature = None
            self._observer_started = False
            self._observer_start_failures = 0
            self._lifecycle = CaptureLifecycle.STOPPED
        return self.health()

    async def ingest(self, event: CaptureEvent) -> bool:
        """Journal an allowed event without exposing or logging its payload."""

        observation_type = str(event.payload.get("observation_type", "")).upper()
        accountless_reference = (
            event.account_id is None
            and event.plant is CapturePlant.TICKER
            and observation_type
            in {"REFERENCE", "REFERENCE_SEARCH", "REFERENCE_TICK_SIZE"}
        )
        if event.account_id is None and not accountless_reference:
            self._runtime_blockers.add("unscoped_non_reference_event_rejected")
            return False
        if event.account_id is not None and event.account_id not in self.config.account_allowlist:
            self._runtime_blockers.add("event_for_non_allowlisted_account_rejected")
            return False
        if event.plant not in self.config.enabled_plants:
            self._runtime_blockers.add("event_for_disabled_plant_rejected")
            return False
        normalized = event.payload.get("normalized")
        if (
            event.account_id is not None
            and isinstance(normalized, Mapping)
            and normalized.get("authorization_revoked") is True
        ):
            self._runtime_blockers.add("account_authorization_revoked")
            self._reconciled_accounts.discard(event.account_id)

        overflow_events: tuple[CaptureEvent, ...] = ()
        if event.source in {CaptureSource.LIVE, CaptureSource.UNKNOWN} and event.account_id:
            async with self._buffer_lock:
                if event.account_id in self._reconciling_accounts:
                    buffer = self._buffers.setdefault(event.account_id, [])
                    event_ids = self._buffer_event_ids.setdefault(event.account_id, set())
                    if event.event_id in event_ids:
                        return False
                    if len(buffer) >= self.config.max_buffered_events:
                        self._runtime_blockers.add("live_event_buffer_overflow")
                        self._overflowed_accounts.add(event.account_id)
                        self._reconciled_accounts.discard(event.account_id)
                        overflow_events = (*buffer, event)
                        self._reconciling_accounts.discard(event.account_id)
                        self._buffers[event.account_id] = list(overflow_events)
                        self._buffer_event_ids[event.account_id] = {
                            item.event_id for item in overflow_events
                        }
                    else:
                        buffer.append(event)
                        event_ids.add(event.event_id)
                        return True

        if overflow_events:
            # Preserve the raw observations durably in their receive order, but
            # do not allow the overflowing generation to checkpoint or become
            # READY.  Raising invalidates its transports through recovery abort.
            await self._persist_failed_buffers([event.account_id])
            raise CaptureBufferOverflow(
                "live recovery buffer exceeded its configured bound"
            )

        return await self.journal.append(event)

    async def wait_until_ready(self, timeout: float = 5.0) -> bool:
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            if self.health().ready:
                return True
            await asyncio.sleep(min(self.config.poll_seconds, 0.05))
        return False

    @property
    def ticker_reference_available(self) -> bool:
        """Whether the optional Ticker Plant can accept read-only queries."""

        status = self._plant_health.get(CapturePlant.TICKER)
        return bool(
            self._lifecycle == CaptureLifecycle.RUNNING
            and CapturePlant.TICKER in self.config.enabled_plants
            and status
            and status.connected
            and status.authenticated
            and status.generation_id
            and not status.reconnecting
            and not status.blocker
        )

    async def search_reference_symbols(
        self,
        search_text: str,
        *,
        exchange: str | None = None,
        product_code: str | None = None,
        timeout_seconds: float,
    ) -> tuple[Mapping[str, Any], ...]:
        if not self.ticker_reference_available:
            raise ConnectionError("reference query service is unavailable")
        return await self._bounded_reference_query(
            lambda: self.observer.search_symbols(
                    search_text,
                    exchange=exchange,
                    product_code=product_code,
                ),
            timeout_seconds,
        )

    async def contract_reference(
        self,
        symbol: str,
        exchange: str,
        *,
        timeout_seconds: float,
    ) -> Any:
        if not self.ticker_reference_available:
            raise ConnectionError("reference query service is unavailable")
        return await self._bounded_reference_query(
            lambda: self.observer.reference_data(symbol, exchange),
            timeout_seconds,
        )

    async def reference_tick_sizes(
        self,
        tick_size_type: str,
        *,
        timeout_seconds: float,
    ) -> tuple[Any, ...]:
        if not self.ticker_reference_available:
            raise ConnectionError("reference query service is unavailable")
        return await self._bounded_reference_query(
            lambda: self.observer.tick_size_table(tick_size_type),
            timeout_seconds,
        )

    async def _bounded_reference_query(self, operation, timeout_seconds: float):
        async def run():
            async with self._reference_query_slots:
                # Authentication can disappear while a request waits for its
                # bounded concurrency slot.
                if not self.ticker_reference_available:
                    raise ConnectionError("reference query service is unavailable")
                return await operation()

        # The timeout covers both queueing and the R|Protocol round trip.
        return await asyncio.wait_for(run(), timeout=timeout_seconds)

    async def _supervise(self) -> None:
        while not self._stopping.is_set():
            if not self._observer_started:
                delay = min(
                    30.0,
                    max(1.0, self.config.poll_seconds)
                    * (2 ** min(self._observer_start_failures - 1, 5)),
                )
                try:
                    await asyncio.wait_for(self._stopping.wait(), timeout=delay)
                    continue
                except asyncio.TimeoutError:
                    pass
                try:
                    await asyncio.wait_for(
                        self.observer.start(self.ingest),
                        timeout=self.config.reconcile_timeout_seconds,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    self._observer_start_failures += 1
                    self._runtime_blockers.add("observer_start_failed")
                    continue
                self._observer_started = True
                self._observer_start_failures = 0
                self._runtime_blockers.discard("observer_start_failed")
                self._lifecycle = CaptureLifecycle.RUNNING

            try:
                raw_health = self.observer.plant_health()
                self._plant_health = {
                    CapturePlant(plant): status for plant, status in raw_health.items()
                }
                if self._required_plants_connected():
                    signature = self._current_generation_signature()
                    if signature != self._attempted_generation_signature:
                        # A failed recovery is not snapshot-polled repeatedly.
                        # A new transport generation (or process restart) is the
                        # next automatic recovery boundary.
                        self._attempted_generation_signature = signature
                        self._generation_signature = None
                        self._reconciled_accounts.clear()
                        self._overflowed_accounts.clear()
                        await self._reconcile(signature)
                        if self._generation_signature == signature:
                            self._runtime_blockers.difference_update(
                                {
                                    "capture_supervision_failed",
                                    "reconciliation_timed_out",
                                    "live_event_buffer_overflow",
                                    "account_authorization_revoked",
                                    "failed_recovery_buffer_persistence_failed",
                                    "failed_reconciliation_checkpoint_persistence_failed",
                                    "recovery_abort_failed",
                                }
                            )
                else:
                    self._generation_signature = None
                    self._attempted_generation_signature = None
                    self._reconciled_accounts.clear()
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError:
                self._runtime_blockers.add("reconciliation_timed_out")
                self._reconciled_accounts.clear()
            except Exception:
                self._runtime_blockers.add("capture_supervision_failed")
                self._reconciled_accounts.clear()
            await asyncio.sleep(self.config.poll_seconds)

    def _required_plants_connected(self) -> bool:
        for plant in self.config.required_plants:
            status = self._plant_health.get(plant)
            if not status or not status.connected or not status.authenticated or not status.generation_id:
                return False
        return True

    def _current_generation_signature(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            sorted(
                (plant.value, self._plant_health[plant].generation_id or "")
                for plant in self.config.required_plants
            )
        )

    def _generation_map(self) -> dict[CapturePlant, str]:
        return {
            plant: self._plant_health[plant].generation_id or ""
            for plant in self.config.required_plants
        }

    async def _reconcile(self, signature: tuple[tuple[str, str], ...]) -> None:
        async with self._reconcile_lock:
            if signature != self._current_generation_signature():
                return

            generations = self._generation_map()
            allowed = set(self.config.account_allowlist)
            target_accounts: list[str] = []
            begun_accounts: list[str] = []
            completed: set[str] = set()
            try:
                retained_accounts = [
                    account_id
                    for account_id, events in self._buffers.items()
                    if events
                ]
                if retained_accounts:
                    await self._recovery_phase(
                        self._persist_failed_buffers(retained_accounts)
                    )
                accounts = await self._recovery_phase(self.observer.discover_accounts())
                discovered_ids = {account.account_id for account in accounts}
                target_accounts = sorted(discovered_ids & allowed)
                self._discovered_allowed = set(target_accounts)
                missing = allowed - discovered_ids
                if missing:
                    self._runtime_blockers.add("allowlisted_account_not_discovered")
                else:
                    self._runtime_blockers.discard("allowlisted_account_not_discovered")
                if not target_accounts:
                    self._runtime_blockers.add("no_allowlisted_accounts_discovered")
                    return
                self._runtime_blockers.discard("no_allowlisted_accounts_discovered")

                async with self._buffer_lock:
                    self._reconciling_accounts = set(target_accounts)
                    self._buffers = {account_id: [] for account_id in target_accounts}
                    self._buffer_event_ids = {
                        account_id: set() for account_id in target_accounts
                    }

                for account_id in target_accounts:
                    await self._recovery_phase(
                        self.journal.begin_reconciliation(account_id, generations)
                    )
                    begun_accounts.append(account_id)

                # Allocate every account's recovery batch before any live
                # subscription. This gives later accounts full provenance even
                # while an earlier account is being snapshotted.
                for account_id in target_accounts:
                    await self._recovery_phase(
                        self.observer.prepare_reconciliation(
                            account_id, generations
                        )
                    )

                # This complete subscription pass intentionally precedes every
                # snapshot/replay call, including when more than one account exists.
                for account_id in target_accounts:
                    for plant in sorted(
                        self.config.required_plants, key=lambda item: item.value
                    ):
                        await self._recovery_phase(
                            self.observer.subscribe_account(account_id, plant)
                        )

                for account_id in target_accounts:
                    result = await self._recovery_phase(
                        self.observer.reconcile_account(account_id, generations)
                    )
                    if not result.clean:
                        self._runtime_blockers.add("account_reconciliation_not_clean")
                        raise RuntimeError("account reconciliation was not clean")
                    if account_id in self._overflowed_accounts:
                        raise CaptureBufferOverflow(
                            "live recovery buffer overflowed before checkpoint"
                        )
                    applied, final_result = await self._recovery_phase(
                        self._drain_buffer(account_id, generations)
                    )
                    if not final_result.clean or final_result.discrepancy_count:
                        self._runtime_blockers.add("account_reconciliation_not_clean")
                        raise RuntimeError(
                            "buffered state reconciliation was not clean"
                        )
                    if account_id in self._overflowed_accounts:
                        raise CaptureBufferOverflow(
                            "live recovery buffer overflowed during drain"
                        )
                    await self._recovery_phase(
                        self.journal.save_checkpoint(
                            ReconciliationCheckpoint(
                                account_id=account_id,
                                generations=dict(generations),
                                checkpoint=final_result.checkpoint,
                                buffered_events_applied=applied,
                                discrepancy_count=final_result.discrepancy_count,
                            )
                        )
                    )
                    completed.add(account_id)
            except BaseException as exc:
                self._reconciled_accounts.clear()
                if isinstance(exc, asyncio.CancelledError):
                    failure_reason = "recovery_cancelled"
                elif isinstance(exc, asyncio.TimeoutError):
                    failure_reason = "recovery_phase_timeout"
                elif isinstance(exc, CaptureBufferOverflow):
                    failure_reason = "live_buffer_overflow"
                else:
                    failure_reason = "reconciliation_failed"
                await asyncio.shield(
                    self._abort_and_preserve(
                        generations,
                        target_accounts,
                        begun_accounts,
                        failure_reason,
                    )
                )
                raise
            finally:
                # A failed attempt is not retried in the same generation.  Its
                # transports are invalidated above so reconnect establishes a
                # fresh generation; retaining a replay buffer here would turn a
                # recovery failure into a memory leak.
                async with self._buffer_lock:
                    self._reconciling_accounts.difference_update(target_accounts)
                    for account_id in set(target_accounts) - completed:
                        if not self._buffers.get(account_id):
                            self._buffers.pop(account_id, None)
                            self._buffer_event_ids.pop(account_id, None)
            self._reconciled_accounts = completed
            if completed == allowed and signature == self._current_generation_signature():
                self._generation_signature = signature
                self._runtime_blockers.discard("account_reconciliation_not_clean")

    async def _recovery_phase(self, operation):
        """Bound one recovery operation without imposing a multi-account deadline."""

        return await asyncio.wait_for(
            operation,
            timeout=self.config.reconcile_timeout_seconds,
        )

    async def _abort_and_preserve(
        self,
        generations: Mapping[CapturePlant, str],
        account_ids: list[str],
        begun_account_ids: list[str],
        failure_reason: str,
    ) -> None:
        for account_id in begun_account_ids:
            try:
                await asyncio.wait_for(
                    self.journal.fail_reconciliation(
                        account_id,
                        generations,
                        failure_reason,
                    ),
                    timeout=self.config.reconcile_timeout_seconds,
                )
            except Exception:
                self._runtime_blockers.add(
                    "failed_reconciliation_checkpoint_persistence_failed"
                )
        try:
            await asyncio.wait_for(
                self.observer.abort_recovery(generations),
                timeout=self.config.reconcile_timeout_seconds,
            )
        except Exception:
            self._runtime_blockers.add("recovery_abort_failed")
        try:
            await asyncio.wait_for(
                self._persist_failed_buffers(account_ids),
                timeout=self.config.reconcile_timeout_seconds,
            )
        except Exception:
            self._runtime_blockers.add("failed_recovery_buffer_persistence_failed")

    async def _persist_failed_buffers(self, account_ids: list[str]) -> None:
        for account_id in account_ids:
            while True:
                async with self._buffer_lock:
                    events = self._buffers.get(account_id, [])
                    if not events:
                        self._buffers.pop(account_id, None)
                        self._buffer_event_ids.pop(account_id, None)
                        self._reconciling_accounts.discard(account_id)
                        break
                    event = events[0]
                await self.journal.append(event)
                async with self._buffer_lock:
                    current = self._buffers.get(account_id, [])
                    if current and current[0].event_id == event.event_id:
                        current.pop(0)
                    self._buffer_event_ids.setdefault(account_id, set()).discard(
                        event.event_id
                    )

    async def _drain_buffer(
        self,
        account_id: str,
        generations: Mapping[CapturePlant, str],
    ) -> tuple[int, RecoveryResult]:
        applied = 0
        final_result: RecoveryResult | None = None
        while True:
            async with self._buffer_lock:
                events = tuple(self._buffers.get(account_id, ()))
                self._buffers[account_id] = []
                self._buffer_event_ids[account_id] = set()
                if not events:
                    # Keep the buffer lock across the local fold barrier. A live
                    # event either precedes this point and is folded, or waits
                    # until the account has left recovery and journals normally.
                    final_result = await self.observer.finalize_reconciliation(
                        account_id,
                        generations,
                    )
                    if final_result.clean:
                        self._reconciling_accounts.discard(account_id)
                        self._buffers.pop(account_id, None)
                        self._buffer_event_ids.pop(account_id, None)
                    break
            durable_events: list[CaptureEvent] = []
            for index, event in enumerate(events):
                try:
                    if await self.journal.append(event):
                        durable_events.append(event)
                except BaseException:
                    # The enclosing per-phase timeout may cancel this coroutine
                    # between extraction and durable append. Put the uncertain
                    # item plus every unprocessed successor back ahead of newer
                    # arrivals; journal idempotency handles an append that may
                    # have committed immediately before cancellation.
                    remaining = list(events[index:])
                    async with self._buffer_lock:
                        newer = self._buffers.setdefault(account_id, [])
                        self._buffers[account_id] = remaining + newer
                        event_ids = self._buffer_event_ids.setdefault(
                            account_id, set()
                        )
                        event_ids.update(item.event_id for item in remaining)
                    raise
            await self.observer.apply_buffered(account_id, durable_events)
            applied += len(durable_events)
        if final_result is None:
            raise RuntimeError("reconciliation did not reach its final fold barrier")
        return applied, final_result

    def health(self) -> CaptureHealth:
        blockers = set(self.config.preflight_blockers)
        blockers.update(self._startup_blockers)
        blockers.update(self._runtime_blockers)
        plants: list[PublicPlantHealth] = []
        generation_ready = self._generation_signature is not None
        for plant in sorted(self.config.required_plants, key=lambda item: item.value):
            status = self._plant_health.get(
                plant,
                PlantHealth(plant=plant, connected=False, authenticated=False, blocker="not_connected"),
            )
            reconciled = generation_ready and status.connected and status.authenticated
            plants.append(
                PublicPlantHealth(
                    plant=plant.value,
                    connected=status.connected,
                    authenticated=status.authenticated,
                    reconciled=reconciled,
                    reconnecting=status.reconnecting,
                    generation_present=bool(status.generation_id),
                    last_message_at=status.last_message_at,
                    blocker=status.blocker,
                )
            )
            if status.blocker:
                blockers.add(f"{plant.value.lower()}_plant_unhealthy")

        expected = set(self.config.account_allowlist)
        ready = (
            self._lifecycle == CaptureLifecycle.RUNNING
            and not blockers
            and bool(expected)
            and self._discovered_allowed == expected
            and self._reconciled_accounts == expected
            and self._required_plants_connected()
            and generation_ready
            and self.journal.durable
        )
        return CaptureHealth(
            lifecycle=self._lifecycle.value,
            live=self._lifecycle not in {CaptureLifecycle.CREATED, CaptureLifecycle.STOPPED},
            ready=ready,
            connectivity_enabled=self.config.connectivity_enabled,
            submission_enabled=False,
            configured_account_count=len(expected),
            discovered_allowed_account_count=len(self._discovered_allowed),
            reconciled_account_count=len(self._reconciled_accounts),
            buffered_event_count=sum(len(events) for events in self._buffers.values()),
            journal_depth=self.journal.depth,
            plants=tuple(plants),
            blockers=tuple(sorted(blockers)),
        )
