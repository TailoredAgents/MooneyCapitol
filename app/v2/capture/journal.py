from __future__ import annotations

import asyncio

from app.v2.capture.contracts import (
    CaptureEvent,
    CapturePlant,
    ReconciliationCheckpoint,
)


class InMemoryCaptureJournal:
    """Test/development journal that is intentionally not deployment-ready."""

    durable = False

    def __init__(self) -> None:
        self._events: dict[str, CaptureEvent] = {}
        self._checkpoints: list[ReconciliationCheckpoint] = []
        self._failed_reconciliations: list[
            tuple[str, dict[CapturePlant, str], str]
        ] = []
        self._lock = asyncio.Lock()

    async def append(self, event: CaptureEvent) -> bool:
        async with self._lock:
            if event.event_id in self._events:
                return False
            self._events[event.event_id] = event
            return True

    async def begin_reconciliation(
        self,
        account_id: str,
        generations: dict[CapturePlant, str],
    ) -> None:
        # The durable implementation persists an explicit STARTED checkpoint.
        # This in-memory implementation keeps no recoverable state by design.
        del account_id, generations

    async def fail_reconciliation(
        self,
        account_id: str,
        generations: dict[CapturePlant, str],
        failure_reason: str,
    ) -> None:
        async with self._lock:
            self._failed_reconciliations.append(
                (account_id, dict(generations), failure_reason)
            )

    async def save_checkpoint(self, checkpoint: ReconciliationCheckpoint) -> None:
        async with self._lock:
            self._checkpoints.append(checkpoint)

    @property
    def depth(self) -> int:
        return len(self._events)

    @property
    def events(self) -> tuple[CaptureEvent, ...]:
        return tuple(self._events.values())

    @property
    def checkpoints(self) -> tuple[ReconciliationCheckpoint, ...]:
        return tuple(self._checkpoints)

    @property
    def failed_reconciliations(
        self,
    ) -> tuple[tuple[str, dict[CapturePlant, str], str], ...]:
        return tuple(self._failed_reconciliations)


class DisabledObserver:
    """Explicit no-network observer used by the default Render deployment."""

    def __init__(self) -> None:
        self.started = False

    async def start(self, event_sink) -> None:
        del event_sink
        self.started = True

    async def stop(self) -> None:
        self.started = False

    async def discover_accounts(self):
        return ()

    async def prepare_reconciliation(self, account_id, generations) -> None:
        del account_id, generations
        raise RuntimeError("disabled observer cannot prepare recovery")

    async def subscribe_account(self, account_id, plant) -> None:
        del account_id, plant

    async def reconcile_account(self, account_id, generations):
        raise RuntimeError("disabled observer cannot reconcile")

    async def apply_buffered(self, account_id, events) -> None:
        del account_id, events

    async def finalize_reconciliation(self, account_id, generations):
        del account_id, generations
        raise RuntimeError("disabled observer cannot finalize recovery")

    async def abort_recovery(self, generations) -> None:
        del generations

    async def search_symbols(self, search_text, *, exchange=None, product_code=None):
        del search_text, exchange, product_code
        raise ConnectionError("disabled observer has no Ticker Plant")

    async def reference_data(self, symbol, exchange):
        del symbol, exchange
        raise ConnectionError("disabled observer has no Ticker Plant")

    async def tick_size_table(self, tick_size_type):
        del tick_size_type
        raise ConnectionError("disabled observer has no Ticker Plant")

    def plant_health(self):
        return {}
