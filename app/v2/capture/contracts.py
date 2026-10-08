from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Mapping, Protocol, Sequence


class CapturePlant(str, Enum):
    ORDER = "ORDER"
    PNL = "PNL"
    TICKER = "TICKER"


class CaptureSource(str, Enum):
    LIVE = "LIVE"
    SNAPSHOT = "SNAPSHOT"
    REPLAY = "REPLAY"
    HISTORY = "HISTORY"
    SYSTEM = "SYSTEM"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class CaptureEvent:
    """Opaque event handed from the observer to the durable journal.

    ``payload`` may contain sensitive broker data and must never be included in
    health output or ordinary logs.
    """

    event_id: str
    # System/reference observations are not account-scoped. Broker lifecycle
    # observations always carry an exact allowlisted opaque account ID.
    account_id: str | None
    plant: CapturePlant
    source: CaptureSource
    generation_id: str
    payload: Mapping[str, object] = field(repr=False)
    received_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class ObservedAccount:
    account_id: str


@dataclass(frozen=True)
class PlantHealth:
    plant: CapturePlant
    connected: bool
    authenticated: bool
    generation_id: str | None = None
    last_message_at: datetime | None = None
    reconnecting: bool = False
    blocker: str | None = None


@dataclass(frozen=True)
class RecoveryResult:
    clean: bool
    checkpoint: str
    records_seen: int = 0
    discrepancy_count: int = 0
    blocker: str | None = None


@dataclass(frozen=True)
class ReconciliationCheckpoint:
    account_id: str
    generations: Mapping[CapturePlant, str]
    checkpoint: str
    buffered_events_applied: int
    discrepancy_count: int = 0
    completed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


EventSink = Callable[[CaptureEvent], Awaitable[bool]]


class CaptureBufferOverflow(RuntimeError):
    """A live tail could not be retained, so the generation cannot be ready."""


class ReadOnlyObserver(Protocol):
    """Narrow adapter surface for broker observation.

    The absence of submit/modify/cancel methods is deliberate.  A concrete
    R|Protocol adapter owns transport reconnects and implements the official
    read-only request sequence behind these orchestration hooks.
    """

    async def start(self, event_sink: EventSink) -> None: ...

    async def stop(self) -> None: ...

    async def discover_accounts(self) -> Sequence[ObservedAccount]: ...

    async def prepare_reconciliation(
        self,
        account_id: str,
        generations: Mapping[CapturePlant, str],
    ) -> None: ...

    async def subscribe_account(self, account_id: str, plant: CapturePlant) -> None: ...

    async def reconcile_account(
        self,
        account_id: str,
        generations: Mapping[CapturePlant, str],
    ) -> RecoveryResult: ...

    async def apply_buffered(
        self,
        account_id: str,
        events: Sequence[CaptureEvent],
    ) -> None: ...

    async def finalize_reconciliation(
        self,
        account_id: str,
        generations: Mapping[CapturePlant, str],
    ) -> RecoveryResult: ...

    async def abort_recovery(
        self,
        generations: Mapping[CapturePlant, str],
    ) -> None: ...

    async def search_symbols(
        self,
        search_text: str,
        *,
        exchange: str | None = None,
        product_code: str | None = None,
    ) -> tuple[Mapping[str, Any], ...]: ...

    async def reference_data(self, symbol: str, exchange: str) -> Any: ...

    async def tick_size_table(self, tick_size_type: str) -> tuple[Any, ...]: ...

    def plant_health(self) -> Mapping[CapturePlant, PlantHealth]: ...


class CaptureJournal(Protocol):
    """Append-only journal contract.  Implementations must deduplicate IDs."""

    durable: bool

    async def append(self, event: CaptureEvent) -> bool:
        """Persist an event and return whether it was newly inserted."""

    async def begin_reconciliation(
        self,
        account_id: str,
        generations: Mapping[CapturePlant, str],
    ) -> None: ...

    async def fail_reconciliation(
        self,
        account_id: str,
        generations: Mapping[CapturePlant, str],
        failure_reason: str,
    ) -> None: ...

    async def save_checkpoint(self, checkpoint: ReconciliationCheckpoint) -> None: ...

    @property
    def depth(self) -> int: ...
