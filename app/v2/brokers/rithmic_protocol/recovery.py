from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Hashable
from uuid import UUID

from .errors import InvalidStateTransition


class RecoveryPhase(str, Enum):
    IDLE = "IDLE"
    SUBSCRIBING = "SUBSCRIBING"
    BUFFERING = "BUFFERING"
    SNAPSHOTTING = "SNAPSHOTTING"
    RECONCILING = "RECONCILING"
    READY = "READY"
    FAILED = "FAILED"


@dataclass(frozen=True)
class RecoveryEvent:
    stream: str
    source_kind: str
    dedupe_key: Hashable
    value: Any
    received_ordinal: int


@dataclass(frozen=True)
class ReconciliationCheckpoint:
    generation_id: str
    completed_at: datetime
    streams: tuple[str, ...]
    snapshot_event_count: int
    buffered_event_count: int
    applied_event_count: int
    digest: str


def _json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (datetime, UUID, Enum)):
        return str(getattr(value, "value", value))
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    return repr(value)


def stable_fingerprint(value: Any) -> str:
    serialized = json.dumps(
        value,
        default=_json_default,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


class RecoveryCoordinator:
    """Subscribe-first barrier with replay/live overlap deduplication."""

    def __init__(self, required_streams: set[str] | frozenset[str]) -> None:
        if not required_streams:
            raise ValueError("at least one recovery stream is required")
        self.required_streams = frozenset(required_streams)
        self.generation_id: str | None = None
        self.phase = RecoveryPhase.IDLE
        self._subscribed: set[str] = set()
        self._snapshot_started: set[str] = set()
        self._snapshot_complete: set[str] = set()
        self._snapshots: list[RecoveryEvent] = []
        self._live_buffer: list[RecoveryEvent] = []
        self._ordinal = 0
        self._applied_keys: set[Hashable] = set()
        self._applied_after_ready: list[RecoveryEvent] = []
        self.last_checkpoint: ReconciliationCheckpoint | None = None

    def begin(self, generation_id: str | UUID) -> None:
        self.generation_id = str(generation_id)
        self.phase = RecoveryPhase.SUBSCRIBING
        self._subscribed.clear()
        self._snapshot_started.clear()
        self._snapshot_complete.clear()
        self._snapshots.clear()
        self._live_buffer.clear()
        self._applied_keys.clear()
        self._applied_after_ready.clear()
        self._ordinal = 0
        self.last_checkpoint = None

    def mark_subscribed(self, stream: str) -> None:
        self._require_stream(stream)
        if self.phase not in {RecoveryPhase.SUBSCRIBING, RecoveryPhase.BUFFERING}:
            raise InvalidStateTransition("subscriptions must precede snapshots")
        self._subscribed.add(stream)
        if self._subscribed == self.required_streams:
            self.phase = RecoveryPhase.BUFFERING

    def begin_snapshot(self, stream: str) -> None:
        self._require_stream(stream)
        if self._subscribed != self.required_streams:
            raise InvalidStateTransition("all live streams must be subscribed before any snapshot")
        if stream not in self._subscribed:
            raise InvalidStateTransition("stream was not subscribed")
        if stream in self._snapshot_complete:
            raise InvalidStateTransition("snapshot was already completed")
        self._snapshot_started.add(stream)
        self.phase = RecoveryPhase.SNAPSHOTTING

    def record_live(self, stream: str, value: Any, dedupe_key: Hashable) -> RecoveryEvent | None:
        self._require_stream(stream)
        event = self._event(stream, "LIVE", value, dedupe_key)
        if self.phase is RecoveryPhase.READY:
            if dedupe_key in self._applied_keys:
                return None
            self._applied_keys.add(dedupe_key)
            self._applied_after_ready.append(event)
            return event
        if self.phase not in {
            RecoveryPhase.SUBSCRIBING,
            RecoveryPhase.BUFFERING,
            RecoveryPhase.SNAPSHOTTING,
            RecoveryPhase.RECONCILING,
        }:
            raise InvalidStateTransition("live event received outside an active recovery")
        self._live_buffer.append(event)
        return None

    def record_snapshot(self, stream: str, value: Any, dedupe_key: Hashable) -> None:
        self._require_stream(stream)
        if stream not in self._snapshot_started or stream in self._snapshot_complete:
            raise InvalidStateTransition("snapshot row received outside its open snapshot")
        self._snapshots.append(self._event(stream, "SNAPSHOT", value, dedupe_key))

    def complete_snapshot(self, stream: str) -> None:
        self._require_stream(stream)
        if stream not in self._snapshot_started:
            raise InvalidStateTransition("cannot complete a snapshot that never started")
        self._snapshot_complete.add(stream)
        if self._snapshot_complete == self.required_streams:
            self.phase = RecoveryPhase.RECONCILING

    def reconcile(self, *, now: datetime | None = None) -> tuple[RecoveryEvent, ...]:
        if self.phase is not RecoveryPhase.RECONCILING:
            raise InvalidStateTransition("all required snapshots must complete before reconciliation")
        applied: list[RecoveryEvent] = []
        snapshot_count = 0
        buffered_count = len(self._live_buffer)
        # Snapshot/replay establishes the base, then buffered live updates are
        # applied in exact receive order. Identical overlap is applied once.
        for event in (*self._snapshots, *sorted(self._live_buffer, key=lambda item: item.received_ordinal)):
            if event.dedupe_key in self._applied_keys:
                continue
            self._applied_keys.add(event.dedupe_key)
            applied.append(event)
            if event.source_kind == "SNAPSHOT":
                snapshot_count += 1
        digest = stable_fingerprint([str(event.dedupe_key) for event in applied])
        self.last_checkpoint = ReconciliationCheckpoint(
            generation_id=self.generation_id or "",
            completed_at=now or datetime.now(timezone.utc),
            streams=tuple(sorted(self.required_streams)),
            snapshot_event_count=snapshot_count,
            buffered_event_count=buffered_count,
            applied_event_count=len(applied),
            digest=digest,
        )
        self._snapshots.clear()
        self._live_buffer.clear()
        self.phase = RecoveryPhase.READY
        return tuple(applied)

    def fail(self) -> None:
        self.phase = RecoveryPhase.FAILED

    def _event(self, stream: str, source: str, value: Any, key: Hashable) -> RecoveryEvent:
        self._ordinal += 1
        return RecoveryEvent(stream, source, key, value, self._ordinal)

    def _require_stream(self, stream: str) -> None:
        if stream not in self.required_streams:
            raise ValueError(f"unknown recovery stream: {stream}")

    @property
    def buffered_count(self) -> int:
        return len(self._live_buffer)

    @property
    def ready(self) -> bool:
        return self.phase is RecoveryPhase.READY and self.last_checkpoint is not None
