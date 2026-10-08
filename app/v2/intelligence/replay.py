from __future__ import annotations

import inspect
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Awaitable, Callable, Iterable, Mapping

from app.v2.intelligence.observation import require_aware, require_exact_contract
from app.v2.market_data import AvailabilityMode, MarketDataEvent, MarketEventKind


class ReplayAccessMode(str, Enum):
    BLIND_POINT_IN_TIME = "BLIND_POINT_IN_TIME"
    FINALIZED_MARKET_RECONSTRUCTION = "FINALIZED_MARKET_RECONSTRUCTION"


@dataclass(frozen=True)
class MultiContractReplayRequest:
    nq_contract_id: str
    es_contract_id: str
    start: datetime
    end: datetime
    required_event_kinds: frozenset[MarketEventKind]
    mode: ReplayAccessMode
    replay_version: str

    def __post_init__(self) -> None:
        require_exact_contract(self.nq_contract_id, "NQ")
        require_exact_contract(self.es_contract_id, "ES")
        require_aware(self.start, "start")
        require_aware(self.end, "end")
        if self.nq_contract_id == self.es_contract_id:
            raise ValueError("replay requires distinct exact NQ and ES contracts")
        if self.end <= self.start:
            raise ValueError("replay end must follow start")
        if not self.required_event_kinds or not self.replay_version:
            raise ValueError("replay requires event kinds and a version")
        if any(not isinstance(kind, MarketEventKind) for kind in self.required_event_kinds):
            raise ValueError("replay event kinds must use MarketEventKind")

    @property
    def contract_ids(self) -> frozenset[str]:
        return frozenset({self.nq_contract_id, self.es_contract_id})


@dataclass(frozen=True)
class ReplayManifest:
    replay_id: str
    request: MultiContractReplayRequest
    provider: str
    provider_dataset: str | None
    retrieved_at: datetime
    entitlement_snapshot: Mapping[str, Any]
    source_lineage: Mapping[str, Any]

    def __post_init__(self) -> None:
        require_aware(self.retrieved_at, "retrieved_at")
        if not self.replay_id or not self.provider or not self.source_lineage:
            raise ValueError("replay manifest requires provider and source lineage")


@dataclass(frozen=True)
class ReplayFrame:
    replay_sequence: int
    replay_clock_at: datetime
    event: MarketDataEvent
    mode: ReplayAccessMode

    @property
    def behavior_learning_eligible(self) -> bool:
        return self.mode == ReplayAccessMode.BLIND_POINT_IN_TIME


def prepare_replay_frames(
    request: MultiContractReplayRequest,
    events: Iterable[MarketDataEvent],
) -> tuple[ReplayFrame, ...]:
    """Validate and deterministically merge exact-contract NQ/ES events.

    Blind replay advances on when a fact was actually available. Finalized
    reconstruction advances on event time and is deliberately ineligible for
    behavior learning.
    """

    prepared: list[tuple[datetime, MarketDataEvent]] = []
    for event in events:
        if event.contract_id not in request.contract_ids:
            raise ValueError("replay event contract is outside the requested NQ/ES pair")
        if event.kind not in request.required_event_kinds:
            continue
        if not request.start <= event.source_timestamp < request.end:
            raise ValueError("replay event is outside the requested half-open interval")
        if request.mode == ReplayAccessMode.BLIND_POINT_IN_TIME:
            if event.availability_mode != AvailabilityMode.POINT_IN_TIME:
                raise ValueError("blind replay cannot consume finalized historical facts")
            release_at = event.available_timestamp or event.received_timestamp
            if release_at >= request.end:
                continue
        else:
            if event.availability_mode != AvailabilityMode.FINALIZED_HISTORICAL:
                raise ValueError("finalized reconstruction requires explicitly tagged historical facts")
            release_at = event.source_timestamp
        prepared.append((release_at, event))

    prepared.sort(key=lambda item: _replay_order(item[0], item[1]))
    return tuple(
        ReplayFrame(index, release_at, event, request.mode)
        for index, (release_at, event) in enumerate(prepared, start=1)
    )


class HistoricalReplayEngine:
    """Read-only dispatcher; no strategy, label, fill, risk, or order hooks."""

    async def run(
        self,
        manifest: ReplayManifest,
        events: Iterable[MarketDataEvent],
        observer: Callable[[ReplayFrame], Awaitable[None] | None],
    ) -> int:
        frames = prepare_replay_frames(manifest.request, events)
        for frame in frames:
            result = observer(frame)
            if inspect.isawaitable(result):
                await result
        return len(frames)


def _replay_order(release_at: datetime, event: MarketDataEvent) -> tuple[Any, ...]:
    raw_timestamp = event.source_timestamp_ns
    if raw_timestamp is None:
        raw_timestamp = int(event.source_timestamp.timestamp() * 1_000_000_000)
    return (
        release_at,
        raw_timestamp,
        event.sequence if event.sequence is not None else -1,
        event.contract_id,
        event.kind,
        event.provider_event_id or "",
        event.revision if event.revision is not None else -1,
    )
