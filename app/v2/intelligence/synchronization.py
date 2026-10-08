from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Mapping, Sequence

from app.v2.intelligence.observation import ObservationPurpose, require_aware, require_exact_contract
from app.v2.market_data import BarPayload, DataQuality, MarketDataEvent, MarketEventKind


@dataclass(frozen=True)
class MarketSeriesRequirement:
    series_id: str
    kind: MarketEventKind
    bar_resolution: str | None = None

    def __post_init__(self) -> None:
        if not self.series_id:
            raise ValueError("market series requires an id")
        if self.kind == MarketEventKind.BAR and not self.bar_resolution:
            raise ValueError("bar series require an explicit resolution")
        if self.kind != MarketEventKind.BAR and self.bar_resolution is not None:
            raise ValueError("only bar series may declare a resolution")


@dataclass(frozen=True)
class SynchronizationPolicy:
    version: str
    required_series: tuple[MarketSeriesRequirement, ...]
    maximum_event_age: timedelta
    maximum_leg_skew: timedelta
    allow_finalized_historical: bool = False
    purpose: ObservationPurpose = ObservationPurpose.LIVE_SHADOW

    def __post_init__(self) -> None:
        if not self.version or not self.required_series:
            raise ValueError("synchronization policy requires a version and market series")
        series_ids = [item.series_id for item in self.required_series]
        if len(series_ids) != len(set(series_ids)):
            raise ValueError("synchronization market series ids must be unique")
        if self.maximum_event_age < timedelta(0) or self.maximum_leg_skew < timedelta(0):
            raise ValueError("synchronization durations cannot be negative")
        if self.allow_finalized_historical and self.purpose not in {
            ObservationPurpose.OUTCOME_RESEARCH,
            ObservationPurpose.HISTORICAL_REPLAY,
        }:
            raise ValueError("finalized historical facts cannot support live or behavior observations")


@dataclass(frozen=True)
class SynchronizedEventSlice:
    cutoff_at: datetime
    nq_contract_id: str
    es_contract_id: str
    events: Mapping[str, Mapping[str, MarketDataEvent]]
    quality: DataQuality
    policy_version: str

    @property
    def is_complete(self) -> bool:
        return not self.quality.is_gap and not self.quality.is_late


def synchronize_point_in_time(
    events: Sequence[MarketDataEvent],
    *,
    nq_contract_id: str,
    es_contract_id: str,
    cutoff_at: datetime,
    policy: SynchronizationPolicy,
) -> SynchronizedEventSlice:
    """Select latest facts at-or-before cutoff without backward-looking future fill."""
    require_aware(cutoff_at, "cutoff_at")
    require_exact_contract(nq_contract_id, "NQ")
    require_exact_contract(es_contract_id, "ES")
    if nq_contract_id == es_contract_id:
        raise ValueError("synchronization requires distinct NQ and ES contracts")
    selected: dict[str, dict[str, MarketDataEvent]] = {nq_contract_id: {}, es_contract_id: {}}
    for event in events:
        if event.contract_id not in selected:
            continue
        require_aware(event.source_timestamp, "event.source_timestamp")
        if not event.eligible_at(cutoff_at, allow_finalized_historical=policy.allow_finalized_historical):
            continue
        for requirement in policy.required_series:
            if not _matches_series(event, requirement):
                continue
            current = selected[event.contract_id].get(requirement.series_id)
            if current is None or _event_order(event) > _event_order(current):
                selected[event.contract_id][requirement.series_id] = event

    notes: list[str] = []
    late = False
    gap = False
    corrected = False
    for contract_id, by_kind in selected.items():
        missing = {item.series_id for item in policy.required_series} - set(by_kind)
        if missing:
            gap = True
            notes.append(f"{contract_id}:missing:{','.join(sorted(missing))}")
        for kind, event in by_kind.items():
            if cutoff_at - event.source_timestamp > policy.maximum_event_age:
                late = True
                notes.append(f"{contract_id}:stale:{kind}")
            corrected = corrected or event.quality.is_corrected or (event.revision is not None and event.revision > 0)

    for requirement in policy.required_series:
        nq_event = selected[nq_contract_id].get(requirement.series_id)
        es_event = selected[es_contract_id].get(requirement.series_id)
        if nq_event and es_event and abs(nq_event.source_timestamp - es_event.source_timestamp) > policy.maximum_leg_skew:
            late = True
            notes.append(f"leg_skew:{requirement.series_id}")

    return SynchronizedEventSlice(
        cutoff_at,
        nq_contract_id,
        es_contract_id,
        {contract: dict(values) for contract, values in selected.items()},
        DataQuality(is_gap=gap, is_late=late, is_corrected=corrected, notes=tuple(notes)),
        policy.version,
    )


def _event_order(event: MarketDataEvent) -> tuple[datetime, int, int, int]:
    return (
        event.source_timestamp,
        event.source_timestamp_ns if event.source_timestamp_ns is not None else -1,
        event.sequence if event.sequence is not None else -1,
        event.revision if event.revision is not None else -1,
    )


def _matches_series(event: MarketDataEvent, requirement: MarketSeriesRequirement) -> bool:
    if event.kind != requirement.kind:
        return False
    if requirement.kind != MarketEventKind.BAR:
        return True
    payload: Any = event.payload
    if isinstance(payload, BarPayload):
        return payload.resolution.lower() == requirement.bar_resolution.lower()
    if isinstance(payload, Mapping):
        resolution = payload.get("resolution")
        return isinstance(resolution, str) and resolution.lower() == requirement.bar_resolution.lower()
    return False
