from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Mapping

from app.v2.intelligence.datasets import ObservationOrigin
from app.v2.intelligence.observation import require_aware


@dataclass(frozen=True)
class HorizonDefinition:
    horizon_id: str
    offset_before_decision: timedelta

    def __post_init__(self) -> None:
        if not self.horizon_id or self.offset_before_decision < timedelta(0):
            raise ValueError("horizon requires an id and non-negative offset")


@dataclass(frozen=True)
class HorizonSet:
    version: str
    horizons: tuple[HorizonDefinition, ...]

    def __post_init__(self) -> None:
        if not self.version or not self.horizons:
            raise ValueError("horizon set requires a version and at least one horizon")
        ids = [item.horizon_id for item in self.horizons]
        offsets = [item.offset_before_decision for item in self.horizons]
        if len(ids) != len(set(ids)) or len(offsets) != len(set(offsets)):
            raise ValueError("horizon ids and offsets must be unique")


@dataclass(frozen=True)
class LeadupSnapshotSlot:
    subject_id: str
    horizon_set_version: str
    horizon_id: str
    decision_at: datetime
    feature_cutoff_at: datetime
    offset_before_decision: timedelta


class SnapshotSubjectKind(str, Enum):
    MASTER_TRADE = "MASTER_TRADE"
    SCOUT_CANDIDATE = "SCOUT_CANDIDATE"


@dataclass(frozen=True)
class LeadupSnapshotRecord:
    """A materialized, versioned pre-decision state.

    ``actual_cutoff_at`` may precede the configured target when the provider has
    no event at the exact instant; it may never follow the target. A master trade
    can retain these snapshots even if no Scout candidate existed at the time.
    """

    snapshot_id: str
    subject_id: str
    subject_kind: SnapshotSubjectKind
    horizon_set_version: str
    horizon_id: str
    decision_at: datetime
    target_cutoff_at: datetime
    actual_cutoff_at: datetime
    market_state_id: str
    origin: ObservationOrigin
    captured_at: datetime
    source_lineage: Mapping[str, Any]
    candidate_id: str | None = None

    def __post_init__(self) -> None:
        for label, value in (
            ("decision_at", self.decision_at),
            ("target_cutoff_at", self.target_cutoff_at),
            ("actual_cutoff_at", self.actual_cutoff_at),
            ("captured_at", self.captured_at),
        ):
            require_aware(value, label)
        if self.target_cutoff_at > self.decision_at or self.actual_cutoff_at > self.target_cutoff_at:
            raise ValueError("lead-up snapshot cannot use market state after its target or decision")
        if not all((self.snapshot_id, self.subject_id, self.horizon_set_version, self.horizon_id, self.market_state_id)):
            raise ValueError("lead-up snapshot identifiers are required")
        if not self.source_lineage:
            raise ValueError("lead-up snapshot requires source lineage")
        if self.subject_kind == SnapshotSubjectKind.SCOUT_CANDIDATE and self.candidate_id != self.subject_id:
            raise ValueError("candidate snapshots must identify the candidate subject")

    @property
    def alignment_error(self) -> timedelta:
        return self.target_cutoff_at - self.actual_cutoff_at

    @property
    def prospective_lead_time_eligible(self) -> bool:
        return self.origin in {ObservationOrigin.LIVE, ObservationOrigin.BLIND_REPLAY}


def build_leadup_schedule(subject_id: str, decision_at: datetime, horizon_set: HorizonSet) -> tuple[LeadupSnapshotSlot, ...]:
    require_aware(decision_at, "decision_at")
    return tuple(
        LeadupSnapshotSlot(
            subject_id,
            horizon_set.version,
            horizon.horizon_id,
            decision_at,
            decision_at - horizon.offset_before_decision,
            horizon.offset_before_decision,
        )
        for horizon in sorted(horizon_set.horizons, key=lambda item: item.offset_before_decision, reverse=True)
    )
