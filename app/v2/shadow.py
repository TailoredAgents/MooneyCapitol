from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping

from app.v2.domain.models import OpportunitySnapshot


@dataclass(frozen=True)
class ShadowObservation:
    observation_id: str
    opportunity: OpportunitySnapshot
    feature_cutoff_at: datetime
    behavior_output: Mapping[str, Any] | None = None
    outcome_distribution: Mapping[str, Any] | None = None
    conviction_output: Mapping[str, Any] | None = None
    model_versions: Mapping[str, str] = field(default_factory=dict)
    conner_action: str | None = None

    def __post_init__(self) -> None:
        if self.feature_cutoff_at != self.opportunity.feature_cutoff_at:
            raise ValueError("shadow observation must preserve the opportunity feature cutoff")


@dataclass(frozen=True)
class ShadowResult:
    observation_id: str
    eventual_outcome: Mapping[str, Any]
    outcome_observed_at: datetime


@dataclass(frozen=True)
class ReplayRequest:
    contract_id: str
    start: datetime
    end: datetime
    required_event_kinds: frozenset[str]

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError("replay end must follow start")


class FuturesReplayEngine:
    """Event dispatcher only; fill models and strategy rules are intentionally absent."""

    async def run(self, request: ReplayRequest, events, observer) -> int:
        count = 0
        async for event in events:
            if event.contract_id != request.contract_id:
                raise ValueError("replay event contract mismatch")
            if not request.start <= event.source_timestamp <= request.end:
                raise ValueError("replay event outside requested interval")
            await observer(event)
            count += 1
        return count
