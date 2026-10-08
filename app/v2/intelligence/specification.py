from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time, timedelta
from enum import Enum
from typing import Any, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.v2.intelligence.observation import CONNER_NQ_SCHEMA_NAME
from app.v2.market_data import MarketDataCapability


class ResearchStatus(str, Enum):
    """Whether a configuration is a research candidate or supported by evidence."""

    CANDIDATE = "CANDIDATE"
    EMPIRICALLY_SUPPORTED = "EMPIRICALLY_SUPPORTED"
    RETIRED = "RETIRED"


@dataclass(frozen=True)
class StrategyWindowCandidate:
    """Versioned strategy-time hypothesis, separate from CME trade-date accounting."""

    anchor_id: str
    version: str
    timezone_name: str
    anchor_local_time: time
    duration: timedelta
    status: ResearchStatus = ResearchStatus.CANDIDATE
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.anchor_id or not self.version:
            raise ValueError("strategy window requires an id and version")
        try:
            ZoneInfo(self.timezone_name)
        except ZoneInfoNotFoundError as exc:
            raise ValueError("strategy window requires a valid IANA timezone") from exc
        if self.anchor_local_time.tzinfo is not None:
            raise ValueError("anchor_local_time is wall-clock time; timezone_name supplies the zone")
        if self.duration <= timedelta(0):
            raise ValueError("strategy window duration must be positive")


CONNER_NQ_REQUIRED_CAPABILITIES = frozenset(
    {
        MarketDataCapability.CONTRACT_REFERENCE,
        MarketDataCapability.SCHEDULES,
        MarketDataCapability.BARS,
        MarketDataCapability.TRADES,
        MarketDataCapability.BBO,
        MarketDataCapability.HISTORICAL_REPLAY,
    }
)


@dataclass(frozen=True)
class ObservationSpec:
    """Provider-neutral configuration for reproducible Scout observations.

    It deliberately names measurement families and versions, not trading rules.
    No session anchor is supplied by default because Conner's exact interpretation
    of the New York window has not been established.
    """

    spec_id: str
    version: str
    feature_schema_version: str
    horizon_set_version: str
    synchronization_policy_version: str
    timeframes: tuple[str, ...]
    measurement_versions: Mapping[str, tuple[str, ...]]
    required_capabilities: frozenset[MarketDataCapability] = CONNER_NQ_REQUIRED_CAPABILITIES
    strategy_window_candidates: tuple[StrategyWindowCandidate, ...] = ()
    parameters: Mapping[str, Any] = field(default_factory=dict)
    strategy_namespace: str = CONNER_NQ_SCHEMA_NAME

    def __post_init__(self) -> None:
        if self.strategy_namespace != CONNER_NQ_SCHEMA_NAME:
            raise ValueError("observation spec must use the conner_nq_v1 namespace")
        if not all(
            (
                self.spec_id,
                self.version,
                self.feature_schema_version,
                self.horizon_set_version,
                self.synchronization_policy_version,
            )
        ):
            raise ValueError("observation spec identifiers and versions are required")
        if not self.timeframes or any(not item for item in self.timeframes):
            raise ValueError("observation spec requires at least one named timeframe")
        if len(self.timeframes) != len(set(self.timeframes)):
            raise ValueError("observation timeframes must be unique")
        if not CONNER_NQ_REQUIRED_CAPABILITIES.issubset(self.required_capabilities):
            raise ValueError("conner_nq_v1 requires reference, schedules, bars, trades, BBO, and replay")
        if MarketDataCapability.DEPTH in self.required_capabilities or MarketDataCapability.MBO in self.required_capabilities:
            raise ValueError("depth and MBO are not required dependencies for conner_nq_v1")
        if not self.measurement_versions:
            raise ValueError("observation spec requires versioned candidate measurement definitions")
        for family, versions in self.measurement_versions.items():
            if not family or not versions or any(not version for version in versions):
                raise ValueError("measurement families require one or more explicit versions")
        window_keys = [(item.anchor_id, item.version) for item in self.strategy_window_candidates]
        if len(window_keys) != len(set(window_keys)):
            raise ValueError("strategy window candidate versions must be unique")
