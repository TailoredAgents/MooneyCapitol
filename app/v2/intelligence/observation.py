from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any, Mapping

from enum import Enum

from app.v2.market_data import AvailabilityMode, DataQuality, parse_exact_contract_id


CONNER_NQ_SCHEMA_NAME = "conner_nq_v1"
CONNER_NQ_SCHEMA_VERSION = "1.0.0"


class ObservationPurpose(str, Enum):
    LIVE_SHADOW = "LIVE_SHADOW"
    BEHAVIOR_TRAINING = "BEHAVIOR_TRAINING"
    OUTCOME_RESEARCH = "OUTCOME_RESEARCH"
    HISTORICAL_REPLAY = "HISTORICAL_REPLAY"


@dataclass(frozen=True)
class ResearchContractPair:
    """Versioned NQ/ES contextual pairing; this is not an execution mapping."""

    pair_id: str
    trade_date: date
    nq_contract_id: str
    es_contract_id: str
    selection_version: str
    selected_at: datetime
    source_lineage: Mapping[str, Any]

    def __post_init__(self) -> None:
        require_exact_contract(self.nq_contract_id, "NQ")
        require_exact_contract(self.es_contract_id, "ES")
        require_aware(self.selected_at, "selected_at")
        if self.nq_contract_id == self.es_contract_id:
            raise ValueError("NQ and ES contextual contracts must differ")
        if not self.pair_id or not self.selection_version or not self.source_lineage:
            raise ValueError("contract pair requires versioned selection lineage")


def require_aware(value: datetime, label: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")


def require_exact_contract(contract_id: str, product_code: str) -> None:
    try:
        _, identity_product, _ = parse_exact_contract_id(contract_id)
    except ValueError as exc:
        raise ValueError(f"{product_code} requires a matching expiration-aware internal contract identity") from exc
    if identity_product != product_code.upper():
        raise ValueError(f"{product_code} requires a matching expiration-aware internal contract identity")


@dataclass(frozen=True)
class BarObservation:
    event_id: str
    contract_id: str
    timeframe: str
    window_start: datetime
    window_end: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    trade_count: int | None
    source: str
    source_timestamp: datetime
    received_timestamp: datetime
    session_end_date: date
    available_at: datetime | None = None
    availability_mode: AvailabilityMode = AvailabilityMode.POINT_IN_TIME
    revision: int | None = None
    is_final: bool = True
    window_start_ns: int | None = None
    window_end_ns: int | None = None

    def __post_init__(self) -> None:
        for label, value in (
            ("window_start", self.window_start),
            ("window_end", self.window_end),
            ("source_timestamp", self.source_timestamp),
            ("received_timestamp", self.received_timestamp),
        ):
            require_aware(value, label)
        if self.window_end <= self.window_start:
            raise ValueError("bar window_end must follow window_start")
        if self.source_timestamp < self.window_end:
            raise ValueError("bar source timestamp cannot precede its completed window")
        if self.received_timestamp < self.source_timestamp:
            raise ValueError("bar cannot be received before its source timestamp")
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close) or self.high < self.low:
            raise ValueError("invalid OHLC geometry")
        if self.volume < 0 or (self.trade_count is not None and self.trade_count < 0):
            raise ValueError("bar volume/trade count cannot be negative")
        if self.available_at:
            require_aware(self.available_at, "available_at")
            if self.available_at < self.source_timestamp:
                raise ValueError("bar cannot be available before its source timestamp")
        if self.revision is not None and self.revision < 0:
            raise ValueError("bar revision cannot be negative")
        if self.window_start_ns is not None and self.window_start_ns < 0:
            raise ValueError("bar window_start_ns cannot be negative")
        if self.window_end_ns is not None and self.window_end_ns < 0:
            raise ValueError("bar window_end_ns cannot be negative")
        if self.window_start_ns is not None and self.window_end_ns is not None and self.window_end_ns <= self.window_start_ns:
            raise ValueError("bar nanosecond window must be ordered")

    def eligible_at(self, cutoff: datetime, purpose: ObservationPurpose) -> bool:
        if self.window_end > cutoff:
            return False
        if self.availability_mode == AvailabilityMode.FINALIZED_HISTORICAL:
            return purpose in {ObservationPurpose.OUTCOME_RESEARCH, ObservationPurpose.HISTORICAL_REPLAY}
        return (self.available_at or self.received_timestamp) <= cutoff


@dataclass(frozen=True)
class TradeObservation:
    event_id: str
    contract_id: str
    price: Decimal
    quantity: int
    source_timestamp: datetime
    received_timestamp: datetime
    sequence: int | None
    source: str
    available_at: datetime | None = None
    availability_mode: AvailabilityMode = AvailabilityMode.POINT_IN_TIME
    revision: int | None = None
    source_timestamp_ns: int | None = None

    def __post_init__(self) -> None:
        require_aware(self.source_timestamp, "source_timestamp")
        require_aware(self.received_timestamp, "received_timestamp")
        if self.price <= 0 or self.quantity <= 0:
            raise ValueError("trade price and quantity must be positive")
        if self.received_timestamp < self.source_timestamp:
            raise ValueError("trade cannot be received before it occurs")
        if self.available_at:
            require_aware(self.available_at, "available_at")
            if self.available_at < self.source_timestamp:
                raise ValueError("trade cannot be available before it occurs")
        if self.revision is not None and self.revision < 0:
            raise ValueError("trade revision cannot be negative")
        if self.source_timestamp_ns is not None and self.source_timestamp_ns < 0:
            raise ValueError("trade source_timestamp_ns cannot be negative")


@dataclass(frozen=True)
class BboObservation:
    event_id: str
    contract_id: str
    bid_price: Decimal | None
    bid_quantity: int | None
    ask_price: Decimal | None
    ask_quantity: int | None
    source_timestamp: datetime
    received_timestamp: datetime
    sequence: int | None
    source: str
    available_at: datetime | None = None
    availability_mode: AvailabilityMode = AvailabilityMode.POINT_IN_TIME
    revision: int | None = None
    source_timestamp_ns: int | None = None

    def __post_init__(self) -> None:
        require_aware(self.source_timestamp, "source_timestamp")
        require_aware(self.received_timestamp, "received_timestamp")
        if self.bid_price is not None and self.ask_price is not None and self.bid_price > self.ask_price:
            raise ValueError("BBO bid cannot exceed ask")
        if any(value is not None and value < 0 for value in (self.bid_quantity, self.ask_quantity)):
            raise ValueError("BBO quantities cannot be negative")
        if self.received_timestamp < self.source_timestamp:
            raise ValueError("BBO cannot be received before it occurs")
        if self.available_at:
            require_aware(self.available_at, "available_at")
            if self.available_at < self.source_timestamp:
                raise ValueError("BBO cannot be available before it occurs")
        if self.revision is not None and self.revision < 0:
            raise ValueError("BBO revision cannot be negative")
        if self.source_timestamp_ns is not None and self.source_timestamp_ns < 0:
            raise ValueError("BBO source_timestamp_ns cannot be negative")


@dataclass(frozen=True)
class InstrumentMarketState:
    product_code: str
    contract_id: str
    feature_cutoff_at: datetime
    as_of: datetime
    bars: tuple[BarObservation, ...] = ()
    recent_trades: tuple[TradeObservation, ...] = ()
    latest_bbo: BboObservation | None = None
    source_event_ids: tuple[str, ...] = ()
    purpose: ObservationPurpose = ObservationPurpose.LIVE_SHADOW

    def __post_init__(self) -> None:
        product = self.product_code.upper()
        if product not in {"NQ", "ES"}:
            raise ValueError("conner_nq_v1 synchronized state supports exact NQ and ES contracts")
        require_exact_contract(self.contract_id, product)
        require_aware(self.feature_cutoff_at, "feature_cutoff_at")
        require_aware(self.as_of, "as_of")
        if self.as_of > self.feature_cutoff_at:
            raise ValueError("instrument state cannot be newer than its decision cutoff")
        for bar in self.bars:
            if bar.contract_id != self.contract_id or not bar.eligible_at(self.feature_cutoff_at, self.purpose):
                raise ValueError("bar is ineligible for this contract, cutoff, or observation purpose")
            if bar.window_end > self.as_of:
                raise ValueError("bar extends beyond instrument as_of")
        for trade in self.recent_trades:
            if trade.contract_id != self.contract_id or not _tick_eligible(
                trade.source_timestamp,
                trade.available_at or trade.received_timestamp,
                trade.availability_mode,
                self.feature_cutoff_at,
                self.purpose,
            ):
                raise ValueError("trade is ineligible for this contract, cutoff, or observation purpose")
            if trade.source_timestamp > self.as_of:
                raise ValueError("trade is newer than instrument as_of")
        if self.latest_bbo and (
            self.latest_bbo.contract_id != self.contract_id
            or not _tick_eligible(
                self.latest_bbo.source_timestamp,
                self.latest_bbo.available_at or self.latest_bbo.received_timestamp,
                self.latest_bbo.availability_mode,
                self.feature_cutoff_at,
                self.purpose,
            )
        ):
            raise ValueError("BBO is ineligible for this contract, cutoff, or observation purpose")
        if self.latest_bbo and self.latest_bbo.source_timestamp > self.as_of:
            raise ValueError("BBO is newer than instrument as_of")
        modes = {bar.availability_mode for bar in self.bars}
        modes.update(trade.availability_mode for trade in self.recent_trades)
        if self.latest_bbo:
            modes.add(self.latest_bbo.availability_mode)
        if len(modes) > 1:
            raise ValueError("an instrument state cannot mix point-in-time and finalized historical facts")

    @property
    def availability_mode(self) -> AvailabilityMode:
        modes = [bar.availability_mode for bar in self.bars]
        modes.extend(trade.availability_mode for trade in self.recent_trades)
        if self.latest_bbo:
            modes.append(self.latest_bbo.availability_mode)
        return modes[0] if modes else AvailabilityMode.POINT_IN_TIME

    @property
    def max_available_at(self) -> datetime:
        timestamps = [bar.available_at or bar.received_timestamp for bar in self.bars]
        timestamps.extend(trade.available_at or trade.received_timestamp for trade in self.recent_trades)
        if self.latest_bbo:
            timestamps.append(self.latest_bbo.available_at or self.latest_bbo.received_timestamp)
        return max(timestamps, default=self.as_of)


@dataclass(frozen=True)
class SynchronizedNqEsState:
    state_id: str
    trade_date: date
    feature_cutoff_at: datetime
    nq: InstrumentMarketState
    es: InstrumentMarketState
    synchronization_tolerance: timedelta
    data_quality: DataQuality
    source_lineage: Mapping[str, Any]
    schema_name: str = CONNER_NQ_SCHEMA_NAME
    schema_version: str = CONNER_NQ_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_aware(self.feature_cutoff_at, "feature_cutoff_at")
        if self.schema_name != CONNER_NQ_SCHEMA_NAME:
            raise ValueError("unexpected observation schema")
        if self.nq.product_code.upper() != "NQ" or self.es.product_code.upper() != "ES":
            raise ValueError("synchronized state requires NQ as traded leg and ES as context leg")
        if self.nq.contract_id == self.es.contract_id:
            raise ValueError("NQ and ES contextual contracts must differ")
        if self.nq.feature_cutoff_at != self.feature_cutoff_at or self.es.feature_cutoff_at != self.feature_cutoff_at:
            raise ValueError("both legs must preserve the same decision cutoff")
        if self.nq.purpose != self.es.purpose:
            raise ValueError("both synchronized legs must share one observation purpose")
        if self.nq.availability_mode != self.es.availability_mode:
            raise ValueError("both synchronized legs must share one availability mode")
        if self.synchronization_tolerance < timedelta(0):
            raise ValueError("synchronization tolerance cannot be negative")
        if abs(self.nq.as_of - self.es.as_of) > self.synchronization_tolerance:
            raise ValueError("NQ and ES state exceed synchronization tolerance")
        if not self.source_lineage:
            raise ValueError("synchronized state requires source lineage")


@dataclass(frozen=True)
class ConnerNqObservation:
    observation_id: str
    state: SynchronizedNqEsState
    observed_at: datetime
    candidate_id: str | None = None
    measurement_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_aware(self.observed_at, "observed_at")
        if self.observed_at < self.state.feature_cutoff_at:
            raise ValueError("observation cannot be recorded before its cutoff")


def _tick_eligible(
    event_at: datetime,
    available_at: datetime,
    mode: AvailabilityMode,
    cutoff: datetime,
    purpose: ObservationPurpose,
) -> bool:
    if event_at > cutoff:
        return False
    if mode == AvailabilityMode.FINALIZED_HISTORICAL:
        return purpose in {ObservationPurpose.OUTCOME_RESEARCH, ObservationPurpose.HISTORICAL_REPLAY}
    return available_at <= cutoff
