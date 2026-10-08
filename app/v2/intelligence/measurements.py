from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Mapping

from app.v2.intelligence.observation import (
    BarObservation,
    ObservationPurpose,
    require_aware,
    require_exact_contract,
)
from app.v2.market_data import AvailabilityMode, parse_exact_contract_id


ZERO = Decimal("0")


class MeasurementFamily(str, Enum):
    CANDLE_GEOMETRY = "CANDLE_GEOMETRY"
    RELATIVE_STRUCTURE = "RELATIVE_STRUCTURE"
    SMT_CANDIDATE = "SMT_CANDIDATE"
    FIBONACCI_CANDIDATE = "FIBONACCI_CANDIDATE"
    LEVEL_CANDIDATE = "LEVEL_CANDIDATE"
    VOLATILITY_CONTEXT = "VOLATILITY_CONTEXT"
    TIME_CONTEXT = "TIME_CONTEXT"


@dataclass(frozen=True)
class ResearchMeasurement:
    measurement_id: str
    family: MeasurementFamily
    definition_version: str
    feature_cutoff_at: datetime
    max_input_event_at: datetime
    max_input_available_at: datetime | None
    computed_at: datetime
    definition_available_at: datetime
    value: Decimal | str | bool | None
    units: str | None
    parameters: Mapping[str, Any]
    source_event_ids: tuple[str, ...]
    source_lineage: Mapping[str, Any]
    availability_mode: AvailabilityMode = AvailabilityMode.POINT_IN_TIME
    missing_reason: str | None = None

    def __post_init__(self) -> None:
        for label, timestamp in (
            ("feature_cutoff_at", self.feature_cutoff_at),
            ("max_input_event_at", self.max_input_event_at),
            ("computed_at", self.computed_at),
            ("definition_available_at", self.definition_available_at),
        ):
            require_aware(timestamp, label)
        if self.max_input_available_at:
            require_aware(self.max_input_available_at, "max_input_available_at")
        if self.max_input_event_at > self.feature_cutoff_at:
            raise ValueError("measurement uses a market event after its cutoff")
        if self.definition_available_at > self.feature_cutoff_at:
            raise ValueError("measurement definition was unavailable at its cutoff")
        if self.computed_at < self.max_input_event_at:
            raise ValueError("measurement cannot be computed before its latest input event")
        if self.computed_at < self.definition_available_at:
            raise ValueError("measurement cannot predate its definition")
        if self.availability_mode == AvailabilityMode.POINT_IN_TIME:
            if self.max_input_available_at is None or self.max_input_available_at > self.feature_cutoff_at:
                raise ValueError("measurement uses information unavailable at its cutoff")
            if self.max_input_available_at < self.max_input_event_at:
                raise ValueError("measurement input cannot be available before its event time")
        if not self.definition_version or not self.source_lineage:
            raise ValueError("measurement requires a definition version and lineage")
        if self.value is None and not self.missing_reason:
            raise ValueError("missing measurements require an explicit reason")
        if self.value is not None and self.missing_reason is not None:
            raise ValueError("present measurements cannot carry a missing reason")

    def eligible_for(self, purpose: ObservationPurpose) -> bool:
        if self.availability_mode == AvailabilityMode.FINALIZED_HISTORICAL:
            return purpose in {ObservationPurpose.OUTCOME_RESEARCH, ObservationPurpose.HISTORICAL_REPLAY}
        return True


@dataclass(frozen=True)
class CandleGeometry:
    contract_id: str
    bar_event_id: str
    timeframe: str
    total_range: Decimal
    body_size: Decimal
    upper_wick: Decimal
    lower_wick: Decimal
    upper_wick_to_body: Decimal | None
    lower_wick_to_body: Decimal | None
    close_location: Decimal | None
    volatility_normalized_range: Decimal | None
    feature_cutoff_at: datetime
    definition_version: str
    source_lineage: Mapping[str, Any]


def measure_candle_geometry(
    bar: BarObservation,
    *,
    feature_cutoff_at: datetime,
    purpose: ObservationPurpose,
    definition_version: str,
    source_lineage: Mapping[str, Any],
    volatility_scale: Decimal | None = None,
) -> CandleGeometry:
    require_aware(feature_cutoff_at, "feature_cutoff_at")
    _, product, _ = parse_exact_contract_id(bar.contract_id)
    if product not in {"NQ", "ES"}:
        raise ValueError("conner_nq_v1 candle measurements support NQ or ES")
    if not bar.eligible_at(feature_cutoff_at, purpose):
        raise ValueError("candle geometry bar is unavailable at its cutoff")
    if not definition_version or not source_lineage:
        raise ValueError("candle geometry requires version and source lineage")
    total_range = bar.high - bar.low
    body = abs(bar.close - bar.open)
    upper = bar.high - max(bar.open, bar.close)
    lower = min(bar.open, bar.close) - bar.low
    close_location = None if total_range == ZERO else (bar.close - bar.low) / total_range
    normalized = None
    if volatility_scale is not None:
        if volatility_scale <= ZERO:
            raise ValueError("volatility_scale must be positive")
        normalized = total_range / volatility_scale
    return CandleGeometry(
        bar.contract_id,
        bar.event_id,
        bar.timeframe,
        total_range,
        body,
        upper,
        lower,
        None if body == ZERO else upper / body,
        None if body == ZERO else lower / body,
        close_location,
        normalized,
        feature_cutoff_at,
        definition_version,
        source_lineage,
    )


@dataclass(frozen=True)
class CandleRelationshipMeasurement:
    """Raw multi-candle geometry, not a rejection-block decision."""

    contract_id: str
    timeframe: str
    current_bar_event_id: str
    prior_bar_event_ids: tuple[str, ...]
    overlap_amount: Decimal
    overlap_fraction_of_current: Decimal | None
    close_displacement: Decimal
    body_displacement: Decimal
    same_direction_streak: int
    volatility_normalized_close_displacement: Decimal | None
    feature_cutoff_at: datetime
    definition_version: str
    source_lineage: Mapping[str, Any]


def measure_candle_relationship(
    current: BarObservation,
    prior_bars: tuple[BarObservation, ...],
    *,
    feature_cutoff_at: datetime,
    purpose: ObservationPurpose,
    definition_version: str,
    source_lineage: Mapping[str, Any],
    volatility_scale: Decimal | None = None,
) -> CandleRelationshipMeasurement:
    require_aware(feature_cutoff_at, "feature_cutoff_at")
    _, product, _ = parse_exact_contract_id(current.contract_id)
    if product not in {"NQ", "ES"}:
        raise ValueError("conner_nq_v1 candle measurements support NQ or ES")
    if not prior_bars:
        raise ValueError("candle relationship requires at least one prior bar")
    bars = (*prior_bars, current)
    if any(bar.contract_id != current.contract_id or bar.timeframe != current.timeframe for bar in bars):
        raise ValueError("candle relationship bars must share contract and timeframe")
    if any(not bar.eligible_at(feature_cutoff_at, purpose) for bar in bars):
        raise ValueError("candle relationship contains post-cutoff or unavailable bars")
    if any(left.window_end > right.window_start for left, right in zip(bars, bars[1:])):
        raise ValueError("candle relationship bars must be chronologically ordered without overlap")
    prior = prior_bars[-1]
    overlap = max(ZERO, min(prior.high, current.high) - max(prior.low, current.low))
    current_range = current.high - current.low
    close_displacement = current.close - prior.close
    body_midpoint = lambda bar: (bar.open + bar.close) / Decimal("2")
    direction = lambda bar: Decimal("1") if bar.close > bar.open else Decimal("-1") if bar.close < bar.open else ZERO
    current_direction = direction(current)
    streak = 1
    if current_direction != ZERO:
        for bar in reversed(prior_bars):
            if direction(bar) != current_direction:
                break
            streak += 1
    normalized = None
    if volatility_scale is not None:
        if volatility_scale <= ZERO:
            raise ValueError("volatility_scale must be positive")
        normalized = close_displacement / volatility_scale
    if not definition_version or not source_lineage:
        raise ValueError("candle relationship requires a definition version and source lineage")
    return CandleRelationshipMeasurement(
        current.contract_id,
        current.timeframe,
        current.event_id,
        tuple(bar.event_id for bar in prior_bars),
        overlap,
        None if current_range == ZERO else overlap / current_range,
        close_displacement,
        body_midpoint(current) - body_midpoint(prior),
        streak,
        normalized,
        feature_cutoff_at,
        definition_version,
        source_lineage,
    )


@dataclass(frozen=True)
class SwingCandidate:
    swing_id: str
    contract_id: str
    product_code: str
    kind: str
    price: Decimal
    event_at: datetime
    confirmed_at: datetime
    available_at: datetime
    selector_version: str
    selector_parameters: Mapping[str, Any] = field(default_factory=dict)
    source_event_ids: tuple[str, ...] = ()
    source_lineage: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for label, value in (("event_at", self.event_at), ("confirmed_at", self.confirmed_at), ("available_at", self.available_at)):
            require_aware(value, label)
        if self.available_at < self.confirmed_at or self.confirmed_at < self.event_at:
            raise ValueError("swing event/confirmation/availability times are inconsistent")
        if self.kind not in {"HIGH", "LOW"}:
            raise ValueError("swing kind must be HIGH or LOW")
        if self.price <= ZERO:
            raise ValueError("swing price must be positive")
        product = self.product_code.upper()
        if product not in {"NQ", "ES"}:
            raise ValueError("relative structure candidates support NQ or ES")
        require_exact_contract(self.contract_id, product)
        if not self.selector_version:
            raise ValueError("swing candidate requires a selector version")
        if not self.source_event_ids or not self.source_lineage:
            raise ValueError("swing candidate requires source events and lineage")


@dataclass(frozen=True)
class RelativeStructureCandidate:
    definition_version: str
    nq_prior_swing_id: str
    nq_current_swing_id: str
    es_prior_swing_id: str
    es_current_swing_id: str
    nq_change: Decimal
    es_change: Decimal
    normalized_difference: Decimal
    timing_separation_seconds: Decimal
    candidate_pattern: str
    feature_cutoff_at: datetime
    parameters: Mapping[str, Any]
    source_lineage: Mapping[str, Any]

    def __post_init__(self) -> None:
        require_aware(self.feature_cutoff_at, "feature_cutoff_at")
        if not self.definition_version or not self.source_lineage:
            raise ValueError("relative structure requires version and lineage")
        if self.timing_separation_seconds < ZERO:
            raise ValueError("swing timing separation cannot be negative")


def compare_swing_candidates(
    *,
    nq_prior: SwingCandidate,
    nq_current: SwingCandidate,
    es_prior: SwingCandidate,
    es_current: SwingCandidate,
    nq_scale: Decimal,
    es_scale: Decimal,
    definition_version: str,
    feature_cutoff_at: datetime,
    source_lineage: Mapping[str, Any],
) -> RelativeStructureCandidate:
    require_aware(feature_cutoff_at, "feature_cutoff_at")
    if not definition_version or not source_lineage:
        raise ValueError("relative structure requires a definition version and source lineage")
    if nq_scale <= ZERO or es_scale <= ZERO:
        raise ValueError("normalization scales must be positive")
    swings = (nq_prior, nq_current, es_prior, es_current)
    if any(item.product_code.upper() != "NQ" for item in (nq_prior, nq_current)) or any(
        item.product_code.upper() != "ES" for item in (es_prior, es_current)
    ):
        raise ValueError("relative structure requires NQ swings and corresponding ES swings")
    if any(item.available_at > feature_cutoff_at for item in swings):
        raise ValueError("relative structure uses a swing unavailable at cutoff")
    if {nq_prior.kind, nq_current.kind, es_prior.kind, es_current.kind} not in ({"HIGH"}, {"LOW"}):
        raise ValueError("relative structure compares corresponding high or low candidates")
    nq_change = nq_current.price - nq_prior.price
    es_change = es_current.price - es_prior.price
    nq_normalized = nq_change / nq_scale
    es_normalized = es_change / es_scale
    if nq_current.kind == "HIGH":
        pattern = _high_pattern(nq_change, es_change)
    else:
        pattern = _low_pattern(nq_change, es_change)
    separation = Decimal(str(abs((nq_current.event_at - es_current.event_at).total_seconds())))
    return RelativeStructureCandidate(
        definition_version,
        nq_prior.swing_id,
        nq_current.swing_id,
        es_prior.swing_id,
        es_current.swing_id,
        nq_change,
        es_change,
        nq_normalized - es_normalized,
        separation,
        pattern,
        feature_cutoff_at,
        {"nq_scale": str(nq_scale), "es_scale": str(es_scale)},
        source_lineage,
    )


@dataclass(frozen=True)
class DivergencePersistenceMeasurement:
    relative_structure_version: str
    candidate_pattern: str
    first_observed_at: datetime
    observed_at: datetime
    persistence_seconds: Decimal
    feature_cutoff_at: datetime
    definition_version: str
    source_lineage: Mapping[str, Any]

    def __post_init__(self) -> None:
        for label, timestamp in (
            ("first_observed_at", self.first_observed_at),
            ("observed_at", self.observed_at),
            ("feature_cutoff_at", self.feature_cutoff_at),
        ):
            require_aware(timestamp, label)
        if not self.first_observed_at <= self.observed_at <= self.feature_cutoff_at:
            raise ValueError("divergence persistence must be measured point-in-time")
        expected = Decimal(str((self.observed_at - self.first_observed_at).total_seconds()))
        if self.persistence_seconds != expected or self.persistence_seconds < ZERO:
            raise ValueError("divergence persistence must match its observed interval")
        if not self.definition_version or not self.source_lineage:
            raise ValueError("divergence persistence requires version and lineage")


@dataclass(frozen=True)
class FibonacciCandidate:
    candidate_id: str
    contract_id: str
    selector_version: str
    selector_parameters: Mapping[str, Any]
    anchor_a_id: str
    anchor_a_price: Decimal
    anchor_a_at: datetime
    anchor_b_id: str
    anchor_b_price: Decimal
    anchor_b_at: datetime
    direction: str
    ratio: Decimal
    level_price: Decimal
    current_price: Decimal
    signed_distance: Decimal
    feature_cutoff_at: datetime
    source_lineage: Mapping[str, Any]


def calculate_fibonacci_candidate(
    *,
    candidate_id: str,
    contract_id: str,
    selector_version: str,
    selector_parameters: Mapping[str, Any],
    anchor_a_id: str,
    anchor_a_price: Decimal,
    anchor_a_at: datetime,
    anchor_b_id: str,
    anchor_b_price: Decimal,
    anchor_b_at: datetime,
    ratio: Decimal,
    current_price: Decimal,
    feature_cutoff_at: datetime,
    source_lineage: Mapping[str, Any],
) -> FibonacciCandidate:
    for label, timestamp in (
        ("anchor_a_at", anchor_a_at),
        ("anchor_b_at", anchor_b_at),
        ("feature_cutoff_at", feature_cutoff_at),
    ):
        require_aware(timestamp, label)
    if not candidate_id or not selector_version or not source_lineage:
        raise ValueError("Fibonacci candidates require a versioned selector and source lineage")
    _, product, _ = parse_exact_contract_id(contract_id)
    if product not in {"NQ", "ES"}:
        raise ValueError("conner_nq_v1 Fibonacci candidates support NQ or ES")
    if any(price <= ZERO for price in (anchor_a_price, anchor_b_price, current_price)):
        raise ValueError("Fibonacci prices must be positive")
    if anchor_a_at > feature_cutoff_at or anchor_b_at > feature_cutoff_at:
        raise ValueError("Fibonacci anchor is after cutoff")
    direction = "A_TO_B_UP" if anchor_b_price >= anchor_a_price else "A_TO_B_DOWN"
    level = anchor_a_price + (anchor_b_price - anchor_a_price) * ratio
    return FibonacciCandidate(
        candidate_id,
        contract_id,
        selector_version,
        selector_parameters,
        anchor_a_id,
        anchor_a_price,
        anchor_a_at,
        anchor_b_id,
        anchor_b_price,
        anchor_b_at,
        direction,
        ratio,
        level,
        current_price,
        current_price - level,
        feature_cutoff_at,
        source_lineage,
    )


@dataclass(frozen=True)
class CandidateLevelMeasurement:
    level_id: str
    contract_id: str
    source_algorithm: str
    source_version: str
    price: Decimal
    current_price: Decimal
    signed_distance: Decimal
    first_observed_at: datetime
    available_at: datetime
    feature_cutoff_at: datetime
    parameters: Mapping[str, Any]
    source_lineage: Mapping[str, Any]

    def __post_init__(self) -> None:
        for label, timestamp in (
            ("first_observed_at", self.first_observed_at),
            ("available_at", self.available_at),
            ("feature_cutoff_at", self.feature_cutoff_at),
        ):
            require_aware(timestamp, label)
        if self.available_at > self.feature_cutoff_at or self.first_observed_at > self.feature_cutoff_at:
            raise ValueError("candidate level was unavailable at cutoff")
        if self.price <= ZERO or self.current_price <= ZERO:
            raise ValueError("candidate level prices must be positive")
        _, product, _ = parse_exact_contract_id(self.contract_id)
        if product not in {"NQ", "ES"}:
            raise ValueError("conner_nq_v1 candidate levels support NQ or ES")
        if not self.source_algorithm or not self.source_version or not self.source_lineage:
            raise ValueError("candidate levels require versioned parameters and source lineage")


def _high_pattern(nq_change: Decimal, es_change: Decimal) -> str:
    if nq_change > ZERO and es_change <= ZERO:
        return "NQ_HIGHER_HIGH_ES_NOT"
    if es_change > ZERO and nq_change <= ZERO:
        return "ES_HIGHER_HIGH_NQ_NOT"
    return "NO_DIRECTIONAL_DIVERGENCE"


def _low_pattern(nq_change: Decimal, es_change: Decimal) -> str:
    if nq_change < ZERO and es_change >= ZERO:
        return "NQ_LOWER_LOW_ES_NOT"
    if es_change < ZERO and nq_change >= ZERO:
        return "ES_LOWER_LOW_NQ_NOT"
    return "NO_DIRECTIONAL_DIVERGENCE"
