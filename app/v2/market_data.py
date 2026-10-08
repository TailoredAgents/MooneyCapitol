from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
import re
from typing import Any, AsyncIterator, Mapping, TypeAlias


class MarketDataCapability(str, Enum):
    CONTRACT_REFERENCE = "CONTRACT_REFERENCE"
    SCHEDULES = "SCHEDULES"
    TRADES = "TRADES"
    BARS = "BARS"
    BBO = "BBO"
    DEPTH = "DEPTH"
    MBO = "MBO"
    HISTORICAL_REPLAY = "HISTORICAL_REPLAY"
    LIVE_STREAMING = "LIVE_STREAMING"


class MarketEventKind(str, Enum):
    BAR = "BAR"
    TRADE = "TRADE"
    BBO = "BBO"


class AvailabilityMode(str, Enum):
    POINT_IN_TIME = "POINT_IN_TIME"
    FINALIZED_HISTORICAL = "FINALIZED_HISTORICAL"


class MarketDataError(RuntimeError):
    """Base error for the provider-neutral market-data boundary."""


class MarketDataEntitlementError(MarketDataError):
    """The configured provider access cannot honestly satisfy a request."""


class UnsupportedMarketDataCapability(MarketDataError):
    """The adapter does not implement the requested data capability."""


def _require_aware(value: datetime, label: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")


def datetime_to_epoch_ns(value: datetime) -> int:
    """Convert an aware datetime without passing through a float timestamp."""
    _require_aware(value, "timestamp")
    utc = value.astimezone(timezone.utc)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = utc - epoch
    return ((delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds) * 1_000


def epoch_ns_to_datetime(value: int) -> datetime:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("epoch nanoseconds must be a non-negative integer")
    seconds, nanoseconds = divmod(value, 1_000_000_000)
    return datetime.fromtimestamp(seconds, tz=timezone.utc).replace(microsecond=nanoseconds // 1_000)


_EXACT_CONTRACT_ID_RE = re.compile(
    r"^(?P<exchange>[A-Z0-9]+):(?P<product>[A-Z0-9]+):(?P<expiration>[0-9]{4}-[0-9]{2}-[0-9]{2})$",
    re.IGNORECASE,
)


def parse_exact_contract_id(contract_id: str) -> tuple[str, str, date]:
    if not isinstance(contract_id, str):
        raise ValueError("contract identity must be a string")
    match = _EXACT_CONTRACT_ID_RE.fullmatch(contract_id.strip())
    if match is None:
        raise ValueError("contract identity must use EXCHANGE:PRODUCT:YYYY-MM-DD")
    try:
        expiration = date.fromisoformat(match.group("expiration"))
    except ValueError as exc:
        raise ValueError("contract identity contains an invalid expiration date") from exc
    return match.group("exchange").upper(), match.group("product").upper(), expiration


@dataclass(frozen=True)
class ProviderContractBinding:
    """Maps one canonical exact contract to one provider's exact symbol."""

    contract_id: str
    product_code: str
    provider_symbol: str
    trading_venue: str | None = None

    def __post_init__(self) -> None:
        if not self.contract_id or not self.product_code or not self.provider_symbol:
            raise ValueError("contract binding fields cannot be blank")
        _, identity_product, _ = parse_exact_contract_id(self.contract_id)
        if identity_product != self.product_code.upper():
            raise ValueError("canonical contract identity product does not match its binding")
        if self.provider_symbol.upper() == self.product_code.upper():
            raise ValueError("provider symbol must identify an exact expiry")


@dataclass(frozen=True)
class ContractReference:
    contract_id: str
    product_code: str
    provider_symbol: str
    trading_venue: str
    first_trade_date: date | None
    last_trade_date: date | None
    settlement_date: date | None
    trade_tick_size: Decimal
    settlement_tick_size: Decimal | None = None
    spread_tick_size: Decimal | None = None
    active_as_of: date | None = None
    source: str = ""

    def __post_init__(self) -> None:
        if self.trade_tick_size <= 0:
            raise ValueError("trade_tick_size must be positive")
        if self.first_trade_date and self.last_trade_date and self.first_trade_date > self.last_trade_date:
            raise ValueError("first_trade_date cannot follow last_trade_date")


@dataclass(frozen=True)
class ScheduleEvent:
    event: str
    timestamp: datetime

    def __post_init__(self) -> None:
        if not self.event:
            raise ValueError("schedule event cannot be blank")
        _require_aware(self.timestamp, "schedule timestamp")


@dataclass(frozen=True)
class FuturesSchedule:
    contract_id: str
    product_code: str
    trade_date: date
    trading_venue: str
    events: tuple[ScheduleEvent, ...]
    source: str
    retrieved_at: datetime

    def __post_init__(self) -> None:
        _require_aware(self.retrieved_at, "schedule retrieved_at")
        if not self.events:
            raise ValueError("provider schedule must contain at least one event")
        if tuple(sorted(self.events, key=lambda item: (item.timestamp, item.event))) != self.events:
            raise ValueError("schedule events must be deterministically ordered")


@dataclass(frozen=True)
class BarPayload:
    resolution: str
    starts_at: datetime
    ends_at: datetime
    window_start_ns: int
    window_end_ns: int
    trade_date: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    transactions: int

    def __post_init__(self) -> None:
        _require_aware(self.starts_at, "bar starts_at")
        _require_aware(self.ends_at, "bar ends_at")
        if self.ends_at <= self.starts_at or self.window_end_ns <= self.window_start_ns:
            raise ValueError("bar end must follow bar start")
        if self.low > self.high or not self.low <= self.open <= self.high or not self.low <= self.close <= self.high:
            raise ValueError("invalid OHLC bounds")
        if self.volume < 0 or self.transactions < 0:
            raise ValueError("bar counts cannot be negative")


@dataclass(frozen=True)
class TradePayload:
    occurred_at: datetime
    occurred_at_ns: int
    trade_date: date
    price: Decimal
    size: int
    report_sequence: int | None = None
    channel: int | None = None

    def __post_init__(self) -> None:
        _require_aware(self.occurred_at, "trade occurred_at")
        if self.occurred_at_ns < 0:
            raise ValueError("trade nanosecond timestamp cannot be negative")
        if self.price <= 0 or self.size <= 0:
            raise ValueError("trade price and size must be positive")


@dataclass(frozen=True)
class BboPayload:
    occurred_at: datetime
    occurred_at_ns: int
    trade_date: date
    bid_price: Decimal | None
    bid_size: int | None
    ask_price: Decimal | None
    ask_size: int | None
    bid_timestamp_ns: int | None = None
    ask_timestamp_ns: int | None = None
    report_sequence: int | None = None
    channel: int | None = None

    def __post_init__(self) -> None:
        _require_aware(self.occurred_at, "BBO occurred_at")
        if self.occurred_at_ns < 0:
            raise ValueError("BBO nanosecond timestamp cannot be negative")
        if self.bid_price is None and self.ask_price is None:
            raise ValueError("BBO must contain at least one side")
        for label, price in (("bid_price", self.bid_price), ("ask_price", self.ask_price)):
            if price is not None and price <= 0:
                raise ValueError(f"{label} must be positive")
        for label, size in (("bid_size", self.bid_size), ("ask_size", self.ask_size)):
            if size is not None and size < 0:
                raise ValueError(f"{label} cannot be negative")


MarketDataPayload: TypeAlias = BarPayload | TradePayload | BboPayload


@dataclass(frozen=True)
class MarketDataLineage:
    provider: str
    provider_symbol: str
    endpoint: str
    schema_version: str
    request_id: str | None
    retrieved_at: datetime
    revision: int = 0

    def __post_init__(self) -> None:
        _require_aware(self.retrieved_at, "lineage retrieved_at")
        if self.revision < 0:
            raise ValueError("lineage revision cannot be negative")


@dataclass(frozen=True)
class DataQuality:
    is_gap: bool = False
    is_late: bool = False
    is_corrected: bool = False
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class MarketDataEvent:
    contract_id: str
    kind: MarketEventKind | str
    source: str
    source_timestamp: datetime
    received_timestamp: datetime
    sequence: int | None
    payload: MarketDataPayload | Mapping[str, Any]
    quality: DataQuality = DataQuality()
    available_timestamp: datetime | None = None
    availability_mode: AvailabilityMode = AvailabilityMode.POINT_IN_TIME
    revision: int | None = None
    is_final: bool = True
    source_timestamp_ns: int | None = None
    provider_event_id: str | None = None
    lineage: MarketDataLineage | None = None

    def __post_init__(self) -> None:
        for label, value in (
            ("source_timestamp", self.source_timestamp),
            ("received_timestamp", self.received_timestamp),
        ):
            _require_aware(value, label)
        if self.available_timestamp:
            _require_aware(self.available_timestamp, "available_timestamp")
            if self.available_timestamp < self.source_timestamp:
                raise ValueError("market fact cannot be available before its event timestamp")
        if self.received_timestamp < self.source_timestamp:
            raise ValueError("market fact cannot be received before its event timestamp")
        if self.revision is not None and self.revision < 0:
            raise ValueError("revision cannot be negative")
        if self.source_timestamp_ns is not None and self.source_timestamp_ns < 0:
            raise ValueError("source_timestamp_ns cannot be negative")
        if isinstance(self.kind, str) and not isinstance(self.kind, MarketEventKind):
            try:
                object.__setattr__(self, "kind", MarketEventKind(self.kind.upper()))
            except ValueError as exc:
                raise ValueError(f"unsupported market event kind: {self.kind}") from exc
        expected_payload = {
            MarketEventKind.BAR: BarPayload,
            MarketEventKind.TRADE: TradePayload,
            MarketEventKind.BBO: BboPayload,
        }
        if not isinstance(self.payload, Mapping) and not isinstance(self.payload, expected_payload[self.kind]):
            raise ValueError(f"{self.kind.value} event has the wrong typed payload")
        if isinstance(self.payload, BarPayload):
            if self.source_timestamp != self.payload.ends_at or self.source_timestamp_ns != self.payload.window_end_ns:
                raise ValueError("bar event timestamp must be its completed window end")
        elif isinstance(self.payload, TradePayload):
            if self.source_timestamp != self.payload.occurred_at or self.source_timestamp_ns != self.payload.occurred_at_ns:
                raise ValueError("trade event timestamps must match the typed payload")
        elif isinstance(self.payload, BboPayload):
            if self.source_timestamp != self.payload.occurred_at or self.source_timestamp_ns != self.payload.occurred_at_ns:
                raise ValueError("BBO event timestamps must match the typed payload")
        if self.lineage and self.revision is not None and self.lineage.revision != self.revision:
            raise ValueError("event and lineage revisions must agree")

    @property
    def event_timestamp(self) -> datetime:
        return self.source_timestamp

    @property
    def available_at(self) -> datetime:
        return self.available_timestamp or self.received_timestamp

    def eligible_at(self, cutoff: datetime, *, allow_finalized_historical: bool = False) -> bool:
        _require_aware(cutoff, "cutoff")
        if self.source_timestamp > cutoff:
            return False
        if self.availability_mode == AvailabilityMode.FINALIZED_HISTORICAL:
            return allow_finalized_historical
        return self.available_at <= cutoff


_RESOLUTION_RE = re.compile(r"^(?P<count>[1-9][0-9]*)(?P<unit>sec|min|hour)$", re.IGNORECASE)


def fixed_resolution_ns(resolution: str) -> int:
    """Return a fixed intraday resolution or fail closed for variable periods."""
    match = _RESOLUTION_RE.fullmatch(resolution.strip())
    if match is None:
        raise ValueError("historical point-in-time bars require a fixed sec/min/hour resolution")
    count = int(match.group("count"))
    seconds_per_unit = {"sec": 1, "min": 60, "hour": 3_600}[match.group("unit").lower()]
    return count * seconds_per_unit * 1_000_000_000


@dataclass(frozen=True)
class HistoricalMarketDataRequest:
    contract_ids: tuple[str, ...]
    start: datetime
    end: datetime
    kinds: frozenset[MarketEventKind]
    bar_resolutions: tuple[str, ...] = ("1min",)

    def __post_init__(self) -> None:
        _require_aware(self.start, "historical request start")
        _require_aware(self.end, "historical request end")
        if self.end <= self.start:
            raise ValueError("historical request end must follow start")
        if not self.contract_ids or len(set(self.contract_ids)) != len(self.contract_ids):
            raise ValueError("historical request requires unique exact contracts")
        for contract_id in self.contract_ids:
            parse_exact_contract_id(contract_id)
        if not self.kinds:
            raise ValueError("historical request requires at least one event kind")
        if any(not isinstance(kind, MarketEventKind) for kind in self.kinds):
            raise ValueError("historical request kinds must use MarketEventKind")
        normalized = tuple(item.strip().lower() for item in self.bar_resolutions)
        if MarketEventKind.BAR in self.kinds:
            if not normalized:
                raise ValueError("bar requests require at least one resolution")
            for resolution in normalized:
                fixed_resolution_ns(resolution)
        object.__setattr__(self, "bar_resolutions", normalized)


@dataclass(frozen=True)
class MarketDataEntitlement:
    """Explicit configured access; defaults deliberately grant nothing."""

    enabled: bool = False
    capabilities: frozenset[MarketDataCapability] = field(default_factory=frozenset)
    history_starts_at: datetime | None = None
    delivery_delay: timedelta = timedelta(0)

    def __post_init__(self) -> None:
        if self.history_starts_at:
            _require_aware(self.history_starts_at, "history_starts_at")
        if self.delivery_delay < timedelta(0):
            raise ValueError("delivery_delay cannot be negative")

    def require(self, required: set[MarketDataCapability]) -> None:
        if not self.enabled:
            raise MarketDataEntitlementError("market-data entitlement is disabled")
        missing = required - set(self.capabilities)
        if missing:
            names = ", ".join(sorted(item.value for item in missing))
            raise MarketDataEntitlementError(f"market-data entitlement lacks: {names}")

    def require_historical_window(self, start: datetime, end: datetime, *, now: datetime) -> None:
        _require_aware(now, "entitlement evaluation time")
        if self.history_starts_at and start < self.history_starts_at:
            raise MarketDataEntitlementError("request predates entitled market-data history")
        if end > now - self.delivery_delay:
            raise MarketDataEntitlementError("request extends beyond currently entitled finalized data")


def market_event_sort_key(event: MarketDataEvent) -> tuple[int, str, str, int, str]:
    timestamp_ns = event.source_timestamp_ns
    if timestamp_ns is None:
        timestamp_ns = datetime_to_epoch_ns(event.source_timestamp)
    sequence = event.sequence if event.sequence is not None else -1
    return (timestamp_ns, event.contract_id, event.kind.value, sequence, event.provider_event_id or "")


class FuturesMarketDataProvider(ABC):
    @abstractmethod
    def capabilities(self) -> frozenset[MarketDataCapability]: ...

    @abstractmethod
    async def contract_reference(self, contract_id: str, *, as_of: date | None = None) -> ContractReference: ...

    @abstractmethod
    async def schedule(self, contract_id: str, trade_date: date) -> FuturesSchedule: ...

    @abstractmethod
    def replay(self, request: HistoricalMarketDataRequest) -> AsyncIterator[MarketDataEvent]: ...

    @abstractmethod
    def stream(
        self, contract_ids: tuple[str, ...], kinds: frozenset[MarketEventKind]
    ) -> AsyncIterator[MarketDataEvent]: ...


def require_capabilities(provider: FuturesMarketDataProvider, required: set[MarketDataCapability]) -> None:
    missing = required - set(provider.capabilities())
    if missing:
        names = ", ".join(sorted(item.value for item in missing))
        raise UnsupportedMarketDataCapability(f"market-data provider lacks required capabilities: {names}")
