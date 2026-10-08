from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Mapping


class BrokerPlant(str, Enum):
    SYSTEM = "SYSTEM"
    ORDER = "ORDER"
    PNL = "PNL"
    TICKER = "TICKER"


class ObservationSource(str, Enum):
    LIVE = "LIVE"
    SNAPSHOT = "SNAPSHOT"
    REPLAY = "REPLAY"
    HISTORY = "HISTORY"
    CACHE = "CACHE"
    UNKNOWN = "UNKNOWN"


class ObservedOrderState(str, Enum):
    PENDING = "PENDING"
    WORKING = "WORKING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    COMPLETED_UNKNOWN = "COMPLETED_UNKNOWN"
    UNKNOWN = "UNKNOWN"


class CommandOutcome(str, Enum):
    ACKNOWLEDGED = "ACKNOWLEDGED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    OUTCOME_UNKNOWN = "OUTCOME_UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ExecutionAdjustmentKind(str, Enum):
    FILL = "FILL"
    BUST = "BUST"
    CORRECTION = "CORRECTION"
    UNKNOWN = "UNKNOWN"


class FactFreshness(str, Enum):
    FRESH = "FRESH"
    STALE = "STALE"
    UNKNOWN = "UNKNOWN"


def decimal_from_wire(value: Any, *, field_name: str) -> Decimal | None:
    """Parse an explicitly present broker monetary value without inventing zero."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} cannot be boolean")
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"invalid decimal for {field_name}") from exc


@dataclass(frozen=True)
class BrokerAccountIdentity:
    fcm_id: str
    ib_id: str
    account_id: str

    def __post_init__(self) -> None:
        if not self.fcm_id or not self.ib_id or not self.account_id:
            raise ValueError("FCM, IB, and account identifiers are required")

    @property
    def opaque_key(self) -> tuple[str, str, str]:
        return self.fcm_id, self.ib_id, self.account_id


@dataclass(frozen=True)
class ObservedBrokerAccount:
    identity: BrokerAccountIdentity
    connection_id: str
    observed_at: datetime
    display_name: str | None = None
    currency: str | None = None
    account_status: str | None = None
    access_type: str | None = None
    user_id: str | None = None
    user_type: str | None = None
    order_copy_status: str | None = None
    ticker_session_max: int | None = None
    order_session_max: int | None = None
    allowed: bool = False


@dataclass(frozen=True)
class OrderOrigin:
    user_id: str | None = None
    application: str | None = None
    version: str | None = None
    originator_application: str | None = None
    originator_version: str | None = None
    window_name: str | None = None
    originator_window_name: str | None = None
    manual_or_auto: str | None = None
    user_tag: str | None = field(default=None, repr=False)


@dataclass(frozen=True)
class BrokerTimestamps:
    broker_seconds: int | None = None
    broker_microseconds: int | None = None
    source_seconds: int | None = None
    source_microseconds: int | None = None
    source_nanoseconds: int | None = None
    exchange_seconds: int | None = None
    exchange_microseconds: int | None = None
    exchange_nanoseconds: int | None = None


@dataclass(frozen=True)
class BrokerOrderObservation:
    event_id: str
    connection_generation: str
    source: ObservationSource
    account: BrokerAccountIdentity
    template_id: int
    received_at: datetime
    basket_id: str
    normalized_state: ObservedOrderState
    raw_notify_type: str | None = None
    raw_status: str | None = None
    completion_reason: str | None = None
    report_type: str | None = None
    command_outcome: CommandOutcome = CommandOutcome.NOT_APPLICABLE
    original_basket_id: str | None = None
    linked_basket_ids: tuple[str, ...] = ()
    exchange_order_id: str | None = None
    ticker_plant_exchange_order_id: str | None = None
    symbol: str | None = None
    exchange: str | None = None
    side: str | None = None
    order_type: str | None = None
    duration: str | None = None
    quantity: int | None = None
    cumulative_fill_quantity: int | None = None
    leaves_quantity: int | None = None
    last_fill_id: str | None = None
    last_fill_quantity: int | None = None
    last_fill_price: Decimal | None = None
    average_fill_price: Decimal | None = None
    limit_price: Decimal | None = None
    trigger_price: Decimal | None = None
    sequence_number: str | None = None
    original_sequence_number: str | None = None
    correlation_sequence_number: str | None = None
    confirmed_id: str | None = None
    modify_id: str | None = None
    cancelled_id: str | None = None
    trigger_id: str | None = None
    origin: OrderOrigin = field(default_factory=OrderOrigin)
    timestamps: BrokerTimestamps = field(default_factory=BrokerTimestamps)
    is_snapshot: bool = False
    replay_batch_id: str | None = None
    broker_text: str | None = None
    payload_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if not self.event_id or not self.connection_generation or not self.basket_id:
            raise ValueError("event, generation, and basket identities are required")
        for label, value in (
            ("quantity", self.quantity),
            ("cumulative_fill_quantity", self.cumulative_fill_quantity),
            ("leaves_quantity", self.leaves_quantity),
            ("last_fill_quantity", self.last_fill_quantity),
        ):
            if value is not None and (isinstance(value, bool) or value < 0):
                raise ValueError(f"{label} cannot be negative")


@dataclass(frozen=True)
class BrokerExecutionObservation:
    execution_event_id: str
    connection_generation: str
    source: ObservationSource
    adjustment_kind: ExecutionAdjustmentKind
    account: BrokerAccountIdentity
    basket_id: str
    fill_id: str
    received_at: datetime
    quantity: int
    price: Decimal
    symbol: str | None = None
    exchange: str | None = None
    exchange_order_id: str | None = None
    sequence_number: str | None = None
    correction_of_fill_id: str | None = None
    trade_date: str | None = None
    timestamps: BrokerTimestamps = field(default_factory=BrokerTimestamps)
    replay_batch_id: str | None = None

    def __post_init__(self) -> None:
        if not self.fill_id or not self.basket_id:
            raise ValueError("fill and basket identities are required")
        if isinstance(self.quantity, bool) or self.quantity < 0:
            raise ValueError("execution quantity cannot be negative")


@dataclass(frozen=True)
class BrokerPositionObservation:
    account: BrokerAccountIdentity
    connection_generation: str
    source: ObservationSource
    captured_at: datetime
    symbol: str
    exchange: str
    net_quantity: int | None = None
    open_quantity: int | None = None
    closed_quantity: int | None = None
    working_buy_quantity: int | None = None
    working_sell_quantity: int | None = None
    average_open_fill_price: Decimal | None = None
    open_position_pnl: Decimal | None = None
    closed_position_pnl: Decimal | None = None
    day_open_pnl: Decimal | None = None
    day_closed_pnl: Decimal | None = None
    day_total_pnl: Decimal | None = None
    currency: str | None = None
    freshness: FactFreshness = FactFreshness.UNKNOWN
    is_snapshot: bool = False
    replay_batch_id: str | None = None


@dataclass(frozen=True)
class BrokerAccountPnlObservation:
    account: BrokerAccountIdentity
    connection_generation: str
    source: ObservationSource
    captured_at: datetime
    account_balance: Decimal | None = None
    cash_on_hand: Decimal | None = None
    margin_balance: Decimal | None = None
    available_buying_power: Decimal | None = None
    used_buying_power: Decimal | None = None
    reserved_buying_power: Decimal | None = None
    excess_buy_margin: Decimal | None = None
    excess_sell_margin: Decimal | None = None
    open_position_pnl: Decimal | None = None
    closed_position_pnl: Decimal | None = None
    day_open_pnl: Decimal | None = None
    day_closed_pnl: Decimal | None = None
    day_total_pnl: Decimal | None = None
    commission: Decimal | None = None
    currency: str | None = None
    freshness: FactFreshness = FactFreshness.UNKNOWN
    is_snapshot: bool = False
    replay_batch_id: str | None = None


@dataclass(frozen=True)
class BrokerRmsObservation:
    account: BrokerAccountIdentity
    connection_generation: str
    observed_at: datetime
    source: ObservationSource
    product_code: str | None = None
    currency: str | None = None
    account_status: str | None = None
    loss_limit: Decimal | None = None
    max_order_quantity: int | None = None
    buy_limit: int | None = None
    sell_limit: int | None = None
    buy_margin_rate: Decimal | None = None
    sell_margin_rate: Decimal | None = None
    commission_rate: Decimal | None = None
    auto_liquidate_criteria: str | None = None
    auto_liquidate_threshold: Decimal | None = None
    auto_liquidate_current_value: Decimal | None = None
    peak_account_balance: Decimal | None = None
    freshness: FactFreshness = FactFreshness.UNKNOWN


@dataclass(frozen=True)
class BrokerBracketTierObservation:
    level: int
    kind: str
    quantity: int | None = None
    released_quantity: int | None = None
    ticks: Decimal | None = None
    trailing: bool | None = None

    def __post_init__(self) -> None:
        if self.level < 0:
            raise ValueError("bracket tier level cannot be negative")


@dataclass(frozen=True)
class BrokerBracketObservation:
    account: BrokerAccountIdentity
    connection_generation: str
    source: ObservationSource
    observed_at: datetime
    parent_basket_id: str
    bracket_type: str | None = None
    order_operation_type: str | None = None
    linked_basket_ids: tuple[str, ...] = ()
    tiers: tuple[BrokerBracketTierObservation, ...] = ()
    is_snapshot: bool = False
    replay_batch_id: str | None = None


@dataclass(frozen=True)
class BrokerContractReference:
    observed_at: datetime
    source: str
    symbol: str
    exchange: str
    product_code: str | None = None
    trading_symbol: str | None = None
    trading_exchange: str | None = None
    instrument_type: str | None = None
    underlying_symbol: str | None = None
    expiration: date | None = None
    currency: str | None = None
    tick_size_type: int | None = None
    minimum_quoted_price_change: Decimal | None = None
    minimum_feed_price_change: Decimal | None = None
    single_point_value: Decimal | None = None
    minimum_size_increment: Decimal | None = None
    size_multiplier: Decimal | None = None
    price_display_format: str | None = None
    is_tradable: bool | None = None


@dataclass(frozen=True)
class PlantHealth:
    plant: BrokerPlant
    required: bool
    generation_id: str | None
    connected: bool
    authenticated: bool
    reconciled: bool
    reconnecting: bool
    last_message_at: datetime | None = None
    last_heartbeat_at: datetime | None = None
    last_error_code: str | None = None

    @property
    def ready(self) -> bool:
        return (not self.required) or (self.connected and self.authenticated and self.reconciled and not self.reconnecting)


@dataclass(frozen=True)
class ReconciliationCheckpoint:
    checkpoint_id: str
    connection_generation: str
    account: BrokerAccountIdentity
    started_at: datetime
    completed_at: datetime | None
    subscriptions_started: bool
    order_snapshot_complete: bool
    execution_replay_complete: bool
    fill_history_complete: bool
    bracket_snapshot_complete: bool
    pnl_snapshot_complete: bool
    buffered_event_count: int = 0
    clean: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return all(
            (
                self.subscriptions_started,
                self.order_snapshot_complete,
                self.execution_replay_complete,
                self.fill_history_complete,
                self.bracket_snapshot_complete,
                self.pnl_snapshot_complete,
                self.clean,
                self.completed_at is not None,
            )
        )
