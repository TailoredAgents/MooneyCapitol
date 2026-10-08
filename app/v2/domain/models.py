from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Mapping


ZERO = Decimal("0")


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class PositionSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class OrderStatus(str, Enum):
    CREATED = "CREATED"
    PENDING = "PENDING"
    WORKING = "WORKING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    UNKNOWN = "UNKNOWN"

    @property
    def terminal(self) -> bool:
        return self in {self.FILLED, self.CANCELLED, self.REJECTED, self.EXPIRED}


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class ManagementEventType(str, Enum):
    ADD = "ADD"
    SCALE_OUT = "SCALE_OUT"
    STOP_CHANGE = "STOP_CHANGE"
    TARGET_CHANGE = "TARGET_CHANGE"
    FLATTEN = "FLATTEN"
    REVERSAL = "REVERSAL"


def _positive_decimal(value: Decimal, label: str) -> None:
    if value <= ZERO:
        raise ValueError(f"{label} must be positive")


def _positive_contracts(value: int, label: str = "contracts") -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")


@dataclass(frozen=True)
class ContractSpecification:
    product_code: str
    exchange: str
    point_value: Decimal
    tick_size: Decimal
    tick_value: Decimal
    currency: str = "USD"

    def __post_init__(self) -> None:
        _positive_decimal(self.point_value, "point_value")
        _positive_decimal(self.tick_size, "tick_size")
        _positive_decimal(self.tick_value, "tick_value")
        if self.tick_value != self.point_value * self.tick_size:
            raise ValueError("tick_value must equal point_value * tick_size")


@dataclass(frozen=True)
class FuturesContract:
    contract_id: str
    product_code: str
    exchange: str
    expiration: date
    specification: ContractSpecification
    provider_symbols: Mapping[str, str] = field(default_factory=dict)
    first_trade_date: date | None = None
    last_trade_date: date | None = None
    trading_symbol: str | None = None
    trading_exchange: str | None = None
    instrument_type: str | None = None
    tick_size_type: int | None = None
    is_tradable: bool | None = None
    reference_source: str | None = None
    reference_observed_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.product_code.upper() in {"NQ", "MNQ", "ES"} and self.contract_id.upper() == self.product_code.upper():
            raise ValueError("bare product codes are not valid expiration-aware contract identities")
        if self.specification.product_code.upper() != self.product_code.upper():
            raise ValueError("contract specification product mismatch")
        if self.specification.exchange.upper() != self.exchange.upper():
            raise ValueError("contract specification exchange mismatch")
        if self.first_trade_date and self.last_trade_date and self.first_trade_date > self.last_trade_date:
            raise ValueError("first_trade_date cannot follow last_trade_date")

    def is_available_on(self, trade_date: date) -> bool:
        if self.first_trade_date and trade_date < self.first_trade_date:
            return False
        if self.last_trade_date and trade_date > self.last_trade_date:
            return False
        return trade_date <= self.expiration


@dataclass(frozen=True)
class ContractMapping:
    source_contract_id: str
    target_contract_id: str
    source_product: str
    target_product: str
    expiration: date

    def __post_init__(self) -> None:
        allowed = {self.source_product.upper(), self.target_product.upper()}
        if allowed != {"NQ", "MNQ"}:
            raise ValueError("V2 foundation only seeds explicit NQ/MNQ product mappings")
        if self.source_contract_id == self.target_contract_id:
            raise ValueError("source and target contracts must differ")


@dataclass(frozen=True)
class BrokerConnection:
    connection_id: str
    broker: str
    environment: str
    enabled: bool = False
    credential_ref: str | None = field(default=None, repr=False)


@dataclass(frozen=True)
class BrokerAccount:
    account_id: str
    connection_id: str
    broker_account_ref: str = field(repr=False)
    display_name: str = ""
    enabled: bool = False
    fcm_id: str | None = field(default=None, repr=False)
    ib_id: str | None = field(default=None, repr=False)
    currency: str | None = None
    account_status: str | None = None
    access_type: str | None = None
    user_type: str | None = None


@dataclass(frozen=True)
class RiskProfile:
    profile_id: str
    sizing_mode: str
    sizing_value: Decimal
    max_risk_per_trade: Decimal
    max_contracts: int
    max_concurrent_open_risk: Decimal
    per_product_max_contracts: Mapping[str, int] = field(default_factory=dict)
    global_kill_switch: bool = True
    account_kill_switch: bool = True
    daily_realized_loss_ceiling: Decimal = ZERO
    daily_total_loss_ceiling: Decimal = ZERO
    margin_headroom_fraction: Decimal = ZERO


@dataclass(frozen=True)
class OpportunitySnapshot:
    opportunity_id: str
    contract_id: str
    observed_at: datetime
    feature_cutoff_at: datetime
    trade_date: date
    rule_version: str
    feature_schema_version: str
    market_data_lineage: Mapping[str, Any]
    surfaced_to_trader_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.feature_cutoff_at > self.observed_at:
            raise ValueError("feature cutoff cannot be after opportunity observation")


@dataclass(frozen=True)
class TradePlanVersion:
    plan_id: str
    trade_id: str
    version: int
    recorded_at: datetime
    contract_id: str
    direction: PositionSide
    planned_entry: Decimal | None
    original_stop: Decimal | None
    planned_targets: tuple[Decimal, ...] = ()
    planned_quantity: int | None = None
    planned_risk_dollars: Decimal | None = None
    source: str = "trader"

    def __post_init__(self) -> None:
        if self.version < 1:
            raise ValueError("plan version must be at least 1")
        if self.planned_quantity is not None:
            _positive_contracts(self.planned_quantity, "planned_quantity")
        if self.original_stop is None and self.planned_risk_dollars is not None:
            raise ValueError("planned risk requires a valid original stop")


@dataclass(frozen=True)
class NormalizedOrderEvent:
    event_id: str
    broker: str
    account_id: str
    broker_order_id: str
    contract_id: str
    revision: int
    status: OrderStatus
    side: Side
    quantity: int
    event_at: datetime
    received_at: datetime
    order_type: OrderType | None = None
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None
    filled_quantity: int = 0
    average_fill_price: Decimal | None = None
    raw_reference: str | None = None

    def __post_init__(self) -> None:
        _positive_contracts(self.quantity, "quantity")
        if self.revision < 0:
            raise ValueError("revision cannot be negative")
        if self.filled_quantity < 0 or self.filled_quantity > self.quantity:
            raise ValueError("filled quantity must be between zero and order quantity")


MasterOrderEvent = NormalizedOrderEvent


@dataclass(frozen=True)
class MasterOrder:
    master_order_id: str
    account_id: str
    contract_id: str
    side: Side
    quantity: int
    status: OrderStatus
    latest_revision: int
    plan_id: str | None = None
    parent_order_id: str | None = None

    def __post_init__(self) -> None:
        _positive_contracts(self.quantity)
        if self.latest_revision < 0:
            raise ValueError("latest_revision cannot be negative")


@dataclass(frozen=True)
class MasterExecution:
    execution_id: str
    master_order_id: str
    account_id: str
    contract_id: str
    side: Side
    quantity: int
    price: Decimal
    executed_at: datetime
    received_at: datetime
    fee: Decimal = ZERO

    def __post_init__(self) -> None:
        _positive_contracts(self.quantity)
        _positive_decimal(self.price, "price")
        if self.fee < ZERO:
            raise ValueError("fee cannot be negative")


@dataclass(frozen=True)
class TradeManagementEvent:
    management_event_id: str
    trade_id: str
    event_type: ManagementEventType
    occurred_at: datetime
    quantity: int | None = None
    previous_price: Decimal | None = None
    new_price: Decimal | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.quantity is not None:
            _positive_contracts(self.quantity)


@dataclass(frozen=True)
class MasterTrade:
    trade_id: str
    account_id: str
    contract_id: str
    direction: PositionSide
    original_plan_id: str | None
    original_stop: Decimal | None
    original_risk_dollars: Decimal | None
    entry_vwap: Decimal | None
    exit_vwap: Decimal | None
    initial_quantity: int
    maximum_quantity: int
    opened_at: datetime | None
    closed_at: datetime | None
    realized_gross_pnl: Decimal | None = None
    realized_net_pnl: Decimal | None = None
    realized_r: Decimal | None = None
    mfe_r: Decimal | None = None
    mae_r: Decimal | None = None

    def __post_init__(self) -> None:
        r_metrics = (self.realized_r, self.mfe_r, self.mae_r)
        if self.original_stop is None and self.original_risk_dollars is not None:
            raise ValueError("original risk requires a valid original stop")
        if any(value is not None for value in r_metrics) and self.original_risk_dollars is None:
            raise ValueError("R metrics require an original risk denominator")
        if self.original_risk_dollars is not None and self.original_risk_dollars <= ZERO:
            raise ValueError("original risk denominator must be positive")
        if self.initial_quantity < 0 or self.maximum_quantity < self.initial_quantity:
            raise ValueError("invalid trade quantities")


@dataclass(frozen=True)
class CopyIntent:
    copy_intent_id: str
    master_order_id: str
    follower_account_id: str
    source_contract_id: str
    target_contract_id: str
    intended_quantity: int
    created_at: datetime
    planned_risk_dollars: Decimal | None = None

    def __post_init__(self) -> None:
        _positive_contracts(self.intended_quantity, "intended_quantity")


@dataclass(frozen=True)
class FollowerOrder:
    follower_order_id: str
    copy_intent_id: str
    account_id: str
    contract_id: str
    side: Side
    quantity: int
    status: OrderStatus
    latest_revision: int = 0

    def __post_init__(self) -> None:
        _positive_contracts(self.quantity)
        if self.latest_revision < 0:
            raise ValueError("latest_revision cannot be negative")


@dataclass(frozen=True)
class OrderLink:
    link_id: str
    parent_order_id: str
    child_order_id: str
    link_type: str


@dataclass(frozen=True)
class OCOGroup:
    group_id: str
    order_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.order_ids) < 2:
            raise ValueError("OCO groups require at least two orders")


@dataclass(frozen=True)
class FollowerExecution:
    execution_id: str
    follower_order_id: str
    copy_intent_id: str
    account_id: str
    contract_id: str
    quantity: int
    price: Decimal
    executed_at: datetime
    received_at: datetime
    fee: Decimal = ZERO

    def __post_init__(self) -> None:
        _positive_contracts(self.quantity)
        _positive_decimal(self.price, "price")
        if self.fee < ZERO:
            raise ValueError("fee cannot be negative")


@dataclass(frozen=True)
class FollowerTrade:
    follower_trade_id: str
    master_trade_id: str
    account_id: str
    contract_id: str
    entry_vwap: Decimal | None
    exit_vwap: Decimal | None
    realized_net_pnl: Decimal | None
    realized_r: Decimal | None
    execution_quality: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PositionSnapshot:
    account_id: str
    contract_id: str
    captured_at: datetime
    quantity: int
    average_price: Decimal | None
    mark_price: Decimal | None
    unrealized_pnl: Decimal


@dataclass(frozen=True)
class AccountRiskSnapshot:
    account_id: str
    captured_at: datetime
    equity: Decimal
    available_margin: Decimal
    open_risk: Decimal
    realized_pnl_trade_date: Decimal
    total_pnl_trade_date: Decimal
    global_kill_switch: bool
    account_kill_switch: bool
    drawdown_trade_date: Decimal = ZERO


@dataclass(frozen=True)
class ReconciliationIncident:
    incident_id: str
    account_id: str
    detected_at: datetime
    incident_type: str
    severity: str
    expected: Mapping[str, Any]
    actual: Mapping[str, Any]
    resolved_at: datetime | None = None


def apply_order_event(current: MasterOrder | FollowerOrder, event: NormalizedOrderEvent):
    """Apply a monotonic lifecycle event without allowing terminal rollback."""
    if event.revision <= current.latest_revision:
        return current
    if current.status.terminal and event.status != current.status:
        return current
    values = {**current.__dict__, "status": event.status, "latest_revision": event.revision}
    return type(current)(**values)
