from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from .constants import Template
from .fields import (
    FieldView,
    enum_name,
    is_exact_success_response,
    optional_bool,
    optional_decimal,
    optional_int,
    optional_text,
)


class SourceKind(str, Enum):
    LIVE = "LIVE"
    SNAPSHOT = "SNAPSHOT"
    HISTORY = "HISTORY"
    UNKNOWN = "UNKNOWN"


class OrderLifecycleState(str, Enum):
    PENDING = "PENDING"
    WORKING = "WORKING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    UNKNOWN = "UNKNOWN"
    OUTCOME_UNKNOWN = "OUTCOME_UNKNOWN"


class ExecutionEffect(str, Enum):
    NONE = "NONE"
    APPLY_FILL = "APPLY_FILL"
    BUST_FILL = "BUST_FILL"
    CORRECT_FILL = "CORRECT_FILL"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class BrokerAccountKey:
    fcm_id: str
    ib_id: str
    account_id: str

    def __post_init__(self) -> None:
        if not self.fcm_id or not self.ib_id or not self.account_id:
            raise ValueError("FCM, IB and account identifiers are required opaque strings")


@dataclass(frozen=True)
class TimestampEvidence:
    ssboe: int | None = None
    usecs: int | None = None
    source_ssboe: int | None = None
    source_usecs: int | None = None
    source_nsecs: int | None = None
    exchange_receipt_ssboe: int | None = None
    exchange_receipt_nsecs: int | None = None
    server_received_ssboe: int | None = None
    server_received_usecs: int | None = None


@dataclass(frozen=True)
class OriginMetadata:
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
class SystemInfoObservation:
    systems: tuple[str, ...]
    has_aggregated_quotes: tuple[bool, ...]
    response_codes: tuple[str, ...]


@dataclass(frozen=True)
class GatewayInfoObservation:
    system_name: str | None
    gateway_names: tuple[str, ...]
    gateway_uris: tuple[str, ...] = field(repr=False)
    response_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class LoginObservation:
    success: bool
    response_codes: tuple[str, ...]
    template_version: str | None
    fcm_id: str | None
    ib_id: str | None
    unique_user_id: str | None = field(default=None, repr=False)
    heartbeat_interval_seconds: Decimal | None = None


@dataclass(frozen=True)
class LoginInfoObservation:
    fcm_id: str | None
    ib_id: str | None
    user: str | None = field(default=None, repr=False)
    user_type: str | None = None
    user_type_code: int | None = None
    status: str | None = None
    order_copy_status: str | None = None
    ticker_session_max: int | None = None
    order_session_max: int | None = None
    country_code: str | None = None
    state_code: str | None = None
    sensitive_metadata: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({}), repr=False
    )


@dataclass(frozen=True)
class AccountObservation:
    key: BrokerAccountKey
    account_name: str | None
    currency: str | None
    loss_limit: Decimal | None
    auto_liquidate: str | None
    auto_liquidate_threshold_current: Decimal | None
    creation_ssboe: int | None
    creation_usecs: int | None
    account_status: str | None = None
    access_type: str | None = None
    user_id: str | None = field(default=None, repr=False)
    user_type: str | None = None
    user_status: str | None = None
    order_copy_status: str | None = None
    ticker_session_max: int | None = None
    order_session_max: int | None = None
    country_code: str | None = None
    state_code: str | None = None
    sensitive_metadata: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({}), repr=False
    )


@dataclass(frozen=True)
class OrderObservation:
    template_id: int
    source_kind: SourceKind
    account: BrokerAccountKey | None
    basket_id: str | None
    original_basket_id: str | None
    linked_basket_ids: str | None
    exchange_order_id: str | None
    ticker_plant_exchange_order_id: str | None
    fill_id: str | None
    raw_notify_type: str | None
    normalized_state: OrderLifecycleState | None
    command_failure: bool
    execution_effect: ExecutionEffect
    status: str | None
    completion_reason: str | None
    report_type: str | None
    quantity: int | None
    fill_size: int | None
    cumulative_fill_size: int | None
    unfilled_size: int | None
    confirmed_size: int | None
    modified_size: int | None
    cancelled_size: int | None
    price: Decimal | None
    trigger_price: Decimal | None
    fill_price: Decimal | None
    average_fill_price: Decimal | None
    symbol: str | None
    exchange: str | None
    trade_exchange: str | None
    trade_route: str | None
    transaction_type: str | None
    duration: str | None
    price_type: str | None
    original_price_type: str | None
    bracket_type: str | None
    sequence_number: str | None
    original_sequence_number: str | None
    correlation_sequence_number: str | None
    origin: OriginMetadata
    timestamps: TimestampEvidence
    text: str | None = field(default=None, repr=False)
    report_text: str | None = field(default=None, repr=False)
    remarks: str | None = field(default=None, repr=False)

    @property
    def stable_identity(self) -> tuple[object, ...]:
        if self.fill_id:
            return (
                "fill",
                self.account,
                self.basket_id,
                self.fill_id,
                self.report_type,
                self.sequence_number,
            )
        return (
            "order",
            self.account,
            self.basket_id,
            self.template_id,
            self.raw_notify_type,
            self.sequence_number,
            self.timestamps.source_ssboe,
            self.timestamps.source_nsecs,
            self.timestamps.ssboe,
            self.timestamps.usecs,
        )


@dataclass(frozen=True)
class InstrumentPnlObservation:
    source_kind: SourceKind
    account: BrokerAccountKey
    symbol: str | None
    exchange: str | None
    product_code: str | None
    instrument_type: str | None
    fill_buy_quantity: int | None
    fill_sell_quantity: int | None
    working_buy_quantity: int | None
    working_sell_quantity: int | None
    buy_quantity: int | None
    sell_quantity: int | None
    open_position_quantity: int | None
    closed_position_quantity: int | None
    net_quantity: int | None
    average_open_fill_price: Decimal | None
    open_position_pnl: Decimal | None
    closed_position_pnl: Decimal | None
    day_open_pnl: Decimal | None
    day_closed_pnl: Decimal | None
    day_pnl: Decimal | None
    day_open_pnl_offset: Decimal | None
    day_closed_pnl_offset: Decimal | None
    timestamps: TimestampEvidence


@dataclass(frozen=True)
class AccountPnlObservation:
    source_kind: SourceKind
    account: BrokerAccountKey
    fill_buy_quantity: int | None
    fill_sell_quantity: int | None
    working_buy_quantity: int | None
    working_sell_quantity: int | None
    buy_quantity: int | None
    sell_quantity: int | None
    open_position_quantity: int | None
    closed_position_quantity: int | None
    net_quantity: int | None
    open_position_pnl: Decimal | None
    closed_position_pnl: Decimal | None
    day_open_pnl: Decimal | None
    day_closed_pnl: Decimal | None
    day_pnl: Decimal | None
    day_open_pnl_offset: Decimal | None
    day_closed_pnl_offset: Decimal | None
    cash_on_hand: Decimal | None
    account_balance: Decimal | None
    margin_balance: Decimal | None
    minimum_margin_balance: Decimal | None
    minimum_account_balance: Decimal | None
    available_buying_power: Decimal | None
    used_buying_power: Decimal | None
    reserved_buying_power: Decimal | None
    excess_buy_margin: Decimal | None
    excess_sell_margin: Decimal | None
    commission: Decimal | None
    timestamps: TimestampEvidence


@dataclass(frozen=True)
class AccountRmsObservation:
    account: BrokerAccountKey
    presence_bits: int | None
    currency: str | None
    status: str | None
    algorithm: str | None
    auto_liquidate_criteria: str | None
    auto_liquidate: str | None
    disable_on_auto_liquidate: str | None
    auto_liquidate_threshold: Decimal | None
    auto_liquidate_max_min_balance: Decimal | None
    loss_limit: Decimal | None
    minimum_account_balance: Decimal | None
    minimum_margin_balance: Decimal | None
    default_commission: Decimal | None
    buy_limit: int | None
    sell_limit: int | None
    maximum_order_quantity: int | None
    check_minimum_account_balance: bool | None


@dataclass(frozen=True)
class ProductRmsObservation:
    account: BrokerAccountKey
    product_code: str | None
    presence_bits: int | None
    loss_limit: Decimal | None
    commission_fill_rate: Decimal | None
    buy_margin_rate: Decimal | None
    sell_margin_rate: Decimal | None
    buy_limit: int | None
    sell_limit: int | None
    maximum_order_quantity: int | None


@dataclass(frozen=True)
class RmsUpdateObservation:
    account: BrokerAccountKey
    update_bits: int | None
    current_auto_liquidate_threshold: Decimal | None
    peak_account_balance: Decimal | None
    peak_account_balance_ssboe: int | None


@dataclass(frozen=True)
class BracketObservation:
    account: BrokerAccountKey | None
    parent_basket_id: str | None
    bracket_type: str | None
    linked_basket_ids: str | None
    stop_ticks: int | None
    stop_quantity: int | None
    stop_quantity_released: int | None
    target_ticks: int | None
    target_quantity: int | None
    target_quantity_released: int | None
    trailing_field_id: str | None
    trailing_stop_trigger_ticks: int | None


@dataclass(frozen=True)
class ReferenceObservation:
    symbol: str | None
    exchange: str | None
    exchange_symbol: str | None
    symbol_name: str | None
    trading_symbol: str | None
    trading_exchange: str | None
    product_code: str | None
    instrument_type: str | None
    underlying_symbol: str | None
    expiration_date: str | None
    currency: str | None
    tick_size_type: str | None
    price_display_format: str | None
    is_tradable: str | None
    minimum_quoted_price_change: Decimal | None
    minimum_feed_price_change: Decimal | None
    single_point_value: Decimal | None
    quote_to_feed_price_factor: Decimal | None
    feed_to_quote_price_factor: Decimal | None
    presence_bits: int | None


@dataclass(frozen=True)
class TickSizeObservation:
    tick_size_type: str | None
    minimum_feed_price_change: Decimal | None
    first_price: Decimal | None
    last_price: Decimal | None
    first_price_operator: str | None
    last_price_operator: str | None
    presence_bits: int | None


_USER_TYPES = {0: "USER_TYPE_ADMIN", 1: "USER_TYPE_FCM", 2: "USER_TYPE_IB", 3: "USER_TYPE_TRADER"}
_ORDER_NOTIFY_TYPES = {
    1: "ORDER_RCVD_FROM_CLNT",
    2: "MODIFY_RCVD_FROM_CLNT",
    3: "CANCEL_RCVD_FROM_CLNT",
    4: "OPEN_PENDING",
    5: "MODIFY_PENDING",
    6: "CANCEL_PENDING",
    7: "ORDER_RCVD_BY_EXCH_GTWY",
    8: "MODIFY_RCVD_BY_EXCH_GTWY",
    9: "CANCEL_RCVD_BY_EXCH_GTWY",
    10: "ORDER_SENT_TO_EXCH",
    11: "MODIFY_SENT_TO_EXCH",
    12: "CANCEL_SENT_TO_EXCH",
    13: "OPEN",
    14: "MODIFIED",
    15: "COMPLETE",
    16: "MODIFICATION_FAILED",
    17: "CANCELLATION_FAILED",
    18: "TRIGGER_PENDING",
    19: "GENERIC",
    20: "LINK_ORDERS_FAILED",
}
_EXCHANGE_NOTIFY_TYPES = {
    1: "STATUS",
    2: "MODIFY",
    3: "CANCEL",
    4: "TRIGGER",
    5: "FILL",
    6: "REJECT",
    7: "NOT_MODIFIED",
    8: "NOT_CANCELLED",
    9: "GENERIC",
}
_TRANSACTION_TYPES = {1: "BUY", 2: "SELL", 3: "SS"}
_DURATIONS = {1: "DAY", 2: "GTC", 3: "IOC", 4: "FOK"}
_PRICE_TYPES = {1: "LIMIT", 2: "MARKET", 3: "STOP_LIMIT", 4: "STOP_MARKET"}
_BRACKET_TYPES = {
    1: "STOP_ONLY",
    2: "TARGET_ONLY",
    3: "TARGET_AND_STOP",
    4: "STOP_ONLY_STATIC",
    5: "TARGET_ONLY_STATIC",
    6: "TARGET_AND_STOP_STATIC",
}
_PLACEMENT_TYPES = {1: "MANUAL", 2: "AUTO"}
_AUTO_LIQUIDATE = {1: "ENABLED", 2: "DISABLED"}


def _account(view: FieldView) -> BrokerAccountKey | None:
    fcm = optional_text(view.get("fcm_id"))
    ib = optional_text(view.get("ib_id"))
    account = optional_text(view.get("account_id"))
    if fcm is None and ib is None and account is None:
        return None
    if fcm is None or ib is None or account is None:
        return None
    return BrokerAccountKey(fcm, ib, account)


def _required_account(view: FieldView) -> BrokerAccountKey:
    account = _account(view)
    if account is None:
        raise ValueError("complete FCM/IB/account identity is required")
    return account


def _source(view: FieldView, template_id: int) -> SourceKind:
    if template_id == Template.FILL_HISTORY_RESPONSE:
        return SourceKind.HISTORY
    snapshot = optional_bool(view.get("is_snapshot"))
    if snapshot is True:
        return SourceKind.SNAPSHOT
    if snapshot is False:
        return SourceKind.LIVE
    return SourceKind.UNKNOWN


def _timestamps(view: FieldView) -> TimestampEvidence:
    return TimestampEvidence(
        ssboe=optional_int(view.get("ssboe")),
        usecs=optional_int(view.get("usecs")),
        source_ssboe=optional_int(view.get("source_ssboe")),
        source_usecs=optional_int(view.get("source_usecs")),
        source_nsecs=optional_int(view.get("source_nsecs")),
        exchange_receipt_ssboe=optional_int(view.get("exch_receipt_ssboe")),
        exchange_receipt_nsecs=optional_int(view.get("exch_receipt_nsecs")),
        server_received_ssboe=optional_int(view.get("proto_srvr_rcvd_ssboe")),
        server_received_usecs=optional_int(view.get("proto_srvr_rcvd_usecs")),
    )


def normalize_system_info(message: Mapping[str, Any] | Any) -> SystemInfoObservation:
    view = FieldView(message)
    flags = tuple(bool(item) for item in (view.get("has_aggregated_quotes", ()) or ()))
    return SystemInfoObservation(view.strings("system_name"), flags, view.strings("rp_code"))


def normalize_gateway_info(message: Mapping[str, Any] | Any) -> GatewayInfoObservation:
    view = FieldView(message)
    return GatewayInfoObservation(
        optional_text(view.get("system_name")),
        view.strings("gateway_name"),
        view.strings("gateway_uri"),
        view.strings("rp_code"),
    )


def normalize_login(message: Mapping[str, Any] | Any) -> LoginObservation:
    view = FieldView(message)
    codes = view.strings("rp_code")
    return LoginObservation(
        success=is_exact_success_response(codes),
        response_codes=codes,
        template_version=optional_text(view.get("template_version")),
        fcm_id=optional_text(view.get("fcm_id")),
        ib_id=optional_text(view.get("ib_id")),
        unique_user_id=optional_text(view.get("unique_user_id")),
        heartbeat_interval_seconds=optional_decimal(view.get("heartbeat_interval")),
    )


def normalize_login_info(message: Mapping[str, Any] | Any) -> LoginInfoObservation:
    view = FieldView(message)
    raw_user_type = view.first("user_type", "type")
    user_type_code = (
        optional_int(raw_user_type)
        if not isinstance(raw_user_type, str) or raw_user_type.strip().isdigit()
        else None
    )
    sensitive_names = (
        "first_name",
        "last_name",
        "email_address",
        "address_street_1",
        "address_street_2",
        "address_city",
        "address_state",
        "address_country",
        "address_zip",
        "phone_residence",
        "phone_work",
        "phone_mobile",
    )
    sensitive = {
        name: value
        for name in sensitive_names
        if (value := optional_text(view.get(name))) is not None
    }
    return LoginInfoObservation(
        fcm_id=optional_text(view.get("fcm_id")),
        ib_id=optional_text(view.get("ib_id")),
        user=optional_text(view.get("user")),
        user_type=enum_name(
            user_type_code if user_type_code is not None else raw_user_type,
            _USER_TYPES,
        ),
        user_type_code=user_type_code,
        status=optional_text(view.get("status")),
        order_copy_status=optional_text(view.get("order_copy_status")),
        ticker_session_max=optional_int(view.get("tp_max_session_count")),
        order_session_max=optional_int(view.get("op_max_session_count")),
        country_code=optional_text(view.get("country_code")),
        state_code=optional_text(view.get("state_code")),
        sensitive_metadata=MappingProxyType(sensitive),
    )


def normalize_account(message: Mapping[str, Any] | Any) -> AccountObservation:
    view = FieldView(message)
    return AccountObservation(
        key=_required_account(view),
        account_name=optional_text(view.get("account_name")),
        currency=optional_text(view.get("account_currency")),
        loss_limit=optional_decimal(view.get("loss_limit")),
        auto_liquidate=optional_text(view.get("account_auto_liquidate")),
        auto_liquidate_threshold_current=optional_decimal(
            view.get("auto_liq_threshold_current_value")
        ),
        creation_ssboe=optional_int(view.get("account_creation_ssboe")),
        creation_usecs=optional_int(view.get("account_creation_usecs")),
    )


def _execution_effect(report_type: str | None, notify_type: str | None) -> ExecutionEffect:
    report = (report_type or "").strip().lower()
    if report == "bust":
        return ExecutionEffect.BUST_FILL
    if report == "trade correct":
        return ExecutionEffect.CORRECT_FILL
    if report == "fill" or (not report and notify_type == "FILL"):
        return ExecutionEffect.APPLY_FILL
    if notify_type == "FILL":
        return ExecutionEffect.UNKNOWN
    return ExecutionEffect.NONE


def _state(
    *,
    template_id: int,
    notify_type: str | None,
    completion_reason: str | None,
    cumulative: int | None,
    unfilled: int | None,
    effect: ExecutionEffect,
) -> tuple[OrderLifecycleState | None, bool]:
    command_failures = {
        "MODIFICATION_FAILED",
        "CANCELLATION_FAILED",
        "LINK_ORDERS_FAILED",
        "NOT_MODIFIED",
        "NOT_CANCELLED",
    }
    if notify_type in command_failures:
        return None, True
    if template_id == Template.RITHMIC_ORDER_NOTIFICATION:
        if notify_type in {
            "ORDER_RCVD_FROM_CLNT",
            "MODIFY_RCVD_FROM_CLNT",
            "CANCEL_RCVD_FROM_CLNT",
            "OPEN_PENDING",
            "MODIFY_PENDING",
            "CANCEL_PENDING",
            "ORDER_RCVD_BY_EXCH_GTWY",
            "MODIFY_RCVD_BY_EXCH_GTWY",
            "CANCEL_RCVD_BY_EXCH_GTWY",
            "ORDER_SENT_TO_EXCH",
            "MODIFY_SENT_TO_EXCH",
            "CANCEL_SENT_TO_EXCH",
            "TRIGGER_PENDING",
        }:
            return OrderLifecycleState.PENDING, False
        if notify_type in {"OPEN", "MODIFIED"}:
            if cumulative and cumulative > 0:
                return OrderLifecycleState.PARTIALLY_FILLED, False
            return OrderLifecycleState.WORKING, False
        if notify_type == "COMPLETE":
            reason = (completion_reason or "").upper()
            if reason == "F":
                return OrderLifecycleState.FILLED, False
            if reason in {"C", "PFBC"}:
                return OrderLifecycleState.CANCELLED, False
            if reason in {"R", "FA"}:
                return OrderLifecycleState.REJECTED, False
            return OrderLifecycleState.UNKNOWN, False
        return None, False

    if notify_type in {"STATUS", "MODIFY", "TRIGGER"}:
        if cumulative and cumulative > 0 and (unfilled is None or unfilled > 0):
            return OrderLifecycleState.PARTIALLY_FILLED, False
        return OrderLifecycleState.WORKING, False
    if notify_type == "CANCEL":
        return OrderLifecycleState.CANCELLED, False
    if notify_type == "REJECT":
        return OrderLifecycleState.REJECTED, False
    if notify_type == "FILL":
        if effect in {ExecutionEffect.BUST_FILL, ExecutionEffect.CORRECT_FILL, ExecutionEffect.UNKNOWN}:
            return None, False
        if unfilled == 0:
            return OrderLifecycleState.FILLED, False
        if unfilled is not None and unfilled > 0:
            return OrderLifecycleState.PARTIALLY_FILLED, False
        return None, False
    return None, False


def normalize_order(message: Mapping[str, Any] | Any, *, template_id: int | None = None) -> OrderObservation:
    view = FieldView(message)
    resolved_template = int(template_id if template_id is not None else view.get("template_id"))
    if resolved_template == Template.RITHMIC_ORDER_NOTIFICATION:
        notify = enum_name(view.get("notify_type"), _ORDER_NOTIFY_TYPES)
    elif resolved_template == Template.EXCHANGE_ORDER_NOTIFICATION:
        notify = enum_name(view.get("notify_type"), _EXCHANGE_NOTIFY_TYPES)
    else:
        notify = optional_text(view.get("notify_type"))
    report_type = optional_text(view.get("report_type"))
    effect = _execution_effect(report_type, notify)
    cumulative = optional_int(view.first("total_fill_size_64", "total_fill_size"), unsigned=True)
    unfilled = optional_int(view.first("total_unfilled_size_64", "total_unfilled_size"), unsigned=True)
    completion = optional_text(view.get("completion_reason"))
    state, command_failure = _state(
        template_id=resolved_template,
        notify_type=notify,
        completion_reason=completion,
        cumulative=cumulative,
        unfilled=unfilled,
        effect=effect,
    )
    fill_size_names = (
        ("fill_size",) if resolved_template == Template.FILL_HISTORY_RESPONSE else ("fill_size_64",)
    )
    return OrderObservation(
        template_id=resolved_template,
        source_kind=_source(view, resolved_template),
        account=_account(view),
        basket_id=optional_text(view.get("basket_id")),
        original_basket_id=optional_text(view.get("original_basket_id")),
        linked_basket_ids=optional_text(view.get("linked_basket_ids")),
        exchange_order_id=optional_text(view.get("exchange_order_id")),
        ticker_plant_exchange_order_id=optional_text(view.get("tp_exchange_order_id")),
        fill_id=optional_text(view.get("fill_id")),
        raw_notify_type=notify,
        normalized_state=state,
        command_failure=command_failure,
        execution_effect=effect,
        status=optional_text(view.get("status")),
        completion_reason=completion,
        report_type=report_type,
        quantity=optional_int(view.get("quantity_64"), unsigned=True),
        fill_size=optional_int(view.first(*fill_size_names), unsigned=True),
        cumulative_fill_size=cumulative,
        unfilled_size=unfilled,
        confirmed_size=optional_int(view.get("confirmed_size_64"), unsigned=True),
        modified_size=optional_int(view.get("modified_size_64"), unsigned=True),
        cancelled_size=optional_int(view.get("cancelled_size_64"), unsigned=True),
        price=optional_decimal(view.get("price")),
        trigger_price=optional_decimal(view.get("trigger_price")),
        fill_price=optional_decimal(view.get("fill_price")),
        average_fill_price=optional_decimal(view.get("avg_fill_price")),
        symbol=optional_text(view.get("symbol")),
        exchange=optional_text(view.get("exchange")),
        trade_exchange=optional_text(view.get("trade_exchange")),
        trade_route=optional_text(view.get("trade_route")),
        transaction_type=enum_name(view.get("transaction_type"), _TRANSACTION_TYPES),
        duration=enum_name(view.get("duration"), _DURATIONS),
        price_type=enum_name(view.get("price_type"), _PRICE_TYPES),
        original_price_type=enum_name(view.get("orig_price_type"), _PRICE_TYPES),
        bracket_type=enum_name(view.get("bracket_type"), _BRACKET_TYPES),
        sequence_number=optional_text(view.get("sequence_number")),
        original_sequence_number=optional_text(view.get("orig_sequence_number")),
        correlation_sequence_number=optional_text(view.get("cor_sequence_number")),
        origin=OriginMetadata(
            user_id=optional_text(view.get("user_id")),
            application=optional_text(view.get("application")),
            version=optional_text(view.get("version")),
            originator_application=optional_text(view.get("originator_application")),
            originator_version=optional_text(view.get("originator_version")),
            window_name=optional_text(view.get("window_name")),
            originator_window_name=optional_text(view.get("originator_window_name")),
            manual_or_auto=enum_name(view.get("manual_or_auto"), _PLACEMENT_TYPES),
            user_tag=optional_text(view.get("user_tag")),
        ),
        timestamps=_timestamps(view),
        text=optional_text(view.get("text")),
        report_text=optional_text(view.get("report_text")),
        remarks=optional_text(view.get("remarks")),
    )


def normalize_instrument_pnl(message: Mapping[str, Any] | Any) -> InstrumentPnlObservation:
    view = FieldView(message)
    return InstrumentPnlObservation(
        source_kind=_source(view, Template.INSTRUMENT_PNL_UPDATE),
        account=_required_account(view),
        symbol=optional_text(view.get("symbol")),
        exchange=optional_text(view.get("exchange")),
        product_code=optional_text(view.get("product_code")),
        instrument_type=optional_text(view.get("instrument_type")),
        fill_buy_quantity=optional_int(view.get("fill_buy_qty")),
        fill_sell_quantity=optional_int(view.get("fill_sell_qty")),
        working_buy_quantity=optional_int(view.get("order_buy_qty")),
        working_sell_quantity=optional_int(view.get("order_sell_qty")),
        buy_quantity=optional_int(view.get("buy_qty")),
        sell_quantity=optional_int(view.get("sell_qty")),
        open_position_quantity=optional_int(view.get("open_position_quantity")),
        closed_position_quantity=optional_int(view.get("closed_position_quantity")),
        net_quantity=optional_int(view.get("net_quantity")),
        average_open_fill_price=optional_decimal(view.get("avg_open_fill_price")),
        open_position_pnl=optional_decimal(view.get("open_position_pnl")),
        closed_position_pnl=optional_decimal(view.get("closed_position_pnl")),
        day_open_pnl=optional_decimal(view.get("day_open_pnl")),
        day_closed_pnl=optional_decimal(view.get("day_closed_pnl")),
        day_pnl=optional_decimal(view.get("day_pnl")),
        day_open_pnl_offset=optional_decimal(view.get("day_open_pnl_offset")),
        day_closed_pnl_offset=optional_decimal(view.get("day_closed_pnl_offset")),
        timestamps=_timestamps(view),
    )


def normalize_account_pnl(message: Mapping[str, Any] | Any) -> AccountPnlObservation:
    view = FieldView(message)
    return AccountPnlObservation(
        source_kind=_source(view, Template.ACCOUNT_PNL_UPDATE),
        account=_required_account(view),
        fill_buy_quantity=optional_int(view.get("fill_buy_qty")),
        fill_sell_quantity=optional_int(view.get("fill_sell_qty")),
        working_buy_quantity=optional_int(view.get("order_buy_qty")),
        working_sell_quantity=optional_int(view.get("order_sell_qty")),
        buy_quantity=optional_int(view.get("buy_qty")),
        sell_quantity=optional_int(view.get("sell_qty")),
        open_position_quantity=optional_int(view.get("open_position_quantity")),
        closed_position_quantity=optional_int(view.get("closed_position_quantity")),
        net_quantity=optional_int(view.get("net_quantity")),
        open_position_pnl=optional_decimal(view.get("open_position_pnl")),
        closed_position_pnl=optional_decimal(view.get("closed_position_pnl")),
        day_open_pnl=optional_decimal(view.get("day_open_pnl")),
        day_closed_pnl=optional_decimal(view.get("day_closed_pnl")),
        day_pnl=optional_decimal(view.get("day_pnl")),
        day_open_pnl_offset=optional_decimal(view.get("day_open_pnl_offset")),
        day_closed_pnl_offset=optional_decimal(view.get("day_closed_pnl_offset")),
        cash_on_hand=optional_decimal(view.get("cash_on_hand")),
        account_balance=optional_decimal(view.get("account_balance")),
        margin_balance=optional_decimal(view.get("margin_balance")),
        minimum_margin_balance=optional_decimal(view.get("min_margin_balance")),
        minimum_account_balance=optional_decimal(view.get("min_account_balance")),
        available_buying_power=optional_decimal(view.get("available_buying_power")),
        used_buying_power=optional_decimal(view.get("used_buying_power")),
        reserved_buying_power=optional_decimal(view.get("reserved_buying_power")),
        excess_buy_margin=optional_decimal(view.get("excess_buy_margin")),
        excess_sell_margin=optional_decimal(view.get("excess_sell_margin")),
        commission=optional_decimal(view.get("rms_account_commission")),
        timestamps=_timestamps(view),
    )


def normalize_account_rms(message: Mapping[str, Any] | Any) -> AccountRmsObservation:
    view = FieldView(message)
    return AccountRmsObservation(
        account=_required_account(view),
        presence_bits=optional_int(view.get("presence_bits"), unsigned=True),
        currency=optional_text(view.get("currency")),
        status=optional_text(view.get("status")),
        algorithm=optional_text(view.get("algorithm")),
        auto_liquidate_criteria=optional_text(view.get("auto_liquidate_criteria")),
        auto_liquidate=enum_name(view.get("auto_liquidate"), _AUTO_LIQUIDATE),
        disable_on_auto_liquidate=enum_name(view.get("disable_on_auto_liquidate"), _AUTO_LIQUIDATE),
        auto_liquidate_threshold=optional_decimal(view.get("auto_liquidate_threshold")),
        auto_liquidate_max_min_balance=optional_decimal(
            view.get("auto_liquidate_max_min_account_balance")
        ),
        loss_limit=optional_decimal(view.get("loss_limit")),
        minimum_account_balance=optional_decimal(view.get("min_account_balance")),
        minimum_margin_balance=optional_decimal(view.get("min_margin_balance")),
        default_commission=optional_decimal(view.get("default_commission")),
        buy_limit=optional_int(view.get("buy_limit")),
        sell_limit=optional_int(view.get("sell_limit")),
        maximum_order_quantity=optional_int(view.get("max_order_quantity")),
        check_minimum_account_balance=optional_bool(view.get("check_min_account_balance")),
    )


def normalize_product_rms(message: Mapping[str, Any] | Any) -> ProductRmsObservation:
    view = FieldView(message)
    return ProductRmsObservation(
        account=_required_account(view),
        product_code=optional_text(view.get("product_code")),
        presence_bits=optional_int(view.get("presence_bits"), unsigned=True),
        loss_limit=optional_decimal(view.get("loss_limit")),
        commission_fill_rate=optional_decimal(view.get("commission_fill_rate")),
        buy_margin_rate=optional_decimal(view.get("buy_margin_rate")),
        sell_margin_rate=optional_decimal(view.get("sell_margin_rate")),
        buy_limit=optional_int(view.get("buy_limit")),
        sell_limit=optional_int(view.get("sell_limit")),
        maximum_order_quantity=optional_int(view.get("max_order_quantity")),
    )


def normalize_rms_update(message: Mapping[str, Any] | Any) -> RmsUpdateObservation:
    view = FieldView(message)
    return RmsUpdateObservation(
        account=_required_account(view),
        update_bits=optional_int(view.get("update_bits"), unsigned=True),
        current_auto_liquidate_threshold=optional_decimal(
            view.get("auto_liq_threshold_current_value")
        ),
        peak_account_balance=optional_decimal(view.get("auto_liq_peak_account_balance")),
        peak_account_balance_ssboe=optional_int(
            view.get("auto_liq_peak_account_balance_ssboe")
        ),
    )


def normalize_bracket(message: Mapping[str, Any] | Any) -> BracketObservation:
    view = FieldView(message)
    return BracketObservation(
        account=_account(view),
        parent_basket_id=optional_text(view.get("basket_id")),
        bracket_type=enum_name(view.get("bracket_type"), _BRACKET_TYPES),
        linked_basket_ids=optional_text(view.get("linked_basket_ids")),
        stop_ticks=optional_int(view.get("stop_ticks")),
        stop_quantity=optional_int(view.get("stop_quantity"), unsigned=True),
        stop_quantity_released=optional_int(view.get("stop_quantity_released"), unsigned=True),
        target_ticks=optional_int(view.get("target_ticks")),
        target_quantity=optional_int(view.get("target_quantity"), unsigned=True),
        target_quantity_released=optional_int(
            view.get("target_quantity_released"), unsigned=True
        ),
        trailing_field_id=optional_text(view.get("bracket_trailing_field_id")),
        trailing_stop_trigger_ticks=optional_int(view.get("trailing_stop_trigger_ticks")),
    )


def normalize_reference(message: Mapping[str, Any] | Any) -> ReferenceObservation:
    view = FieldView(message)
    return ReferenceObservation(
        symbol=optional_text(view.get("symbol")),
        exchange=optional_text(view.get("exchange")),
        exchange_symbol=optional_text(view.get("exchange_symbol")),
        symbol_name=optional_text(view.get("symbol_name")),
        trading_symbol=optional_text(view.get("trading_symbol")),
        trading_exchange=optional_text(view.get("trading_exchange")),
        product_code=optional_text(view.get("product_code")),
        instrument_type=optional_text(view.get("instrument_type")),
        underlying_symbol=optional_text(view.get("underlying_symbol")),
        expiration_date=optional_text(view.get("expiration_date")),
        currency=optional_text(view.get("currency")),
        tick_size_type=optional_text(view.get("tick_size_type")),
        price_display_format=optional_text(view.get("price_display_format")),
        is_tradable=optional_text(view.get("is_tradable")),
        minimum_quoted_price_change=optional_decimal(view.get("min_qprice_change")),
        minimum_feed_price_change=optional_decimal(view.get("min_fprice_change")),
        single_point_value=optional_decimal(view.get("single_point_value")),
        quote_to_feed_price_factor=optional_decimal(view.get("qtof_price")),
        feed_to_quote_price_factor=optional_decimal(view.get("ftoq_price")),
        presence_bits=optional_int(view.get("presence_bits"), unsigned=True),
    )


def normalize_tick_size(message: Mapping[str, Any] | Any) -> TickSizeObservation:
    view = FieldView(message)
    return TickSizeObservation(
        tick_size_type=optional_text(view.get("tick_size_type")),
        minimum_feed_price_change=optional_decimal(view.get("min_fprice_change")),
        first_price=optional_decimal(view.get("tick_size_first_price")),
        last_price=optional_decimal(view.get("tick_size_last_price")),
        first_price_operator=optional_text(view.get("tick_size_fp_operator")),
        last_price_operator=optional_text(view.get("tick_size_lp_operator")),
        presence_bits=optional_int(view.get("presence_bits"), unsigned=True),
    )
