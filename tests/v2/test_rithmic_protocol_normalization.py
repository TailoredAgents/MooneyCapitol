from __future__ import annotations

from decimal import Decimal

from app.v2.brokers.rithmic_protocol.constants import Template
from app.v2.brokers.rithmic_protocol.normalization import (
    ExecutionEffect,
    OrderLifecycleState,
    SourceKind,
    normalize_account,
    normalize_account_pnl,
    normalize_account_rms,
    normalize_bracket,
    normalize_instrument_pnl,
    normalize_login_info,
    normalize_order,
    normalize_product_rms,
    normalize_reference,
    normalize_rms_update,
    normalize_tick_size,
)


ACCOUNT = {"fcm_id": "f", "ib_id": "i", "account_id": "a"}


def test_account_and_login_discovery_preserve_opaque_ids_and_metadata():
    login = normalize_login_info(
        {
            "fcm_id": "f",
            "ib_id": "i",
            "user": "secret-user",
            "user_type": 3,
            "order_copy_status": "enabled",
            "tp_max_session_count": 2,
            "op_max_session_count": 1,
            "email_address": "private@example.test",
        }
    )
    assert login.user_type == "USER_TYPE_TRADER"
    assert login.user_type_code == 3
    assert login.order_session_max == 1
    assert login.sensitive_metadata["email_address"] == "private@example.test"
    current_user = normalize_login_info(
        {
            "fcm_id": "f",
            "ib_id": "i",
            "user": "secret-user",
            "type": "3",
            "status": "enabled",
        }
    )
    assert current_user.user_type == "USER_TYPE_TRADER"
    assert current_user.user_type_code == 3
    assert current_user.status == "enabled"
    account = normalize_account(
        {
            **ACCOUNT,
            "account_name": "Test",
            "account_currency": "USD",
            "loss_limit": "1500.25",
            "account_auto_liquidate": "enabled",
        }
    )
    assert account.key.account_id == "a"
    assert account.loss_limit == Decimal("1500.25")


def test_order_normalization_uses_64_bit_fields_and_preserves_origin_and_sequences():
    quantity = 2**40
    observation = normalize_order(
        {
            "template_id": 352,
            **ACCOUNT,
            "notify_type": 5,
            "is_snapshot": False,
            "basket_id": "basket",
            "exchange_order_id": "exchange",
            "tp_exchange_order_id": "ticker-exchange",
            "fill_id": "fill-1",
            "report_type": "fill",
            "quantity_64": quantity,
            "fill_size_64": quantity - 2,
            "total_fill_size_64": quantity - 2,
            "total_unfilled_size_64": 2,
            "fill_price": "20123.25",
            "avg_fill_price": "20123.125",
            "sequence_number": "00000000000000000009",
            "orig_sequence_number": "7",
            "cor_sequence_number": "8",
            "application": "R|Trader",
            "originator_application": "R|Trader",
            "manual_or_auto": 1,
            "user_tag": "opaque-tag",
            "source_ssboe": 1,
            "source_nsecs": 999,
        }
    )
    assert observation.quantity == quantity
    assert observation.fill_size == quantity - 2
    assert observation.normalized_state is OrderLifecycleState.PARTIALLY_FILLED
    assert observation.execution_effect is ExecutionEffect.APPLY_FILL
    assert observation.origin.manual_or_auto == "MANUAL"
    assert observation.origin.application == "R|Trader"
    assert observation.sequence_number == "00000000000000000009"
    assert observation.fill_price == Decimal("20123.25")


def test_fill_complete_bust_correction_command_failure_and_unknown_are_not_flattened():
    filled = normalize_order(
        {
            "template_id": 352,
            **ACCOUNT,
            "notify_type": "FILL",
            "report_type": "fill",
            "fill_size_64": 1,
            "total_fill_size_64": 1,
            "total_unfilled_size_64": 0,
        }
    )
    assert filled.normalized_state is OrderLifecycleState.FILLED

    bust = normalize_order(
        {
            "template_id": 352,
            **ACCOUNT,
            "notify_type": 5,
            "report_type": "bust",
            "fill_size_64": 1,
            "total_unfilled_size_64": 0,
        }
    )
    assert bust.execution_effect is ExecutionEffect.BUST_FILL
    assert bust.normalized_state is None

    corrected = normalize_order(
        {"template_id": 352, **ACCOUNT, "notify_type": 5, "report_type": "trade correct"}
    )
    assert corrected.execution_effect is ExecutionEffect.CORRECT_FILL
    assert corrected.normalized_state is None

    command_failure = normalize_order(
        {"template_id": 351, **ACCOUNT, "notify_type": 17, "basket_id": "b"}
    )
    assert command_failure.command_failure and command_failure.normalized_state is None

    unknown_complete = normalize_order(
        {
            "template_id": 351,
            **ACCOUNT,
            "notify_type": 15,
            "completion_reason": "future-code",
        }
    )
    assert unknown_complete.normalized_state is OrderLifecycleState.UNKNOWN


def test_completion_reason_controls_complete_instead_of_blindly_marking_filled():
    expected = {
        "F": OrderLifecycleState.FILLED,
        "C": OrderLifecycleState.CANCELLED,
        "PFBC": OrderLifecycleState.CANCELLED,
        "R": OrderLifecycleState.REJECTED,
        "FA": OrderLifecycleState.REJECTED,
        "U": OrderLifecycleState.UNKNOWN,
    }
    for reason, state in expected.items():
        observation = normalize_order(
            {"template_id": 351, **ACCOUNT, "notify_type": "COMPLETE", "completion_reason": reason}
        )
        assert observation.normalized_state is state


def test_fill_history_uint64_field_name_and_history_source_are_supported():
    observation = normalize_order(
        {
            "template_id": 3513,
            **ACCOUNT,
            "basket_id": "b",
            "fill_id": "f",
            "report_type": "fill",
            "fill_size": 2**39,
            "total_fill_size": 2**39,
            "total_unfilled_size": 0,
        }
    )
    assert observation.source_kind is SourceKind.HISTORY
    assert observation.fill_size == 2**39


def test_pnl_fields_are_distinct_decimals_and_not_invented_equity():
    instrument = normalize_instrument_pnl(
        {
            **ACCOUNT,
            "is_snapshot": True,
            "symbol": "NQZ6",
            "exchange": "CME",
            "net_quantity": -2,
            "order_buy_qty": 3,
            "order_sell_qty": 4,
            "avg_open_fill_price": "20000.25",
            "open_position_pnl": "12.34",
            "closed_position_pnl": "56.78",
            "day_pnl": "69.12",
        }
    )
    assert instrument.source_kind is SourceKind.SNAPSHOT
    assert instrument.net_quantity == -2
    assert instrument.average_open_fill_price == Decimal("20000.25")
    assert instrument.open_position_pnl == Decimal("12.34")

    account = normalize_account_pnl(
        {
            **ACCOUNT,
            "is_snapshot": False,
            "account_balance": "10000.10",
            "cash_on_hand": "9000.20",
            "margin_balance": "8000.30",
            "available_buying_power": "7000.40",
            "used_buying_power": "1000.50",
            "reserved_buying_power": "200.60",
            "day_open_pnl": "1.1",
            "day_closed_pnl": "2.2",
            "day_pnl": "3.3",
        }
    )
    assert account.account_balance == Decimal("10000.10")
    assert account.available_buying_power == Decimal("7000.40")
    assert not hasattr(account, "equity")


def test_rms_bracket_and_reference_observation_preserve_official_facts():
    account_rms = normalize_account_rms(
        {
            **ACCOUNT,
            "currency": "USD",
            "status": "enabled",
            "auto_liquidate": 1,
            "disable_on_auto_liquidate": 2,
            "loss_limit": "2500",
            "max_order_quantity": 10,
        }
    )
    assert account_rms.auto_liquidate == "ENABLED"
    assert account_rms.loss_limit == Decimal("2500")
    product_rms = normalize_product_rms(
        {
            **ACCOUNT,
            "product_code": "NQ",
            "buy_margin_rate": "1000.25",
            "sell_margin_rate": "1001.25",
            "commission_fill_rate": "2.50",
        }
    )
    assert product_rms.buy_margin_rate == Decimal("1000.25")
    update = normalize_rms_update(
        {
            **ACCOUNT,
            "auto_liq_threshold_current_value": "500.5",
            "auto_liq_peak_account_balance": "12000",
        }
    )
    assert update.current_auto_liquidate_threshold == Decimal("500.5")

    bracket = normalize_bracket(
        {
            **ACCOUNT,
            "basket_id": "parent",
            "stop_ticks": "20",
            "stop_quantity": "2",
            "stop_quantity_released": "1",
            "target_ticks": "40",
            "target_quantity": "2",
            "target_quantity_released": "1",
            "trailing_stop_trigger_ticks": "5",
        }
    )
    assert bracket.parent_basket_id == "parent"
    assert bracket.stop_quantity_released == 1
    assert bracket.trailing_stop_trigger_ticks == 5

    reference = normalize_reference(
        {
            "symbol": "NQZ6",
            "exchange": "CME",
            "trading_symbol": "NQZ6",
            "trading_exchange": "CME",
            "product_code": "NQ",
            "expiration_date": "20261218",
            "min_qprice_change": "0.25",
            "min_fprice_change": "0.25",
            "single_point_value": "20",
            "is_tradable": "yes",
        }
    )
    assert reference.minimum_quoted_price_change == Decimal("0.25")
    assert reference.single_point_value == Decimal("20")
    tick = normalize_tick_size(
        {"tick_size_type": "x", "min_fprice_change": "0.25", "tick_size_first_price": "0"}
    )
    assert tick.minimum_feed_price_change == Decimal("0.25")
