from datetime import datetime, timezone

from app.copier.engine import CopyEngine, CopyTarget
from app.copier.models import MasterExecutionEvent
from app.copier.risk import RiskPolicy, RiskUsage, validate_copy
from app.copier.sizing import SizingPolicy, master_account_trade_pct, size_child_order


class FakeTradingClient:
    def __init__(self, fail=False):
        self.orders = []
        self.fail = fail

    def place_equity_order(self, account_id, order):
        if self.fail:
            raise RuntimeError("broker unavailable")
        self.orders.append((account_id, order))
        return {"ok": True, "client_order_id": order.client_order_id, "order_id": "child-order-1"}


def _master(side="BUY"):
    return MasterExecutionEvent(
        broker="webull",
        account_id="master",
        execution_id="exec-1",
        order_id="order-1",
        client_order_id="master-client",
        symbol="AAPL",
        side=side,
        quantity=10,
        price=25.0,
        executed_at=datetime(2026, 5, 16, 14, 30, tzinfo=timezone.utc),
        raw_payload={},
    )


def test_master_execution_event_normalizes_webull_payload():
    event = MasterExecutionEvent.from_webull_payload(
        {
            "account_id": "acct-master",
            "order_id": "order-1",
            "client_order_id": "client-1",
            "symbol": "aapl",
            "side": "buy",
            "filled_qty": "3",
            "avg_fill_price": "12.50",
            "filled_at": "2026-05-16T14:30:00+00:00",
        }
    )

    assert event.symbol == "AAPL"
    assert event.side == "BUY"
    assert event.quantity == 3.0
    assert event.price == 12.5
    assert event.execution_id == "order-1"


def test_sizing_fixed_multiplier_and_percent_equity():
    master = _master()

    assert size_child_order(master, SizingPolicy(mode="fixed_multiplier", value=0.5)) == 5.0
    assert size_child_order(
        master,
        SizingPolicy(mode="percent_equity", value=0),
        master_equity=100_000,
        target_equity=25_000,
    ) == 2.0
    assert size_child_order(
        master,
        SizingPolicy(mode="equity_ratio", value=0),
        master_equity=100_000,
        target_equity=25_000,
    ) == 2.0
    assert master_account_trade_pct(master, 5_000) == 0.05
    assert size_child_order(master, SizingPolicy(mode="disabled", value=1)) == 0.0


def test_percent_equity_sizing_matches_master_account_percentage():
    master = _master()
    # Master buys $250 of a $5,000 account = 5%. Target is $2,000, so target notional is $100.
    assert size_child_order(
        master,
        SizingPolicy(mode="percent_equity"),
        master_equity=5_000,
        target_equity=2_000,
    ) == 4.0


def test_risk_blocks_kill_switch_and_notional():
    master = _master()

    assert validate_copy(master, 1, RiskPolicy(global_kill_switch=True)).reason == "global_kill_switch"
    assert (
        validate_copy(master, 10, RiskPolicy(global_kill_switch=False, max_notional_per_trade=100)).reason
        == "max_notional_exceeded"
    )
    assert (
        validate_copy(
            master,
            5,
            RiskPolicy(global_kill_switch=False, max_position_pct=0.10),
            target_equity=1_000,
        ).reason
        == "max_position_pct_exceeded"
    )
    assert (
        validate_copy(
            master,
            5,
            RiskPolicy(global_kill_switch=False, max_daily_notional=500),
            usage=RiskUsage(daily_notional=400),
        ).reason
        == "max_daily_notional_exceeded"
    )
    assert (
        validate_copy(
            master,
            5,
            RiskPolicy(global_kill_switch=False, max_daily_trades=3),
            usage=RiskUsage(daily_orders=3),
        ).reason
        == "max_daily_trades_exceeded"
    )
    assert (
        validate_copy(
            master,
            5,
            RiskPolicy(global_kill_switch=False, max_orders_per_minute=2),
            usage=RiskUsage(minute_orders=2),
        ).reason
        == "max_orders_per_minute_exceeded"
    )


def test_risk_allows_sell_when_target_has_long_position():
    decision = validate_copy(
        _master(side="SELL"),
        4,
        RiskPolicy(global_kill_switch=False, shorting_enabled=False),
        target_position_qty=4,
    )

    assert decision.allowed is True


def test_risk_blocks_sell_that_would_create_short_without_permission():
    decision = validate_copy(
        _master(side="SELL"),
        5,
        RiskPolicy(global_kill_switch=False, shorting_enabled=False),
        target_position_qty=4,
    )

    assert decision.allowed is False
    assert decision.reason == "short_copy_disabled"


def test_copy_engine_places_market_order_for_allowed_target():
    client = FakeTradingClient()
    engine = CopyEngine()
    target = CopyTarget(
        name="personal",
        account_id="copy-acct",
        client=client,
        sizing=SizingPolicy(mode="fixed_multiplier", value=0.5),
        risk=RiskPolicy(enabled=True, global_kill_switch=False, max_notional_per_trade=1_000),
    )

    results = engine.copy_execution(_master(), [target])

    assert len(results) == 1
    assert results[0].allowed is True
    assert results[0].submitted is True
    assert results[0].quantity == 5.0
    assert results[0].client_order_id is not None
    assert results[0].order is not None
    assert results[0].latency_ms is not None
    assert len(client.orders) == 1
    account_id, order = client.orders[0]
    assert account_id == "copy-acct"
    assert order.symbol == "AAPL"
    assert order.side == "BUY"


def test_copy_engine_plan_execution_does_not_place_order():
    client = FakeTradingClient()
    engine = CopyEngine()
    target = CopyTarget(
        name="personal",
        account_id="copy-acct",
        client=client,
        sizing=SizingPolicy(mode="fixed_multiplier", value=0.5),
        risk=RiskPolicy(enabled=True, global_kill_switch=False, max_notional_per_trade=1_000),
    )

    results = engine.plan_execution(_master(), [target])

    assert results[0].allowed is True
    assert results[0].submitted is False
    assert results[0].reason == "read_only"
    assert results[0].quantity == 5.0
    assert results[0].client_order_id is not None
    assert results[0].order is not None
    assert len(client.orders) == 0


def test_copy_engine_places_sell_order_for_position_exit():
    client = FakeTradingClient()
    engine = CopyEngine()
    target = CopyTarget(
        name="personal",
        account_id="copy-acct",
        client=client,
        sizing=SizingPolicy(mode="fixed_quantity", value=4),
        risk=RiskPolicy(enabled=True, global_kill_switch=False, shorting_enabled=False),
        positions={"AAPL": 4},
    )

    results = engine.copy_execution(_master(side="SELL"), [target])

    assert results[0].allowed is True
    assert results[0].submitted is True
    assert client.orders[0][1].side == "SELL"


def test_copy_engine_blocks_sell_when_no_position_and_shorts_disabled():
    client = FakeTradingClient()
    engine = CopyEngine()
    target = CopyTarget(
        name="personal",
        account_id="copy-acct",
        client=client,
        sizing=SizingPolicy(mode="fixed_quantity", value=1),
        risk=RiskPolicy(enabled=True, global_kill_switch=False, shorting_enabled=False),
        positions={},
    )

    results = engine.copy_execution(_master(side="SELL"), [target])

    assert results[0].submitted is False
    assert results[0].reason == "short_copy_disabled"
    assert len(client.orders) == 0


def test_copy_engine_handles_multiple_targets_independently():
    allowed_client = FakeTradingClient()
    blocked_client = FakeTradingClient()
    engine = CopyEngine()

    targets = [
        CopyTarget(
            name="personal",
            account_id="copy-1",
            client=allowed_client,
            sizing=SizingPolicy(mode="fixed_quantity", value=2),
            risk=RiskPolicy(enabled=True, global_kill_switch=False, max_notional_per_trade=1_000),
        ),
        CopyTarget(
            name="future",
            account_id="copy-2",
            client=blocked_client,
            sizing=SizingPolicy(mode="fixed_quantity", value=2),
            risk=RiskPolicy(enabled=False, global_kill_switch=False, max_notional_per_trade=1_000),
        ),
    ]

    results = engine.copy_execution(_master(), targets)

    assert [result.allowed for result in results] == [True, False]
    assert results[1].reason == "target_disabled"
    assert len(allowed_client.orders) == 1
    assert len(blocked_client.orders) == 0


def test_copy_engine_isolates_broker_submit_failure():
    failing_client = FakeTradingClient(fail=True)
    healthy_client = FakeTradingClient()
    engine = CopyEngine()

    targets = [
        CopyTarget(
            name="personal",
            account_id="copy-1",
            client=failing_client,
            sizing=SizingPolicy(mode="fixed_quantity", value=2),
            risk=RiskPolicy(enabled=True, global_kill_switch=False, max_notional_per_trade=1_000),
        ),
        CopyTarget(
            name="backup",
            account_id="copy-2",
            client=healthy_client,
            sizing=SizingPolicy(mode="fixed_quantity", value=2),
            risk=RiskPolicy(enabled=True, global_kill_switch=False, max_notional_per_trade=1_000),
        ),
    ]

    results = engine.copy_execution(_master(), targets)

    assert [result.submitted for result in results] == [False, True]
    assert results[0].reason == "broker_submit_failed"
    assert results[0].error == "broker unavailable"
    assert len(healthy_client.orders) == 1


def test_copy_engine_blocks_when_daily_usage_limit_is_reached():
    client = FakeTradingClient()
    engine = CopyEngine()
    target = CopyTarget(
        name="personal",
        account_id="copy-1",
        client=client,
        sizing=SizingPolicy(mode="fixed_quantity", value=2),
        risk=RiskPolicy(enabled=True, global_kill_switch=False, max_daily_notional=100),
        risk_usage=RiskUsage(daily_notional=75),
    )

    results = engine.copy_execution(_master(), [target])

    assert results[0].submitted is False
    assert results[0].reason == "max_daily_notional_exceeded"
    assert results[0].client_order_id is not None
    assert len(client.orders) == 0


def test_copy_engine_plan_execution_records_blocked_decision_without_order():
    client = FakeTradingClient()
    engine = CopyEngine()
    target = CopyTarget(
        name="personal",
        account_id="copy-1",
        client=client,
        sizing=SizingPolicy(mode="fixed_quantity", value=2),
        risk=RiskPolicy(enabled=True, global_kill_switch=False, max_daily_notional=100),
        risk_usage=RiskUsage(daily_notional=75),
    )

    results = engine.plan_execution(_master(), [target])

    assert results[0].allowed is False
    assert results[0].submitted is False
    assert results[0].reason == "max_daily_notional_exceeded"
    assert results[0].client_order_id is not None
    assert len(client.orders) == 0
