from dataclasses import FrozenInstanceError
from datetime import date, datetime
from decimal import Decimal

import pytest

from app.v2.calendar import CmeEquityIndexCalendar, EASTERN, SessionOverride
from app.v2.domain.models import MasterOrder, MasterTrade, NormalizedOrderEvent, OrderStatus, PositionSide, Side, apply_order_event
from app.v2.reference import FuturesContractRegistry, MNQ_SPECIFICATION, NQ_SPECIFICATION


def test_verified_static_product_economics_and_expiry_identity():
    assert NQ_SPECIFICATION.point_value == Decimal("20")
    assert NQ_SPECIFICATION.tick_size == Decimal("0.25")
    assert NQ_SPECIFICATION.tick_value == Decimal("5")
    assert MNQ_SPECIFICATION.point_value == Decimal("2")
    assert MNQ_SPECIFICATION.tick_value == Decimal("0.50")
    registry = FuturesContractRegistry()
    with pytest.raises(ValueError, match="bare product"):
        registry.register_contract(contract_id="NQ", product_code="NQ", expiration=date(2027, 3, 19))


def test_nq_mnq_mapping_must_preserve_expiry_and_does_not_roll():
    registry = FuturesContractRegistry()
    nq = registry.register_contract(contract_id="CME:NQ:2027-03-19", product_code="NQ", expiration=date(2027, 3, 19))
    mnq = registry.register_contract(contract_id="CME:MNQ:2027-03-19", product_code="MNQ", expiration=date(2027, 3, 19))
    other = registry.register_contract(contract_id="CME:MNQ:2027-06-18", product_code="MNQ", expiration=date(2027, 6, 18))
    registry.map_same_expiry(nq.contract_id, mnq.contract_id)
    assert registry.mapped_contract(nq.contract_id, "MNQ") == mnq
    with pytest.raises(ValueError, match="exact expiry"):
        registry.map_same_expiry(nq.contract_id, other.contract_id)


def test_cme_trade_date_spans_prior_evening_and_maintenance_break():
    calendar = CmeEquityIndexCalendar()
    sunday_evening = datetime(2026, 10, 4, 18, 1, tzinfo=EASTERN)
    monday_close = datetime(2026, 10, 5, 17, 0, tzinfo=EASTERN)
    assert calendar.trade_date_at(sunday_evening) == date(2026, 10, 5)
    assert calendar.trade_date_at(monday_close) is None
    schedule = calendar.schedule_for(date(2026, 10, 5))
    assert schedule.maintenance_break_after.start_at == monday_close
    assert schedule.maintenance_break_after.end_at.hour == 18


def test_calendar_accepts_provider_holidays_and_early_closes():
    holiday = SessionOverride(date(2026, 12, 25), None, None, holiday=True, source="provider")
    early = SessionOverride(
        date(2026, 11, 27),
        datetime(2026, 11, 26, 18, tzinfo=EASTERN),
        datetime(2026, 11, 27, 13, 15, tzinfo=EASTERN),
        early_close=True,
        source="provider",
    )
    calendar = CmeEquityIndexCalendar([holiday, early])
    assert calendar.schedule_for(holiday.trade_date).holiday
    assert calendar.schedule_for(early.trade_date).early_close


def test_stale_revision_and_terminal_rollback_are_ignored():
    now = datetime.now(EASTERN)
    current = MasterOrder("o1", "a1", "CME:NQ:2027-03-19", Side.BUY, 1, OrderStatus.FILLED, 5)
    stale = NormalizedOrderEvent("e1", "broker", "a1", "b1", current.contract_id, 4, OrderStatus.WORKING, Side.BUY, 1, now, now)
    rollback = NormalizedOrderEvent("e2", "broker", "a1", "b1", current.contract_id, 6, OrderStatus.WORKING, Side.BUY, 1, now, now)
    assert apply_order_event(current, stale) is current
    assert apply_order_event(current, rollback) is current
    with pytest.raises(FrozenInstanceError):
        current.status = OrderStatus.WORKING


def test_r_metrics_require_the_immutable_original_denominator():
    values = dict(
        trade_id="t1",
        account_id="a1",
        contract_id="CME:NQ:2027-03-19",
        direction=PositionSide.LONG,
        original_plan_id="p1",
        original_stop=Decimal("19990"),
        original_risk_dollars=None,
        entry_vwap=Decimal("20000"),
        exit_vwap=Decimal("20010"),
        initial_quantity=1,
        maximum_quantity=1,
        opened_at=None,
        closed_at=None,
        realized_r=Decimal("1"),
    )
    with pytest.raises(ValueError, match="denominator"):
        MasterTrade(**values)
