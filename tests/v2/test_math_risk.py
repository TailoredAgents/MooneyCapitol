from datetime import date
from decimal import Decimal
import random

from app.v2.domain.models import PositionSide, Side
from app.v2.math import (
    FillLot,
    PositionTransitionType,
    RoundingPolicy,
    contract_risk,
    equivalent_contracts,
    mae,
    mfe,
    planned_risk_dollars,
    realized_gross_pnl,
    realized_net_pnl,
    realized_r,
    split_reversal,
    apply_position_fill,
    weighted_entry,
    weighted_exit,
)
from app.v2.reference import FuturesContractRegistry, MNQ_SPECIFICATION, NQ_SPECIFICATION
from app.v2.risk import InstrumentSelection, RiskLimits, RiskState, SizingCandidate, SizingMode, evaluate_futures_risk


D = Decimal


def _contracts():
    registry = FuturesContractRegistry()
    expiry = date(2027, 3, 19)
    return (
        registry.register_contract(contract_id="CME:NQ:2027-03-19", product_code="NQ", expiration=expiry),
        registry.register_contract(contract_id="CME:MNQ:2027-03-19", product_code="MNQ", expiration=expiry),
    )


def test_decimal_math_for_risk_vwap_scale_out_costs_and_excursions():
    assert contract_risk(D("20000"), D("19990"), NQ_SPECIFICATION) == D("200")
    assert planned_risk_dollars(D("20000"), None, 2, NQ_SPECIFICATION) is None
    assert planned_risk_dollars(D("20000"), D("19990"), 2, NQ_SPECIFICATION, cost_buffer_per_contract=D("3")) == D("406")
    entries = [FillLot(D("100"), 2, D("1")), FillLot(D("101"), 1, D("0.5"))]
    exits = [FillLot(D("102"), 1, D("0.5")), FillLot(D("103"), 2, D("1"))]
    assert weighted_entry(entries) == D("301") / D("3")
    assert weighted_exit(exits) == D("308") / D("3")
    gross = realized_gross_pnl(PositionSide.LONG, entries, exits, NQ_SPECIFICATION)
    assert gross == D("140")
    net = realized_net_pnl(gross, entries + exits, extra_cost_buffer=D("2"))
    assert net == D("135")
    assert realized_r(net, D("270")) == D("0.5")
    assert mfe(PositionSide.LONG, D("100"), D("103"), 2, NQ_SPECIFICATION, D("100")).ticks == D("12")
    assert mae(PositionSide.SHORT, D("100"), D("102"), 1, NQ_SPECIFICATION, D("100")).dollars == D("40")


def test_partial_exit_add_and_reversal_are_explicit():
    assert realized_gross_pnl(
        PositionSide.LONG,
        [FillLot(D("100"), 1), FillLot(D("99"), 1)],
        [FillLot(D("101"), 1)],
        NQ_SPECIFICATION,
    ) == D("30")
    assert split_reversal(2, Side.SELL, 5).close_quantity == 2
    assert split_reversal(2, Side.SELL, 5).open_quantity == 3
    assert split_reversal(2, Side.SELL, 5).resulting_signed_quantity == -3
    assert apply_position_fill(0, Side.BUY, 2).transition == PositionTransitionType.OPEN
    assert apply_position_fill(2, Side.BUY, 1).transition == PositionTransitionType.ADD
    assert apply_position_fill(3, Side.SELL, 1).transition == PositionTransitionType.PARTIAL_EXIT
    assert apply_position_fill(3, Side.SELL, 3).transition == PositionTransitionType.CLOSE
    assert apply_position_fill(3, Side.SELL, 4).transition == PositionTransitionType.REVERSAL


def test_nq_mnq_equivalence_never_exceeds_source_risk():
    assert equivalent_contracts(1, NQ_SPECIFICATION, MNQ_SPECIFICATION, D("10")) == 10
    assert equivalent_contracts(1, MNQ_SPECIFICATION, NQ_SPECIFICATION, D("10")) == 0


def test_auto_selection_preserves_expiry_and_stays_under_hard_budget():
    nq, mnq = _contracts()
    limits = RiskLimits(
        sizing_mode=SizingMode.FIXED_DOLLAR,
        sizing_value=D("150"),
        max_risk_per_trade=D("150"),
        max_contracts=20,
        max_concurrent_open_risk=D("500"),
        per_product_max_contracts={"NQ": 3, "MNQ": 20},
        daily_realized_loss_ceiling=D("500"),
        daily_total_loss_ceiling=D("600"),
        instrument_selection=InstrumentSelection.AUTO_NQ_MNQ,
    )
    state = RiskState(D("100000"), D("10000"), global_kill_switch=False, account_kill_switch=False)
    decision = evaluate_futures_risk(
        limits,
        state,
        [SizingCandidate(nq, D("200"), D("1000")), SizingCandidate(mnq, D("20"), D("100"))],
    )
    assert decision.allowed and decision.contract_id == mnq.contract_id
    assert decision.contracts == 7 and decision.economic_risk == D("140")
    assert decision.economic_risk <= limits.max_risk_per_trade


def test_risk_property_never_exceeds_ceiling_across_rounding_modes():
    _, mnq = _contracts()
    rng = random.Random(41)
    state = RiskState(D("50000"), D("100000"), global_kill_switch=False, account_kill_switch=False)
    for _ in range(250):
        ceiling = D(rng.randint(1, 5000))
        unit = D(rng.randint(1, 500))
        policy = rng.choice(list(RoundingPolicy))
        limits = RiskLimits(
            sizing_mode=SizingMode.FIXED_DOLLAR,
            sizing_value=ceiling,
            max_risk_per_trade=ceiling,
            max_contracts=1000,
            max_concurrent_open_risk=ceiling,
            daily_realized_loss_ceiling=D("10000"),
            daily_total_loss_ceiling=D("10000"),
            rounding=policy,
        )
        decision = evaluate_futures_risk(limits, state, [SizingCandidate(mnq, unit, D("0"))])
        assert not decision.allowed or decision.economic_risk <= ceiling


def test_risk_fail_closed_switches_losses_and_missing_ceilings():
    _, mnq = _contracts()
    base = dict(
        sizing_mode=SizingMode.FIXED_DOLLAR,
        sizing_value=D("100"),
        max_risk_per_trade=D("100"),
        max_contracts=10,
        max_concurrent_open_risk=D("100"),
        daily_realized_loss_ceiling=D("100"),
        daily_total_loss_ceiling=D("100"),
    )
    candidate = [SizingCandidate(mnq, D("20"), D("0"))]
    assert evaluate_futures_risk(RiskLimits(**base), RiskState(D("1"), D("1")), candidate).reason == "global_kill_switch"
    no_ceiling = RiskLimits(**{**base, "max_risk_per_trade": D("0")})
    state = RiskState(D("1"), D("1"), global_kill_switch=False, account_kill_switch=False)
    assert evaluate_futures_risk(no_ceiling, state, candidate).reason == "max_risk_per_trade_not_configured"
    loss_state = RiskState(D("1"), D("1"), daily_realized_pnl=D("-100"), global_kill_switch=False, account_kill_switch=False)
    assert evaluate_futures_risk(RiskLimits(**base), loss_state, candidate).reason == "daily_realized_loss_ceiling_reached"
    drawdown_state = RiskState(D("1"), D("1"), daily_drawdown=D("100"), global_kill_switch=False, account_kill_switch=False)
    assert evaluate_futures_risk(RiskLimits(**base), drawdown_state, candidate).reason == "daily_total_loss_ceiling_reached"
