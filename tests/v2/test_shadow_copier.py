from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from app.v2.domain.models import Side
from app.v2.execution.copier import (
    CopiedPositionState,
    CopierMode,
    CumulativeFollowerState,
    FollowerContractCandidate,
    FollowerPolicy,
    FollowerState,
    MasterTrade,
    MasterPositionTarget,
    PositionPlanAction,
    plan_cumulative_copy,
    plan_shadow_copy,
)
from app.v2.execution.copier_config import ShadowCopierConfig
from app.v2.reference import FuturesContractRegistry


D = Decimal


def _contracts():
    registry = FuturesContractRegistry()
    expiration = date(2027, 3, 19)
    nq = registry.register_contract(
        contract_id="CME:NQ:2027-03-19",
        product_code="NQ",
        expiration=expiration,
    )
    mnq = registry.register_contract(
        contract_id="CME:MNQ:2027-03-19",
        product_code="MNQ",
        expiration=expiration,
    )
    registry.map_same_expiry(nq.contract_id, mnq.contract_id)
    return nq, mnq


def _policy(**overrides) -> FollowerPolicy:
    values = {
        "max_risk_fraction": D("0.10"),
        "max_contracts": 20,
        "max_concurrent_open_risk_fraction": D("0.20"),
        "daily_realized_loss_fraction": D("0.10"),
        "daily_total_loss_fraction": D("0.15"),
        "max_relative_parity_error": D("0"),
        "margin_headroom_fraction": D("0.20"),
        "allow_over_target": False,
    }
    values.update(overrides)
    return FollowerPolicy(**values)


def _candidates(nq, mnq):
    return (
        FollowerContractCandidate(nq, D("1000")),
        FollowerContractCandidate(mnq, D("100")),
    )


def test_one_nq_to_one_mnq_exactly_matches_ten_to_one_equity() -> None:
    nq, mnq = _contracts()
    master = MasterTrade(
        account_id="master-secret",
        contract=nq,
        side=Side.BUY,
        quantity=1,
        entry_price=D("20000"),
        stop_price=D("19980"),
        equity=D("5000"),
    )
    follower = FollowerState(
        account_id="follower-secret",
        equity=D("500"),
        available_margin=D("500"),
        account_kill_switch=False,
    )

    plan = plan_shadow_copy(
        master,
        [follower],
        _candidates(nq, mnq),
        _policy(),
        global_kill_switch=False,
    )

    item = plan.followers[0]
    assert plan.submission_enabled is False
    assert item.allowed is True
    assert item.product_code == "MNQ"
    assert item.quantity == 1
    assert item.target_risk_fraction == D("0.08")
    assert item.actual_risk_fraction == D("0.08")
    assert item.parity_error == D("0")
    assert item.would_submit is False


def test_underfunded_follower_reports_untradeable_contract_granularity() -> None:
    nq, mnq = _contracts()
    master = MasterTrade(
        account_id="master",
        contract=nq,
        side=Side.BUY,
        quantity=1,
        entry_price=D("20000"),
        stop_price=D("19980"),
        equity=D("30000"),
    )
    follower = FollowerState(
        account_id="small-follower",
        equity=D("500"),
        available_margin=D("500"),
        account_kill_switch=False,
    )

    plan = plan_shadow_copy(
        master,
        [follower],
        _candidates(nq, mnq),
        _policy(),
        global_kill_switch=False,
    )

    item = plan.followers[0]
    assert item.allowed is False
    assert item.reason == "minimum_contract_exceeds_target"
    assert item.product_code == "MNQ"
    assert item.quantity == 1
    assert item.over_target is True
    assert item.parity_error == D("5")
    assert item.target_risk_dollars == follower.equity * master.risk_fraction


def test_nearest_mode_can_report_over_target_but_tolerance_still_controls() -> None:
    nq, mnq = _contracts()
    master = MasterTrade(
        account_id="master",
        contract=nq,
        side=Side.SELL,
        quantity=1,
        entry_price=D("20000"),
        stop_price=D("20025"),
        equity=D("100000"),
    )
    follower = FollowerState(
        account_id="follower",
        equity=D("27000"),
        available_margin=D("10000"),
        account_kill_switch=False,
    )

    under = plan_shadow_copy(
        master,
        [follower],
        _candidates(nq, mnq),
        _policy(max_relative_parity_error=D("0.30")),
        global_kill_switch=False,
    ).followers[0]
    nearest = plan_shadow_copy(
        master,
        [follower],
        _candidates(nq, mnq),
        _policy(
            max_relative_parity_error=D("0.12"),
            allow_over_target=True,
        ),
        global_kill_switch=False,
    ).followers[0]

    assert under.product_code == "MNQ" and under.quantity == 2
    assert under.over_target is False
    assert under.allowed is True
    assert nearest.product_code == "MNQ" and nearest.quantity == 3
    assert nearest.over_target is True
    assert nearest.allowed is True
    assert nearest.parity_error < under.parity_error


def test_over_target_policy_applies_to_exposure_even_when_risk_is_lower() -> None:
    nq, mnq = _contracts()
    master = MasterTrade(
        account_id="master",
        contract=nq,
        side=Side.BUY,
        quantity=1,
        entry_price=D("20000"),
        stop_price=D("19999"),
        equity=D("10000"),
        cost_buffer_per_contract=D("180"),
    )
    follower = FollowerState(
        account_id="follower",
        equity=D("900"),
        available_margin=D("500"),
        account_kill_switch=False,
    )

    plan = plan_shadow_copy(
        master,
        [follower],
        [FollowerContractCandidate(mnq, D("100"))],
        _policy(max_relative_parity_error=D("1")),
        global_kill_switch=False,
    ).followers[0]

    assert plan.actual_risk_fraction < plan.target_risk_fraction
    assert plan.actual_point_exposure_fraction > plan.target_point_exposure_fraction
    assert plan.allowed is False
    assert plan.over_target is True
    assert plan.reason == "minimum_contract_exceeds_target"


def test_shadow_planner_fails_closed_on_switches_and_account_limits() -> None:
    nq, mnq = _contracts()
    master = MasterTrade(
        account_id="master",
        contract=nq,
        side=Side.BUY,
        quantity=1,
        entry_price=D("20000"),
        stop_price=D("19980"),
        equity=D("5000"),
    )
    follower = FollowerState(
        account_id="follower",
        equity=D("500"),
        available_margin=D("500"),
        account_kill_switch=False,
    )

    globally_blocked = plan_shadow_copy(
        master,
        [follower],
        _candidates(nq, mnq),
        _policy(),
    ).followers[0]
    account_blocked = plan_shadow_copy(
        master,
        [FollowerState("follower", D("500"), D("500"))],
        _candidates(nq, mnq),
        _policy(),
        global_kill_switch=False,
    ).followers[0]
    hard_limit = plan_shadow_copy(
        master,
        [follower],
        _candidates(nq, mnq),
        _policy(max_risk_fraction=D("0.05")),
        global_kill_switch=False,
    ).followers[0]

    assert globally_blocked.reason == "global_kill_switch"
    assert account_blocked.reason == "account_kill_switch"
    assert hard_limit.reason == "master_risk_exceeds_account_limit"


def test_master_stop_and_same_expiry_are_mandatory() -> None:
    nq, mnq = _contracts()
    with pytest.raises(ValueError, match="stop below entry"):
        MasterTrade(
            "master",
            nq,
            Side.BUY,
            1,
            D("20000"),
            D("20000"),
            D("5000"),
        )

    later = FuturesContractRegistry().register_contract(
        contract_id="CME:MNQ:2027-06-18",
        product_code="MNQ",
        expiration=date(2027, 6, 18),
    )
    master = MasterTrade(
        "master",
        nq,
        Side.BUY,
        1,
        D("20000"),
        D("19980"),
        D("5000"),
    )
    with pytest.raises(ValueError, match="preserve the master contract expiry"):
        plan_shadow_copy(
            master,
            [FollowerState("follower", D("500"), D("500"), account_kill_switch=False)],
            [FollowerContractCandidate(later, D("100"))],
            _policy(),
            global_kill_switch=False,
        )


def test_shadow_config_has_fixed_topology_and_requires_hard_limits() -> None:
    empty = ShadowCopierConfig.from_mapping({})
    assert empty.effective_mode == CopierMode.SHADOW
    assert empty.ready_for_shadow is False
    assert empty.topology.ready_for_simulation is True
    assert len(empty.topology.followers) == 4
    assert "current_broker_margins_required" in empty.preflight_blockers

    ready = ShadowCopierConfig.from_mapping(
        {
            "V2_COPIER_MODE": "SHADOW",
            "V2_COPIER_MAX_RISK_FRACTION": "0.02",
            "V2_COPIER_MAX_CONTRACTS": "5",
            "V2_COPIER_MAX_CONCURRENT_RISK_FRACTION": "0.05",
            "V2_COPIER_DAILY_REALIZED_LOSS_FRACTION": "0.03",
            "V2_COPIER_DAILY_TOTAL_LOSS_FRACTION": "0.04",
            "V2_COPIER_MAX_RELATIVE_PARITY_ERROR": "0.05",
            "V2_COPIER_MARGIN_HEADROOM_FRACTION": "0.20",
            "V2_COPIER_NQ_INITIAL_MARGIN": "1000",
            "V2_COPIER_MNQ_INITIAL_MARGIN": "100",
        }
    )
    assert ready.ready_for_shadow is True
    assert ready.preflight_blockers == ()
    assert ready.follower_policy().max_contracts == 5
    assert ready.topology.ready_for_test_connectivity is False

    live = ShadowCopierConfig.from_mapping({"V2_COPIER_MODE": "LIVE"})
    assert live.effective_mode == CopierMode.OFF
    assert "broker_submission_not_implemented" in live.preflight_blockers


def _master_position(nq, signed_quantity: int, *, equity: str = "10000"):
    return MasterPositionTarget(
        account_id="master",
        contract=nq,
        signed_quantity=signed_quantity,
        equity=D(equity),
        entry_price=D("20000") if signed_quantity else None,
        stop_price=(
            D("19980")
            if signed_quantity > 0
            else D("20020")
            if signed_quantity < 0
            else None
        ),
    )


def _cumulative_follower(
    contract=None,
    signed_quantity: int = 0,
    *,
    equity: str = "1000",
    available_margin: str = "1000",
    open_risk: str = "0",
    account_kill_switch: bool = False,
):
    copied_risk = D(open_risk)
    return CumulativeFollowerState(
        account=FollowerState(
            account_id="follower",
            equity=D(equity),
            available_margin=D(available_margin),
            current_open_risk=copied_risk,
            account_kill_switch=account_kill_switch,
        ),
        copied_position=CopiedPositionState(
            contract=contract,
            signed_quantity=signed_quantity,
            open_risk_dollars=copied_risk,
        ),
    )


def test_cumulative_sizing_plans_total_position_deltas_for_add_and_reduction() -> None:
    nq, mnq = _contracts()
    candidates = _candidates(nq, mnq)
    added = plan_cumulative_copy(
        _master_position(nq, 2),
        [_cumulative_follower(mnq, 1, open_risk="40")],
        candidates,
        _policy(),
        global_kill_switch=False,
    ).followers[0]
    reduced = plan_cumulative_copy(
        _master_position(nq, 1),
        [_cumulative_follower(mnq, 2, open_risk="80")],
        candidates,
        _policy(),
        global_kill_switch=False,
    ).followers[0]

    assert added.allowed is True
    assert added.action == PositionPlanAction.ADD
    assert added.achievable_target_signed_quantity == 2
    assert added.delta_signed_quantity == 1
    assert added.delta_side == Side.BUY
    assert reduced.allowed is True
    assert reduced.action == PositionPlanAction.REDUCE
    assert reduced.achievable_target_signed_quantity == 1
    assert reduced.delta_signed_quantity == -1
    assert reduced.delta_side == Side.SELL


def test_cumulative_sizing_plans_reversal_from_total_not_event_delta() -> None:
    nq, mnq = _contracts()
    item = plan_cumulative_copy(
        _master_position(nq, -1),
        [_cumulative_follower(mnq, 1, open_risk="40")],
        _candidates(nq, mnq),
        _policy(),
        global_kill_switch=False,
    ).followers[0]

    assert item.allowed is True
    assert item.action == PositionPlanAction.REVERSE
    assert item.evaluated_target_signed_quantity == -1
    assert item.achievable_target_signed_quantity == -1
    assert item.delta_signed_quantity == -2
    assert item.delta_side == Side.SELL
    assert item.delta_quantity == 2


def test_cumulative_sizing_closes_but_never_converts_an_open_contract() -> None:
    nq, mnq = _contracts()
    item = plan_cumulative_copy(
        _master_position(nq, 1),
        [
            _cumulative_follower(
                nq,
                1,
                equity="3000",
                open_risk="400",
            )
        ],
        _candidates(nq, mnq),
        _policy(max_risk_fraction=D("0.20")),
        global_kill_switch=False,
    ).followers[0]

    assert item.allowed is True
    assert item.reason == "no_safe_nonzero_target_close"
    assert item.action == PositionPlanAction.CLOSE
    assert item.evaluated_product_code == "NQ"
    assert item.achievable_target_signed_quantity == 0
    assert item.delta_signed_quantity == -1


def test_cumulative_sizing_closes_followers_when_master_is_flat() -> None:
    nq, mnq = _contracts()
    item = plan_cumulative_copy(
        _master_position(nq, 0),
        [_cumulative_follower(mnq, -2, open_risk="80")],
        _candidates(nq, mnq),
        _policy(),
        global_kill_switch=True,
    ).followers[0]

    assert item.allowed is True
    assert item.reason == "master_flat_close"
    assert item.action == PositionPlanAction.CLOSE
    assert item.delta_signed_quantity == 2
    assert item.delta_side == Side.BUY


def test_risk_gate_still_allows_a_cumulative_reduction() -> None:
    nq, mnq = _contracts()
    item = plan_cumulative_copy(
        _master_position(nq, 1),
        [_cumulative_follower(mnq, 2, open_risk="80")],
        _candidates(nq, mnq),
        _policy(),
        global_kill_switch=True,
    ).followers[0]

    assert item.allowed is True
    assert item.reason == "risk_reducing_plan_allowed"
    assert item.action == PositionPlanAction.REDUCE
    assert item.delta_signed_quantity == -1


def test_funding_report_quantifies_one_mnq_equity_and_margin_shortfalls() -> None:
    nq, mnq = _contracts()
    item = plan_cumulative_copy(
        _master_position(nq, 1, equity="30000"),
        [
            _cumulative_follower(
                equity="500",
                available_margin="50",
            )
        ],
        _candidates(nq, mnq),
        _policy(),
        global_kill_switch=False,
    ).followers[0]

    report = item.funding
    assert report.funding_adequate is False
    assert report.currently_eligible is False
    assert report.one_mnq_risk_dollars == D("40")
    assert report.minimum_equity == D("3000")
    assert report.equity_shortfall == D("2500")
    assert report.minimum_available_margin == D("125")
    assert report.available_margin_shortfall == D("75")
    assert "equity_below_one_mnq_parity" in report.blockers
    assert "one_mnq_margin_capacity_insufficient" in report.blockers


def test_cumulative_sizing_blocks_expiry_changes_until_position_is_flat() -> None:
    nq, mnq = _contracts()
    later_mnq = FuturesContractRegistry().register_contract(
        contract_id="CME:MNQ:2027-06-18",
        product_code="MNQ",
        expiration=date(2027, 6, 18),
    )
    item = plan_cumulative_copy(
        _master_position(nq, 1),
        [_cumulative_follower(later_mnq, 1, open_risk="40")],
        _candidates(nq, mnq),
        _policy(),
        global_kill_switch=False,
    ).followers[0]

    assert item.allowed is False
    assert item.reason == "open_position_expiry_mismatch_no_auto_roll"
    assert item.action == PositionPlanAction.BLOCKED
    assert item.delta_signed_quantity == 0
