from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from app.v2.domain.models import Side
from app.v2.execution.copier import (
    CopierMode,
    FollowerContractCandidate,
    FollowerPolicy,
    FollowerState,
    MasterTrade,
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


def test_shadow_config_requires_one_master_four_followers_and_hard_limits() -> None:
    empty = ShadowCopierConfig.from_mapping({})
    assert empty.effective_mode == CopierMode.SHADOW
    assert empty.ready_for_shadow is False
    assert "master_account_required" in empty.preflight_blockers
    assert "exactly_four_followers_required" in empty.preflight_blockers
    assert "current_broker_margins_required" in empty.preflight_blockers

    ready = ShadowCopierConfig.from_mapping(
        {
            "V2_COPIER_MODE": "SHADOW",
            "V2_COPIER_MASTER_ACCOUNT_ID": "master-secret",
            "V2_COPIER_FOLLOWER_ACCOUNT_IDS": "f1,f2,f3,f4",
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
    assert "master-secret" not in repr(ready)

    live = ShadowCopierConfig.from_mapping(
        {**{
            "V2_COPIER_MODE": "LIVE",
            "V2_COPIER_MASTER_ACCOUNT_ID": "master",
            "V2_COPIER_FOLLOWER_ACCOUNT_IDS": "f1,f2,f3,f4",
        }}
    )
    assert live.effective_mode == CopierMode.OFF
    assert "broker_submission_not_implemented" in live.preflight_blockers
