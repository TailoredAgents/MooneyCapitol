from __future__ import annotations

from dataclasses import dataclass, field, replace
from decimal import Decimal, ROUND_FLOOR
from enum import Enum
from typing import Mapping, Sequence

from app.v2.domain.models import FuturesContract, Side, ZERO
from app.v2.math import contract_risk


ONE = Decimal("1")


class CopierMode(str, Enum):
    OFF = "OFF"
    SHADOW = "SHADOW"
    TEST = "TEST"
    LIVE = "LIVE"


@dataclass(frozen=True)
class MasterTrade:
    account_id: str = field(repr=False)
    contract: FuturesContract
    side: Side
    quantity: int
    entry_price: Decimal
    stop_price: Decimal
    equity: Decimal
    cost_buffer_per_contract: Decimal = ZERO

    def __post_init__(self) -> None:
        if not self.account_id.strip():
            raise ValueError("master account is required")
        if isinstance(self.quantity, bool) or self.quantity <= 0:
            raise ValueError("master quantity must be a positive integer")
        if self.entry_price <= ZERO or self.stop_price <= ZERO:
            raise ValueError("entry and stop prices must be positive")
        if self.equity <= ZERO:
            raise ValueError("master equity must be positive")
        if self.cost_buffer_per_contract < ZERO:
            raise ValueError("master cost buffer cannot be negative")
        if self.side == Side.BUY and self.stop_price >= self.entry_price:
            raise ValueError("long master trades require a stop below entry")
        if self.side == Side.SELL and self.stop_price <= self.entry_price:
            raise ValueError("short master trades require a stop above entry")

    @property
    def risk_per_contract(self) -> Decimal:
        return contract_risk(
            self.entry_price,
            self.stop_price,
            self.contract.specification,
            cost_buffer_per_contract=self.cost_buffer_per_contract,
        )

    @property
    def risk_dollars(self) -> Decimal:
        return self.risk_per_contract * Decimal(self.quantity)

    @property
    def risk_fraction(self) -> Decimal:
        return self.risk_dollars / self.equity

    @property
    def point_exposure_fraction(self) -> Decimal:
        return (
            self.contract.specification.point_value
            * Decimal(self.quantity)
            / self.equity
        )


@dataclass(frozen=True)
class FollowerState:
    account_id: str = field(repr=False)
    equity: Decimal
    available_margin: Decimal
    current_open_risk: Decimal = ZERO
    daily_realized_pnl: Decimal = ZERO
    daily_total_pnl: Decimal = ZERO
    daily_drawdown: Decimal = ZERO
    account_kill_switch: bool = True

    def __post_init__(self) -> None:
        if not self.account_id.strip():
            raise ValueError("follower account is required")
        if self.equity <= ZERO:
            raise ValueError("follower equity must be positive")
        if self.available_margin < ZERO or self.current_open_risk < ZERO:
            raise ValueError("available margin and open risk cannot be negative")
        if self.daily_drawdown < ZERO:
            raise ValueError("daily drawdown cannot be negative")


@dataclass(frozen=True)
class FollowerPolicy:
    max_risk_fraction: Decimal
    max_contracts: int
    max_concurrent_open_risk_fraction: Decimal
    daily_realized_loss_fraction: Decimal
    daily_total_loss_fraction: Decimal
    max_relative_parity_error: Decimal
    margin_headroom_fraction: Decimal = Decimal("0.20")
    allow_over_target: bool = False
    per_product_max_contracts: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        required_positive = (
            self.max_risk_fraction,
            self.max_concurrent_open_risk_fraction,
            self.daily_realized_loss_fraction,
            self.daily_total_loss_fraction,
        )
        if any(value <= ZERO or value > ONE for value in required_positive):
            raise ValueError("risk and loss fractions must be in (0, 1]")
        if not ZERO <= self.max_relative_parity_error <= ONE:
            raise ValueError("parity tolerance must be in [0, 1]")
        if not ZERO <= self.margin_headroom_fraction < ONE:
            raise ValueError("margin headroom must be in [0, 1)")
        if isinstance(self.max_contracts, bool) or self.max_contracts <= 0:
            raise ValueError("max contracts must be a positive integer")
        if any(isinstance(value, bool) or value <= 0 for value in self.per_product_max_contracts.values()):
            raise ValueError("per-product contract limits must be positive integers")


@dataclass(frozen=True)
class FollowerContractCandidate:
    contract: FuturesContract
    initial_margin_per_contract: Decimal
    cost_buffer_per_contract: Decimal = ZERO

    def __post_init__(self) -> None:
        if self.initial_margin_per_contract <= ZERO:
            raise ValueError("candidate margin must be positive")
        if self.cost_buffer_per_contract < ZERO:
            raise ValueError("candidate cost buffer cannot be negative")


@dataclass(frozen=True)
class FollowerCopyPlan:
    account_id: str = field(repr=False)
    allowed: bool = False
    reason: str = "not_evaluated"
    contract_id: str | None = None
    product_code: str | None = None
    quantity: int = 0
    target_risk_dollars: Decimal = ZERO
    target_risk_fraction: Decimal = ZERO
    actual_risk_dollars: Decimal = ZERO
    actual_risk_fraction: Decimal = ZERO
    target_point_exposure_fraction: Decimal = ZERO
    actual_point_exposure_fraction: Decimal = ZERO
    risk_parity_error: Decimal = ONE
    exposure_parity_error: Decimal = ONE
    parity_error: Decimal = ONE
    over_target: bool = False
    margin_required: Decimal = ZERO
    would_submit: bool = False


@dataclass(frozen=True)
class ShadowCopyPlan:
    mode: CopierMode
    master_account_id: str = field(repr=False)
    master_contract_id: str
    master_product_code: str
    master_quantity: int
    master_risk_dollars: Decimal
    master_risk_fraction: Decimal
    master_point_exposure_fraction: Decimal
    followers: tuple[FollowerCopyPlan, ...]
    submission_enabled: bool = False


class PositionPlanAction(str, Enum):
    BLOCKED = "BLOCKED"
    HOLD = "HOLD"
    OPEN = "OPEN"
    ADD = "ADD"
    REDUCE = "REDUCE"
    CLOSE = "CLOSE"
    REVERSE = "REVERSE"


@dataclass(frozen=True)
class MasterPositionTarget:
    """Authoritative cumulative master position used for total-target sizing."""

    account_id: str = field(repr=False)
    contract: FuturesContract
    signed_quantity: int
    equity: Decimal
    entry_price: Decimal | None = None
    stop_price: Decimal | None = None
    cost_buffer_per_contract: Decimal = ZERO

    def __post_init__(self) -> None:
        if not self.account_id.strip():
            raise ValueError("master account is required")
        if isinstance(self.signed_quantity, bool) or not isinstance(
            self.signed_quantity, int
        ):
            raise ValueError("master signed quantity must be an integer")
        if self.equity <= ZERO:
            raise ValueError("master equity must be positive")
        if self.cost_buffer_per_contract < ZERO:
            raise ValueError("master cost buffer cannot be negative")
        if self.signed_quantity == 0:
            if self.entry_price is not None or self.stop_price is not None:
                raise ValueError("flat master positions cannot carry entry or stop prices")
            return
        if self.entry_price is None or self.stop_price is None:
            raise ValueError("open master positions require entry and stop prices")
        if self.entry_price <= ZERO or self.stop_price <= ZERO:
            raise ValueError("entry and stop prices must be positive")
        if self.signed_quantity > 0 and self.stop_price >= self.entry_price:
            raise ValueError("long master positions require a stop below entry")
        if self.signed_quantity < 0 and self.stop_price <= self.entry_price:
            raise ValueError("short master positions require a stop above entry")

    @property
    def is_flat(self) -> bool:
        return self.signed_quantity == 0

    def as_trade(self) -> MasterTrade:
        if self.is_flat or self.entry_price is None or self.stop_price is None:
            raise ValueError("flat master positions do not have an active trade")
        return MasterTrade(
            account_id=self.account_id,
            contract=self.contract,
            side=Side.BUY if self.signed_quantity > 0 else Side.SELL,
            quantity=abs(self.signed_quantity),
            entry_price=self.entry_price,
            stop_price=self.stop_price,
            equity=self.equity,
            cost_buffer_per_contract=self.cost_buffer_per_contract,
        )


@dataclass(frozen=True)
class CopiedPositionState:
    """Authoritative current copied position and its stop-defined risk."""

    contract: FuturesContract | None = None
    signed_quantity: int = 0
    open_risk_dollars: Decimal = ZERO

    def __post_init__(self) -> None:
        if isinstance(self.signed_quantity, bool) or not isinstance(
            self.signed_quantity, int
        ):
            raise ValueError("copied signed quantity must be an integer")
        if self.open_risk_dollars < ZERO:
            raise ValueError("copied open risk cannot be negative")
        if self.signed_quantity == 0:
            if self.contract is not None or self.open_risk_dollars != ZERO:
                raise ValueError("flat copied positions cannot carry contract or risk")
        elif self.contract is None:
            raise ValueError("open copied positions require an exact contract")


@dataclass(frozen=True)
class CumulativeFollowerState:
    account: FollowerState = field(repr=False)
    copied_position: CopiedPositionState = field(default_factory=CopiedPositionState)


@dataclass(frozen=True)
class FundingAdequacyReport:
    account_id: str = field(repr=False)
    mnq_contract_id: str | None
    funding_adequate: bool
    currently_eligible: bool
    blockers: tuple[str, ...]
    one_mnq_risk_dollars: Decimal = ZERO
    one_mnq_margin_required: Decimal = ZERO
    minimum_equity: Decimal = ZERO
    equity_shortfall: Decimal = ZERO
    minimum_available_margin: Decimal = ZERO
    available_margin_shortfall: Decimal = ZERO
    usable_margin: Decimal = ZERO


@dataclass(frozen=True)
class CumulativeFollowerPlan:
    account_id: str = field(repr=False)
    allowed: bool
    reason: str
    action: PositionPlanAction
    current_contract_id: str | None
    current_product_code: str | None
    current_signed_quantity: int
    evaluated_contract_id: str | None
    evaluated_product_code: str | None
    desired_target_quantity: Decimal
    evaluated_target_signed_quantity: int | None
    achievable_target_signed_quantity: int
    delta_signed_quantity: int
    delta_side: Side | None
    delta_quantity: int
    desired_risk_dollars: Decimal
    desired_risk_fraction: Decimal
    achievable_risk_dollars: Decimal
    achievable_risk_fraction: Decimal
    desired_point_exposure_fraction: Decimal
    achievable_point_exposure_fraction: Decimal
    risk_parity_error: Decimal
    exposure_parity_error: Decimal
    parity_error: Decimal
    over_target: bool
    target_margin_required: Decimal
    funding: FundingAdequacyReport
    would_submit: bool = False


@dataclass(frozen=True)
class CumulativeCopyPlan:
    mode: CopierMode
    master_account_id: str = field(repr=False)
    master_contract_id: str
    master_product_code: str
    master_signed_quantity: int
    master_risk_dollars: Decimal
    master_risk_fraction: Decimal
    master_point_exposure_fraction: Decimal
    followers: tuple[CumulativeFollowerPlan, ...]
    submission_enabled: bool = False


@dataclass(frozen=True)
class _Option:
    candidate: FollowerContractCandidate
    quantity: int
    risk_dollars: Decimal
    risk_fraction: Decimal
    point_exposure_fraction: Decimal
    risk_error: Decimal
    exposure_error: Decimal
    parity_error: Decimal
    over_target: bool
    margin_required: Decimal


def plan_shadow_copy(
    master: MasterTrade,
    followers: Sequence[FollowerState],
    candidates: Sequence[FollowerContractCandidate],
    policy: FollowerPolicy,
    *,
    mode: CopierMode = CopierMode.SHADOW,
    global_kill_switch: bool = True,
) -> ShadowCopyPlan:
    """Produce deterministic follower plans without broker side effects.

    The target is the master's *actual* stop-defined risk and point exposure as
    a fraction of master equity. Whole-contract granularity is explicit: a
    candidate may be reported but remains blocked when it falls outside the
    configured parity tolerance or any independent account ceiling.
    """
    if mode not in {CopierMode.OFF, CopierMode.SHADOW}:
        raise ValueError("broker submission modes are unavailable in the shadow foundation")
    if not followers:
        raise ValueError("at least one follower is required")
    account_ids = [follower.account_id for follower in followers]
    if len(set(account_ids)) != len(account_ids):
        raise ValueError("follower accounts must be unique")
    if master.account_id in set(account_ids):
        raise ValueError("master account cannot also be a follower")

    normalized_candidates = _validate_candidates(master, candidates)
    plans = tuple(
        _plan_follower(
            master,
            follower,
            normalized_candidates,
            policy,
            mode=mode,
            global_kill_switch=global_kill_switch,
        )
        for follower in followers
    )
    return ShadowCopyPlan(
        mode=mode,
        master_account_id=master.account_id,
        master_contract_id=master.contract.contract_id,
        master_product_code=master.contract.product_code,
        master_quantity=master.quantity,
        master_risk_dollars=master.risk_dollars,
        master_risk_fraction=master.risk_fraction,
        master_point_exposure_fraction=master.point_exposure_fraction,
        followers=plans,
    )


def plan_cumulative_copy(
    master: MasterPositionTarget,
    followers: Sequence[CumulativeFollowerState],
    candidates: Sequence[FollowerContractCandidate],
    policy: FollowerPolicy,
    *,
    mode: CopierMode = CopierMode.SHADOW,
    global_kill_switch: bool = True,
) -> CumulativeCopyPlan:
    """Plan desired totals and deltas without producing broker side effects.

    Sizing always starts from the master's current cumulative position. Existing
    follower exposure is subtracted only after the whole-contract target is
    chosen, which prevents fill-by-fill rounding drift on adds and reductions.
    An open follower contract is pinned until flat; the planner never rolls or
    converts an open copied position automatically.
    """

    if mode not in {CopierMode.OFF, CopierMode.SHADOW}:
        raise ValueError("broker submission modes are unavailable in the sizing foundation")
    if not followers:
        raise ValueError("at least one follower is required")
    account_ids = [item.account.account_id for item in followers]
    if len(set(account_ids)) != len(account_ids):
        raise ValueError("follower accounts must be unique")
    if master.account_id in set(account_ids):
        raise ValueError("master account cannot also be a follower")

    if master.is_flat:
        normalized_candidates = tuple(candidates)
        if not normalized_candidates:
            raise ValueError("at least one follower contract candidate is required")
        active_master = None
    else:
        active_master = master.as_trade()
        normalized_candidates = _validate_candidates(active_master, candidates)

    plans = tuple(
        _plan_cumulative_follower(
            master,
            active_master,
            follower,
            normalized_candidates,
            policy,
            mode=mode,
            global_kill_switch=global_kill_switch,
        )
        for follower in followers
    )
    return CumulativeCopyPlan(
        mode=mode,
        master_account_id=master.account_id,
        master_contract_id=master.contract.contract_id,
        master_product_code=master.contract.product_code,
        master_signed_quantity=master.signed_quantity,
        master_risk_dollars=(active_master.risk_dollars if active_master else ZERO),
        master_risk_fraction=(active_master.risk_fraction if active_master else ZERO),
        master_point_exposure_fraction=(
            active_master.point_exposure_fraction if active_master else ZERO
        ),
        followers=plans,
    )


def _plan_cumulative_follower(
    master_target: MasterPositionTarget,
    master: MasterTrade | None,
    state: CumulativeFollowerState,
    candidates: Sequence[FollowerContractCandidate],
    policy: FollowerPolicy,
    *,
    mode: CopierMode,
    global_kill_switch: bool,
) -> CumulativeFollowerPlan:
    follower = state.account
    position = state.copied_position
    if position.open_risk_dollars > follower.current_open_risk:
        raise ValueError("copied position risk cannot exceed total open risk")

    funding = _funding_adequacy(
        master,
        follower,
        position,
        candidates,
        policy,
        mode=mode,
        global_kill_switch=global_kill_switch,
    )
    if master is None:
        return _flat_master_plan(follower, position, funding, mode)

    eligible_candidates = tuple(candidates)
    current_candidate: FollowerContractCandidate | None = None
    if position.signed_quantity != 0:
        assert position.contract is not None
        if (
            position.contract.expiration != master.contract.expiration
            or position.contract.exchange.upper() != master.contract.exchange.upper()
        ):
            return _cumulative_result(
                master,
                follower,
                position,
                funding,
                option=None,
                allowed=False,
                reason="open_position_expiry_mismatch_no_auto_roll",
            )
        current_candidate = next(
            (
                item
                for item in candidates
                if item.contract.contract_id == position.contract.contract_id
            ),
            None,
        )
        if current_candidate is None:
            return _cumulative_result(
                master,
                follower,
                position,
                funding,
                option=None,
                allowed=False,
                reason="open_position_contract_not_in_exact_candidates",
            )
        # Keep the current exact contract until it is flat. This prevents an
        # NQ/MNQ conversion or an expiry roll from appearing as a sizing delta.
        eligible_candidates = (current_candidate,)

    other_open_risk = follower.current_open_risk - position.open_risk_dollars
    hard_risk_budget = min(
        follower.equity * policy.max_risk_fraction,
        follower.equity * policy.max_concurrent_open_risk_fraction
        - other_open_risk,
    )
    existing_margin_credit = ZERO
    if current_candidate is not None:
        existing_margin_credit = (
            current_candidate.initial_margin_per_contract
            * Decimal(abs(position.signed_quantity))
        )

    options: tuple[_Option, ...] = ()
    if hard_risk_budget > ZERO:
        options = _options(
            master,
            follower,
            eligible_candidates,
            policy,
            hard_risk_budget,
            existing_margin_credit=existing_margin_credit,
        )
    if not options:
        diagnostic: _Option | None = None
        if hard_risk_budget > ZERO and not policy.allow_over_target:
            over_target_options = _options(
                master,
                follower,
                eligible_candidates,
                replace(policy, allow_over_target=True),
                hard_risk_budget,
                existing_margin_credit=existing_margin_credit,
            )
            if over_target_options:
                diagnostic = _best_option(over_target_options)
        if position.signed_quantity != 0 and mode == CopierMode.SHADOW:
            return _cumulative_result(
                master,
                follower,
                position,
                funding,
                option=diagnostic,
                allowed=True,
                reason="no_safe_nonzero_target_close",
                achievable_signed_quantity=0,
            )
        reason = (
            "minimum_contract_exceeds_target"
            if diagnostic is not None
            else "no_contract_within_hard_limits"
        )
        return _cumulative_result(
            master,
            follower,
            position,
            funding,
            option=diagnostic,
            allowed=False,
            reason=reason,
        )

    best = _best_option(options)
    target_sign = 1 if master.side == Side.BUY else -1
    evaluated_signed_quantity = target_sign * best.quantity
    if evaluated_signed_quantity == position.signed_quantity:
        return _cumulative_result(
            master,
            follower,
            position,
            funding,
            option=best,
            allowed=True,
            reason="already_at_cumulative_target",
            achievable_signed_quantity=evaluated_signed_quantity,
        )

    entry_blocker = _entry_blocker(
        master,
        follower,
        policy,
        mode=mode,
        global_kill_switch=global_kill_switch,
    )
    risk_reducing = _is_risk_reducing(
        position.signed_quantity,
        evaluated_signed_quantity,
    )
    if entry_blocker is not None and not (
        mode == CopierMode.SHADOW and risk_reducing
    ):
        return _cumulative_result(
            master,
            follower,
            position,
            funding,
            option=best,
            allowed=False,
            reason=entry_blocker,
        )

    parity_allowed = best.parity_error <= policy.max_relative_parity_error
    if not parity_allowed:
        current_error = _current_parity_error(master, follower, position)
        improves_safely = (
            risk_reducing
            and not best.over_target
            and best.parity_error < current_error
        )
        if not improves_safely:
            return _cumulative_result(
                master,
                follower,
                position,
                funding,
                option=best,
                allowed=False,
                reason="parity_tolerance_exceeded",
            )
        reason = "risk_reducing_parity_improvement"
    elif entry_blocker is not None:
        reason = "risk_reducing_plan_allowed"
    else:
        reason = "cumulative_plan_ready"

    return _cumulative_result(
        master,
        follower,
        position,
        funding,
        option=best,
        allowed=True,
        reason=reason,
        achievable_signed_quantity=evaluated_signed_quantity,
    )


def _flat_master_plan(
    follower: FollowerState,
    position: CopiedPositionState,
    funding: FundingAdequacyReport,
    mode: CopierMode,
) -> CumulativeFollowerPlan:
    allowed = mode == CopierMode.SHADOW
    target = 0 if allowed else position.signed_quantity
    delta = target - position.signed_quantity
    action = (
        _position_action(position.signed_quantity, target)
        if allowed
        else PositionPlanAction.BLOCKED
    )
    return CumulativeFollowerPlan(
        account_id=follower.account_id,
        allowed=allowed,
        reason=(
            "already_flat"
            if allowed and position.signed_quantity == 0
            else "master_flat_close"
            if allowed
            else "copier_off"
        ),
        action=action,
        current_contract_id=(position.contract.contract_id if position.contract else None),
        current_product_code=(position.contract.product_code if position.contract else None),
        current_signed_quantity=position.signed_quantity,
        evaluated_contract_id=None,
        evaluated_product_code=None,
        desired_target_quantity=ZERO,
        evaluated_target_signed_quantity=0,
        achievable_target_signed_quantity=target,
        delta_signed_quantity=delta,
        delta_side=_delta_side(delta),
        delta_quantity=abs(delta),
        desired_risk_dollars=ZERO,
        desired_risk_fraction=ZERO,
        achievable_risk_dollars=(ZERO if target == 0 else position.open_risk_dollars),
        achievable_risk_fraction=(
            ZERO if target == 0 else position.open_risk_dollars / follower.equity
        ),
        desired_point_exposure_fraction=ZERO,
        achievable_point_exposure_fraction=(
            ZERO if target == 0 else _position_exposure_fraction(follower, position)
        ),
        risk_parity_error=ZERO if target == 0 else ONE,
        exposure_parity_error=ZERO if target == 0 else ONE,
        parity_error=ZERO if target == 0 else ONE,
        over_target=target != 0,
        target_margin_required=ZERO,
        funding=funding,
    )


def _cumulative_result(
    master: MasterTrade,
    follower: FollowerState,
    position: CopiedPositionState,
    funding: FundingAdequacyReport,
    *,
    option: _Option | None,
    allowed: bool,
    reason: str,
    achievable_signed_quantity: int | None = None,
) -> CumulativeFollowerPlan:
    target_sign = 1 if master.side == Side.BUY else -1
    evaluated_signed = target_sign * option.quantity if option else None
    achievable_signed = (
        achievable_signed_quantity
        if allowed and achievable_signed_quantity is not None
        else position.signed_quantity
    )
    if option is not None:
        evaluated_contract = option.candidate.contract
        desired_quantity = (
            Decimal(master.quantity)
            * follower.equity
            / master.equity
            * master.contract.specification.point_value
            / evaluated_contract.specification.point_value
        )
        risk_error = option.risk_error
        exposure_error = option.exposure_error
        parity_error = option.parity_error
        over_target = option.over_target
    else:
        evaluated_contract = None
        desired_quantity = ZERO
        risk_error = _relative_fraction_error(
            position.open_risk_dollars,
            follower.equity,
            master.risk_dollars,
            master.equity,
        )
        current_exposure = _position_exposure_dollars(position)
        exposure_error = _relative_fraction_error(
            current_exposure,
            follower.equity,
            master.contract.specification.point_value * Decimal(master.quantity),
            master.equity,
        )
        parity_error = max(risk_error, exposure_error)
        over_target = (
            _fraction_comparison(
                position.open_risk_dollars,
                follower.equity,
                master.risk_dollars,
                master.equity,
            )
            > ZERO
            or _fraction_comparison(
                current_exposure,
                follower.equity,
                master.contract.specification.point_value * Decimal(master.quantity),
                master.equity,
            )
            > ZERO
        )

    if achievable_signed == 0:
        achievable_risk = ZERO
        achievable_exposure_fraction = ZERO
        target_margin = ZERO
    elif option is not None and achievable_signed == evaluated_signed:
        achievable_risk = option.risk_dollars
        achievable_exposure_fraction = option.point_exposure_fraction
        target_margin = option.margin_required
    else:
        achievable_risk = position.open_risk_dollars
        achievable_exposure_fraction = _position_exposure_fraction(follower, position)
        target_margin = _current_margin_required(position, option)

    delta = achievable_signed - position.signed_quantity
    return CumulativeFollowerPlan(
        account_id=follower.account_id,
        allowed=allowed,
        reason=reason,
        action=(
            _position_action(position.signed_quantity, achievable_signed)
            if allowed
            else PositionPlanAction.BLOCKED
        ),
        current_contract_id=(position.contract.contract_id if position.contract else None),
        current_product_code=(position.contract.product_code if position.contract else None),
        current_signed_quantity=position.signed_quantity,
        evaluated_contract_id=(evaluated_contract.contract_id if evaluated_contract else None),
        evaluated_product_code=(evaluated_contract.product_code if evaluated_contract else None),
        desired_target_quantity=desired_quantity,
        evaluated_target_signed_quantity=evaluated_signed,
        achievable_target_signed_quantity=achievable_signed,
        delta_signed_quantity=delta,
        delta_side=_delta_side(delta),
        delta_quantity=abs(delta),
        desired_risk_dollars=follower.equity * master.risk_fraction,
        desired_risk_fraction=master.risk_fraction,
        achievable_risk_dollars=achievable_risk,
        achievable_risk_fraction=achievable_risk / follower.equity,
        desired_point_exposure_fraction=master.point_exposure_fraction,
        achievable_point_exposure_fraction=achievable_exposure_fraction,
        risk_parity_error=risk_error,
        exposure_parity_error=exposure_error,
        parity_error=parity_error,
        over_target=over_target,
        target_margin_required=target_margin,
        funding=funding,
    )


def _funding_adequacy(
    master: MasterTrade | None,
    follower: FollowerState,
    position: CopiedPositionState,
    candidates: Sequence[FollowerContractCandidate],
    policy: FollowerPolicy,
    *,
    mode: CopierMode,
    global_kill_switch: bool,
) -> FundingAdequacyReport:
    mnq = next(
        (item for item in candidates if item.contract.product_code.upper() == "MNQ"),
        None,
    )
    usable_margin = follower.available_margin * (ONE - policy.margin_headroom_fraction)
    if mnq is None:
        return FundingAdequacyReport(
            account_id=follower.account_id,
            mnq_contract_id=None,
            funding_adequate=False,
            currently_eligible=False,
            blockers=("mnq_candidate_missing",),
            usable_margin=usable_margin,
        )

    existing_margin_credit = ZERO
    if (
        position.contract is not None
        and position.contract.contract_id == mnq.contract.contract_id
    ):
        existing_margin_credit = (
            mnq.initial_margin_per_contract * Decimal(abs(position.signed_quantity))
        )
    required_new_margin = max(
        ZERO,
        mnq.initial_margin_per_contract - existing_margin_credit,
    )
    minimum_available_margin = (
        required_new_margin / (ONE - policy.margin_headroom_fraction)
    )
    margin_shortfall = max(ZERO, minimum_available_margin - follower.available_margin)

    if master is None:
        return FundingAdequacyReport(
            account_id=follower.account_id,
            mnq_contract_id=mnq.contract.contract_id,
            funding_adequate=False,
            currently_eligible=False,
            blockers=("master_flat_no_stop_risk",),
            one_mnq_margin_required=mnq.initial_margin_per_contract,
            minimum_available_margin=minimum_available_margin,
            available_margin_shortfall=margin_shortfall,
            usable_margin=usable_margin,
        )

    unit_risk = contract_risk(
        master.entry_price,
        master.stop_price,
        mnq.contract.specification,
        cost_buffer_per_contract=mnq.cost_buffer_per_contract,
    )
    other_open_risk = follower.current_open_risk - position.open_risk_dollars
    parity_equity = max(
        unit_risk * master.equity / master.risk_dollars,
        mnq.contract.specification.point_value
        * master.equity
        / (
            master.contract.specification.point_value
            * Decimal(master.quantity)
        ),
    )
    per_trade_equity = unit_risk / policy.max_risk_fraction
    concurrent_equity = (
        unit_risk + other_open_risk
    ) / policy.max_concurrent_open_risk_fraction
    minimum_equity = max(parity_equity, per_trade_equity, concurrent_equity)
    equity_shortfall = max(ZERO, minimum_equity - follower.equity)

    funding_blockers: list[str] = []
    if follower.equity < parity_equity:
        funding_blockers.append("equity_below_one_mnq_parity")
    if unit_risk > follower.equity * policy.max_risk_fraction:
        funding_blockers.append("one_mnq_exceeds_per_trade_risk_capacity")
    if (
        unit_risk
        > follower.equity * policy.max_concurrent_open_risk_fraction
        - other_open_risk
    ):
        funding_blockers.append("one_mnq_exceeds_concurrent_risk_capacity")
    if margin_shortfall > ZERO:
        funding_blockers.append("one_mnq_margin_capacity_insufficient")

    operational_blocker = _entry_blocker(
        master,
        follower,
        policy,
        mode=mode,
        global_kill_switch=global_kill_switch,
    )
    blockers = list(funding_blockers)
    if operational_blocker is not None and operational_blocker not in blockers:
        blockers.append(operational_blocker)
    funding_adequate = not funding_blockers
    return FundingAdequacyReport(
        account_id=follower.account_id,
        mnq_contract_id=mnq.contract.contract_id,
        funding_adequate=funding_adequate,
        currently_eligible=funding_adequate and operational_blocker is None,
        blockers=tuple(blockers),
        one_mnq_risk_dollars=unit_risk,
        one_mnq_margin_required=mnq.initial_margin_per_contract,
        minimum_equity=minimum_equity,
        equity_shortfall=equity_shortfall,
        minimum_available_margin=minimum_available_margin,
        available_margin_shortfall=margin_shortfall,
        usable_margin=usable_margin + existing_margin_credit,
    )


def _entry_blocker(
    master: MasterTrade,
    follower: FollowerState,
    policy: FollowerPolicy,
    *,
    mode: CopierMode,
    global_kill_switch: bool,
) -> str | None:
    if mode == CopierMode.OFF:
        return "copier_off"
    if global_kill_switch:
        return "global_kill_switch"
    if follower.account_kill_switch:
        return "account_kill_switch"
    if master.risk_fraction > policy.max_risk_fraction:
        return "master_risk_exceeds_account_limit"
    daily_realized_limit = follower.equity * policy.daily_realized_loss_fraction
    daily_total_limit = follower.equity * policy.daily_total_loss_fraction
    if follower.daily_realized_pnl <= -daily_realized_limit:
        return "daily_realized_loss_ceiling_reached"
    if (
        follower.daily_total_pnl <= -daily_total_limit
        or follower.daily_drawdown >= daily_total_limit
    ):
        return "daily_total_loss_ceiling_reached"
    return None


def _current_parity_error(
    master: MasterTrade,
    follower: FollowerState,
    position: CopiedPositionState,
) -> Decimal:
    risk_error = _relative_fraction_error(
        position.open_risk_dollars,
        follower.equity,
        master.risk_dollars,
        master.equity,
    )
    exposure_error = _relative_fraction_error(
        _position_exposure_dollars(position),
        follower.equity,
        master.contract.specification.point_value * Decimal(master.quantity),
        master.equity,
    )
    return max(risk_error, exposure_error)


def _position_exposure_dollars(position: CopiedPositionState) -> Decimal:
    if position.contract is None:
        return ZERO
    return (
        position.contract.specification.point_value
        * Decimal(abs(position.signed_quantity))
    )


def _position_exposure_fraction(
    follower: FollowerState,
    position: CopiedPositionState,
) -> Decimal:
    return _position_exposure_dollars(position) / follower.equity


def _current_margin_required(
    position: CopiedPositionState,
    option: _Option | None,
) -> Decimal:
    if position.contract is None or option is None:
        return ZERO
    if option.candidate.contract.contract_id != position.contract.contract_id:
        return ZERO
    return (
        option.candidate.initial_margin_per_contract
        * Decimal(abs(position.signed_quantity))
    )


def _is_risk_reducing(current: int, target: int) -> bool:
    if current == 0:
        return False
    if target == 0:
        return True
    return current * target > 0 and abs(target) < abs(current)


def _position_action(current: int, target: int) -> PositionPlanAction:
    if current == target:
        return PositionPlanAction.HOLD
    if current == 0:
        return PositionPlanAction.OPEN
    if target == 0:
        return PositionPlanAction.CLOSE
    if current * target < 0:
        return PositionPlanAction.REVERSE
    if abs(target) > abs(current):
        return PositionPlanAction.ADD
    return PositionPlanAction.REDUCE


def _delta_side(delta: int) -> Side | None:
    if delta > 0:
        return Side.BUY
    if delta < 0:
        return Side.SELL
    return None


def _validate_candidates(
    master: MasterTrade,
    candidates: Sequence[FollowerContractCandidate],
) -> tuple[FollowerContractCandidate, ...]:
    if not candidates:
        raise ValueError("at least one follower contract candidate is required")
    allowed_products = {"NQ", "MNQ"}
    if master.contract.product_code.upper() not in allowed_products:
        raise ValueError("shadow copier currently supports only NQ and MNQ")

    identities: set[str] = set()
    normalized: list[FollowerContractCandidate] = []
    for candidate in candidates:
        contract = candidate.contract
        if contract.product_code.upper() not in allowed_products:
            raise ValueError("follower candidates are restricted to NQ and MNQ")
        if contract.expiration != master.contract.expiration:
            raise ValueError("follower candidates must preserve the master contract expiry")
        if contract.exchange.upper() != master.contract.exchange.upper():
            raise ValueError("follower candidates must preserve the master exchange")
        if contract.contract_id in identities:
            raise ValueError("follower contract candidates must be unique")
        identities.add(contract.contract_id)
        normalized.append(candidate)
    return tuple(normalized)


def _plan_follower(
    master: MasterTrade,
    follower: FollowerState,
    candidates: Sequence[FollowerContractCandidate],
    policy: FollowerPolicy,
    *,
    mode: CopierMode,
    global_kill_switch: bool,
) -> FollowerCopyPlan:
    target_risk = follower.equity * master.risk_fraction
    base = dict(
        account_id=follower.account_id,
        target_risk_dollars=target_risk,
        target_risk_fraction=master.risk_fraction,
        target_point_exposure_fraction=master.point_exposure_fraction,
    )
    blocker = _entry_blocker(
        master,
        follower,
        policy,
        mode=mode,
        global_kill_switch=global_kill_switch,
    )
    if blocker is not None:
        return FollowerCopyPlan(reason=blocker, **base)

    hard_risk_budget = min(
        follower.equity * policy.max_risk_fraction,
        follower.equity * policy.max_concurrent_open_risk_fraction
        - follower.current_open_risk,
    )
    if hard_risk_budget <= ZERO:
        return FollowerCopyPlan(reason="concurrent_open_risk_exhausted", **base)

    options = _options(master, follower, candidates, policy, hard_risk_budget)
    if not options:
        if not policy.allow_over_target:
            over_target_options = _options(
                master,
                follower,
                candidates,
                replace(policy, allow_over_target=True),
                hard_risk_budget,
            )
            if over_target_options:
                best = _best_option(over_target_options)
                return _plan_from_option(
                    base,
                    best,
                    allowed=False,
                    reason="minimum_contract_exceeds_target",
                )
        return FollowerCopyPlan(reason="no_contract_within_hard_limits", **base)

    best = _best_option(options)
    allowed = best.parity_error <= policy.max_relative_parity_error
    return _plan_from_option(
        base,
        best,
        allowed=allowed,
        reason="shadow_plan_ready" if allowed else "parity_tolerance_exceeded",
    )


def _best_option(options: Sequence[_Option]) -> _Option:
    return min(
        options,
        key=lambda option: (
            option.parity_error,
            option.over_target,
            option.risk_error,
            option.quantity,
            option.margin_required,
            option.candidate.contract.contract_id,
        ),
    )


def _plan_from_option(
    base: dict[str, object],
    option: _Option,
    *,
    allowed: bool,
    reason: str,
) -> FollowerCopyPlan:
    return FollowerCopyPlan(
        allowed=allowed,
        reason=reason,
        contract_id=option.candidate.contract.contract_id,
        product_code=option.candidate.contract.product_code,
        quantity=option.quantity,
        actual_risk_dollars=option.risk_dollars,
        actual_risk_fraction=option.risk_fraction,
        actual_point_exposure_fraction=option.point_exposure_fraction,
        risk_parity_error=option.risk_error,
        exposure_parity_error=option.exposure_error,
        parity_error=option.parity_error,
        over_target=option.over_target,
        margin_required=option.margin_required,
        would_submit=False,
        **base,
    )


def _options(
    master: MasterTrade,
    follower: FollowerState,
    candidates: Sequence[FollowerContractCandidate],
    policy: FollowerPolicy,
    hard_risk_budget: Decimal,
    *,
    existing_margin_credit: Decimal = ZERO,
) -> tuple[_Option, ...]:
    usable_margin = (
        follower.available_margin * (ONE - policy.margin_headroom_fraction)
        + existing_margin_credit
    )
    options: list[_Option] = []
    for candidate in candidates:
        product = candidate.contract.product_code.upper()
        contract_limit = min(
            policy.max_contracts,
            policy.per_product_max_contracts.get(product, policy.max_contracts),
        )
        margin_limit = int(
            (usable_margin / candidate.initial_margin_per_contract).to_integral_value(
                rounding=ROUND_FLOOR
            )
        )
        maximum = min(contract_limit, margin_limit)
        if maximum <= 0:
            continue
        unit_risk = contract_risk(
            master.entry_price,
            master.stop_price,
            candidate.contract.specification,
            cost_buffer_per_contract=candidate.cost_buffer_per_contract,
        )
        for quantity in range(1, maximum + 1):
            risk_dollars = unit_risk * Decimal(quantity)
            if risk_dollars > hard_risk_budget:
                break
            risk_fraction = risk_dollars / follower.equity
            point_exposure = (
                candidate.contract.specification.point_value
                * Decimal(quantity)
                / follower.equity
            )
            risk_comparison = _fraction_comparison(
                risk_dollars,
                follower.equity,
                master.risk_dollars,
                master.equity,
            )
            exposure_comparison = _fraction_comparison(
                candidate.contract.specification.point_value * Decimal(quantity),
                follower.equity,
                master.contract.specification.point_value
                * Decimal(master.quantity),
                master.equity,
            )
            over_target = risk_comparison > ZERO or exposure_comparison > ZERO
            if over_target and not policy.allow_over_target:
                continue
            risk_error = _relative_fraction_error(
                risk_dollars,
                follower.equity,
                master.risk_dollars,
                master.equity,
            )
            exposure_error = _relative_fraction_error(
                candidate.contract.specification.point_value * Decimal(quantity),
                follower.equity,
                master.contract.specification.point_value
                * Decimal(master.quantity),
                master.equity,
            )
            options.append(
                _Option(
                    candidate=candidate,
                    quantity=quantity,
                    risk_dollars=risk_dollars,
                    risk_fraction=risk_fraction,
                    point_exposure_fraction=point_exposure,
                    risk_error=risk_error,
                    exposure_error=exposure_error,
                    parity_error=max(risk_error, exposure_error),
                    over_target=over_target,
                    margin_required=(
                        candidate.initial_margin_per_contract * Decimal(quantity)
                    ),
                )
            )
    return tuple(options)


def _fraction_comparison(
    actual_numerator: Decimal,
    actual_denominator: Decimal,
    target_numerator: Decimal,
    target_denominator: Decimal,
) -> Decimal:
    return (
        actual_numerator * target_denominator
        - target_numerator * actual_denominator
    )


def _relative_fraction_error(
    actual_numerator: Decimal,
    actual_denominator: Decimal,
    target_numerator: Decimal,
    target_denominator: Decimal,
) -> Decimal:
    target_scaled = target_numerator * actual_denominator
    if target_scaled <= ZERO:
        return ONE
    actual_scaled = actual_numerator * target_denominator
    return abs(actual_scaled - target_scaled) / target_scaled
