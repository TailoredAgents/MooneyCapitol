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
    if mode == CopierMode.OFF:
        return FollowerCopyPlan(reason="copier_off", **base)
    if global_kill_switch:
        return FollowerCopyPlan(reason="global_kill_switch", **base)
    if follower.account_kill_switch:
        return FollowerCopyPlan(reason="account_kill_switch", **base)
    if master.risk_fraction > policy.max_risk_fraction:
        return FollowerCopyPlan(reason="master_risk_exceeds_account_limit", **base)

    daily_realized_limit = follower.equity * policy.daily_realized_loss_fraction
    daily_total_limit = follower.equity * policy.daily_total_loss_fraction
    if follower.daily_realized_pnl <= -daily_realized_limit:
        return FollowerCopyPlan(reason="daily_realized_loss_ceiling_reached", **base)
    if follower.daily_total_pnl <= -daily_total_limit or follower.daily_drawdown >= daily_total_limit:
        return FollowerCopyPlan(reason="daily_total_loss_ceiling_reached", **base)

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
) -> tuple[_Option, ...]:
    usable_margin = follower.available_margin * (ONE - policy.margin_headroom_fraction)
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
