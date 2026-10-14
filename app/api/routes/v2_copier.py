from __future__ import annotations

import hashlib
from datetime import date
from decimal import Decimal
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel, Field, model_validator

from app.api.auth import require_operator
from app.v2.domain.models import Side
from app.v2.execution.copier import (
    CopierMode,
    FollowerContractCandidate,
    FollowerPolicy,
    FollowerState,
    MasterTrade,
    ShadowCopyPlan,
    plan_shadow_copy,
)
from app.v2.execution.copier_config import ShadowCopierConfig
from app.v2.reference import FuturesContractRegistry


router = APIRouter(
    prefix="/v2/copier",
    tags=["v2-copier"],
    dependencies=[Depends(require_operator)],
)

Fraction = Annotated[Decimal, Field(ge=Decimal("0"), le=Decimal("1"))]
PositiveFraction = Annotated[Decimal, Field(gt=Decimal("0"), le=Decimal("1"))]
PositiveDecimal = Annotated[Decimal, Field(gt=Decimal("0"))]
NonNegativeDecimal = Annotated[Decimal, Field(ge=Decimal("0"))]


class ShadowPreviewMaster(BaseModel):
    account_label: str = Field(min_length=1, max_length=128)
    product_code: Literal["NQ", "MNQ"]
    expiration: date
    side: Literal["BUY", "SELL"]
    quantity: int = Field(ge=1, le=10_000)
    entry_price: PositiveDecimal
    stop_price: PositiveDecimal
    equity: PositiveDecimal
    cost_buffer_per_contract: NonNegativeDecimal = Decimal("0")

    @model_validator(mode="after")
    def validate_protective_stop(self) -> "ShadowPreviewMaster":
        if self.side == "BUY" and self.stop_price >= self.entry_price:
            raise ValueError("long master trades require a stop below entry")
        if self.side == "SELL" and self.stop_price <= self.entry_price:
            raise ValueError("short master trades require a stop above entry")
        return self


class ShadowPreviewFollower(BaseModel):
    account_label: str = Field(min_length=1, max_length=128)
    equity: PositiveDecimal
    available_margin: NonNegativeDecimal
    current_open_risk: NonNegativeDecimal = Decimal("0")
    daily_realized_pnl: Decimal = Decimal("0")
    daily_total_pnl: Decimal = Decimal("0")
    daily_drawdown: NonNegativeDecimal = Decimal("0")
    account_kill_switch: bool = False


class ShadowPreviewPolicy(BaseModel):
    max_risk_fraction: PositiveFraction
    max_contracts: int = Field(ge=1, le=100_000)
    max_concurrent_open_risk_fraction: PositiveFraction
    daily_realized_loss_fraction: PositiveFraction
    daily_total_loss_fraction: PositiveFraction
    max_relative_parity_error: Fraction
    margin_headroom_fraction: Fraction = Decimal("0.20")
    allow_over_target: bool = False
    nq_max_contracts: int | None = Field(default=None, ge=1, le=100_000)
    mnq_max_contracts: int | None = Field(default=None, ge=1, le=100_000)

    @model_validator(mode="after")
    def validate_margin_headroom(self) -> "ShadowPreviewPolicy":
        if self.margin_headroom_fraction >= Decimal("1"):
            raise ValueError("margin headroom must be less than one")
        return self


class ShadowPreviewContracts(BaseModel):
    nq_initial_margin: PositiveDecimal
    mnq_initial_margin: PositiveDecimal
    nq_cost_buffer: NonNegativeDecimal = Decimal("0")
    mnq_cost_buffer: NonNegativeDecimal = Decimal("0")


class ShadowPreviewRequest(BaseModel):
    master: ShadowPreviewMaster
    followers: list[ShadowPreviewFollower] = Field(min_length=1, max_length=4)
    policy: ShadowPreviewPolicy
    contracts: ShadowPreviewContracts
    global_kill_switch: bool = True

    @model_validator(mode="after")
    def validate_accounts(self) -> "ShadowPreviewRequest":
        follower_labels = [item.account_label for item in self.followers]
        if len(set(follower_labels)) != len(follower_labels):
            raise ValueError("follower labels must be unique")
        if self.master.account_label in set(follower_labels):
            raise ValueError("master cannot also be a follower")
        return self


@router.get("/status")
def shadow_copier_status(response: Response) -> dict[str, Any]:
    response.headers["Cache-Control"] = "no-store"
    return build_shadow_copier_status(ShadowCopierConfig.from_mapping())


@router.post("/preview")
def preview_shadow_copy(
    payload: ShadowPreviewRequest,
    response: Response,
) -> dict[str, Any]:
    """Calculate a no-side-effect risk/ROI parity plan.

    This endpoint does not read credentials, connect to Rithmic, persist a copy
    intent, or expose any method capable of broker submission.
    """
    registry = FuturesContractRegistry()
    contracts = {
        product: registry.register_contract(
            contract_id=f"PREVIEW:CME:{product}:{payload.master.expiration.isoformat()}",
            product_code=product,
            expiration=payload.master.expiration,
        )
        for product in ("NQ", "MNQ")
    }
    registry.map_same_expiry(
        contracts["NQ"].contract_id,
        contracts["MNQ"].contract_id,
    )
    master = MasterTrade(
        account_id=payload.master.account_label,
        contract=contracts[payload.master.product_code],
        side=Side(payload.master.side),
        quantity=payload.master.quantity,
        entry_price=payload.master.entry_price,
        stop_price=payload.master.stop_price,
        equity=payload.master.equity,
        cost_buffer_per_contract=payload.master.cost_buffer_per_contract,
    )
    followers = tuple(
        FollowerState(
            account_id=item.account_label,
            equity=item.equity,
            available_margin=item.available_margin,
            current_open_risk=item.current_open_risk,
            daily_realized_pnl=item.daily_realized_pnl,
            daily_total_pnl=item.daily_total_pnl,
            daily_drawdown=item.daily_drawdown,
            account_kill_switch=item.account_kill_switch,
        )
        for item in payload.followers
    )
    product_limits = {
        product: value
        for product, value in (
            ("NQ", payload.policy.nq_max_contracts),
            ("MNQ", payload.policy.mnq_max_contracts),
        )
        if value is not None
    }
    policy = FollowerPolicy(
        max_risk_fraction=payload.policy.max_risk_fraction,
        max_contracts=payload.policy.max_contracts,
        max_concurrent_open_risk_fraction=(
            payload.policy.max_concurrent_open_risk_fraction
        ),
        daily_realized_loss_fraction=payload.policy.daily_realized_loss_fraction,
        daily_total_loss_fraction=payload.policy.daily_total_loss_fraction,
        max_relative_parity_error=payload.policy.max_relative_parity_error,
        margin_headroom_fraction=payload.policy.margin_headroom_fraction,
        allow_over_target=payload.policy.allow_over_target,
        per_product_max_contracts=product_limits,
    )
    candidates = (
        FollowerContractCandidate(
            contracts["NQ"],
            payload.contracts.nq_initial_margin,
            payload.contracts.nq_cost_buffer,
        ),
        FollowerContractCandidate(
            contracts["MNQ"],
            payload.contracts.mnq_initial_margin,
            payload.contracts.mnq_cost_buffer,
        ),
    )
    plan = plan_shadow_copy(
        master,
        followers,
        candidates,
        policy,
        mode=CopierMode.SHADOW,
        global_kill_switch=payload.global_kill_switch,
    )
    response.headers["Cache-Control"] = "no-store"
    return _serialize_plan(plan)


def build_shadow_copier_status(config: ShadowCopierConfig) -> dict[str, Any]:
    return {
        "requested_mode": config.requested_mode,
        "effective_mode": config.effective_mode.value,
        "ready_for_shadow": config.ready_for_shadow,
        "submission_enabled": False,
        "master_configured": bool(config.master_account_id),
        "configured_follower_count": len(config.follower_account_ids),
        "expected_follower_count": 4,
        "mandatory_initial_stop": True,
        "allowed_products": ["NQ", "MNQ"],
        "same_expiry_mapping_required": True,
        "master_risk_parity": True,
        "allow_over_target": config.allow_over_target,
        "max_relative_parity_error": str(config.max_relative_parity_error),
        "broker_margins_configured": bool(
            config.nq_initial_margin > Decimal("0")
            and config.mnq_initial_margin > Decimal("0")
        ),
        "planner_implemented": True,
        "master_event_orchestration_implemented": False,
        "execution_adapter_implemented": False,
        "blockers": list(config.preflight_blockers),
    }


def _serialize_plan(plan: ShadowCopyPlan) -> dict[str, Any]:
    return {
        "mode": plan.mode.value,
        "submission_enabled": plan.submission_enabled,
        "master": {
            "account": _opaque_alias(plan.master_account_id, "Master"),
            "contract": plan.master_contract_id,
            "product": plan.master_product_code,
            "quantity": plan.master_quantity,
            "risk_dollars": _decimal_text(plan.master_risk_dollars),
            "risk_fraction": _decimal_text(plan.master_risk_fraction),
            "point_exposure_fraction": _decimal_text(
                plan.master_point_exposure_fraction
            ),
        },
        "followers": [
            {
                "account": _opaque_alias(item.account_id, "Follower"),
                "allowed": item.allowed,
                "reason": item.reason,
                "contract": item.contract_id,
                "product": item.product_code,
                "quantity": item.quantity,
                "target_risk_dollars": _decimal_text(item.target_risk_dollars),
                "target_risk_fraction": _decimal_text(item.target_risk_fraction),
                "actual_risk_dollars": _decimal_text(item.actual_risk_dollars),
                "actual_risk_fraction": _decimal_text(item.actual_risk_fraction),
                "risk_parity_error": _decimal_text(item.risk_parity_error),
                "exposure_parity_error": _decimal_text(
                    item.exposure_parity_error
                ),
                "parity_error": _decimal_text(item.parity_error),
                "over_target": item.over_target,
                "margin_required": _decimal_text(item.margin_required),
                "would_submit": item.would_submit,
            }
            for item in plan.followers
        ],
    }


def _opaque_alias(value: object, prefix: str) -> str:
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:8].upper()
    return f"{prefix} {digest}"


def _decimal_text(value: Decimal) -> str:
    if value == Decimal("0"):
        return "0"
    return format(value.normalize(), "f")
