from __future__ import annotations

import hashlib
from datetime import date
from decimal import Decimal
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field, model_validator
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.api.auth import (
    OperatorIdentity,
    SESSION_COOKIE_NAME,
    create_csrf_token,
    require_operator,
    require_operator_mutation,
)
from app.db.session import get_session
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
from app.v2.execution.copy_control import (
    DEFAULT_COPY_PROFILE_ID,
    CopyControlAuditSnapshot,
    CopyProfileSnapshot,
    FollowerControlAction,
    InvalidControlTransition,
)
from app.v2.execution.copy_control_store import (
    ControlRevisionConflict,
    CopyControlStore,
    CopyControlTopologyCorrupt,
    CopyProfileNotFound,
    DuplicateControlRequest,
    FollowerControlNotFound,
    SqlAlchemyCopyControlStore,
)
from app.v2.reference import FuturesContractRegistry


router = APIRouter(
    prefix="/v2/copier",
    tags=["v2-copier"],
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


OperatorFollowerAction = Literal[
    "ENABLE",
    "PAUSE",
    "REQUEST_FLATTEN_AND_DISABLE",
    "LOCK",
    "UNLOCK_TO_PAUSED",
]
ControlReasonCode = Literal[
    "operator_requested",
    "simulation",
    "scheduled_pause",
    "connection_issue",
    "risk_review",
    "reconciliation_issue",
    "emergency",
    "recovery",
]


class FollowerControlRequest(BaseModel):
    action: OperatorFollowerAction
    expected_revision: int = Field(ge=1)
    request_id: str = Field(min_length=1, max_length=96, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
    reason_code: ControlReasonCode | None = None


class GlobalPauseRequest(BaseModel):
    paused: bool
    expected_revision: int = Field(ge=1)
    request_id: str = Field(min_length=1, max_length=96, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
    reason_code: ControlReasonCode


def control_db_session():
    with get_session() as session:
        yield session


def copy_control_store(
    session: Session = Depends(control_db_session),
) -> CopyControlStore:
    return SqlAlchemyCopyControlStore(session)


@router.get("/status", dependencies=[Depends(require_operator)])
def shadow_copier_status(response: Response) -> dict[str, Any]:
    response.headers["Cache-Control"] = "no-store"
    return build_shadow_copier_status(ShadowCopierConfig.from_mapping())


@router.post("/preview", dependencies=[Depends(require_operator)])
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


@router.get("/control/csrf")
def copy_control_csrf(
    request: Request,
    response: Response,
    identity: OperatorIdentity = Depends(require_operator),
) -> dict[str, Any]:
    response.headers["Cache-Control"] = "no-store"
    if identity.auth_method != "dashboard_session":
        return {"required": False, "token": None}
    session_token = request.cookies.get(SESSION_COOKIE_NAME, "")
    return {"required": True, "token": create_csrf_token(session_token)}


@router.get("/control")
def get_copy_control(
    response: Response,
    _identity: OperatorIdentity = Depends(require_operator),
    store: CopyControlStore = Depends(copy_control_store),
) -> dict[str, Any]:
    try:
        profile = store.get_profile(DEFAULT_COPY_PROFILE_ID)
    except (CopyProfileNotFound, CopyControlTopologyCorrupt, SQLAlchemyError) as exc:
        raise HTTPException(
            status_code=503,
            detail="V2 copy control store is unavailable or incomplete",
        ) from exc
    response.headers["Cache-Control"] = "no-store"
    return _serialize_control_profile(profile)


@router.post("/control/followers/{connection_id}/actions")
def update_follower_control(
    connection_id: str,
    payload: FollowerControlRequest,
    response: Response,
    identity: OperatorIdentity = Depends(require_operator_mutation),
    store: CopyControlStore = Depends(copy_control_store),
) -> dict[str, Any]:
    try:
        profile = store.transition_follower(
            profile_id=DEFAULT_COPY_PROFILE_ID,
            connection_id=connection_id,
            action=FollowerControlAction(payload.action),
            expected_revision=payload.expected_revision,
            actor=identity.actor,
            auth_method=identity.auth_method,
            request_id=payload.request_id,
            reason_code=payload.reason_code,
        )
    except (
        ControlRevisionConflict,
        DuplicateControlRequest,
        InvalidControlTransition,
        FollowerControlNotFound,
        CopyProfileNotFound,
        CopyControlTopologyCorrupt,
        SQLAlchemyError,
        ValueError,
    ) as exc:
        _raise_control_api_error(exc)
    response.headers["Cache-Control"] = "no-store"
    return _serialize_control_profile(profile)


@router.post("/control/global-pause")
def update_global_pause(
    payload: GlobalPauseRequest,
    response: Response,
    identity: OperatorIdentity = Depends(require_operator_mutation),
    store: CopyControlStore = Depends(copy_control_store),
) -> dict[str, Any]:
    try:
        profile = store.set_global_pause(
            profile_id=DEFAULT_COPY_PROFILE_ID,
            paused=payload.paused,
            expected_revision=payload.expected_revision,
            actor=identity.actor,
            auth_method=identity.auth_method,
            request_id=payload.request_id,
            reason_code=payload.reason_code,
        )
    except (
        ControlRevisionConflict,
        DuplicateControlRequest,
        CopyProfileNotFound,
        CopyControlTopologyCorrupt,
        SQLAlchemyError,
        ValueError,
    ) as exc:
        _raise_control_api_error(exc)
    response.headers["Cache-Control"] = "no-store"
    return _serialize_control_profile(profile)


@router.get("/control/audit")
def get_copy_control_audit(
    response: Response,
    limit: int = Query(default=100, ge=1, le=500),
    _identity: OperatorIdentity = Depends(require_operator),
    store: CopyControlStore = Depends(copy_control_store),
) -> dict[str, Any]:
    try:
        items = store.list_audit(
            profile_id=DEFAULT_COPY_PROFILE_ID,
            limit=limit,
        )
    except (CopyProfileNotFound, CopyControlTopologyCorrupt, SQLAlchemyError) as exc:
        raise HTTPException(
            status_code=503,
            detail="V2 copy control audit store is unavailable",
        ) from exc
    response.headers["Cache-Control"] = "no-store"
    return {
        "count": len(items),
        "items": [_serialize_control_audit(item) for item in items],
    }


def build_shadow_copier_status(config: ShadowCopierConfig) -> dict[str, Any]:
    topology = config.topology
    return {
        "architecture": "five_independent_account_connections",
        "requested_mode": config.requested_mode,
        "effective_mode": config.effective_mode.value,
        "ready_for_shadow": config.ready_for_shadow,
        "ready_for_simulation": topology.ready_for_simulation,
        "ready_for_test_connectivity": topology.ready_for_test_connectivity,
        "submission_enabled": False,
        "connectivity_enabled": topology.connectivity_enabled,
        "environment": topology.environment,
        "connection_slot_count": len(topology.accounts),
        "master_configured": topology.master.account_id_present,
        "configured_follower_count": sum(
            follower.account_id_present for follower in topology.followers
        ),
        "expected_follower_count": 4,
        "credential_complete_account_count": (
            topology.credential_complete_account_count
        ),
        "connection_settings_complete_account_count": (
            topology.connection_settings_complete_account_count
        ),
        "accounts": [
            {
                "connection": account.display_name,
                "connection_id": account.connection_id,
                "role": account.role.value,
                "follower_number": account.follower_number,
                "environment": account.environment,
                "required_plants": ["ORDER", "PNL"],
                "account_selector_present": account.account_id_present,
                "credentials_present": account.credentials_present,
                "connection_settings_present": (
                    account.connection_settings_present
                    and account.discovery_uri_valid
                ),
                "ready_for_test_connectivity": bool(
                    account.environment == "TEST"
                    and account.secrets_present
                    and account.connection_settings_present
                    and account.discovery_uri_valid
                ),
            }
            for account in topology.accounts
        ],
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
        "connection_supervisor_implemented": True,
        "simulated_connections_implemented": True,
        "required_supervised_session_count": len(topology.accounts) * 2,
        "supervisor_runtime_health_attached": False,
        "durable_control_plane_implemented": True,
        "control_defaults_fail_closed": True,
        "operator_control_ui_implemented": False,
        "master_event_orchestration_implemented": False,
        "execution_adapter_implemented": False,
        "blockers": list(config.preflight_blockers),
        "connectivity_blockers": list(topology.connectivity_blockers),
    }


def _raise_control_api_error(exc: Exception) -> None:
    if isinstance(exc, ControlRevisionConflict):
        raise HTTPException(
            status_code=409,
            detail={
                "code": "control_revision_conflict",
                "current_revision": exc.current_revision,
            },
        ) from exc
    if isinstance(exc, DuplicateControlRequest):
        raise HTTPException(
            status_code=409,
            detail={"code": "duplicate_control_request"},
        ) from exc
    if isinstance(exc, InvalidControlTransition):
        raise HTTPException(
            status_code=409,
            detail={"code": "invalid_control_transition", "message": str(exc)},
        ) from exc
    if isinstance(exc, FollowerControlNotFound):
        raise HTTPException(status_code=404, detail="Follower control not found") from exc
    if isinstance(
        exc,
        (CopyProfileNotFound, CopyControlTopologyCorrupt, SQLAlchemyError),
    ):
        raise HTTPException(
            status_code=503,
            detail="V2 copy control store is unavailable or incomplete",
        ) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    raise exc


def _serialize_control_profile(profile: CopyProfileSnapshot) -> dict[str, Any]:
    return {
        "profile_id": profile.profile_id,
        "name": profile.name,
        "master_connection_id": profile.master_connection_id,
        "global_paused": profile.global_paused,
        "revision": profile.revision,
        "entry_enabled_connection_ids": list(
            profile.entry_enabled_connection_ids
        ),
        "created_at": profile.created_at.isoformat(),
        "updated_at": profile.updated_at.isoformat(),
        "followers": [
            {
                "connection_id": follower.connection_id,
                "follower_number": follower.follower_number,
                "state": follower.state.value,
                "restart_directive": follower.restart_directive.value,
                "permits_new_entries": follower.state.permits_new_entries,
                "requires_lifecycle_management": (
                    follower.state.requires_lifecycle_management
                ),
                "revision": follower.revision,
                "reason_code": follower.reason_code,
                "flatten_requested_at": (
                    follower.flatten_requested_at.isoformat()
                    if follower.flatten_requested_at
                    else None
                ),
                "locked_at": (
                    follower.locked_at.isoformat() if follower.locked_at else None
                ),
                "updated_at": follower.updated_at.isoformat(),
            }
            for follower in profile.followers
        ],
    }


def _serialize_control_audit(item: CopyControlAuditSnapshot) -> dict[str, Any]:
    return {
        "audit_id": item.audit_id,
        "profile_id": item.profile_id,
        "connection_id": item.connection_id,
        "action": item.action,
        "actor": item.actor,
        "auth_method": item.auth_method,
        "expected_revision": item.expected_revision,
        "previous_revision": item.previous_revision,
        "resulting_revision": item.resulting_revision,
        "previous_state": item.previous_state,
        "resulting_state": item.resulting_state,
        "reason_code": item.reason_code,
        "request_id": item.request_id,
        "occurred_at": item.occurred_at.isoformat(),
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
