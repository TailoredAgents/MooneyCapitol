from __future__ import annotations

from datetime import datetime
import os

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import desc, or_, select
from sqlalchemy.orm import Session

from app.api.auth import require_operator
from app.copier.audit import record_audit_event
from app.copier.readiness import evaluate_copier_readiness
from app.copier.state import get_copier_status
from app.core.config import CopierConfig, CopyTargetAccountConfig
from app.core.config_store import CONFIG, persist_config, refresh_config
from app.db.models import CopierAuditEvent, CopyOrder, CopyReconciliation, CopyTargetAccount, MasterExecution
from app.db.session import get_session
from app.services.kv_store import StateStoreError


router = APIRouter(
    prefix="/copier",
    tags=["copier"],
    dependencies=[Depends(require_operator)],
)

SIZING_MODES = {"disabled", "fixed_quantity", "fixed_multiplier", "percent_equity", "equity_ratio"}
COPIER_MODES = {"read_only", "test", "live"}
CONFIRM_ENABLE_COPIER = "ENABLE_COPIER"
CONFIRM_DISABLE_KILL_SWITCH = "DISABLE_KILL_SWITCH"
CONFIRM_SET_LIVE_MODE = "SET_LIVE_MODE"


class CopierSettingsUpdate(BaseModel):
    enabled: bool | None = None
    mode: str | None = None
    master_equity: float | None = Field(default=None, ge=0)
    regular_hours_only: bool | None = None
    copy_shorts: bool | None = None
    max_orders_per_minute: int | None = Field(default=None, ge=1, le=500)


class CopyTargetUpdate(BaseModel):
    enabled: bool | None = None
    account_ref: str | None = None
    equity: float | None = Field(default=None, ge=0)
    sizing_mode: str | None = None
    sizing_value: float | None = Field(default=None, ge=0)
    min_notional: float | None = Field(default=None, ge=0)
    max_notional_per_trade: float | None = Field(default=None, ge=0)
    max_position_pct: float | None = Field(default=None, ge=0)
    max_daily_notional: float | None = Field(default=None, ge=0)
    max_daily_trades: int | None = Field(default=None, ge=0)
    regular_hours_only: bool | None = None
    shorting_enabled: bool | None = None
    allowlist: list[str] | None = None
    blocklist: list[str] | None = None


def db_session():
    with get_session() as session:
        yield session


def _refresh_or_503() -> None:
    try:
        refresh_config()
    except StateStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _persist_or_503() -> None:
    try:
        persist_config()
    except StateStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/status")
def copier_status():
    _refresh_or_503()
    return get_copier_status()


@router.get("/targets")
def copier_targets():
    _refresh_or_503()
    return {
        "count": len(CONFIG.copier.targets),
        "items": [_serialize_target_config(target) for target in CONFIG.copier.targets],
    }


@router.get("/readiness")
def copier_readiness(session: Session = Depends(db_session)):
    _refresh_or_503()
    return evaluate_copier_readiness(session)


@router.patch("/settings")
def update_copier_settings(
    payload: CopierSettingsUpdate,
    x_actor: str | None = Header(default=None),
    x_confirm: str | None = Header(default=None),
):
    _refresh_or_503()
    before = CONFIG.copier.model_dump()
    if payload.enabled is True and not CONFIG.copier.enabled:
        _require_confirmation(x_confirm, CONFIRM_ENABLE_COPIER)
    if payload.mode is not None and payload.mode.lower() == "live" and CONFIG.copier.mode != "live":
        _require_confirmation(x_confirm, CONFIRM_SET_LIVE_MODE)
    if payload.mode is not None:
        mode = payload.mode.lower()
        if mode not in COPIER_MODES:
            raise HTTPException(status_code=400, detail=f"Unsupported copier mode: {payload.mode}")
        CONFIG.copier.mode = mode
    if payload.enabled is not None:
        CONFIG.copier.enabled = payload.enabled
    if payload.master_equity is not None:
        CONFIG.copier.master_equity = payload.master_equity or None
    if payload.regular_hours_only is not None:
        CONFIG.copier.regular_hours_only = payload.regular_hours_only
    if payload.copy_shorts is not None:
        CONFIG.copier.copy_shorts = payload.copy_shorts
    if payload.max_orders_per_minute is not None:
        CONFIG.copier.max_orders_per_minute = payload.max_orders_per_minute
    try:
        _validate_copier_settings()
    except HTTPException:
        CONFIG.copier = CopierConfig.model_validate(before)
        raise
    _persist_or_503()
    record_audit_event(
        event_type="copier_settings_updated",
        actor=x_actor,
        message="Copier settings updated",
        payload={"before": before, "after": CONFIG.copier.model_dump()},
    )
    return get_copier_status()


@router.patch("/targets/{target_name}")
def update_copier_target(
    target_name: str,
    payload: CopyTargetUpdate,
    x_actor: str | None = Header(default=None),
    x_confirm: str | None = Header(default=None),
):
    _refresh_or_503()
    target = _find_target_or_404(target_name)
    before = target.model_dump()
    if payload.enabled is True and not target.enabled:
        _require_confirmation(x_confirm, _target_confirmation(target.name))

    if payload.enabled is not None:
        target.enabled = payload.enabled
    if payload.account_ref is not None:
        target.account_ref = payload.account_ref or None
    if payload.equity is not None:
        target.equity = payload.equity or None
    if payload.sizing_mode is not None:
        mode = payload.sizing_mode.lower()
        if mode not in SIZING_MODES:
            raise HTTPException(status_code=400, detail=f"Unsupported sizing mode: {payload.sizing_mode}")
        target.sizing_mode = mode
    if payload.sizing_value is not None:
        target.sizing_value = payload.sizing_value
    if payload.min_notional is not None:
        target.min_notional = payload.min_notional
    if payload.max_notional_per_trade is not None:
        target.max_notional_per_trade = payload.max_notional_per_trade
    if payload.max_position_pct is not None:
        target.max_position_pct = payload.max_position_pct
    if payload.max_daily_notional is not None:
        target.max_daily_notional = payload.max_daily_notional
    if payload.max_daily_trades is not None:
        target.max_daily_trades = payload.max_daily_trades
    if payload.regular_hours_only is not None:
        target.regular_hours_only = payload.regular_hours_only
    if payload.shorting_enabled is not None:
        target.shorting_enabled = payload.shorting_enabled
    if payload.allowlist is not None:
        target.allowlist = _normalize_symbol_list(payload.allowlist)
    if payload.blocklist is not None:
        target.blocklist = _normalize_symbol_list(payload.blocklist)

    try:
        _validate_target(target)
    except HTTPException:
        replacement = CopyTargetAccountConfig.model_validate(before)
        idx = CONFIG.copier.targets.index(target)
        CONFIG.copier.targets[idx] = replacement
        raise
    _persist_or_503()
    record_audit_event(
        event_type="copy_target_updated",
        actor=x_actor,
        message=f"Copy target {target.name} updated",
        payload={"target": target.name, "before": before, "after": target.model_dump()},
    )
    return _serialize_target_config(target)


@router.get("/master-executions")
def recent_master_executions(
    limit: int = Query(default=50, ge=1, le=250),
    symbol: str | None = None,
    session: Session = Depends(db_session),
):
    stmt = select(MasterExecution).order_by(desc(MasterExecution.executed_at)).limit(limit)
    if symbol:
        stmt = (
            select(MasterExecution)
            .where(MasterExecution.symbol == symbol.upper())
            .order_by(desc(MasterExecution.executed_at))
            .limit(limit)
        )
    rows = session.execute(stmt).scalars().all()
    return {"count": len(rows), "items": [_serialize_master_execution(row) for row in rows]}


@router.get("/copy-orders")
def recent_copy_orders(
    limit: int = Query(default=50, ge=1, le=250),
    status: str | None = None,
    target: str | None = None,
    session: Session = Depends(db_session),
):
    stmt = select(CopyOrder, CopyTargetAccount.name).join(
        CopyTargetAccount,
        CopyOrder.target_account_id == CopyTargetAccount.id,
    )
    if status:
        stmt = stmt.where(CopyOrder.status == status)
    if target:
        stmt = stmt.where(CopyTargetAccount.name == target)
    stmt = stmt.order_by(desc(CopyOrder.submitted_at), desc(CopyOrder.id)).limit(limit)
    rows = session.execute(stmt).all()
    return {
        "count": len(rows),
        "items": [_serialize_copy_order(order, target_name) for order, target_name in rows],
    }


@router.get("/trades")
def recent_copier_trades(
    limit: int = Query(default=100, ge=1, le=500),
    symbol: str | None = None,
    target: str | None = None,
    status: str | None = None,
    session: Session = Depends(db_session),
):
    stmt = (
        select(MasterExecution, CopyOrder, CopyTargetAccount.name)
        .join(CopyOrder, CopyOrder.master_execution_id == MasterExecution.id)
        .join(CopyTargetAccount, CopyOrder.target_account_id == CopyTargetAccount.id)
    )
    if symbol:
        stmt = stmt.where(MasterExecution.symbol == symbol.upper())
    if target:
        stmt = stmt.where(CopyTargetAccount.name == target)
    if status:
        stmt = stmt.where(CopyOrder.status == status)
    stmt = stmt.order_by(desc(MasterExecution.executed_at), desc(CopyOrder.id)).limit(limit)
    rows = session.execute(stmt).all()
    return {
        "count": len(rows),
        "items": [
            _serialize_copier_trade(master, order, target_name)
            for master, order, target_name in rows
        ],
    }


@router.get("/audit-events")
def recent_audit_events(
    limit: int = Query(default=50, ge=1, le=250),
    event_type: str | None = None,
    session: Session = Depends(db_session),
):
    stmt = select(CopierAuditEvent).order_by(desc(CopierAuditEvent.created_at)).limit(limit)
    if event_type:
        stmt = (
            select(CopierAuditEvent)
            .where(CopierAuditEvent.event_type == event_type)
            .order_by(desc(CopierAuditEvent.created_at))
            .limit(limit)
        )
    rows = session.execute(stmt).scalars().all()
    return {"count": len(rows), "items": [_serialize_audit_event(row) for row in rows]}


@router.get("/errors")
def recent_copier_errors(
    limit: int = Query(default=50, ge=1, le=250),
    session: Session = Depends(db_session),
):
    failed_orders_stmt = (
        select(CopyOrder, CopyTargetAccount.name)
        .join(CopyTargetAccount, CopyOrder.target_account_id == CopyTargetAccount.id)
        .where(
            or_(
                CopyOrder.status.in_(["submit_failed", "rejected", "cancelled"]),
                CopyOrder.reject_reason.is_not(None),
            )
        )
        .order_by(desc(CopyOrder.submitted_at), desc(CopyOrder.id))
        .limit(limit)
    )
    audit_stmt = (
        select(CopierAuditEvent)
        .where(
            or_(
                CopierAuditEvent.event_type.ilike("%error%"),
                CopierAuditEvent.event_type.ilike("%failed%"),
                CopierAuditEvent.event_type.ilike("%blocked%"),
                CopierAuditEvent.event_type.ilike("%missing%"),
            )
        )
        .order_by(desc(CopierAuditEvent.created_at))
        .limit(limit)
    )
    failed_orders = session.execute(failed_orders_stmt).all()
    audit_events = session.execute(audit_stmt).scalars().all()
    return {
        "copy_orders": [_serialize_copy_order(order, target_name) for order, target_name in failed_orders],
        "audit_events": [_serialize_audit_event(row) for row in audit_events],
    }


@router.get("/reconciliations")
def recent_reconciliations(
    limit: int = Query(default=50, ge=1, le=250),
    status: str | None = "open",
    severity: str | None = None,
    session: Session = Depends(db_session),
):
    stmt = select(CopyReconciliation).order_by(desc(CopyReconciliation.detected_at)).limit(limit)
    if status:
        stmt = stmt.where(CopyReconciliation.status == status)
    if severity:
        stmt = stmt.where(CopyReconciliation.severity == severity)
    rows = session.execute(stmt).scalars().all()
    return {"count": len(rows), "items": [_serialize_reconciliation(row) for row in rows]}


@router.post("/kill-switch/enable")
def enable_kill_switch(x_actor: str | None = Header(default=None)):
    _refresh_or_503()
    CONFIG.copier.global_kill_switch = True
    _persist_or_503()
    record_audit_event(
        event_type="kill_switch_enabled",
        actor=x_actor,
        message="Global copier kill switch enabled",
        payload={"global_kill_switch": True},
    )
    return {"ok": True, "global_kill_switch": True}


@router.post("/kill-switch/disable")
def disable_kill_switch(
    x_actor: str | None = Header(default=None),
    x_confirm: str | None = Header(default=None),
):
    _refresh_or_503()
    _require_confirmation(x_confirm, CONFIRM_DISABLE_KILL_SWITCH)
    if CONFIG.copier.mode == "live" and not CONFIG.copier.live_trading_enabled:
        raise HTTPException(
            status_code=400,
            detail="Cannot disable kill switch in live mode unless live_trading_enabled is true",
        )
    CONFIG.copier.global_kill_switch = False
    _persist_or_503()
    record_audit_event(
        event_type="kill_switch_disabled",
        actor=x_actor,
        message="Global copier kill switch disabled",
        payload={
            "global_kill_switch": False,
            "mode": CONFIG.copier.mode,
            "live_trading_enabled": CONFIG.copier.live_trading_enabled,
        },
    )
    return {"ok": True, "global_kill_switch": False}


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _serialize_master_execution(row: MasterExecution) -> dict:
    return {
        "id": row.id,
        "broker": row.broker,
        "account_ref": row.account_ref,
        "broker_execution_id": row.broker_execution_id,
        "broker_order_id": row.broker_order_id,
        "symbol": row.symbol,
        "side": row.side,
        "qty": row.qty,
        "price": row.price,
        "asset_class": row.asset_class,
        "executed_at": _iso(row.executed_at),
        "received_at": _iso(row.received_at),
    }


def _serialize_copy_order(row: CopyOrder, target_name: str | None = None) -> dict:
    return {
        "id": row.id,
        "master_execution_id": row.master_execution_id,
        "target_account_id": row.target_account_id,
        "target": target_name,
        "broker": row.broker,
        "client_order_id": row.client_order_id,
        "broker_order_id": row.broker_order_id,
        "symbol": row.symbol,
        "side": row.side,
        "qty": row.qty,
        "order_type": row.order_type,
        "time_in_force": row.time_in_force,
        "status": row.status,
        "submitted_at": _iso(row.submitted_at),
        "accepted_at": _iso(row.accepted_at),
        "filled_at": _iso(row.filled_at),
        "filled_qty": row.filled_qty,
        "avg_fill_price": row.avg_fill_price,
        "reject_reason": row.reject_reason,
        "latency_ms": row.latency_ms,
    }


def _serialize_copier_trade(master: MasterExecution, order: CopyOrder, target_name: str | None = None) -> dict:
    copied_notional = None
    if order.qty is not None and master.price is not None:
        copied_notional = float(order.qty) * float(master.price)
    filled_notional = None
    if order.filled_qty is not None and order.avg_fill_price is not None:
        filled_notional = float(order.filled_qty) * float(order.avg_fill_price)
    slippage_bps = None
    if order.avg_fill_price is not None and master.price:
        direction = 1 if master.side == "BUY" else -1
        slippage_bps = ((float(order.avg_fill_price) - float(master.price)) / float(master.price)) * 10_000 * direction
    return {
        "master_execution_id": master.id,
        "master_broker_execution_id": master.broker_execution_id,
        "master_order_id": master.broker_order_id,
        "master_account": master.account_ref,
        "symbol": master.symbol,
        "side": master.side,
        "master_qty": master.qty,
        "master_price": master.price,
        "master_notional": float(master.qty) * float(master.price),
        "master_executed_at": _iso(master.executed_at),
        "master_received_at": _iso(master.received_at),
        "copy_order_id": order.id,
        "target": target_name,
        "client_order_id": order.client_order_id,
        "broker_order_id": order.broker_order_id,
        "copy_qty": order.qty,
        "copy_notional": copied_notional,
        "copy_status": order.status,
        "copy_submitted_at": _iso(order.submitted_at),
        "copy_accepted_at": _iso(order.accepted_at),
        "copy_filled_at": _iso(order.filled_at),
        "copy_filled_qty": order.filled_qty,
        "copy_avg_fill_price": order.avg_fill_price,
        "copy_filled_notional": filled_notional,
        "copy_latency_ms": order.latency_ms,
        "copy_reject_reason": order.reject_reason,
        "copy_slippage_bps": slippage_bps,
    }


def _serialize_audit_event(row: CopierAuditEvent) -> dict:
    return {
        "id": row.id,
        "event_type": row.event_type,
        "actor": row.actor,
        "target_account_id": row.target_account_id,
        "message": row.message,
        "created_at": _iso(row.created_at),
        "payload": row.payload,
    }


def _serialize_reconciliation(row: CopyReconciliation) -> dict:
    return {
        "id": row.id,
        "target_account_id": row.target_account_id,
        "symbol": row.symbol,
        "severity": row.severity,
        "status": row.status,
        "message": row.message,
        "detected_at": _iso(row.detected_at),
        "resolved_at": _iso(row.resolved_at),
        "raw_context": row.raw_context,
    }


def _serialize_target_config(target) -> dict:
    return {
        "name": target.name,
        "broker": target.broker,
        "environment": target.environment,
        "enabled": target.enabled,
        "account_ref": target.account_ref,
        "account_id_env": target.account_id_env,
        "account_configured": bool(target.account_ref or os.getenv(target.account_id_env)),
        "equity": target.equity,
        "equity_env": target.equity_env,
        "equity_configured": _configured_float(target.equity, target.equity_env) is not None,
        "sizing_mode": target.sizing_mode,
        "sizing_value": target.sizing_value,
        "min_notional": target.min_notional,
        "max_notional_per_trade": target.max_notional_per_trade,
        "max_position_pct": target.max_position_pct,
        "max_daily_notional": target.max_daily_notional,
        "max_daily_trades": target.max_daily_trades,
        "regular_hours_only": target.regular_hours_only,
        "shorting_enabled": target.shorting_enabled,
        "allowlist": target.allowlist,
        "blocklist": target.blocklist,
    }


def _readiness_checks(session: Session) -> list[dict]:
    checks: list[dict] = []
    _add_check(checks, "copier_enabled", CONFIG.copier.enabled, "blocker", "Global copier is enabled")
    _add_check(
        checks,
        "safe_mode",
        CONFIG.copier.mode in COPIER_MODES and (CONFIG.copier.mode != "live" or CONFIG.copier.live_trading_enabled),
        "blocker",
        "Copier mode is valid and live mode is explicitly gated",
        {"mode": CONFIG.copier.mode, "live_trading_enabled": CONFIG.copier.live_trading_enabled},
    )
    _add_check(
        checks,
        "master_account",
        bool(CONFIG.copier.master_account or os.getenv(CONFIG.copier.master_account_env)),
        "blocker",
        "Webull master account id is configured",
        {"env": CONFIG.copier.master_account_env},
    )
    _add_check(
        checks,
        "master_credentials",
        _env_values_configured(
            [
                CONFIG.copier.master_endpoint_env,
                CONFIG.copier.master_app_key_env,
                CONFIG.copier.master_app_secret_env,
            ]
        ),
        "blocker",
        "Webull master API endpoint, app key, and secret are configured",
    )
    _add_check(
        checks,
        "master_equity",
        _configured_float(CONFIG.copier.master_equity, CONFIG.copier.master_equity_env) is not None,
        "blocker",
        "Master account equity is configured for percent-equity sizing",
        {"env": CONFIG.copier.master_equity_env},
    )
    _add_check(
        checks,
        "kill_switch_on",
        CONFIG.copier.global_kill_switch,
        "warning",
        "Global kill switch is currently on",
    )

    enabled_targets = [target for target in CONFIG.copier.targets if target.enabled]
    _add_check(
        checks,
        "enabled_targets",
        bool(enabled_targets),
        "blocker",
        "At least one copy target is enabled",
    )
    for target in CONFIG.copier.targets:
        target_prefix = f"target:{target.name}"
        if not target.enabled:
            _add_check(checks, f"{target_prefix}:disabled", True, "info", f"Target {target.name} is disabled")
            continue
        _add_check(
            checks,
            f"{target_prefix}:account",
            bool(target.account_ref or os.getenv(target.account_id_env)),
            "blocker",
            f"Target {target.name} Webull account id is configured",
            {"env": target.account_id_env},
        )
        _add_check(
            checks,
            f"{target_prefix}:credentials",
            _env_values_configured([target.endpoint_env, target.api_key_env, target.api_secret_env]),
            "blocker",
            f"Target {target.name} Webull endpoint, app key, and secret are configured",
        )
        _add_check(
            checks,
            f"{target_prefix}:sizing",
            target.sizing_mode in {"percent_equity", "equity_ratio"},
            "blocker",
            f"Target {target.name} has percent-equity mirror sizing configured",
            {"sizing_mode": target.sizing_mode},
        )
        if target.sizing_mode in {"percent_equity", "equity_ratio"}:
            _add_check(
                checks,
                f"{target_prefix}:equity",
                _configured_float(target.equity, target.equity_env) is not None,
                "blocker",
                f"Target {target.name} equity is configured",
                {"env": target.equity_env},
            )
        _add_check(
            checks,
            f"{target_prefix}:mirror_sizing",
            True,
            "info",
            f"Target {target.name} mirrors the master account percentage",
            {
                "max_notional_per_trade": target.max_notional_per_trade,
                "max_position_pct": target.max_position_pct,
            },
        )
        _add_check(
            checks,
            f"{target_prefix}:shorts_off",
            not target.shorting_enabled,
            "warning",
            f"Target {target.name} short copying is disabled",
        )

    _add_check(
        checks,
        "open_reconciliations",
        not _has_open_reconciliations(session),
        "blocker",
        "No open copier reconciliation issues",
    )
    _add_check(
        checks,
        "failed_copy_orders",
        not _has_unresolved_failed_copy_orders(session),
        "blocker",
        "No unresolved failed copied orders",
    )
    return checks


def _add_check(
    checks: list[dict],
    key: str,
    ok: bool,
    severity: str,
    label: str,
    context: dict | None = None,
) -> None:
    checks.append({"key": key, "ok": bool(ok), "severity": severity, "label": label, "context": context or {}})


def _find_target_or_404(target_name: str):
    for target in CONFIG.copier.targets:
        if target.name == target_name:
            return target
    raise HTTPException(status_code=404, detail=f"Unknown copier target: {target_name}")


def _require_confirmation(actual: str | None, expected: str) -> None:
    raw = actual if isinstance(actual, str) else ""
    confirmations = {item.strip() for item in raw.split(",") if item.strip()}
    if expected not in confirmations:
        raise HTTPException(status_code=400, detail=f"Confirmation required: {expected}")


def _target_confirmation(target_name: str) -> str:
    return f"ENABLE_TARGET:{target_name}"


def _validate_copier_settings() -> None:
    if CONFIG.copier.mode == "live" and not CONFIG.copier.live_trading_enabled:
        raise HTTPException(status_code=400, detail="Live mode requires live_trading_enabled=true in config")
    for target in CONFIG.copier.targets:
        _validate_target(target)


def _validate_target(target) -> None:
    if target.sizing_mode not in SIZING_MODES:
        raise HTTPException(status_code=400, detail=f"Unsupported sizing mode for {target.name}: {target.sizing_mode}")
    if target.max_position_pct and target.max_position_pct < 0:
        raise HTTPException(status_code=400, detail="max_position_pct must be greater than or equal to 0")
    if target.enabled:
        if not CONFIG.copier.enabled:
            raise HTTPException(status_code=400, detail="Enable the global copier before enabling a target")
        if CONFIG.copier.mode == "live" and not CONFIG.copier.live_trading_enabled:
            raise HTTPException(status_code=400, detail="Cannot enable live target unless live_trading_enabled=true")
        if target.sizing_mode not in {"percent_equity", "equity_ratio"}:
            raise HTTPException(status_code=400, detail=f"{target.name} must use percent_equity sizing")
        if target.sizing_mode in {"percent_equity", "equity_ratio"}:
            if _configured_float(CONFIG.copier.master_equity, CONFIG.copier.master_equity_env) is None:
                raise HTTPException(status_code=400, detail="percent_equity requires master account equity")
            if _configured_float(target.equity, target.equity_env) is None:
                raise HTTPException(status_code=400, detail=f"percent_equity requires target equity for {target.name}")


def _configured_float(value: float | None, env_name: str | None) -> float | None:
    if value is not None and value > 0:
        return value
    if not env_name:
        return None
    raw = os.getenv(env_name)
    if raw in (None, ""):
        return None
    try:
        parsed = float(raw)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _env_values_configured(env_names: list[str | None]) -> bool:
    return all(bool(name and os.getenv(name)) for name in env_names)


def _has_open_reconciliations(session: Session) -> bool:
    try:
        row = (
            session.execute(
                select(CopyReconciliation.id).where(CopyReconciliation.status == "open").limit(1)
            ).scalar_one_or_none()
        )
    except Exception:
        return True
    return row is not None


def _has_unresolved_failed_copy_orders(session: Session) -> bool:
    try:
        row = (
            session.execute(
                select(CopyOrder.id)
                .where(
                    or_(
                        CopyOrder.status.in_(["submit_failed", "rejected", "cancelled", "expired"]),
                        CopyOrder.reject_reason.is_not(None),
                    )
                )
                .limit(1)
            ).scalar_one_or_none()
        )
    except Exception:
        return True
    return row is not None


def _normalize_symbol_list(values: list[str]) -> list[str]:
    return sorted({value.strip().upper() for value in values if value and value.strip()})
