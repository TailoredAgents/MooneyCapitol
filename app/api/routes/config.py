from __future__ import annotations

from fastapi import APIRouter, Depends, Header
from fastapi import HTTPException

from app.api.auth import require_operator
from app.core.config import AppConfig
from app.core.config_store import CONFIG, persist_config, refresh_config
from app.services.kv_store import StateStoreError


router = APIRouter(prefix="", tags=["config"])


@router.get("/config", response_model=AppConfig, dependencies=[Depends(require_operator)])
def get_config():
    try:
        refresh_config()
    except StateStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return CONFIG


@router.put(
    "/config",
    response_model=AppConfig,
    dependencies=[Depends(require_operator)],
)
def update_config(cfg: AppConfig, x_confirm: str | None = Header(default=None)):
    _validate_execution_transition(CONFIG, cfg, x_confirm if isinstance(x_confirm, str) else None)
    CONFIG.session = cfg.session
    CONFIG.universe = cfg.universe
    CONFIG.scan = cfg.scan
    CONFIG.detectors = cfg.detectors
    CONFIG.gating = cfg.gating
    CONFIG.alerts = cfg.alerts
    CONFIG.retention_days = cfg.retention_days
    CONFIG.reports = cfg.reports
    CONFIG.openai = cfg.openai
    CONFIG.depth_provider = cfg.depth_provider
    CONFIG.copier = cfg.copier
    try:
        persist_config()
    except StateStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return CONFIG


def _validate_execution_transition(current: AppConfig, proposed: AppConfig, confirmation: str | None) -> None:
    confirmations = {item.strip() for item in (confirmation or "").split(",") if item.strip()}
    copier = proposed.copier
    if copier.mode not in {"read_only", "test", "live"}:
        raise HTTPException(status_code=400, detail=f"Unsupported copier mode: {copier.mode}")
    if not 1 <= copier.max_orders_per_minute <= 500:
        raise HTTPException(status_code=400, detail="max_orders_per_minute must be between 1 and 500")
    required: list[str] = []
    if copier.enabled and not current.copier.enabled:
        required.append("ENABLE_COPIER")
    if copier.mode == "live" and current.copier.mode != "live":
        required.append("SET_LIVE_MODE")
    if not copier.global_kill_switch and current.copier.global_kill_switch:
        required.append("DISABLE_KILL_SWITCH")
    for target in copier.targets:
        previous = next((item for item in current.copier.targets if item.name == target.name), None)
        if target.enabled and (previous is None or not previous.enabled):
            required.append(f"ENABLE_TARGET:{target.name}")
        if target.enabled and not copier.enabled:
            raise HTTPException(status_code=400, detail="Enable the global copier before enabling a target")
        if target.sizing_mode not in {"disabled", "fixed_quantity", "fixed_multiplier", "percent_equity", "equity_ratio"}:
            raise HTTPException(status_code=400, detail=f"Unsupported sizing mode for {target.name}: {target.sizing_mode}")
        numeric_limits = (
            target.sizing_value,
            target.min_notional,
            target.max_notional_per_trade,
            target.max_position_pct,
            target.max_daily_notional,
            target.max_daily_trades,
        )
        if any(value < 0 for value in numeric_limits):
            raise HTTPException(status_code=400, detail=f"Execution limits cannot be negative for {target.name}")
        if target.enabled and target.sizing_mode not in {"percent_equity", "equity_ratio"}:
            raise HTTPException(status_code=400, detail=f"{target.name} must use percent-equity sizing")
    missing = [item for item in required if item not in confirmations]
    if missing:
        raise HTTPException(status_code=400, detail=f"Confirmation required: {', '.join(missing)}")
    if copier.mode == "live":
        if not copier.live_trading_enabled:
            raise HTTPException(status_code=400, detail="Live mode requires live_trading_enabled=true")
        if copier.live_max_notional_per_order <= 0:
            raise HTTPException(status_code=400, detail="Live mode requires a positive fail-closed notional ceiling")
