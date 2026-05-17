from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.core.config_store import CONFIG
from app.services.kv_store import get_json, set_json


_STATUS_KEY = "copier_status"

_DEFAULT_STATUS: dict[str, Any] = {
    "state": "not_started",
    "master_connected": False,
    "last_master_execution_at": None,
    "last_copy_order_at": None,
    "last_error": None,
    "latency": {},
    "updated_at": None,
}


def ensure_copier_status_initialized() -> None:
    if get_json(_STATUS_KEY) is None:
        set_json(_STATUS_KEY, _default_status())


def _default_status() -> dict[str, Any]:
    payload = dict(_DEFAULT_STATUS)
    payload["updated_at"] = datetime.now(tz=timezone.utc).isoformat()
    return payload


def get_copier_status() -> dict[str, Any]:
    stored = get_json(_STATUS_KEY) or _default_status()
    targets = []
    for target in CONFIG.copier.targets:
        targets.append(
            {
                "name": target.name,
                "broker": target.broker,
                "environment": target.environment,
                "enabled": target.enabled,
                "account_configured": bool(target.account_ref),
                "endpoint_env": target.endpoint_env,
                "equity_configured": bool(target.equity or target.equity_env),
                "sizing_mode": target.sizing_mode,
                "min_notional": target.min_notional,
                "max_position_pct": target.max_position_pct,
                "max_daily_notional": target.max_daily_notional,
                "max_daily_trades": target.max_daily_trades,
                "regular_hours_only": target.regular_hours_only,
                "shorting_enabled": target.shorting_enabled,
            }
        )
    return {
        "enabled": CONFIG.copier.enabled,
        "mode": CONFIG.copier.mode,
        "live_trading_enabled": CONFIG.copier.live_trading_enabled,
        "global_kill_switch": CONFIG.copier.global_kill_switch,
        "max_orders_per_minute": CONFIG.copier.max_orders_per_minute,
        "master": {
            "broker": CONFIG.copier.master_broker,
            "account_configured": bool(CONFIG.copier.master_account),
            "equity_configured": bool(CONFIG.copier.master_equity or CONFIG.copier.master_equity_env),
            "equity": CONFIG.copier.master_equity,
            "endpoint_env": CONFIG.copier.master_endpoint_env,
            "equities_only": CONFIG.copier.equities_only,
            "regular_hours_only": CONFIG.copier.regular_hours_only,
            "copy_shorts": CONFIG.copier.copy_shorts,
        },
        "targets": targets,
        "runtime": stored,
    }


def set_copier_status(**updates: Any) -> dict[str, Any]:
    current = get_json(_STATUS_KEY) or _default_status()
    current.update(updates)
    current["updated_at"] = datetime.now(tz=timezone.utc).isoformat()
    set_json(_STATUS_KEY, current)
    return current
