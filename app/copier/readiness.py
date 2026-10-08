from __future__ import annotations

import os
from sqlalchemy import desc, or_, select
from sqlalchemy.orm import Session

from app.core.config_store import CONFIG
from app.db.models import AccountSnapshot, CopyOrder, CopyReconciliation
from app.copier.environment_safety import environment_safety_blocks


SIZING_MODES = {"disabled", "fixed_quantity", "fixed_multiplier", "percent_equity", "equity_ratio"}
COPIER_MODES = {"read_only", "test", "live"}


def evaluate_copier_readiness(session: Session) -> dict:
    checks = _readiness_checks(session)
    blockers = [check for check in checks if check["severity"] == "blocker" and not check["ok"]]
    warnings = [check for check in checks if check["severity"] == "warning" and not check["ok"]]
    enabled_targets = [target for target in CONFIG.copier.targets if target.enabled]
    return {
        "ready": not blockers,
        "ready_to_disable_kill_switch": not blockers and CONFIG.copier.enabled and bool(enabled_targets),
        "blockers": blockers,
        "warnings": warnings,
        "checks": checks,
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
        "deployment_safety_gates",
        not environment_safety_blocks(CONFIG.copier),
        "blocker",
        "Deployment safety flags permit the persisted copier configuration",
        {"blockers": environment_safety_blocks(CONFIG.copier)},
    )
    _add_check(
        checks,
        "live_notional_ceiling",
        CONFIG.copier.mode != "live" or CONFIG.copier.live_max_notional_per_order > 0,
        "blocker",
        "Live mode has a positive global per-order notional ceiling",
        {"ceiling": CONFIG.copier.live_max_notional_per_order},
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
                CONFIG.copier.master_events_endpoint_env,
                CONFIG.copier.master_app_key_env,
                CONFIG.copier.master_app_secret_env,
            ]
        ),
        "blocker",
        "Webull master API endpoint, events endpoint, app key, and secret are configured",
    )
    _add_check(
        checks,
        "master_equity",
        _equity_available(session, CONFIG.copier.master_account or os.getenv(CONFIG.copier.master_account_env), CONFIG.copier.master_equity, CONFIG.copier.master_equity_env),
        "blocker",
        "Master account equity is available from latest Webull snapshot or fallback config",
        {"env": CONFIG.copier.master_equity_env, "source": _equity_source(session, CONFIG.copier.master_account or os.getenv(CONFIG.copier.master_account_env), CONFIG.copier.master_equity, CONFIG.copier.master_equity_env)},
    )
    _add_check(checks, "kill_switch_on", CONFIG.copier.global_kill_switch, "warning", "Global kill switch is currently on")

    enabled_targets = [target for target in CONFIG.copier.targets if target.enabled]
    _add_check(checks, "enabled_targets", bool(enabled_targets), "blocker", "At least one copy target is enabled")
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
                _equity_available(session, target.account_ref or os.getenv(target.account_id_env), target.equity, target.equity_env),
                "blocker",
                f"Target {target.name} equity is available from latest Webull snapshot or fallback config",
                {"env": target.equity_env, "source": _equity_source(session, target.account_ref or os.getenv(target.account_id_env), target.equity, target.equity_env)},
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
                "max_daily_notional": target.max_daily_notional,
                "max_daily_trades": target.max_daily_trades,
            },
        )
        _add_check(checks, f"{target_prefix}:shorts_off", not target.shorting_enabled, "warning", f"Target {target.name} short copying is disabled")

    _add_check(checks, "open_reconciliations", not _has_open_reconciliations(session), "blocker", "No open copier reconciliation issues")
    _add_check(checks, "failed_copy_orders", not _has_unresolved_failed_copy_orders(session), "blocker", "No unresolved failed copied orders")
    return checks


def _add_check(checks: list[dict], key: str, ok: bool, severity: str, label: str, context: dict | None = None) -> None:
    checks.append({"key": key, "ok": bool(ok), "severity": severity, "label": label, "context": context or {}})


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


def _equity_available(session: Session, account_ref: str | None, value: float | None, env_name: str | None) -> bool:
    return _latest_snapshot_equity(session, account_ref) is not None or _configured_float(value, env_name) is not None


def _equity_source(session: Session, account_ref: str | None, value: float | None, env_name: str | None) -> str | None:
    if _latest_snapshot_equity(session, account_ref) is not None:
        return "webull_snapshot"
    if _configured_float(value, env_name) is not None:
        return "fallback_config"
    return None


def _latest_snapshot_equity(session: Session, account_ref: str | None) -> float | None:
    if not account_ref:
        return None
    try:
        row = (
            session.execute(
                select(AccountSnapshot)
                .where(AccountSnapshot.account_ref == account_ref)
                .order_by(desc(AccountSnapshot.snapshot_time))
                .limit(1)
            )
            .scalars()
            .first()
        )
    except Exception:
        return None
    if row is None:
        return None
    value = row.equity_value if row.equity_value is not None else row.total_value
    return float(value) if value is not None and value > 0 else None


def _env_values_configured(env_names: list[str | None]) -> bool:
    return all(bool(name and os.getenv(name)) for name in env_names)


def _has_open_reconciliations(session: Session) -> bool:
    try:
        row = session.execute(select(CopyReconciliation.id).where(CopyReconciliation.status == "open").limit(1)).scalar_one_or_none()
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
