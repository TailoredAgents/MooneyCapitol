from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.copier.models import WebullCredentials
from app.copier.readiness import evaluate_copier_readiness
from app.copier.webull_client import WebullTradingClient
from app.core.config import CopierConfig, CopyTargetAccountConfig
from app.db.models import AccountSnapshot


ClientFactory = Callable[[WebullCredentials], WebullTradingClient]


@dataclass(frozen=True)
class PreflightCheck:
    key: str
    ok: bool
    severity: str
    label: str
    context: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["context"] = data["context"] or {}
        return data


def run_webull_preflight(
    config: CopierConfig,
    *,
    session: Session | None = None,
    client_factory: ClientFactory | None = None,
    include_network: bool = True,
    include_disabled_targets: bool = False,
) -> dict[str, Any]:
    client_factory = client_factory or (lambda credentials: WebullTradingClient(credentials))
    checks: list[PreflightCheck] = []

    checks.extend(_local_config_checks(config, session=session))
    accounts: list[dict[str, Any]] = [
        _check_master_account(config, client_factory=client_factory, include_network=include_network, session=session)
    ]
    for target in config.targets:
        if target.enabled or include_disabled_targets:
            accounts.append(
                _check_target_account(
                    config,
                    target,
                    client_factory=client_factory,
                    include_network=include_network and target.enabled,
                    session=session,
                )
            )

    for account in accounts:
        checks.extend(PreflightCheck(**item) for item in account.get("checks", []))
    if session is not None:
        checks.extend(_database_checks(session))

    blockers = [check for check in checks if check.severity == "blocker" and not check.ok]
    warnings = [check for check in checks if check.severity == "warning" and not check.ok]
    return {
        "ready": not blockers,
        "network_checked": include_network,
        "accounts": accounts,
        "blockers": [check.to_dict() for check in blockers],
        "warnings": [check.to_dict() for check in warnings],
        "checks": [check.to_dict() for check in checks],
    }


def _local_config_checks(config: CopierConfig, *, session: Session | None = None) -> list[PreflightCheck]:
    enabled_targets = [target for target in config.targets if target.enabled]
    master_account = config.master_account or os.getenv(config.master_account_env)
    return [
        PreflightCheck("copier.enabled", config.enabled, "blocker", "Copier is enabled"),
        PreflightCheck(
            "copier.mode",
            config.mode in {"read_only", "test", "live"},
            "blocker",
            "Copier mode is valid",
            {"mode": config.mode},
        ),
        PreflightCheck(
            "copier.live_gate",
            config.mode != "live" or config.live_trading_enabled,
            "blocker",
            "Live mode is explicitly gated",
            {"mode": config.mode, "live_trading_enabled": config.live_trading_enabled},
        ),
        PreflightCheck("copier.kill_switch", config.global_kill_switch, "warning", "Global kill switch is on"),
        PreflightCheck(
            "copier.master_equity",
            _equity_available(session, master_account, config.master_equity, config.master_equity_env),
            "blocker",
            "Master account equity is available from latest Webull snapshot or fallback config",
            {"env": config.master_equity_env, "source": _equity_source(session, master_account, config.master_equity, config.master_equity_env)},
        ),
        PreflightCheck(
            "copier.enabled_targets",
            bool(enabled_targets),
            "blocker",
            "At least one target account is enabled",
            {"targets": [target.name for target in enabled_targets]},
        ),
    ]


def _check_master_account(
    config: CopierConfig,
    *,
    client_factory: ClientFactory,
    include_network: bool,
    session: Session | None,
) -> dict[str, Any]:
    credentials = _master_credentials(config)
    report = _base_account_report("master", "master", credentials, include_network=include_network)
    report["checks"].extend(
        [
            _check(
                "master.account_id",
                bool(credentials.account_id),
                "blocker",
                "Master Webull account id is configured",
                {"env": config.master_account_env, "account": _mask(credentials.account_id)},
            ),
            _check(
                "master.credentials",
                _has_credentials(credentials, require_events=True),
                "blocker",
                "Master Webull API endpoint, events endpoint, app key, and secret are configured",
                {
                    "endpoint_env": config.master_endpoint_env,
                    "events_endpoint_env": config.master_events_endpoint_env,
                    "app_key_env": config.master_app_key_env,
                    "app_secret_env": config.master_app_secret_env,
                },
            ),
        ]
    )
    _add_network_checks(report, credentials, client_factory=client_factory, include_network=include_network)
    return report


def _check_target_account(
    config: CopierConfig,
    target: CopyTargetAccountConfig,
    *,
    client_factory: ClientFactory,
    include_network: bool,
    session: Session | None,
) -> dict[str, Any]:
    credentials = _target_credentials(target)
    report = _base_account_report("target", target.name, credentials, include_network=include_network)
    report["checks"].extend(
        [
            _check(f"target:{target.name}.enabled", target.enabled, "info", f"Target {target.name} is enabled"),
            _check(
                f"target:{target.name}.account_id",
                bool(credentials.account_id),
                "blocker",
                f"Target {target.name} Webull account id is configured",
                {"env": target.account_id_env, "account": _mask(credentials.account_id)},
            ),
            _check(
                f"target:{target.name}.credentials",
                _has_credentials(credentials),
                "blocker",
                f"Target {target.name} Webull API endpoint, app key, and secret are configured",
                {
                    "endpoint_env": target.endpoint_env,
                    "app_key_env": target.api_key_env,
                    "app_secret_env": target.api_secret_env,
                },
            ),
            _check(
                f"target:{target.name}.sizing",
                target.sizing_mode in {"percent_equity", "equity_ratio"},
                "blocker",
                f"Target {target.name} uses percent-equity mirror sizing",
                {"sizing_mode": target.sizing_mode},
            ),
            _check(
                f"target:{target.name}.equity",
                _equity_available(session, credentials.account_id, target.equity, target.equity_env),
                "blocker",
                f"Target {target.name} account equity is available from latest Webull snapshot or fallback config",
                {"env": target.equity_env, "source": _equity_source(session, credentials.account_id, target.equity, target.equity_env)},
            ),
            _check(
                f"target:{target.name}.shorts",
                target.shorting_enabled and config.copy_shorts,
                "warning",
                f"Target {target.name} short copying is enabled",
                {"target_shorting_enabled": target.shorting_enabled, "global_copy_shorts": config.copy_shorts},
            ),
        ]
    )
    _add_network_checks(report, credentials, client_factory=client_factory, include_network=include_network)
    return report


def _base_account_report(
    role: str,
    name: str,
    credentials: WebullCredentials,
    *,
    include_network: bool,
) -> dict[str, Any]:
    return {
        "role": role,
        "name": name,
        "environment": credentials.environment,
        "account": _mask(credentials.account_id),
        "network_checked": include_network,
        "checks": [],
    }


def _add_network_checks(
    report: dict[str, Any],
    credentials: WebullCredentials,
    *,
    client_factory: ClientFactory,
    include_network: bool,
) -> None:
    prefix = f"{report['role']}:{report['name']}"
    if not include_network:
        report["checks"].append(_check(f"{prefix}.network_skipped", True, "info", "Network checks skipped"))
        return
    if not _has_credentials(credentials):
        report["checks"].append(
            _check(f"{prefix}.network_ready", False, "blocker", "Network checks require endpoint, app key, and secret")
        )
        return

    try:
        client = client_factory(credentials)
        client.warm_up()
        report["checks"].append(_check(f"{prefix}.sdk_client", True, "blocker", "Webull SDK client initialized"))
    except Exception as exc:
        report["checks"].append(
            _check(f"{prefix}.sdk_client", False, "blocker", "Webull SDK client initialized", {"error": str(exc)})
        )
        return

    try:
        accounts = _normalize_account_list(client.get_account_list())
        report["accounts_seen"] = [_mask(_account_id(row)) for row in accounts if _account_id(row)]
        report["checks"].append(
            _check(
                f"{prefix}.account_list",
                True,
                "blocker",
                "Webull account list can be read",
                {"count": len(accounts)},
            )
        )
        if credentials.account_id:
            report["checks"].append(
                _check(
                    f"{prefix}.account_match",
                    any(_account_id(row) == credentials.account_id for row in accounts),
                    "blocker",
                    "Configured account id appears in Webull account list",
                    {"account": _mask(credentials.account_id)},
                )
            )
    except Exception as exc:
        report["checks"].append(
            _check(f"{prefix}.account_list", False, "blocker", "Webull account list can be read", {"error": str(exc)})
        )

    if not credentials.account_id:
        report["checks"].append(
            _check(f"{prefix}.positions_skipped", False, "blocker", "Position check requires configured account id")
        )
        return

    try:
        positions = client.get_account_positions(credentials.account_id)
        report["checks"].append(
            _check(
                f"{prefix}.positions",
                True,
                "blocker",
                "Webull positions endpoint can be read",
                {"count": len(positions)},
            )
        )
    except Exception as exc:
        report["checks"].append(
            _check(
                f"{prefix}.positions",
                False,
                "blocker",
                "Webull positions endpoint can be read",
                {"error": str(exc)},
            )
        )

    try:
        balance = client.get_account_balance(credentials.account_id)
        report["balance_keys"] = sorted(str(key) for key in balance.keys())[:20]
        report["checks"].append(
            _check(f"{prefix}.balance", True, "warning", "Webull balance/account-detail endpoint can be read")
        )
    except Exception as exc:
        report["checks"].append(
            _check(
                f"{prefix}.balance",
                False,
                "warning",
                "Webull balance/account-detail endpoint can be read",
                {"error": str(exc)},
            )
        )


def _database_checks(session: Session) -> list[PreflightCheck]:
    try:
        readiness = evaluate_copier_readiness(session)
    except Exception as exc:
        return [
            PreflightCheck(
                "database.readiness",
                False,
                "blocker",
                "Database copier readiness can be evaluated",
                {"error": str(exc)},
            )
        ]
    return [
        PreflightCheck(
            "database.readiness",
            readiness.get("ready", False),
            "blocker",
            "Database copier readiness is clean",
            {
                "blockers": len(readiness.get("blockers") or []),
                "warnings": len(readiness.get("warnings") or []),
            },
        )
    ]


def _master_credentials(config: CopierConfig) -> WebullCredentials:
    return WebullCredentials(
        app_key=os.getenv(config.master_app_key_env) or "",
        app_secret=os.getenv(config.master_app_secret_env) or "",
        endpoint=os.getenv(config.master_endpoint_env) or "",
        events_endpoint=os.getenv(config.master_events_endpoint_env) or None,
        account_id=config.master_account or os.getenv(config.master_account_env),
        environment=config.mode,
    )


def _target_credentials(target: CopyTargetAccountConfig) -> WebullCredentials:
    return WebullCredentials(
        app_key=os.getenv(target.api_key_env) or "",
        app_secret=os.getenv(target.api_secret_env) or "",
        endpoint=os.getenv(target.endpoint_env) or "",
        account_id=target.account_ref or os.getenv(target.account_id_env),
        environment=target.environment,
    )


def _has_credentials(credentials: WebullCredentials, *, require_events: bool = False) -> bool:
    if not (credentials.app_key and credentials.app_secret and credentials.endpoint):
        return False
    return bool(credentials.events_endpoint) if require_events else True


def _check(key: str, ok: bool, severity: str, label: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
    return PreflightCheck(key=key, ok=ok, severity=severity, label=label, context=context or {}).to_dict()


def _configured_float(value: float | None, env_name: str | None) -> float | None:
    if value is not None and value > 0:
        return float(value)
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


def _equity_available(session: Session | None, account_ref: str | None, value: float | None, env_name: str | None) -> bool:
    return _latest_snapshot_equity(session, account_ref) is not None or _configured_float(value, env_name) is not None


def _equity_source(session: Session | None, account_ref: str | None, value: float | None, env_name: str | None) -> str | None:
    if _latest_snapshot_equity(session, account_ref) is not None:
        return "webull_snapshot"
    if _configured_float(value, env_name) is not None:
        return "fallback_config"
    return None


def _latest_snapshot_equity(session: Session | None, account_ref: str | None) -> float | None:
    if session is None or not account_ref:
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


def _normalize_account_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        for key in ("accounts", "items", "list"):
            value = data.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    for key in ("accounts", "items", "list"):
        value = payload.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
    return []


def _account_id(row: dict[str, Any]) -> str | None:
    for key in ("account_id", "accountId", "account_ref", "accountNo", "account_no", "accountNumber", "id"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def _mask(value: str | None) -> str | None:
    if not value:
        return None
    text = str(value)
    if len(text) <= 4:
        return "*" * len(text)
    return f"{'*' * max(len(text) - 4, 0)}{text[-4:]}"
