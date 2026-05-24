from __future__ import annotations

import os
from typing import Any, Callable

from app.copier.models import WebullCredentials
from app.copier.webull_client import WebullClientError, WebullTradingClient


ClientFactory = Callable[[WebullCredentials], WebullTradingClient]

BROKER_MODE_ENV = "PAPER_TRADER_BROKER_MODE"
WEBULL_PAPER_ENV = {
    "api_endpoint": "WEBULL_AI_PAPER_API_ENDPOINT",
    "events_endpoint": "WEBULL_AI_PAPER_EVENTS_ENDPOINT",
    "app_key": "WEBULL_AI_PAPER_APP_KEY",
    "app_secret": "WEBULL_AI_PAPER_APP_SECRET",
    "account_id": "WEBULL_AI_PAPER_ACCOUNT_ID",
    "display_name": "WEBULL_AI_PAPER_DISPLAY_NAME",
}
INTERNAL_MODES = {"", "internal", "simulation", "sim"}
WEBULL_PAPER_MODES = {"webull_paper", "webull_test", "paper", "test"}
LIVE_MODES = {"live", "webull_live", "production", "prod"}
REQUIRED_WEBULL_KEYS = ("api_endpoint", "app_key", "app_secret", "account_id")


def webull_paper_config() -> dict[str, Any]:
    mode = os.getenv(BROKER_MODE_ENV, "internal").strip().lower()
    values = {key: os.getenv(env_name, "").strip() for key, env_name in WEBULL_PAPER_ENV.items()}
    missing = [WEBULL_PAPER_ENV[key] for key in REQUIRED_WEBULL_KEYS if not values.get(key)]
    configured = not missing
    return {
        "broker_mode": mode or "internal",
        "broker_mode_env": BROKER_MODE_ENV,
        "configured": configured,
        "missing_env": missing,
        "display_name": values["display_name"] or "AI Paper Account",
        "api_endpoint": values["api_endpoint"],
        "events_endpoint": values["events_endpoint"],
        "account_id": values["account_id"],
        "masked_account_id": _mask(values["account_id"]),
        "has_app_key": bool(values["app_key"]),
        "has_app_secret": bool(values["app_secret"]),
    }


def webull_paper_readiness(
    *,
    skip_network: bool = True,
    client_factory: ClientFactory | None = None,
) -> dict[str, Any]:
    config = webull_paper_config()
    mode = str(config["broker_mode"]).lower()
    checks: list[dict[str, Any]] = []

    def add(key: str, ok: bool, label: str, severity: str = "info", context: dict[str, Any] | None = None) -> None:
        checks.append(
            {
                "key": key,
                "ok": bool(ok),
                "label": label,
                "severity": severity,
                "context": context or {},
            }
        )

    safe_mode = mode in INTERNAL_MODES or mode in WEBULL_PAPER_MODES
    add(
        "broker_mode_safe",
        safe_mode and mode not in LIVE_MODES,
        "AI paper trader broker mode is safe",
        "blocker" if mode in LIVE_MODES else "warning",
        {"mode": mode},
    )

    if mode in INTERNAL_MODES:
        add("internal_simulation", True, "AI paper trader is using broker-disconnected internal simulation")
        add("webull_paper_optional", True, "Webull paper account is optional until broker paper mode is enabled")
        return {
            "ready": True,
            "status": "internal",
            "message": "AI paper trading is running in internal simulation mode. No Webull paper orders are submitted.",
            "network_checked": False,
            "account_ready": False,
            "config": config,
            "checks": checks,
        }

    if mode not in WEBULL_PAPER_MODES:
        return {
            "ready": False,
            "status": "invalid_mode",
            "message": f"Unsupported AI paper broker mode: {mode}",
            "network_checked": False,
            "account_ready": False,
            "config": config,
            "checks": checks,
        }

    add(
        "webull_paper_env_configured",
        bool(config["configured"]),
        "Webull AI paper endpoint, app key, app secret, and account id are configured",
        "warning",
        {"missing_env": config["missing_env"]},
    )
    add(
        "webull_paper_events_endpoint_configured",
        bool(config["events_endpoint"]),
        "Webull AI paper events endpoint is configured for future paper order events",
        "warning",
        {"env": WEBULL_PAPER_ENV["events_endpoint"]},
    )

    if not config["configured"]:
        return {
            "ready": False,
            "status": "missing_config",
            "message": "Webull AI paper mode is selected, but required paper account settings are missing.",
            "network_checked": False,
            "account_ready": False,
            "config": config,
            "checks": checks,
        }

    if skip_network:
        return {
            "ready": False,
            "status": "configured",
            "message": "Webull AI paper account settings are present, but read-only network validation still needs to run.",
            "network_checked": False,
            "account_ready": False,
            "config": config,
            "checks": checks,
        }

    try:
        credentials = _credentials_from_env()
        client = client_factory(credentials) if client_factory else WebullTradingClient(credentials)
        accounts = client.get_account_list()
        account_ids = _extract_account_ids(accounts)
        account_visible = credentials.account_id in account_ids
        add(
            "webull_paper_account_visible",
            bool(account_visible),
            "Configured Webull AI paper account is visible to the app credentials",
            "warning",
            {"account_ids": sorted(_mask(value) for value in account_ids)},
        )
        balance = client.get_account_balance(credentials.account_id or "")
        add(
            "webull_paper_balance_available",
            bool(balance),
            "Webull AI paper account balance can be read",
            "warning",
            {"balance_keys": sorted(balance.keys()) if isinstance(balance, dict) else []},
        )
        positions = client.get_account_positions(credentials.account_id or "")
        add(
            "webull_paper_positions_available",
            isinstance(positions, list),
            "Webull AI paper account positions can be read",
            "warning",
            {"position_count": len(positions) if isinstance(positions, list) else None},
        )
        account_ready = bool(account_visible and balance is not None and isinstance(positions, list))
        return {
            "ready": account_ready,
            "status": "validated" if account_ready else "validation_failed",
            "message": "Webull AI paper account read-only validation completed.",
            "network_checked": True,
            "account_ready": account_ready,
            "accounts_found": len(account_ids),
            "config": config,
            "checks": checks,
        }
    except (WebullClientError, ValueError, RuntimeError) as exc:
        add(
            "webull_paper_network_validation",
            False,
            "Webull AI paper account read-only validation completed without errors",
            "warning",
            {"error": str(exc)},
        )
        return {
            "ready": False,
            "status": "validation_error",
            "message": "Webull AI paper account read-only validation failed.",
            "network_checked": True,
            "account_ready": False,
            "config": config,
            "checks": checks,
        }


def _credentials_from_env() -> WebullCredentials:
    endpoint = os.getenv(WEBULL_PAPER_ENV["api_endpoint"], "").strip()
    app_key = os.getenv(WEBULL_PAPER_ENV["app_key"], "").strip()
    app_secret = os.getenv(WEBULL_PAPER_ENV["app_secret"], "").strip()
    account_id = os.getenv(WEBULL_PAPER_ENV["account_id"], "").strip()
    events_endpoint = os.getenv(WEBULL_PAPER_ENV["events_endpoint"], "").strip() or None
    missing = [
        env_name
        for env_name, value in [
            (WEBULL_PAPER_ENV["api_endpoint"], endpoint),
            (WEBULL_PAPER_ENV["app_key"], app_key),
            (WEBULL_PAPER_ENV["app_secret"], app_secret),
            (WEBULL_PAPER_ENV["account_id"], account_id),
        ]
        if not value
    ]
    if missing:
        raise WebullClientError(f"Missing Webull AI paper environment values: {', '.join(missing)}")
    return WebullCredentials(
        app_key=app_key,
        app_secret=app_secret,
        endpoint=endpoint,
        events_endpoint=events_endpoint,
        account_id=account_id,
        environment="test",
    )


def _extract_account_ids(payload: Any) -> set[str]:
    ids: set[str] = set()
    keys = {"account_id", "accountId", "account_no", "accountNo", "account_number", "accountNumber", "account_sid", "accountSid"}
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in keys and value not in (None, ""):
                ids.add(str(value))
            else:
                ids.update(_extract_account_ids(value))
    elif isinstance(payload, list):
        for item in payload:
            ids.update(_extract_account_ids(item))
    return ids


def _mask(value: str | None) -> str | None:
    if not value:
        return None
    text = str(value)
    if len(text) <= 8:
        return "*" * len(text)
    return f"{text[:4]}...{text[-4:]}"
