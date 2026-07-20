from __future__ import annotations

import hashlib
import os
from pathlib import Path
from threading import Lock
from typing import Any, Callable

from app.copier.models import WebullCredentials, WebullEquityOrder


_TOKEN_LOCKS_GUARD = Lock()
_TOKEN_INIT_LOCKS: dict[str, Lock] = {}


class WebullClientError(RuntimeError):
    pass


class WebullSDKUnavailable(WebullClientError):
    pass


class WebullTradingClient:
    """Thin adapter over Webull's official Python SDK.

    The SDK is imported lazily so local tests can validate request construction
    without Webull credentials or the package installed.
    """

    def __init__(
        self,
        credentials: WebullCredentials,
        trade_client: Any | None = None,
        api_client_factory: Callable[..., Any] | None = None,
        trade_client_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.credentials = credentials
        self._trade_client = trade_client
        self._api_client_factory = api_client_factory
        self._trade_client_factory = trade_client_factory

    @classmethod
    def from_env(
        cls,
        app_key_env: str,
        app_secret_env: str,
        endpoint_env: str,
        account_id_env: str | None = None,
        environment: str = "test",
    ) -> "WebullTradingClient":
        app_key = os.getenv(app_key_env)
        app_secret = os.getenv(app_secret_env)
        endpoint = os.getenv(endpoint_env)
        account_id = os.getenv(account_id_env) if account_id_env else None
        missing = [
            name
            for name, value in [
                (app_key_env, app_key),
                (app_secret_env, app_secret),
                (endpoint_env, endpoint),
            ]
            if not value
        ]
        if missing:
            raise WebullClientError(f"Missing Webull environment values: {', '.join(missing)}")
        return cls(
            WebullCredentials(
                app_key=app_key or "",
                app_secret=app_secret or "",
                endpoint=endpoint or "",
                account_id=account_id,
                environment=environment,
            )
        )

    @property
    def trade_client(self) -> Any:
        if self._trade_client is None:
            self._trade_client = self._build_trade_client()
        return self._trade_client

    def warm_up(self) -> None:
        _ = self.trade_client

    def _build_trade_client(self) -> Any:
        api_client_factory = self._api_client_factory
        trade_client_factory = self._trade_client_factory
        if api_client_factory is None or trade_client_factory is None:
            try:
                from webull.core.client import ApiClient
                from webull.trade.trade_client import TradeClient
            except Exception as exc:  # pragma: no cover - depends on optional SDK
                raise WebullSDKUnavailable("webull-openapi-python-sdk is not installed") from exc
            api_client_factory = ApiClient
            trade_client_factory = TradeClient

        api_client = api_client_factory(
            self.credentials.app_key,
            self.credentials.app_secret,
            self.credentials.region_id,
        )
        api_client.add_endpoint(self.credentials.region_id, self.credentials.endpoint)
        token_dir = _token_dir_for_credentials(self.credentials)
        set_token_dir = getattr(api_client, "set_token_dir", None)
        if callable(set_token_dir):
            set_token_dir(token_dir)

        # TradeClient initializes Webull's reusable 2FA token in its constructor.
        # Several worker jobs can initialize the same account concurrently, so
        # serialize that work and let later clients reuse the verified token.
        with _token_init_lock(token_dir):
            return trade_client_factory(api_client)

    def get_account_list(self) -> list[dict[str, Any]]:
        response = self.trade_client.account_v2.get_account_list()
        return self._json_or_raise(response)

    def get_account_balance(self, account_id: str) -> dict[str, Any]:
        account_v2 = self.trade_client.account_v2
        for method_name in ("get_account_balance", "get_account_balances", "get_account_detail", "get_account_info"):
            method = getattr(account_v2, method_name, None)
            if callable(method):
                response = method(account_id)
                payload = self._json_or_raise(response)
                return payload if isinstance(payload, dict) else {"data": payload}
        raise WebullClientError("Webull SDK account_v2 does not expose account balance/detail lookup")

    def place_equity_order(self, account_id: str, order: WebullEquityOrder) -> dict[str, Any]:
        response = self.trade_client.order_v2.place_order(account_id, [order.to_webull_payload()])
        return self._json_or_raise(response)

    def cancel_order(self, account_id: str, client_order_id: str) -> dict[str, Any]:
        response = self.trade_client.order_v2.cancel_order(account_id, client_order_id)
        return self._json_or_raise(response)

    def get_order_detail(self, account_id: str, client_order_id: str) -> dict[str, Any]:
        response = self.trade_client.order_v2.get_order_detail(account_id, client_order_id)
        return self._json_or_raise(response)

    def get_account_positions(self, account_id: str) -> list[dict[str, Any]]:
        account_v2 = self.trade_client.account_v2
        if hasattr(account_v2, "get_account_position"):
            response = account_v2.get_account_position(account_id)
        elif hasattr(account_v2, "get_account_positions"):
            response = account_v2.get_account_positions(account_id)
        else:
            raise WebullClientError("Webull SDK account_v2 does not expose account position lookup")
        payload = self._json_or_raise(response)
        if isinstance(payload, list):
            return payload
        data = payload.get("data") if isinstance(payload, dict) else None
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("positions", "items", "list"):
                value = data.get(key)
                if isinstance(value, list):
                    return value
        if isinstance(payload, dict):
            for key in ("positions", "items", "list"):
                value = payload.get(key)
                if isinstance(value, list):
                    return value
        return []

    def _json_or_raise(self, response: Any) -> Any:
        status_code = getattr(response, "status_code", None)
        try:
            payload = response.json()
        except Exception as exc:
            raise WebullClientError("Webull SDK response did not contain JSON") from exc
        if status_code is not None and not (200 <= int(status_code) < 300):
            raise WebullClientError(f"Webull request failed with status {status_code}: {payload}")
        return payload


def _token_dir_for_credentials(credentials: WebullCredentials) -> str:
    """Return a stable, credential-scoped SDK token directory.

    Webull's SDK otherwise stores every App Key's access token in the same
    ``conf/token.txt`` file. This application connects multiple Webull accounts,
    so the shared default causes one account's token to overwrite another's and
    can trigger repeated Open API verification prompts.
    """

    base_dir = Path(os.getenv("WEBULL_OPENAPI_TOKEN_DIR", "conf/webull_tokens")).expanduser()
    identity = "\0".join(
        [
            credentials.app_key,
            credentials.endpoint,
            credentials.account_id or "",
        ]
    )
    credential_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    return str((base_dir / credential_id).absolute())


def _token_init_lock(token_dir: str) -> Lock:
    with _TOKEN_LOCKS_GUARD:
        lock = _TOKEN_INIT_LOCKS.get(token_dir)
        if lock is None:
            lock = Lock()
            _TOKEN_INIT_LOCKS[token_dir] = lock
        return lock
