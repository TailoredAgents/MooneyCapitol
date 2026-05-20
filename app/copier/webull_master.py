from __future__ import annotations

from typing import Any, Callable, Iterable

from app.copier.models import WebullCredentials
from app.copier.state import set_copier_status


class WebullMasterEventListener:
    """Webull order-status event listener wrapper.

    This is intentionally thin until live credentials are available. It uses
    Webull's documented gRPC `TradeEventsClient` path and accepts an injected
    factory for tests/replay.
    """

    def __init__(
        self,
        credentials: WebullCredentials,
        account_ids: Iterable[str],
        on_event: Callable[[str, str, dict, Any], None],
        events_client: Any | None = None,
        events_client_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.credentials = credentials
        self.account_ids = [account_id for account_id in account_ids if account_id]
        self.on_event = on_event
        self._events_client = events_client
        self._events_client_factory = events_client_factory
        if self._events_client is not None:
            self._events_client.on_events_message = self.on_event

    @property
    def events_client(self) -> Any:
        if self._events_client is None:
            self._events_client = self._build_events_client()
        return self._events_client

    def _build_events_client(self) -> Any:
        factory = self._events_client_factory
        if factory is None:
            try:
                from webull.trade.trade_events_client import TradeEventsClient
            except Exception as exc:  # pragma: no cover - depends on optional SDK
                raise RuntimeError("webull-openapi-python-sdk is not installed") from exc
            factory = TradeEventsClient
        client = factory(
            self.credentials.app_key,
            self.credentials.app_secret,
            self.credentials.region_id,
            host=self.credentials.events_endpoint,
        )
        client.on_events_message = self.on_event
        return client

    def subscribe(self) -> None:
        if not self.account_ids:
            raise ValueError("At least one Webull account_id is required for event subscription")
        set_copier_status(master_connected=False, state="subscribing")
        self.events_client.do_subscribe(self.account_ids)
        set_copier_status(master_connected=True, state="subscribed")
