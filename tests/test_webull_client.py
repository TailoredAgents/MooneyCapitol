from types import SimpleNamespace

import pytest

from app.copier.ids import copy_client_order_id
from app.copier.models import WebullCredentials, WebullEquityOrder
from app.copier.webull_client import WebullClientError, WebullTradingClient
from app.copier.webull_master import WebullMasterEventListener


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


class FakeOrderV2:
    def __init__(self):
        self.placed = []
        self.cancelled = []
        self.details = []

    def place_order(self, account_id, orders):
        self.placed.append((account_id, orders))
        return FakeResponse({"ok": True, "orders": orders})

    def cancel_order(self, account_id, client_order_id):
        self.cancelled.append((account_id, client_order_id))
        return FakeResponse({"ok": True})

    def get_order_detail(self, account_id, client_order_id):
        self.details.append((account_id, client_order_id))
        return FakeResponse({"client_order_id": client_order_id, "status": "FILLED"})


class FakeTradeClient:
    def __init__(self):
        self.order_v2 = FakeOrderV2()
        self.account_v2 = SimpleNamespace(
            get_account_list=lambda: FakeResponse([{"account_id": "acct-1"}]),
            get_account_balance=lambda account_id: FakeResponse({"account_id": account_id, "net_liquidation": "10000"}),
            get_account_position=lambda account_id: FakeResponse(
                {"data": {"positions": [{"symbol": "AAPL", "quantity": "3"}]}}
            ),
        )


def test_copy_client_order_id_is_deterministic_and_webull_sized():
    first = copy_client_order_id("exec-123", "personal")
    second = copy_client_order_id("exec-123", "personal")
    different = copy_client_order_id("exec-123", "other")

    assert first == second
    assert first != different
    assert len(first) == 32
    assert first.startswith("mc")


def test_equity_market_order_payload_matches_webull_shape():
    order = WebullEquityOrder(
        symbol="aapl",
        side="BUY",
        quantity=1,
        client_order_id="mc123",
    )

    payload = order.to_webull_payload()

    assert payload == {
        "combo_type": "NORMAL",
        "client_order_id": "mc123",
        "symbol": "AAPL",
        "instrument_type": "EQUITY",
        "market": "US",
        "order_type": "MARKET",
        "quantity": "1",
        "support_trading_session": "ALL",
        "side": "BUY",
        "time_in_force": "DAY",
        "entrust_type": "QTY",
    }


def test_limit_order_requires_limit_price():
    order = WebullEquityOrder(
        symbol="AAPL",
        side="BUY",
        quantity=1,
        client_order_id="mc123",
        order_type="LIMIT",
    )

    with pytest.raises(ValueError):
        order.to_webull_payload()


def test_webull_trading_client_places_order_with_injected_sdk_client():
    fake = FakeTradeClient()
    client = WebullTradingClient(
        WebullCredentials(app_key="key", app_secret="secret", endpoint="endpoint", account_id="acct-1"),
        trade_client=fake,
    )
    order = WebullEquityOrder(symbol="AAPL", side="SELL", quantity=2, client_order_id="mc123")

    result = client.place_equity_order("acct-1", order)

    assert result["ok"] is True
    account_id, orders = fake.order_v2.placed[0]
    assert account_id == "acct-1"
    assert orders[0]["side"] == "SELL"
    assert orders[0]["quantity"] == "2"


def test_webull_trading_client_gets_account_positions():
    client = WebullTradingClient(
        WebullCredentials(app_key="key", app_secret="secret", endpoint="endpoint", account_id="acct-1"),
        trade_client=FakeTradeClient(),
    )

    positions = client.get_account_positions("acct-1")

    assert positions == [{"symbol": "AAPL", "quantity": "3"}]


def test_webull_trading_client_gets_account_balance():
    client = WebullTradingClient(
        WebullCredentials(app_key="key", app_secret="secret", endpoint="endpoint", account_id="acct-1"),
        trade_client=FakeTradeClient(),
    )

    balance = client.get_account_balance("acct-1")

    assert balance["account_id"] == "acct-1"
    assert balance["net_liquidation"] == "10000"


def test_webull_trading_client_warm_up_builds_trade_client():
    calls = []

    class ApiClient:
        def __init__(self, app_key, app_secret, region_id):
            calls.append(("api", app_key, app_secret, region_id))

        def add_endpoint(self, region_id, endpoint):
            calls.append(("endpoint", region_id, endpoint))

        def set_token_dir(self, token_dir):
            calls.append(("token_dir", token_dir))

    class TradeClient:
        def __init__(self, api_client):
            calls.append(("trade", api_client))
            self.order_v2 = FakeOrderV2()

    client = WebullTradingClient(
        WebullCredentials(app_key="key", app_secret="secret", endpoint="endpoint"),
        api_client_factory=ApiClient,
        trade_client_factory=TradeClient,
    )

    client.warm_up()

    assert calls[0] == ("api", "key", "secret", "us")
    assert calls[1] == ("endpoint", "us", "endpoint")
    assert calls[2][0] == "token_dir"
    assert calls[3][0] == "trade"


def test_webull_trading_clients_use_separate_token_dirs_per_credentials(monkeypatch, tmp_path):
    token_dirs = []

    class ApiClient:
        def __init__(self, app_key, app_secret, region_id):
            pass

        def add_endpoint(self, region_id, endpoint):
            pass

        def set_token_dir(self, token_dir):
            token_dirs.append(token_dir)

    class TradeClient:
        def __init__(self, api_client):
            self.order_v2 = FakeOrderV2()

    monkeypatch.setenv("WEBULL_OPENAPI_TOKEN_DIR", str(tmp_path / "tokens"))
    clients = [
        WebullTradingClient(
            WebullCredentials(
                app_key="master-key",
                app_secret="master-secret",
                endpoint="https://api.webull.com",
                account_id="master-account",
                environment="live",
            ),
            api_client_factory=ApiClient,
            trade_client_factory=TradeClient,
        ),
        WebullTradingClient(
            WebullCredentials(
                app_key="target-key",
                app_secret="target-secret",
                endpoint="https://api.webull.com",
                account_id="target-account",
                environment="live",
            ),
            api_client_factory=ApiClient,
            trade_client_factory=TradeClient,
        ),
    ]

    for client in clients:
        client.warm_up()

    assert len(token_dirs) == 2
    assert token_dirs[0] != token_dirs[1]
    assert all(token_dir.startswith(str(tmp_path / "tokens")) for token_dir in token_dirs)


def test_webull_trading_client_raises_on_non_success_response():
    client = WebullTradingClient(
        WebullCredentials(app_key="key", app_secret="secret", endpoint="endpoint"),
        trade_client=SimpleNamespace(
            account_v2=SimpleNamespace(get_account_list=lambda: FakeResponse({"error": "bad"}, status_code=417))
        ),
    )

    with pytest.raises(WebullClientError):
        client.get_account_list()


def test_webull_master_event_listener_subscribes_with_injected_client(monkeypatch):
    events = []

    class FakeEventsClient:
        def __init__(self):
            self.on_events_message = None
            self.subscribed = None

        def do_subscribe(self, account_ids):
            self.subscribed = account_ids

    fake = FakeEventsClient()
    monkeypatch.setattr("app.copier.webull_master.set_copier_status", lambda **updates: updates)
    listener = WebullMasterEventListener(
        WebullCredentials(app_key="key", app_secret="secret", endpoint="endpoint"),
        account_ids=["acct-master"],
        on_event=lambda *args: events.append(args),
        events_client=fake,
    )

    listener.subscribe()

    assert fake.subscribed == ["acct-master"]
    assert fake.on_events_message is not None


def test_webull_master_event_listener_passes_events_endpoint(monkeypatch):
    calls = []

    class FakeEventsClient:
        def __init__(self, app_key, app_secret, region_id, host=None):
            calls.append((app_key, app_secret, region_id, host))
            self.on_events_message = None
            self.subscribed = None

        def do_subscribe(self, account_ids):
            self.subscribed = account_ids

    monkeypatch.setattr("app.copier.webull_master.set_copier_status", lambda **updates: updates)
    listener = WebullMasterEventListener(
        WebullCredentials(
            app_key="key",
            app_secret="secret",
            endpoint="us-openapi-alb.uat.webullbroker.com",
            events_endpoint="us-openapi-events.uat.webullbroker.com",
        ),
        account_ids=["acct-master"],
        on_event=lambda *args: None,
        events_client_factory=FakeEventsClient,
    )

    listener.subscribe()

    assert calls == [("key", "secret", "us", "us-openapi-events.uat.webullbroker.com")]
