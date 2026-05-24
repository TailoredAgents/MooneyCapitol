from __future__ import annotations

from app.copier.models import WebullCredentials
from app.services.webull_paper import WEBULL_PAPER_ENV, webull_paper_readiness


def _clear_webull_paper_env(monkeypatch):
    monkeypatch.delenv("PAPER_TRADER_BROKER_MODE", raising=False)
    monkeypatch.delenv("AI_LIVE_TRADING_ENABLED", raising=False)
    for env_name in WEBULL_PAPER_ENV.values():
        monkeypatch.delenv(env_name, raising=False)


def test_webull_paper_readiness_defaults_to_internal_simulation(monkeypatch):
    _clear_webull_paper_env(monkeypatch)

    payload = webull_paper_readiness()

    assert payload["ready"] is True
    assert payload["status"] == "internal"
    assert payload["account_ready"] is False
    assert payload["network_checked"] is False
    assert any(check["key"] == "internal_simulation" and check["ok"] for check in payload["checks"])


def test_webull_paper_readiness_requires_explicit_paper_config(monkeypatch):
    _clear_webull_paper_env(monkeypatch)
    monkeypatch.setenv("PAPER_TRADER_BROKER_MODE", "webull_paper")
    monkeypatch.setenv("WEBULL_AI_PAPER_APP_KEY", "key")

    payload = webull_paper_readiness()

    assert payload["ready"] is False
    assert payload["status"] == "missing_config"
    assert "WEBULL_AI_PAPER_ACCOUNT_ID" in payload["config"]["missing_env"]
    assert any(check["key"] == "webull_paper_env_configured" and not check["ok"] for check in payload["checks"])


def test_webull_paper_readiness_validates_account_read_only(monkeypatch):
    _clear_webull_paper_env(monkeypatch)
    monkeypatch.setenv("PAPER_TRADER_BROKER_MODE", "webull_paper")
    monkeypatch.setenv("WEBULL_AI_PAPER_API_ENDPOINT", "us-openapi-alb.uat.webullbroker.com")
    monkeypatch.setenv("WEBULL_AI_PAPER_EVENTS_ENDPOINT", "us-openapi-events.uat.webullbroker.com")
    monkeypatch.setenv("WEBULL_AI_PAPER_APP_KEY", "key")
    monkeypatch.setenv("WEBULL_AI_PAPER_APP_SECRET", "secret")
    monkeypatch.setenv("WEBULL_AI_PAPER_ACCOUNT_ID", "paper-acct-1")

    class FakeClient:
        def __init__(self, credentials: WebullCredentials):
            self.credentials = credentials

        def get_account_list(self):
            return [{"account_id": "paper-acct-1"}]

        def get_account_balance(self, account_id):
            return {"account_id": account_id, "net_liquidation": "10000"}

        def get_account_positions(self, account_id):
            return []

    payload = webull_paper_readiness(skip_network=False, client_factory=FakeClient)

    assert payload["ready"] is True
    assert payload["status"] == "validated"
    assert payload["account_ready"] is True
    assert payload["network_checked"] is True
    assert any(check["key"] == "webull_paper_account_visible" and check["ok"] for check in payload["checks"])
