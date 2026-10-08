from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.auth import (
    OPERATOR_API_TOKEN_ENV,
    OPERATOR_PASSWORD_ENV,
    OPERATOR_USERNAME_ENV,
)
from app.api.routes import v2_operations


def _clear_auth(monkeypatch) -> None:
    for name in (
        OPERATOR_USERNAME_ENV,
        OPERATOR_PASSWORD_ENV,
        OPERATOR_API_TOKEN_ENV,
    ):
        monkeypatch.delenv(name, raising=False)


def test_primary_dashboard_is_rithmic_read_only_console() -> None:
    html = Path("app/templates/rithmic_dashboard.html").read_text(encoding="utf-8")

    assert "Broker observation console" in html
    assert "Rithmic Test · read-only" in html
    assert "Broker submission is structurally disabled" in html
    assert "/v2/operations/overview?limit=40" in html
    assert 'href="/dashboard/legacy"' in html
    assert "orders-body" in html
    assert "executions-body" in html
    assert "reconciliation-body" in html
    assert "account-pnl-body" in html
    assert "positions-body" in html
    assert "risk-body" in html
    assert "brackets-body" in html
    assert "contracts-body" in html
    assert "/copier/settings" not in html
    assert "/pnl/refresh" not in html
    assert "kill-switch" not in html
    assert "Webull" not in html


def test_account_alias_is_stable_and_does_not_expose_broker_id() -> None:
    row = SimpleNamespace(
        account_id="internal-account-id",
        broker_account_id="raw-broker-account-id",
        currency="USD",
        access_type="TRADING",
        account_status="ACTIVE",
        allowlisted=True,
        order_copy_status=None,
        observed_at=None,
    )

    first = v2_operations._serialize_account(row)
    second = v2_operations._serialize_account(row)

    assert first["account"] == second["account"]
    assert first["account"].startswith("Account ")
    assert "internal-account-id" not in str(first)
    assert "raw-broker-account-id" not in str(first)


def test_operations_snapshot_has_complete_empty_state_shape() -> None:
    class EmptyRows:
        def all(self):
            return []

    class EmptySession:
        def scalar(self, _statement):
            return 0

        def scalars(self, _statement):
            return EmptyRows()

    snapshot = v2_operations.build_operations_snapshot(EmptySession(), limit=10)

    assert snapshot["mode"] == {
        "platform": "V2",
        "broker": "Rithmic",
        "environment": "TEST",
        "observation_only": True,
        "submission_enabled": False,
    }
    assert snapshot["storage"]["events"] == 0
    assert snapshot["storage"]["accounts"] == 0
    assert snapshot["storage"]["last_event_at"] is None
    for key in (
        "plants",
        "accounts",
        "orders",
        "executions",
        "brackets",
        "account_pnl",
        "positions",
        "risk",
        "reconciliation",
        "contracts",
    ):
        assert snapshot[key] == []


def test_operations_api_requires_operator_and_returns_read_only_snapshot(
    monkeypatch,
) -> None:
    _clear_auth(monkeypatch)
    monkeypatch.setenv(OPERATOR_API_TOKEN_ENV, "operator-token")

    snapshot = {
        "mode": {
            "platform": "V2",
            "broker": "Rithmic",
            "environment": "TEST",
            "observation_only": True,
            "submission_enabled": False,
        },
        "storage": {"events": 1},
        "plants": [],
        "accounts": [],
        "orders": [],
        "executions": [],
        "brackets": [],
        "account_pnl": [],
        "positions": [],
        "risk": [],
        "reconciliation": [],
        "contracts": [],
    }

    monkeypatch.setattr(
        v2_operations,
        "build_operations_snapshot",
        lambda _session, *, limit: snapshot,
    )

    async def fake_capture_health():
        return {
            "configured": True,
            "reachable": True,
            "live": True,
            "ready": False,
            "submission_enabled": False,
            "plants": [],
            "blockers": ["reconciliation_pending"],
        }

    monkeypatch.setattr(v2_operations, "fetch_capture_health", fake_capture_health)

    app = FastAPI()
    app.include_router(v2_operations.router)
    app.dependency_overrides[v2_operations.db_session] = lambda: object()
    client = TestClient(app)

    assert client.get("/v2/operations/overview").status_code == 401

    response = client.get(
        "/v2/operations/overview",
        headers={"X-Operator-Token": "operator-token"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["mode"]["broker"] == "Rithmic"
    assert body["mode"]["submission_enabled"] is False
    assert body["capture"]["submission_enabled"] is False
    assert body["generated_at"]
    assert response.headers["cache-control"] == "no-store"


def test_capture_health_is_optional(monkeypatch) -> None:
    monkeypatch.delenv("RITHMIC_CAPTURE_BASE_URL", raising=False)

    import asyncio

    health = asyncio.run(v2_operations.fetch_capture_health())

    assert health["configured"] is False
    assert health["reachable"] is False
    assert health["submission_enabled"] is False
    assert health["status"] == "health_url_not_configured"
