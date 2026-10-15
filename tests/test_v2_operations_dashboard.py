from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
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


def test_primary_dashboard_is_rithmic_operator_console() -> None:
    html = Path("app/templates/rithmic_dashboard.html").read_text(encoding="utf-8")

    assert "Five-account copy operator console" in html
    assert "Rithmic Test / operator controls" in html
    assert "Live broker submission remains structurally impossible" in html
    assert "/v2/operations/overview?limit=40" in html
    assert "/v2/copier/status" in html
    assert "/v2/copier/control" in html
    assert "/v2/copier/control/csrf" in html
    assert "/v2/copier/control/audit?limit=100" in html
    assert "Master and follower readiness" in html
    assert "Immutable control history" in html
    assert "REQUEST_FLATTEN_AND_DISABLE" in html
    assert "flatten-confirmation" in html
    assert "Copy lifecycle and parity evidence" in html
    assert "Master-relative sizing" in html
    assert "/dashboard/legacy" not in html
    assert "Legacy V1" not in html
    assert "orders-body" in html
    assert "executions-body" in html
    assert "reconciliation-body" in html
    assert "account-pnl-body" in html
    assert "positions-body" in html
    assert "risk-body" in html
    assert "brackets-body" in html
    assert "contracts-body" in html
    assert "account-cards" in html
    assert "follower-controls-body" in html
    assert "control-audit-body" in html
    assert "copy-intents-body" in html
    assert "copy-events-body" in html
    assert "copy-findings-body" in html
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
    mapped = v2_operations._serialize_account(
        row,
        {"internal-account-id": "follower-1"},
    )

    assert first["account"] == second["account"]
    assert first["account"].startswith("Account ")
    assert first["connection_id"] is None
    assert mapped["connection_id"] == "follower-1"
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
    assert snapshot["copy_execution"] == {
        "intents": [],
        "lifecycle_events": [],
        "delivery": [],
        "latency": [],
        "reconciliation": [],
        "findings": [],
    }


def test_copy_execution_projection_is_redacted_and_reports_latency() -> None:
    now = datetime(2026, 10, 15, 14, 0, tzinfo=timezone.utc)
    command = SimpleNamespace(
        command_id="private-command-id",
        broker_order_id="private-broker-order-id",
        action="BRACKET",
        status="ACKNOWLEDGED",
        side="BUY",
        quantity=2,
        cumulative_filled_quantity=0,
        order_type="MARKET",
        stop_ticks=20,
        target_ticks=40,
        attempt_count=1,
        updated_at=now,
    )
    intent = SimpleNamespace(
        intent_id="private-intent-id",
        follower_connection_id="follower-2",
        master_event_kind="NEW",
        directive="SUBMIT_BRACKET",
        status="ACKNOWLEDGED",
        allowed=True,
        reason="allowed",
        symbol="MNQZ6",
        exchange="CME",
        current_signed_quantity=1,
        evaluated_target_signed_quantity=2,
        achievable_target_signed_quantity=2,
        delta_signed_quantity=1,
        desired_quantity=Decimal("2"),
        desired_risk_dollars=Decimal("50"),
        achievable_risk_dollars=Decimal("50"),
        desired_risk_fraction=Decimal("0.05"),
        achievable_risk_fraction=Decimal("0.05"),
        desired_exposure_fraction=Decimal("0.1"),
        achievable_exposure_fraction=Decimal("0.1"),
        risk_parity_error=Decimal("0"),
        exposure_parity_error=Decimal("0"),
        parity_error=Decimal("0"),
        target_margin_required=Decimal("200"),
        updated_at=now,
    )
    latency = SimpleNamespace(
        follower_connection_id="follower-2",
        event_occurred_at=now,
        event_received_at=now + timedelta(milliseconds=5),
        decision_at=now + timedelta(milliseconds=8),
        submission_started_at=now + timedelta(milliseconds=10),
        dispatch_response_at=now + timedelta(milliseconds=18),
        acknowledged_at=now + timedelta(milliseconds=25),
        first_fill_at=None,
        completed_at=None,
        updated_at=now + timedelta(milliseconds=25),
    )

    serialized = v2_operations._serialize_copy_intent(intent, command)
    timing = v2_operations._serialize_copy_latency(latency)

    assert serialized["connection_id"] == "follower-2"
    assert serialized["copy"].startswith("Copy ")
    assert serialized["command"]["command"].startswith("Command ")
    assert serialized["command"]["stop_ticks"] == 20
    assert serialized["parity_error"] == "0"
    assert "private-intent-id" not in str(serialized)
    assert "private-command-id" not in str(serialized)
    assert "private-broker-order-id" not in str(serialized)
    assert timing["event_to_receive_ms"] == 5
    assert timing["receive_to_decision_ms"] == 3
    assert timing["decision_to_submission_ms"] == 2
    assert timing["submission_to_response_ms"] == 8
    assert timing["event_to_ack_ms"] == 25


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
