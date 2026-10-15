from __future__ import annotations

from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.api.auth import (
    OPERATOR_API_TOKEN_ENV,
    OPERATOR_PASSWORD_ENV,
    OPERATOR_USERNAME_ENV,
    SESSION_COOKIE_NAME,
    create_dashboard_session_token,
)
from app.api.routes import v2_copier
from app.db.models import Base
from app.v2.db_models import (
    V2CopyControlAudit,
    V2CopyFollowerControl,
    V2CopyProfile,
)
from app.v2.execution.copy_control import DEFAULT_COPY_PROFILE_ID
from app.v2.execution.copy_control_store import SqlAlchemyCopyControlStore


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(v2_copier.router)
    return TestClient(app)


def _control_client() -> tuple[TestClient, Session]:
    engine = create_engine(
        "sqlite+pysqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(
        engine,
        tables=[
            V2CopyProfile.__table__,
            V2CopyFollowerControl.__table__,
            V2CopyControlAudit.__table__,
        ],
    )
    session = Session(engine)
    now = datetime(2026, 10, 14, 14, 0, tzinfo=timezone.utc)
    session.add(
        V2CopyProfile(
            profile_id=DEFAULT_COPY_PROFILE_ID,
            name="Mooney five-account copier",
            master_connection_id="master",
            global_paused=True,
            revision=1,
            created_at=now,
            updated_at=now,
        )
    )
    for number in range(1, 5):
        session.add(
            V2CopyFollowerControl(
                profile_id=DEFAULT_COPY_PROFILE_ID,
                connection_id=f"follower-{number}",
                follower_number=number,
                state="DISABLED",
                revision=1,
                updated_at=now,
            )
        )
    session.commit()
    store = SqlAlchemyCopyControlStore(session)
    app = FastAPI()
    app.include_router(v2_copier.router)
    app.dependency_overrides[v2_copier.copy_control_store] = lambda: store
    return TestClient(app), session


def _preview_payload() -> dict:
    return {
        "master": {
            "account_label": "secret-master-id",
            "product_code": "NQ",
            "expiration": "2027-03-19",
            "side": "BUY",
            "quantity": 1,
            "entry_price": "20000",
            "stop_price": "19980",
            "equity": "5000",
        },
        "followers": [
            {
                "account_label": "secret-follower-id",
                "equity": "500",
                "available_margin": "500",
            }
        ],
        "policy": {
            "max_risk_fraction": "0.10",
            "max_contracts": 10,
            "max_concurrent_open_risk_fraction": "0.20",
            "daily_realized_loss_fraction": "0.10",
            "daily_total_loss_fraction": "0.15",
            "max_relative_parity_error": "0",
        },
        "contracts": {
            "nq_initial_margin": "1000",
            "mnq_initial_margin": "100",
        },
        "global_kill_switch": False,
    }


def test_v2_copier_status_requires_auth_and_redacts_configuration(
    monkeypatch,
) -> None:
    monkeypatch.setenv(OPERATOR_API_TOKEN_ENV, "operator-token")
    slots = ("MASTER", "FOLLOWER_1", "FOLLOWER_2", "FOLLOWER_3", "FOLLOWER_4")
    for number, slot in enumerate(slots, start=1):
        monkeypatch.setenv(
            f"V2_RITHMIC_{slot}_ACCOUNT_ID",
            f"raw-private-account-{number}",
        )
        monkeypatch.setenv(
            f"V2_RITHMIC_{slot}_USERNAME",
            f"raw-private-user-{number}",
        )
        monkeypatch.setenv(
            f"V2_RITHMIC_{slot}_PASSWORD",
            f"raw-private-password-{number}",
        )
    client = _client()

    assert client.get("/v2/copier/status").status_code == 401
    response = client.get(
        "/v2/copier/status",
        headers={"X-Operator-Token": "operator-token"},
    )

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert body["architecture"] == "five_independent_account_connections"
    assert body["submission_enabled"] is False
    assert body["master_configured"] is True
    assert body["configured_follower_count"] == 4
    assert body["credential_complete_account_count"] == 5
    assert len(body["accounts"]) == 5
    assert body["accounts"][0]["connection"] == "Master"
    assert body["planner_implemented"] is True
    assert body["connection_supervisor_implemented"] is True
    assert body["simulated_connections_implemented"] is True
    assert body["required_supervised_session_count"] == 10
    assert body["supervisor_runtime_health_attached"] is False
    assert body["durable_control_plane_implemented"] is True
    assert body["control_defaults_fail_closed"] is True
    assert body["operator_control_ui_implemented"] is False
    assert body["accounts"][0]["required_plants"] == ["ORDER", "PNL"]
    assert body["execution_adapter_implemented"] is False
    assert "raw-private-account" not in response.text
    assert "raw-private-user" not in response.text
    assert "raw-private-password" not in response.text


def test_v2_copier_preview_is_exact_shadow_only_and_redacts_accounts(
    monkeypatch,
) -> None:
    monkeypatch.delenv(OPERATOR_API_TOKEN_ENV, raising=False)
    response = _client().post("/v2/copier/preview", json=_preview_payload())

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert body["mode"] == "SHADOW"
    assert body["submission_enabled"] is False
    assert body["master"]["risk_fraction"] == "0.08"
    assert body["followers"][0]["allowed"] is True
    assert body["followers"][0]["product"] == "MNQ"
    assert body["followers"][0]["quantity"] == 1
    assert body["followers"][0]["parity_error"] == "0"
    assert body["followers"][0]["would_submit"] is False
    assert "secret-master-id" not in response.text
    assert "secret-follower-id" not in response.text


def test_v2_copier_preview_defaults_fail_closed_and_requires_valid_stop() -> None:
    client = _client()
    payload = _preview_payload()
    payload.pop("global_kill_switch")

    blocked = client.post("/v2/copier/preview", json=payload)
    assert blocked.status_code == 200
    assert blocked.json()["followers"][0]["reason"] == "global_kill_switch"

    payload = _preview_payload()
    payload["master"]["stop_price"] = payload["master"]["entry_price"]
    invalid = client.post("/v2/copier/preview", json=payload)
    assert invalid.status_code == 422


def test_v2_copy_control_supports_revisions_global_pause_and_audit(
    monkeypatch,
) -> None:
    monkeypatch.delenv(OPERATOR_USERNAME_ENV, raising=False)
    monkeypatch.delenv(OPERATOR_PASSWORD_ENV, raising=False)
    monkeypatch.delenv(OPERATOR_API_TOKEN_ENV, raising=False)
    client, session = _control_client()
    try:
        initial = client.get("/v2/copier/control")
        assert initial.status_code == 200
        assert initial.headers["cache-control"] == "no-store"
        assert initial.json()["global_paused"] is True
        assert [item["state"] for item in initial.json()["followers"]] == [
            "DISABLED"
        ] * 4

        enabled = client.post(
            "/v2/copier/control/followers/follower-1/actions",
            json={
                "action": "ENABLE",
                "expected_revision": 1,
                "request_id": "api-enable-1",
                "reason_code": "simulation",
            },
        )
        assert enabled.status_code == 200
        assert enabled.json()["followers"][0]["state"] == "ENABLED"
        assert enabled.json()["followers"][0]["revision"] == 2

        conflict = client.post(
            "/v2/copier/control/followers/follower-1/actions",
            json={
                "action": "PAUSE",
                "expected_revision": 1,
                "request_id": "api-pause-stale",
            },
        )
        assert conflict.status_code == 409
        assert conflict.json()["detail"] == {
            "code": "control_revision_conflict",
            "current_revision": 2,
        }

        resumed = client.post(
            "/v2/copier/control/global-pause",
            json={
                "paused": False,
                "expected_revision": 2,
                "request_id": "api-global-resume",
                "reason_code": "simulation",
            },
        )
        assert resumed.status_code == 200
        assert resumed.json()["global_paused"] is False
        assert resumed.json()["revision"] == 3
        assert resumed.json()["entry_enabled_connection_ids"] == ["follower-1"]

        audit = client.get("/v2/copier/control/audit")
        assert audit.status_code == 200
        assert audit.json()["count"] == 2
        assert audit.json()["items"][0]["action"] == "CLEAR_GLOBAL_PAUSE"
    finally:
        session.close()


def test_v2_copy_control_cookie_mutation_requires_csrf(monkeypatch) -> None:
    monkeypatch.setenv(OPERATOR_USERNAME_ENV, "operator")
    monkeypatch.setenv(OPERATOR_PASSWORD_ENV, "secret")
    monkeypatch.delenv(OPERATOR_API_TOKEN_ENV, raising=False)
    client, session = _control_client()
    try:
        session_token = create_dashboard_session_token("operator")
        client.cookies.set(SESSION_COOKIE_NAME, session_token)
        blocked = client.post(
            "/v2/copier/control/followers/follower-1/actions",
            json={
                "action": "ENABLE",
                "expected_revision": 1,
                "request_id": "csrf-blocked",
            },
        )
        assert blocked.status_code == 403

        csrf = client.get("/v2/copier/control/csrf")
        assert csrf.status_code == 200
        assert csrf.json()["required"] is True
        allowed = client.post(
            "/v2/copier/control/followers/follower-1/actions",
            headers={"X-CSRF-Token": csrf.json()["token"]},
            json={
                "action": "ENABLE",
                "expected_revision": 1,
                "request_id": "csrf-allowed",
            },
        )
        assert allowed.status_code == 200
    finally:
        session.close()
