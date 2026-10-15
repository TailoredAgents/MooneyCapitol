from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.auth import OPERATOR_API_TOKEN_ENV
from app.api.routes import v2_copier


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(v2_copier.router)
    return TestClient(app)


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
