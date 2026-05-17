"""Tests for operator authentication on dashboard and copier APIs.

Covers two layers:

1. Unit tests of the auth dependency functions themselves
   (``require_dashboard_auth`` and ``require_operator``) — these prove the
   credential-matching and pass-through behavior in isolation.

2. Integration tests that mount the real ``pages_router`` and
   ``copier_router`` on a slim FastAPI test app and exercise them with
   ``TestClient`` — these prove the auth dependency is actually wired up to
   the live routes (no stale registration). They only assert 401 / passthrough
   behavior so they do not require a working database or copier state store.
"""
from __future__ import annotations

import base64

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPBasicCredentials
from fastapi.testclient import TestClient

from app.api import auth as auth_module
from app.api.auth import (
    OPERATOR_API_TOKEN_ENV,
    OPERATOR_PASSWORD_ENV,
    OPERATOR_USERNAME_ENV,
    require_dashboard_auth,
    require_operator,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _clear_auth_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(OPERATOR_USERNAME_ENV, raising=False)
    monkeypatch.delenv(OPERATOR_PASSWORD_ENV, raising=False)
    monkeypatch.delenv(OPERATOR_API_TOKEN_ENV, raising=False)


def _basic_header(user: str, pw: str) -> dict[str, str]:
    raw = f"{user}:{pw}".encode("utf-8")
    return {"Authorization": "Basic " + base64.b64encode(raw).decode("ascii")}


def _basic_credentials(user: str, pw: str) -> HTTPBasicCredentials:
    return HTTPBasicCredentials(username=user, password=pw)


# ---------------------------------------------------------------------------
# unit tests: require_dashboard_auth
# ---------------------------------------------------------------------------


def test_dashboard_passthrough_when_unconfigured(monkeypatch):
    _clear_auth_env(monkeypatch)
    # No credentials, no env vars set -> should not raise.
    require_dashboard_auth(credentials=None)


def test_dashboard_rejects_missing_credentials(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv(OPERATOR_USERNAME_ENV, "op")
    monkeypatch.setenv(OPERATOR_PASSWORD_ENV, "secret")

    with pytest.raises(HTTPException) as exc:
        require_dashboard_auth(credentials=None)
    assert exc.value.status_code == 401
    assert exc.value.headers.get("WWW-Authenticate", "").startswith("Basic")


def test_dashboard_rejects_wrong_password(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv(OPERATOR_USERNAME_ENV, "op")
    monkeypatch.setenv(OPERATOR_PASSWORD_ENV, "secret")

    with pytest.raises(HTTPException) as exc:
        require_dashboard_auth(credentials=_basic_credentials("op", "wrong"))
    assert exc.value.status_code == 401


def test_dashboard_rejects_wrong_username(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv(OPERATOR_USERNAME_ENV, "op")
    monkeypatch.setenv(OPERATOR_PASSWORD_ENV, "secret")

    with pytest.raises(HTTPException):
        require_dashboard_auth(credentials=_basic_credentials("intruder", "secret"))


def test_dashboard_accepts_correct_credentials(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv(OPERATOR_USERNAME_ENV, "op")
    monkeypatch.setenv(OPERATOR_PASSWORD_ENV, "secret")

    # Should not raise.
    require_dashboard_auth(credentials=_basic_credentials("op", "secret"))


def test_dashboard_ignores_token_only_config(monkeypatch):
    """If only the API token is set (no basic creds), dashboard stays open.

    The dashboard prompt only knows how to send Basic credentials, so the
    operator must configure both username + password to lock /dashboard.
    """
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv(OPERATOR_API_TOKEN_ENV, "token-only")

    require_dashboard_auth(credentials=None)


# ---------------------------------------------------------------------------
# unit tests: require_operator
# ---------------------------------------------------------------------------


def test_operator_passthrough_when_unconfigured(monkeypatch):
    _clear_auth_env(monkeypatch)
    require_operator(credentials=None, x_operator_token=None)


def test_operator_rejects_missing_credentials(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv(OPERATOR_API_TOKEN_ENV, "the-token")

    with pytest.raises(HTTPException) as exc:
        require_operator(credentials=None, x_operator_token=None)
    assert exc.value.status_code == 401


def test_operator_accepts_correct_token(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv(OPERATOR_API_TOKEN_ENV, "the-token")

    require_operator(credentials=None, x_operator_token="the-token")


def test_operator_rejects_wrong_token(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv(OPERATOR_API_TOKEN_ENV, "the-token")

    with pytest.raises(HTTPException):
        require_operator(credentials=None, x_operator_token="other")


def test_operator_accepts_basic_credentials_when_configured(monkeypatch):
    """A dashboard browser session sends Basic creds on /copier/* calls.

    The operator dependency must accept them so the dashboard works for
    logged-in users without needing a separate token.
    """
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv(OPERATOR_USERNAME_ENV, "op")
    monkeypatch.setenv(OPERATOR_PASSWORD_ENV, "secret")

    require_operator(
        credentials=_basic_credentials("op", "secret"),
        x_operator_token=None,
    )


def test_operator_rejects_wrong_basic_credentials(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv(OPERATOR_USERNAME_ENV, "op")
    monkeypatch.setenv(OPERATOR_PASSWORD_ENV, "secret")

    with pytest.raises(HTTPException):
        require_operator(
            credentials=_basic_credentials("op", "wrong"),
            x_operator_token=None,
        )


def test_operator_either_credential_works(monkeypatch):
    """When both basic + token are configured, either path satisfies auth."""
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv(OPERATOR_USERNAME_ENV, "op")
    monkeypatch.setenv(OPERATOR_PASSWORD_ENV, "secret")
    monkeypatch.setenv(OPERATOR_API_TOKEN_ENV, "the-token")

    require_operator(credentials=_basic_credentials("op", "secret"), x_operator_token=None)
    require_operator(credentials=None, x_operator_token="the-token")


# ---------------------------------------------------------------------------
# wiring tests: real routers mounted in a slim FastAPI app
# ---------------------------------------------------------------------------


def _build_app_with_real_routers() -> FastAPI:
    """Mount the real pages + copier routers on a fresh FastAPI app.

    No lifespan is attached, so the test app does not call into the state
    store / DB at startup. We only exercise auth behavior; the route bodies
    never execute when auth correctly returns 401.
    """
    from app.api.pages import router as pages_router
    from app.api.routes.copier import router as copier_router

    app = FastAPI()
    app.include_router(pages_router)
    app.include_router(copier_router)
    return app


def test_dashboard_route_returns_401_when_basic_configured(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv(OPERATOR_USERNAME_ENV, "op")
    monkeypatch.setenv(OPERATOR_PASSWORD_ENV, "secret")

    client = TestClient(_build_app_with_real_routers())
    resp = client.get("/dashboard")
    assert resp.status_code == 401
    assert resp.headers.get("www-authenticate", "").lower().startswith("basic")


def test_dashboard_route_rejects_wrong_password(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv(OPERATOR_USERNAME_ENV, "op")
    monkeypatch.setenv(OPERATOR_PASSWORD_ENV, "secret")

    client = TestClient(_build_app_with_real_routers())
    resp = client.get("/dashboard", headers=_basic_header("op", "wrong"))
    assert resp.status_code == 401


def test_copier_route_returns_401_when_token_configured(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv(OPERATOR_API_TOKEN_ENV, "the-token")

    client = TestClient(_build_app_with_real_routers())
    resp = client.get("/copier/status")
    assert resp.status_code == 401


def test_copier_route_rejects_wrong_token(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv(OPERATOR_API_TOKEN_ENV, "the-token")

    client = TestClient(_build_app_with_real_routers())
    resp = client.get("/copier/status", headers={"X-Operator-Token": "wrong"})
    assert resp.status_code == 401


def test_copier_post_requires_auth(monkeypatch):
    """Mutating endpoints are blocked too (kill-switch enable)."""
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv(OPERATOR_API_TOKEN_ENV, "the-token")

    client = TestClient(_build_app_with_real_routers())
    resp = client.post("/copier/kill-switch/enable")
    assert resp.status_code == 401


def test_dashboard_passthrough_when_no_auth_configured(monkeypatch):
    """Existing dev/test flow still works: no env vars -> no gate."""
    _clear_auth_env(monkeypatch)

    client = TestClient(_build_app_with_real_routers())
    resp = client.get("/dashboard")
    # Dashboard route should render the template successfully (200).
    assert resp.status_code == 200
