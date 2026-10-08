"""Tests for operator authentication on dashboard and protected APIs."""
from __future__ import annotations

import time

import pytest
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import HTTPBasicCredentials
from fastapi.testclient import TestClient

from app.api.auth import (
    OPERATOR_API_TOKEN_ENV,
    OPERATOR_PASSWORD_ENV,
    OPERATOR_USERNAME_ENV,
    SESSION_COOKIE_NAME,
    create_dashboard_session_token,
    login_credentials_match,
    require_dashboard_auth,
    require_operator,
)


def _clear_auth_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(OPERATOR_USERNAME_ENV, raising=False)
    monkeypatch.delenv(OPERATOR_PASSWORD_ENV, raising=False)
    monkeypatch.delenv(OPERATOR_API_TOKEN_ENV, raising=False)


def _basic_credentials(user: str, pw: str) -> HTTPBasicCredentials:
    return HTTPBasicCredentials(username=user, password=pw)


def _request(cookies: dict[str, str] | None = None) -> Request:
    headers: list[tuple[bytes, bytes]] = []
    if cookies:
        cookie_value = "; ".join(f"{key}={value}" for key, value in cookies.items())
        headers.append((b"cookie", cookie_value.encode("utf-8")))
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": headers,
            "query_string": b"",
            "scheme": "http",
            "server": ("testserver", 80),
            "client": ("testclient", 50000),
        }
    )


def _build_app_with_auth_routes() -> FastAPI:
    from app.api.pages import router as pages_router

    app = FastAPI()
    app.include_router(pages_router)

    @app.get("/protected", dependencies=[Depends(require_operator)])
    def protected():
        return {"ok": True}

    return app


def test_dashboard_passthrough_when_unconfigured(monkeypatch):
    _clear_auth_env(monkeypatch)

    require_dashboard_auth(_request())


def test_dashboard_rejects_missing_session_cookie(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv(OPERATOR_USERNAME_ENV, "op")
    monkeypatch.setenv(OPERATOR_PASSWORD_ENV, "secret")

    with pytest.raises(HTTPException) as exc:
        require_dashboard_auth(_request())
    assert exc.value.status_code == 401
    assert "WWW-Authenticate" not in (exc.value.headers or {})


def test_dashboard_accepts_valid_session_cookie(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv(OPERATOR_USERNAME_ENV, "op")
    monkeypatch.setenv(OPERATOR_PASSWORD_ENV, "secret")
    token = create_dashboard_session_token("op", now=int(time.time()))

    require_dashboard_auth(_request({SESSION_COOKIE_NAME: token}))


def test_dashboard_rejects_expired_session_cookie(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv(OPERATOR_USERNAME_ENV, "op")
    monkeypatch.setenv(OPERATOR_PASSWORD_ENV, "secret")
    expired_at = int(time.time()) - (13 * 60 * 60)
    token = create_dashboard_session_token("op", now=expired_at)

    with pytest.raises(HTTPException):
        require_dashboard_auth(_request({SESSION_COOKIE_NAME: token}))


def test_login_credentials_match_configured_operator(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv(OPERATOR_USERNAME_ENV, "op")
    monkeypatch.setenv(OPERATOR_PASSWORD_ENV, "secret")

    assert login_credentials_match("op", "secret") is True
    assert login_credentials_match("op", "wrong") is False


def test_operator_passthrough_when_unconfigured(monkeypatch):
    _clear_auth_env(monkeypatch)

    require_operator(request=_request(), credentials=None, x_operator_token=None)


def test_operator_rejects_missing_credentials(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv(OPERATOR_API_TOKEN_ENV, "the-token")

    with pytest.raises(HTTPException) as exc:
        require_operator(request=_request(), credentials=None, x_operator_token=None)
    assert exc.value.status_code == 401


def test_operator_accepts_correct_token(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv(OPERATOR_API_TOKEN_ENV, "the-token")

    require_operator(request=_request(), credentials=None, x_operator_token="the-token")


def test_operator_rejects_wrong_token(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv(OPERATOR_API_TOKEN_ENV, "the-token")

    with pytest.raises(HTTPException):
        require_operator(request=_request(), credentials=None, x_operator_token="other")


def test_operator_accepts_session_cookie(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv(OPERATOR_USERNAME_ENV, "op")
    monkeypatch.setenv(OPERATOR_PASSWORD_ENV, "secret")
    token = create_dashboard_session_token("op", now=int(time.time()))

    require_operator(
        request=_request({SESSION_COOKIE_NAME: token}),
        credentials=None,
        x_operator_token=None,
    )


def test_operator_still_accepts_basic_credentials(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv(OPERATOR_USERNAME_ENV, "op")
    monkeypatch.setenv(OPERATOR_PASSWORD_ENV, "secret")

    require_operator(
        request=_request(),
        credentials=_basic_credentials("op", "secret"),
        x_operator_token=None,
    )


def test_dashboard_route_redirects_to_login_when_configured(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv(OPERATOR_USERNAME_ENV, "op")
    monkeypatch.setenv(OPERATOR_PASSWORD_ENV, "secret")

    client = TestClient(_build_app_with_auth_routes())
    resp = client.get("/dashboard", follow_redirects=False)

    assert resp.status_code == 303
    assert resp.headers["location"] == "/login"
    assert "www-authenticate" not in resp.headers


def test_public_homepage_is_available_without_login_when_dashboard_auth_configured(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv(OPERATOR_USERNAME_ENV, "op")
    monkeypatch.setenv(OPERATOR_PASSWORD_ENV, "secret")

    client = TestClient(_build_app_with_auth_routes())
    resp = client.get("/")

    assert resp.status_code == 200
    assert "Mooney Trading" in resp.text
    assert 'href="/dashboard"' not in resp.text
    assert "Operator Dashboard" not in resp.text
    assert "Read-only Capture" in resp.text
    assert "Recovery &amp; Reconciliation" in resp.text
    assert "Observation Console" in resp.text
    assert "Futures Research" in resp.text
    assert "Trading involves risk" in resp.text
    assert "Copier Settings" not in resp.text


def test_public_robots_and_sitemap_expose_only_public_pages(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv(OPERATOR_USERNAME_ENV, "op")
    monkeypatch.setenv(OPERATOR_PASSWORD_ENV, "secret")

    client = TestClient(_build_app_with_auth_routes())
    robots = client.get("/robots.txt")
    sitemap = client.get("/sitemap.xml")

    assert robots.status_code == 200
    assert "User-agent: *" in robots.text
    assert "Allow: /" in robots.text
    assert "Disallow: /dashboard" in robots.text
    assert "Disallow: /copier" in robots.text
    assert "Disallow: /pnl" in robots.text
    assert "Sitemap: https://mooneytrading.com/sitemap.xml" in robots.text

    assert sitemap.status_code == 200
    assert sitemap.headers["content-type"].startswith("application/xml")
    assert "<loc>https://mooneytrading.com/</loc>" in sitemap.text
    assert "<loc>https://mooneytrading.com/platform</loc>" in sitemap.text
    assert "<loc>https://mooneytrading.com/faq</loc>" in sitemap.text
    assert "<loc>https://mooneytrading.com/risk-disclosure</loc>" in sitemap.text
    assert "<loc>https://mooneytrading.com/privacy</loc>" in sitemap.text
    assert "/dashboard" not in sitemap.text
    assert "/copier" not in sitemap.text


def test_public_platform_page_is_available_without_login(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv(OPERATOR_USERNAME_ENV, "op")
    monkeypatch.setenv(OPERATOR_PASSWORD_ENV, "secret")

    client = TestClient(_build_app_with_auth_routes())
    resp = client.get("/platform")

    assert resp.status_code == 200
    assert "Platform Overview" in resp.text
    assert "What Mooney Trading is" in resp.text
    assert "Observation workflow" in resp.text
    assert "Operator workflow" in resp.text
    assert "Learning and review" in resp.text
    assert "Launch boundaries" in resp.text
    assert "does not provide public account access, public trading signals, or investor onboarding" in resp.text
    assert "authenticated internal console" in resp.text
    assert 'href="/dashboard"' not in resp.text
    assert "Operator Dashboard" not in resp.text
    assert "Copier Settings" not in resp.text
    assert "WEBULL_MASTER" not in resp.text


def test_public_faq_page_is_available_without_login(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv(OPERATOR_USERNAME_ENV, "op")
    monkeypatch.setenv(OPERATOR_PASSWORD_ENV, "secret")

    client = TestClient(_build_app_with_auth_routes())
    resp = client.get("/faq")

    assert resp.status_code == 200
    assert "FAQ" in resp.text
    assert "Is Mooney Trading open to the public?" in resp.text
    assert "What does the platform do?" in resp.text
    assert "Does the system trade by itself?" in resp.text
    assert "What would need to happen before a public product?" in resp.text
    assert "does not provide account access, signup, public trading signals, or investor onboarding" in resp.text
    assert "private internal console" in resp.text
    assert 'href="/dashboard"' not in resp.text
    assert "Operator Dashboard" not in resp.text
    assert "Copier Settings" not in resp.text
    assert "WEBULL_MASTER" not in resp.text


def test_public_risk_and_privacy_pages_are_available_without_login(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv(OPERATOR_USERNAME_ENV, "op")
    monkeypatch.setenv(OPERATOR_PASSWORD_ENV, "secret")

    client = TestClient(_build_app_with_auth_routes())
    risk = client.get("/risk-disclosure")
    privacy = client.get("/privacy")

    assert risk.status_code == 200
    assert "Risk Disclosure" in risk.text
    assert "No financial advice" in risk.text
    assert "No public copy-trading offer" in risk.text
    assert "Technology limitations" in risk.text
    assert "private internal console" in risk.text
    assert 'href="/dashboard"' not in risk.text
    assert "Operator Dashboard" not in risk.text
    assert "Copier Settings" not in risk.text
    assert "WEBULL_MASTER" not in risk.text

    assert privacy.status_code == 200
    assert "Privacy Notice" in privacy.text
    assert "Public site" in privacy.text
    assert "Internal operator console" in privacy.text
    assert "Service providers" in privacy.text
    assert 'href="/dashboard"' not in privacy.text
    assert "Operator Dashboard" not in privacy.text
    assert "Copier Settings" not in privacy.text
    assert "WEBULL_PERSONAL" not in privacy.text


def test_login_page_renders_when_configured(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv(OPERATOR_USERNAME_ENV, "op")
    monkeypatch.setenv(OPERATOR_PASSWORD_ENV, "secret")

    client = TestClient(_build_app_with_auth_routes())
    resp = client.get("/login")

    assert resp.status_code == 200
    assert "Sign in" in resp.text


def test_login_success_sets_cookie_and_opens_dashboard(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv(OPERATOR_USERNAME_ENV, "op")
    monkeypatch.setenv(OPERATOR_PASSWORD_ENV, "secret")

    client = TestClient(_build_app_with_auth_routes())
    login_resp = client.post(
        "/login",
        data={"username": "op", "password": "secret"},
        follow_redirects=False,
    )

    assert login_resp.status_code == 303
    assert login_resp.headers["location"] == "/dashboard"
    assert SESSION_COOKIE_NAME in login_resp.headers.get("set-cookie", "")

    dashboard_resp = client.get("/dashboard")
    assert dashboard_resp.status_code == 200
    assert "Rithmic Test" in dashboard_resp.text
    assert "Broker observation console" in dashboard_resp.text

    legacy_resp = client.get("/dashboard/legacy")
    assert legacy_resp.status_code == 200
    assert "Legacy V1 · Webull equities" in legacy_resp.text
    assert "Copier Settings" in legacy_resp.text


def test_login_rejects_wrong_password(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv(OPERATOR_USERNAME_ENV, "op")
    monkeypatch.setenv(OPERATOR_PASSWORD_ENV, "secret")

    client = TestClient(_build_app_with_auth_routes())
    resp = client.post("/login", data={"username": "op", "password": "wrong"})

    assert resp.status_code == 401
    assert "Invalid username or password." in resp.text


def test_logout_clears_session(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv(OPERATOR_USERNAME_ENV, "op")
    monkeypatch.setenv(OPERATOR_PASSWORD_ENV, "secret")

    client = TestClient(_build_app_with_auth_routes())
    client.post("/login", data={"username": "op", "password": "secret"})

    logout_resp = client.post("/logout", follow_redirects=False)
    dashboard_resp = client.get("/dashboard", follow_redirects=False)

    assert logout_resp.status_code == 303
    assert logout_resp.headers["location"] == "/login"
    assert dashboard_resp.status_code == 303
    assert dashboard_resp.headers["location"] == "/login"


def test_protected_api_accepts_session_cookie_and_token(monkeypatch):
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv(OPERATOR_USERNAME_ENV, "op")
    monkeypatch.setenv(OPERATOR_PASSWORD_ENV, "secret")
    monkeypatch.setenv(OPERATOR_API_TOKEN_ENV, "the-token")

    client = TestClient(_build_app_with_auth_routes())

    assert client.get("/protected").status_code == 401
    assert client.get("/protected", headers={"X-Operator-Token": "the-token"}).status_code == 200

    client.post("/login", data={"username": "op", "password": "secret"})
    assert client.get("/protected").status_code == 200


def test_dashboard_passthrough_when_no_auth_configured(monkeypatch):
    _clear_auth_env(monkeypatch)

    client = TestClient(_build_app_with_auth_routes())
    resp = client.get("/dashboard")

    assert resp.status_code == 200
