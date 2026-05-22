"""Operator authentication for dashboard and copier control endpoints.

This module gates access to the MooneyCapitol dashboard and the /copier
control APIs. It is the v1 "simplest workable" auth layer:

- A single shared operator username + password for the browser-facing
  dashboard, submitted through the app login page.
- A signed, HTTP-only browser session cookie for dashboard users.
- A single shared API token for non-browser callers (scripts, curl, future
  CLI tools) sent as the ``X-Operator-Token`` request header.

Both credentials live in environment variables so they never touch git and
can be rotated by re-deploying with new env values. When none of the auth
env vars are configured, the dependencies short-circuit and let the request
through. This preserves the existing developer workflow (local runs, the
current test suite) without needing test-time monkeypatching.

Env vars
--------
- COWORK_OPERATOR_USERNAME : operator login name for /dashboard.
- COWORK_OPERATOR_PASSWORD : operator password for /dashboard.
- COWORK_OPERATOR_API_TOKEN : shared secret for X-Operator-Token API calls.

Dependencies
------------
- ``require_dashboard_auth``: Signed session cookie only. Use on browser-facing
  HTML routes when a dependency is needed.
- ``require_operator``: Accepts a signed session cookie, Basic credentials, OR
  the X-Operator-Token header. Use on /copier API endpoints so logged-in
  dashboard browser requests work while external scripts can still authenticate
  with the token.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Optional

from fastapi import Depends, Header, HTTPException, Request, Response, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials


OPERATOR_USERNAME_ENV = "COWORK_OPERATOR_USERNAME"
OPERATOR_PASSWORD_ENV = "COWORK_OPERATOR_PASSWORD"
OPERATOR_API_TOKEN_ENV = "COWORK_OPERATOR_API_TOKEN"
SESSION_COOKIE_NAME = "mooney_operator_session"
SESSION_MAX_AGE_SECONDS = 12 * 60 * 60

# auto_error=False so missing credentials hand control back to us; we want to
# produce a single, consistent 401 instead of FastAPI's default Basic prompt.
_basic_scheme = HTTPBasic(auto_error=False)


def _strip(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _operator_username() -> Optional[str]:
    return _strip(os.getenv(OPERATOR_USERNAME_ENV))


def _operator_password() -> Optional[str]:
    # Passwords are not stripped because trailing/leading whitespace may be
    # intentional. An empty string is treated as "not configured".
    value = os.getenv(OPERATOR_PASSWORD_ENV)
    return value if value else None


def _operator_token() -> Optional[str]:
    return _strip(os.getenv(OPERATOR_API_TOKEN_ENV))


def _session_secret() -> Optional[str]:
    return _operator_token() or _operator_password()


def _basic_configured() -> bool:
    return _operator_username() is not None and _operator_password() is not None


def _any_auth_configured() -> bool:
    return _basic_configured() or _operator_token() is not None


def _basic_matches(credentials: Optional[HTTPBasicCredentials]) -> bool:
    if credentials is None:
        return False
    user = _operator_username()
    pwd = _operator_password()
    if user is None or pwd is None:
        return False
    user_ok = secrets.compare_digest(
        credentials.username.encode("utf-8"), user.encode("utf-8")
    )
    pwd_ok = secrets.compare_digest(
        credentials.password.encode("utf-8"), pwd.encode("utf-8")
    )
    return user_ok and pwd_ok


def _urlsafe_b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _urlsafe_b64decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def _session_signature(payload: str) -> Optional[str]:
    secret = _session_secret()
    if secret is None:
        return None
    digest = hmac.new(secret.encode("utf-8"), payload.encode("ascii"), hashlib.sha256).digest()
    return _urlsafe_b64encode(digest)


def create_dashboard_session_token(username: str, now: Optional[int] = None) -> str:
    issued_at = int(now if now is not None else time.time())
    payload = _urlsafe_b64encode(
        json.dumps(
            {"u": username, "exp": issued_at + SESSION_MAX_AGE_SECONDS},
            separators=(",", ":"),
        ).encode("utf-8")
    )
    signature = _session_signature(payload)
    if signature is None:
        raise RuntimeError("Operator credentials are not configured")
    return f"{payload}.{signature}"


def _session_matches(token: Optional[str]) -> bool:
    if not token or "." not in token:
        return False
    payload, signature = token.rsplit(".", 1)
    expected = _session_signature(payload)
    if expected is None or not secrets.compare_digest(signature, expected):
        return False
    try:
        data = json.loads(_urlsafe_b64decode(payload).decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return False
    if not isinstance(data, dict):
        return False
    username = data.get("u")
    expires_at = data.get("exp")
    configured_user = _operator_username()
    if not isinstance(username, str) or not isinstance(expires_at, int):
        return False
    if configured_user is None or not secrets.compare_digest(username, configured_user):
        return False
    return expires_at >= int(time.time())


def _request_is_secure(request: Request) -> bool:
    forwarded_proto = request.headers.get("x-forwarded-proto", "").split(",", 1)[0].strip()
    return request.url.scheme == "https" or forwarded_proto == "https"


def set_dashboard_session_cookie(response: Response, request: Request) -> None:
    username = _operator_username()
    if username is None:
        return
    response.set_cookie(
        SESSION_COOKIE_NAME,
        create_dashboard_session_token(username),
        max_age=SESSION_MAX_AGE_SECONDS,
        httponly=True,
        secure=_request_is_secure(request),
        samesite="lax",
    )


def clear_dashboard_session_cookie(response: Response, request: Request) -> None:
    response.delete_cookie(
        SESSION_COOKIE_NAME,
        httponly=True,
        secure=_request_is_secure(request),
        samesite="lax",
    )


def is_dashboard_authenticated(request: Request) -> bool:
    if not _basic_configured():
        return True
    return _session_matches(request.cookies.get(SESSION_COOKIE_NAME))


def _token_matches(token: Optional[str]) -> bool:
    configured = _operator_token()
    if configured is None or token is None:
        return False
    return secrets.compare_digest(token, configured)


def login_credentials_match(username: str, password: str) -> bool:
    return _basic_matches(HTTPBasicCredentials(username=username, password=password))


def require_dashboard_auth(request: Request) -> None:
    """Require a valid dashboard session cookie for browser-facing pages.

    No-op when COWORK_OPERATOR_USERNAME / COWORK_OPERATOR_PASSWORD are not
    configured so local dev keeps working.
    """
    if is_dashboard_authenticated(request):
        return
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
    )


def require_operator(
    request: Request,
    credentials: Optional[HTTPBasicCredentials] = Depends(_basic_scheme),
    x_operator_token: Optional[str] = Header(default=None),
) -> None:
    """Require operator credentials for protected API endpoints.

    Accepts EITHER:
      - Signed dashboard session cookie from /login, OR
      - HTTP Basic credentials matching the configured operator, OR
      - X-Operator-Token header matching COWORK_OPERATOR_API_TOKEN.

    No-op when no auth env vars are configured. When at least one auth
    method is configured, every request must satisfy one of them.
    """
    if not _any_auth_configured():
        return
    if _token_matches(x_operator_token):
        return
    if _session_matches(request.cookies.get(SESSION_COOKIE_NAME)):
        return
    if _basic_matches(credentials):
        return
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Operator authentication required",
    )
