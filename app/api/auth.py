"""Operator authentication for dashboard and copier control endpoints.

This module gates access to the MooneyCapitol dashboard and the /copier
control APIs. It is the v1 "simplest workable" auth layer:

- A single shared operator username + password for the browser-facing
  dashboard, prompted via HTTP Basic Auth.
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
- ``require_dashboard_auth``: HTTP Basic Auth only. Use on browser-facing
  HTML routes so the browser shows a native login prompt.
- ``require_operator``: Accepts EITHER Basic Auth credentials matching the
  configured operator OR the X-Operator-Token header. Use on /copier API
  endpoints so that a logged-in dashboard browser session is accepted
  automatically (same-origin Basic credentials are replayed by the browser)
  while external scripts can still authenticate with the token.
"""
from __future__ import annotations

import os
import secrets
from typing import Optional

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials


OPERATOR_USERNAME_ENV = "COWORK_OPERATOR_USERNAME"
OPERATOR_PASSWORD_ENV = "COWORK_OPERATOR_PASSWORD"
OPERATOR_API_TOKEN_ENV = "COWORK_OPERATOR_API_TOKEN"

# auto_error=False so missing credentials hand control back to us; we want to
# produce a single, consistent 401 (with WWW-Authenticate on dashboard paths).
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


def _token_matches(token: Optional[str]) -> bool:
    configured = _operator_token()
    if configured is None or token is None:
        return False
    return secrets.compare_digest(token, configured)


def require_dashboard_auth(
    credentials: Optional[HTTPBasicCredentials] = Depends(_basic_scheme),
) -> None:
    """Require HTTP Basic Auth for browser-facing pages.

    No-op when COWORK_OPERATOR_USERNAME / COWORK_OPERATOR_PASSWORD are not
    configured so local dev and the existing test suite keep working.
    """
    if not _basic_configured():
        return
    if _basic_matches(credentials):
        return
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
        headers={"WWW-Authenticate": 'Basic realm="MooneyCapitol"'},
    )


def require_operator(
    credentials: Optional[HTTPBasicCredentials] = Depends(_basic_scheme),
    x_operator_token: Optional[str] = Header(default=None),
) -> None:
    """Require operator credentials for protected API endpoints.

    Accepts EITHER:
      - HTTP Basic credentials matching the configured operator (so a
        dashboard browser session is accepted automatically), OR
      - X-Operator-Token header matching COWORK_OPERATOR_API_TOKEN.

    No-op when no auth env vars are configured. When at least one auth
    method is configured, every request must satisfy one of them.
    """
    if not _any_auth_configured():
        return
    if _token_matches(x_operator_token):
        return
    if _basic_matches(credentials):
        return
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Operator authentication required",
    )
