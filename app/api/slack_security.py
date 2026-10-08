from __future__ import annotations

import hmac
import os
import time
from hashlib import sha256

from fastapi import HTTPException


MAX_REQUEST_AGE_SECONDS = 300


def require_valid_slack_signature(timestamp: str | None, body: bytes, signature: str | None) -> None:
    secret = os.getenv("SLACK_SIGNING_SECRET")
    if not secret:
        raise HTTPException(status_code=503, detail="Slack signature validation is unavailable")
    if not timestamp or not signature:
        raise HTTPException(status_code=400, detail="missing signature headers")
    try:
        request_time = int(timestamp)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="invalid signature timestamp") from exc
    if abs(int(time.time()) - request_time) > MAX_REQUEST_AGE_SECONDS:
        raise HTTPException(status_code=403, detail="stale Slack request")
    basestring = f"v0:{timestamp}:{body.decode()}".encode()
    expected = "v0=" + hmac.new(secret.encode(), basestring, sha256).hexdigest()
    if not hmac.compare_digest(expected, signature):
        raise HTTPException(status_code=403, detail="invalid signature")
