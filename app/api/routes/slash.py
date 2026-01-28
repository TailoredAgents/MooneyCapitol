from __future__ import annotations

import hmac
import os
from hashlib import sha256
from urllib.parse import parse_qs

from fastapi import APIRouter, Header, HTTPException, Request

from app.core.config_store import CONFIG
from app.core.config_store import persist_config, refresh_config
from app.services.kv_store import StateStoreError
router = APIRouter(prefix="", tags=["slash"])


def _verify_signature(secret: str, timestamp: str, body: bytes, signature: str) -> bool:
    basestring = f"v0:{timestamp}:{body.decode()}".encode()
    computed = "v0=" + hmac.new(secret.encode(), basestring, sha256).hexdigest()
    return hmac.compare_digest(computed, signature)


@router.post("/slack/commands")
async def slack_commands(
    request: Request,
    x_slack_signature: str | None = Header(default=None),
    x_slack_request_timestamp: str | None = Header(default=None),
):
    try:
        refresh_config()
    except StateStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    body = await request.body()
    secret = os.getenv("SLACK_SIGNING_SECRET")
    if secret:
        if not x_slack_signature or not x_slack_request_timestamp:
            raise HTTPException(status_code=400, detail="missing signature headers")
        if not _verify_signature(secret, x_slack_request_timestamp, body, x_slack_signature):
            raise HTTPException(status_code=403, detail="invalid signature")

    form = parse_qs(body.decode())
    command = form.get("command", [""])[0]
    text = (form.get("text", [""])[0] or "").strip()

    response_text = "Unsupported command"
    try:
        if command == "/watch":
            response_text = _handle_watch_command(text)
        elif command == "/alerts":
            response_text = _handle_alerts_command(text)
        elif command == "/headsup":
            response_text = _handle_headsup_command(text)
    except StateStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {"response_type": "ephemeral", "text": response_text}


def _handle_watch_command(text: str) -> str:
    parts = text.split()
    if len(parts) < 2:
        return "Usage: /watch add|remove|block TICKER"
    action, ticker = parts[0].lower(), parts[1].upper()
    if action == "add":
        if ticker not in CONFIG.universe.pinned:
            CONFIG.universe.pinned.append(ticker)
        if ticker in CONFIG.universe.blocklist:
            CONFIG.universe.blocklist.remove(ticker)
        persist_config()
        return f"Pinned {ticker}."
    if action == "remove":
        if ticker in CONFIG.universe.pinned:
            CONFIG.universe.pinned.remove(ticker)
            persist_config()
        return f"Removed {ticker} from pinned."
    if action == "block":
        if ticker in CONFIG.universe.pinned:
            CONFIG.universe.pinned.remove(ticker)
        if ticker not in CONFIG.universe.blocklist:
            CONFIG.universe.blocklist.append(ticker)
        persist_config()
        return f"Blocklisted {ticker}."
    return "Unknown /watch action"


def _handle_alerts_command(text: str) -> str:
    parts = text.split()
    if len(parts) != 2 or parts[0].lower() != "budget":
        return "Usage: /alerts budget <number>"
    try:
        value = int(parts[1])
        CONFIG.alerts.per_minute_budget_open = value
        persist_config()
        return f"Alert budget updated to {value} per minute."
    except ValueError:
        return "Invalid number for budget"


def _handle_headsup_command(text: str) -> str:
    value = text.strip().lower()
    if value in {"on", "true", "1"}:
        CONFIG.alerts.heads_up_ping = True
        persist_config()
        return "Heads-Up pings enabled."
    if value in {"off", "false", "0"}:
        CONFIG.alerts.heads_up_ping = False
        persist_config()
        return "Heads-Up pings disabled."
    return "Usage: /headsup on|off"
