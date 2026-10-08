from __future__ import annotations

import json
from datetime import datetime, timezone
from urllib.parse import parse_qs

from fastapi import APIRouter, Header, Request
from sqlalchemy import select

from app.adapters.slack import SlackAdapter
from app.api.slack_security import require_valid_slack_signature
from app.db.models import Alert
from app.db.session import get_session


router = APIRouter(prefix="", tags=["slack"])


@router.post("/webhooks/slack/actions")
async def slack_actions(
    request: Request,
    x_slack_signature: str | None = Header(default=None),
    x_slack_request_timestamp: str | None = Header(default=None),
):
    body = await request.body()
    require_valid_slack_signature(x_slack_request_timestamp, body, x_slack_signature)

    form = parse_qs(body.decode())
    payload_raw = form.get("payload", [None])[0]
    if not payload_raw:
        return {"ok": True}

    data = json.loads(payload_raw)
    actions = data.get("actions", [])
    if not actions:
        return {"ok": True}
    action_value = actions[0].get("value") or actions[0].get("action_id")
    status_map = {
        "acknowledge": "acknowledged",
        "ack": "acknowledged",
        "pass": "passed",
        "discuss": "discuss",
    }
    new_status = status_map.get(action_value)
    if not new_status:
        return {"ok": True}

    message_ts = data.get("message", {}).get("ts")
    if not message_ts:
        return {"ok": True}

    user = data.get("user", {})
    ack_by = user.get("username") or user.get("name") or user.get("id")

    with get_session() as session:
        alert = session.execute(select(Alert).where(Alert.slack_message_ts == message_ts)).scalar_one_or_none()
        if alert:
            alert.status = new_status
            alert.ack_by = ack_by
            alert.ack_ts = datetime.utcnow().replace(tzinfo=timezone.utc)
            session.flush()

            emoji_map = {"acknowledged": "✅", "passed": "🚫", "discuss": "💬"}
            emoji = emoji_map.get(new_status, "")
            blocks = data.get("message", {}).get("blocks", [])
            text = data.get("message", {}).get("text", "")
            if blocks:
                for block in blocks:
                    if block.get("type") == "header" and block.get("text"):
                        base = block["text"].get("text", "")
                        base = base.lstrip("✅🚫💬 ")
                        block["text"]["text"] = f"{emoji} {base}".strip() if emoji else base
                        break
            if emoji and text:
                base_text = text.lstrip("✅🚫💬 ")
                text = f"{emoji} {base_text}".strip()
            update_text = text or emoji or "Status updated"
            slack = SlackAdapter()
            slack.update_message(message_ts, blocks, update_text)

    return {"ok": True}
