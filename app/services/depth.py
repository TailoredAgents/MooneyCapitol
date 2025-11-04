from __future__ import annotations

import os
from typing import Any, Dict


_status: Dict[str, Any] = {"mode": "demo", "status": "init"}


def set_status(status: Dict[str, Any]) -> None:
    global _status
    _status = status


def get_status() -> Dict[str, Any]:
    return dict(_status)


def mark_snapshot(mode: str, source: str | None = None) -> None:
    from datetime import datetime

    status = get_status()
    status.update({
        "mode": mode,
        "status": "ok",
        "last_snapshot_ms": int(datetime.utcnow().timestamp() * 1000),
    })
    if source:
        status["source"] = source
    set_status(status)


def mark_stale(mode: str, reason: str) -> None:
    status = get_status()
    status.update({"mode": mode, "status": "stale", "reason": reason})
    set_status(status)


def compute_status() -> Dict[str, Any]:
    status = get_status()
    if status.get("status") not in {"init"}:
        return status
    mode = os.getenv("DEPTH_MODE", "demo").lower()
    if mode == "demo":
        status = {"mode": "demo", "status": "ok"}
    else:
        creds_ok = all(os.getenv(key) for key in ["IBKR_HOST", "IBKR_PORT", "IBKR_CLIENT_ID"])
        if creds_ok:
            status = {"mode": "ibkr", "status": "initializing"}
        else:
            status = {"mode": "ibkr", "status": "disabled", "reason": "no credentials", "fallback": "demo"}
    set_status(status)
    return status
