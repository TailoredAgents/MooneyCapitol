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
    elif mode == "webull":
        status = {"mode": "webull", "status": "disabled", "reason": "adapter not implemented", "fallback": "demo"}
    else:
        status = {"mode": mode, "status": "disabled", "reason": "unsupported depth mode", "fallback": "demo"}
    set_status(status)
    return status
