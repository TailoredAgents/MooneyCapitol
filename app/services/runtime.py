from __future__ import annotations

from typing import Any, Dict

from app.services.kv_store import get_json, set_json


def ensure_worker_tick_initialized() -> None:
    if get_json("worker_tick") is None:
        update_worker_tick(meta={})


def update_worker_tick(meta: Dict[str, Any] | None = None) -> None:
    from datetime import datetime, timezone

    payload = {"ts": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(), "meta": meta or {}}
    set_json("worker_tick", payload)


def get_worker_tick() -> Dict[str, Any] | None:
    return get_json("worker_tick")
