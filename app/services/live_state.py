from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List

from app.services.kv_store import get_json, set_json


_lanes: Dict[str, Any] = {
    "armed": [],
    "primed": [],
    "active": [],
    "updated_at": datetime.utcnow().isoformat() + "Z",
}

_LANES_KEY = "lanes"

def ensure_lanes_initialized() -> None:
    if get_json(_LANES_KEY) is None:
        set_json(_LANES_KEY, dict(_lanes))


def set_lanes(armed: List[dict], primed: List[dict], active: List[dict], meta: dict | None = None) -> None:
    _lanes["armed"] = armed
    _lanes["primed"] = primed
    _lanes["active"] = active
    _lanes["updated_at"] = datetime.utcnow().isoformat() + "Z"
    if meta:
        _lanes["meta"] = meta
    set_json(_LANES_KEY, dict(_lanes))


def get_lanes() -> Dict[str, Any]:
    stored = get_json(_LANES_KEY)
    if stored:
        return deepcopy(stored)
    return deepcopy(_lanes)

