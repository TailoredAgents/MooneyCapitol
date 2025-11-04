from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List


_lanes: Dict[str, Any] = {
    "armed": [],
    "primed": [],
    "active": [],
    "updated_at": datetime.utcnow().isoformat() + "Z",
}


def set_lanes(armed: List[dict], primed: List[dict], active: List[dict], meta: dict | None = None) -> None:
    _lanes["armed"] = armed
    _lanes["primed"] = primed
    _lanes["active"] = active
    _lanes["updated_at"] = datetime.utcnow().isoformat() + "Z"
    if meta:
        _lanes["meta"] = meta


def get_lanes() -> Dict[str, Any]:
    return deepcopy(_lanes)

