from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Query


router = APIRouter(prefix="", tags=["setups"])

_setups_mem: list[dict] = []  # placeholder store


@router.get("/setups")
def list_setups(state: Literal["armed", "primed", "active", "resolved"] | None = Query(default=None)):
    if state is None:
        return {"items": _setups_mem}
    return {"items": [s for s in _setups_mem if s.get("state") == state]}

