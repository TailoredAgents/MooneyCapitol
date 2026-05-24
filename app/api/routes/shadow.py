from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.api.auth import require_operator
from app.services.shadow_trader import list_shadow_decisions


router = APIRouter(prefix="/shadow", tags=["shadow"], dependencies=[Depends(require_operator)])


@router.get("/decisions")
def get_shadow_decisions(limit: int = Query(default=100, ge=1, le=250)):
    return list_shadow_decisions(limit=limit)
