from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.api.auth import require_operator
from app.services.paper_trader import list_paper_trades


router = APIRouter(prefix="/paper", tags=["paper"], dependencies=[Depends(require_operator)])


@router.get("/trades")
def get_paper_trades(limit: int = Query(default=100, ge=1, le=250)):
    return list_paper_trades(limit=limit)
