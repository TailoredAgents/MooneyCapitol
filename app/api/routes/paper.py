from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.api.auth import require_operator
from app.services.paper_trader import list_paper_trades, paper_promotion_readiness


router = APIRouter(prefix="/paper", tags=["paper"], dependencies=[Depends(require_operator)])


@router.get("/trades")
def get_paper_trades(limit: int = Query(default=100, ge=1, le=250)):
    return list_paper_trades(limit=limit)


@router.get("/readiness")
def get_paper_readiness():
    return paper_promotion_readiness()
