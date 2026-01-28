from __future__ import annotations

from datetime import date

from fastapi import APIRouter, HTTPException, Query

from app.services.learning import get_learning_service
from app.services.kv_store import StateStoreError


router = APIRouter(prefix="", tags=["learning"])


@router.get("/learning/report")
def learning_report(date: date | None = Query(default=None)):
    try:
        report = get_learning_service().load_report()
    except StateStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not report:
        raise HTTPException(status_code=404, detail="report not available")
    return report
