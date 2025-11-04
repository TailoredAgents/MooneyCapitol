from __future__ import annotations

import json
from datetime import date

from fastapi import APIRouter, HTTPException, Query

from app.services.learning import ARTIFACT_DIR, get_learning_service


router = APIRouter(prefix="", tags=["learning"])


@router.get("/learning/report")
def learning_report(date: date | None = Query(default=None)):
    report_path = ARTIFACT_DIR / "learning_report.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="report not available")
    return json.loads(report_path.read_text())
