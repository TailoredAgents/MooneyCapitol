from __future__ import annotations

import os

from fastapi import APIRouter
from sqlalchemy import text

from app.core.config_store import CONFIG
from app.db.session import engine
from app.services.depth import compute_status as get_depth_status
from app.services.live_state import get_lanes
from app.services.runtime import get_worker_tick


router = APIRouter(prefix="", tags=["health"])

@router.get("/health")
def health():
    ok_db = False
    try:
        with engine.begin() as conn:
            conn.execute(text("select 1"))
        ok_db = True
    except Exception:
        ok_db = False
    depth = get_depth_status()
    slack_enabled = bool(os.getenv("SLACK_BOT_TOKEN"))
    polygon_mode = "live" if os.getenv("POLYGON_API_KEY") else "demo"
    worker_tick = get_worker_tick() or {}
    lanes = get_lanes()
    backlog = {
        "armed": len(lanes.get("armed", [])),
        "primed": len(lanes.get("primed", [])),
        "active": len(lanes.get("active", [])),
    }
    return {
        "ok": ok_db,
        "db": {"status": "ok" if ok_db else "error"},
        "depth": depth,
        "slack": {"status": "ok" if slack_enabled else "disabled", "channel": CONFIG.alerts.slack_channel},
        "polygon": {"mode": polygon_mode, "status": "ok"},
        "worker": worker_tick,
        "backlog": backlog,
    }
