from __future__ import annotations

import os

from fastapi import APIRouter
from sqlalchemy import text

from app.core.config_store import CONFIG
from app.db.session import engine
from app.services.depth import compute_status as get_depth_status
from app.services.live_state import get_lanes
from app.services.runtime import get_worker_tick
from app.services.kv_store import StateStoreError, get_store_mode, get_updated_at


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
    worker_tick = {}
    lanes = {"armed": [], "primed": [], "active": []}
    strict = (os.getenv("STATE_STORE") or "auto").lower() in {"db", "postgres", "postgresql"}
    state_store = {"mode": get_store_mode(), "strict": strict}
    try:
        worker_tick = get_worker_tick() or {}
    except StateStoreError as exc:
        state_store["error"] = str(exc)
    try:
        lanes = get_lanes()
    except StateStoreError as exc:
        state_store["error"] = str(exc)
    try:
        state_store["keys_updated_at"] = {
            "app_config": (get_updated_at("app_config").isoformat() if get_updated_at("app_config") else None),
            "worker_tick": (get_updated_at("worker_tick").isoformat() if get_updated_at("worker_tick") else None),
            "lanes": (get_updated_at("lanes").isoformat() if get_updated_at("lanes") else None),
            "learning_report": (get_updated_at("learning_report").isoformat() if get_updated_at("learning_report") else None),
            "learning_thresholds": (get_updated_at("learning_thresholds").isoformat() if get_updated_at("learning_thresholds") else None),
        }
    except StateStoreError as exc:
        state_store["error"] = str(exc)
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
        "state_store": state_store,
        "worker": worker_tick,
        "backlog": backlog,
    }
