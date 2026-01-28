from __future__ import annotations

from fastapi import APIRouter
from fastapi import HTTPException

from app.core.config import AppConfig
from app.core.config_store import CONFIG, persist_config, refresh_config
from app.services.kv_store import StateStoreError


router = APIRouter(prefix="", tags=["config"])


@router.get("/config", response_model=AppConfig)
def get_config():
    try:
        refresh_config()
    except StateStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return CONFIG


@router.put("/config", response_model=AppConfig)
def update_config(cfg: AppConfig):
    CONFIG.session = cfg.session
    CONFIG.universe = cfg.universe
    CONFIG.scan = cfg.scan
    CONFIG.detectors = cfg.detectors
    CONFIG.gating = cfg.gating
    CONFIG.alerts = cfg.alerts
    CONFIG.retention_days = cfg.retention_days
    CONFIG.reports = cfg.reports
    CONFIG.depth_provider = cfg.depth_provider
    try:
        persist_config()
    except StateStoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return CONFIG
