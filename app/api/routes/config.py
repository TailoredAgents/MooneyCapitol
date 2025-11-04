from __future__ import annotations

from fastapi import APIRouter

from app.core.config import AppConfig
from app.core.config_store import CONFIG


router = APIRouter(prefix="", tags=["config"])


@router.get("/config", response_model=AppConfig)
def get_config():
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
    return CONFIG
