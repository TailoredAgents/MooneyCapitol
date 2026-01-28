from __future__ import annotations

from app.core.config import AppConfig, default_config
from app.observability.logging import get_logger
from app.services.kv_store import get_json, set_json


CONFIG: AppConfig = default_config()


logger = get_logger("config")

_CONFIG_KEY = "app_config"

def ensure_config_initialized() -> None:
    """Ensure a persisted config exists.

    In DB-backed mode, API and worker should both call this at startup so they
    converge on a single shared config immediately.
    """
    raw = get_json(_CONFIG_KEY)
    if raw:
        refresh_config()
        return
    persist_config()
    refresh_config()


def persist_config() -> None:
    """Persist current config so API + worker share it in Render."""
    set_json(_CONFIG_KEY, CONFIG.model_dump())


def refresh_config() -> bool:
    """Refresh CONFIG in-place from persisted state.

    Do not rebind CONFIG: many modules do `from ... import CONFIG` and rely on the
    object identity staying stable.
    """
    raw = get_json(_CONFIG_KEY)
    if not raw:
        return False
    try:
        cfg = AppConfig.model_validate(raw)
    except Exception as exc:
        logger.warning("config.load_failed", err=str(exc))
        return False
    CONFIG.session = cfg.session
    CONFIG.universe = cfg.universe
    CONFIG.scan = cfg.scan
    CONFIG.detectors = cfg.detectors
    CONFIG.gating = cfg.gating
    CONFIG.alerts = cfg.alerts
    CONFIG.retention_days = cfg.retention_days
    CONFIG.reports = cfg.reports
    CONFIG.depth_provider = cfg.depth_provider
    return True
