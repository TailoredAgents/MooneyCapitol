from __future__ import annotations

from pydantic import BaseModel
from pydantic_settings import BaseSettings


class SessionConfig(BaseModel):
    rth_only: bool = True
    start_et: str = "09:30"
    end_et: str = "16:00"


class UniverseDiscovery(BaseModel):
    source: str = "gappers"
    top_n: int = 100
    gap_threshold_abs: float = 0.05
    premkt_volume_min: int = 20000
    premkt_rvol_min: float = 1.0
    build_every: str = "2m"
    longlist_freeze_et: str = "09:10"
    finalize_et: str = "09:25"


class UniverseConfig(BaseModel):
    mode: str = "hybrid"
    price_min: float = 0.50
    price_max: float = 20.00
    min_volume_prev_day: int = 100000
    include_types: list[str] = ["common_stock"]
    discovery: UniverseDiscovery = UniverseDiscovery()
    pinned: list[str] = []
    blocklist: list[str] = []


class ScanCadence(BaseModel):
    _1m: str = "10s"
    _2m: str = "10s"
    _5m: str = "60s"
    _15m: str = "60s"


class ScanConfig(BaseModel):
    cadence: dict[str, str] = {"1m": "10s", "2m": "10s", "5m": "60s", "15m": "60s"}
    l2_active_max: int = 5


class ConsolidationDetectorConfig(BaseModel):
    min_bars: int = 8
    max_bars: int = 25
    box_height_caps: list[str] = ["$0.10", "0.35pct_mid", "0.6*ATR14_1m"]
    compression_check: bool = True
    rvol_min_inside: float = 1.0


class BreakoutDetectorConfig(BaseModel):
    close_outside_ticks: int = 1
    volume_burst_rvol: float = 1.5
    retest_max_bars: int = 4


class L2ConfirmConfig(BaseModel):
    levels: int = 5
    window_seconds: int = 3
    imbalance_threshold: float = 0.70
    persistence_seconds: int = 3


class DetectorsConfig(BaseModel):
    consolidation: ConsolidationDetectorConfig = ConsolidationDetectorConfig()
    breakout: BreakoutDetectorConfig = BreakoutDetectorConfig()
    l2_confirm: L2ConfirmConfig = L2ConfirmConfig()


class GatingConfig(BaseModel):
    min_rr: float = 2.0
    alert_score_threshold: int = 70
    per_minute_budget_open: int = 8


class AlertsConfig(BaseModel):
    slack_channel: str = "all-trading"
    mention: str = "@conner"
    heads_up_ping: bool = False
    primed_ping: bool = True
    buttons: list[str] = ["Acknowledge", "Pass", "Discuss"]
    per_minute_budget_open: int = 8


class RetentionConfig(BaseModel):
    l2_snapshots: int = 7
    candles: int = 30
    features_and_setups: int = 365


class ReportsConfig(BaseModel):
    eod_slack_time_et: str = "16:10"


class DepthProviderConfig(BaseModel):
    type: str = "ibkr"
    smart_aggregate: bool = True


class AppConfig(BaseModel):
    session: SessionConfig = SessionConfig()
    universe: UniverseConfig = UniverseConfig()
    scan: ScanConfig = ScanConfig()
    detectors: DetectorsConfig = DetectorsConfig()
    gating: GatingConfig = GatingConfig()
    alerts: AlertsConfig = AlertsConfig()
    retention_days: RetentionConfig = RetentionConfig()
    reports: ReportsConfig = ReportsConfig()
    depth_provider: DepthProviderConfig = DepthProviderConfig()


class Settings(BaseSettings):
    database_url: str
    openai_api_key: str | None = None
    polygon_api_key: str | None = None
    slack_bot_token: str | None = None
    slack_signing_secret: str | None = None
    sentry_dsn: str | None = None

    model_config = {
        "env_file": ".env",
        "env_nested_delimiter": "__",
        "env_prefix": "",
        "case_sensitive": False,
    }


def default_config() -> AppConfig:
    return AppConfig()
