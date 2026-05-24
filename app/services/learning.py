from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from sqlalchemy import select

from app.db.models import AccountSnapshot, Alert, Fill, PaperTrade, Setup, ShadowDecision, Trade
from app.db.session import get_session
from app.observability.logging import get_logger
from app.services.kv_store import get_bytes, get_json, set_bytes, set_json
from app.services.trade_matcher import SetupMatch, best_setup_match


logger = get_logger("learning")

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

_MODEL_KEY = "learning_model_alert_ranker"
_SCALER_KEY = "learning_model_alert_ranker_scaler"
_XGB_MODEL_KEY = "learning_model_alert_ranker_xgb"
_FEATURE_IMPORTANCE_KEY = "learning_feature_importance"
_THRESHOLDS_KEY = "learning_thresholds"
_CANARY_KEY = "learning_thresholds_canary"
_REPORT_KEY = "learning_report"
FEATURE_VERSION = "2026-05-24-connor-size-v1"

# Regime-specific model keys
_REGIME_MODELS_KEY = "learning_regime_models"
_REGIME_META_MODEL_KEY = "learning_regime_meta_model"
_CONNOR_SIZER_KEY = "learning_model_connor_sizer"

FEATURE_COLS = [
    "box_height",
    "box_bars",
    "rvol_break",
    "l2_mean",
    "l2_persist",
    "dist_htf",
    "dist_gap",
    "spread_cents",
    "direction_long",
    "price",
    "price_bucket",
    "time_bucket",
    "score",
    "rr_min",
    # Regime indicators
    "regime_premarket",
    "regime_opening", 
    "regime_morning",
    "regime_midday",
    "regime_afternoon",
    "regime_closing",
    # Regime-specific features
    "gap_strength",
    "premarket_rvol",
    "opening_volume",
    "l2_opening_pressure", 
    "consolidation_quality",
    "htf_support",
    "momentum_strength",
    # Enhanced discretionary trading features
    "l2_absorption_score",
    "bid_ask_imbalance", 
    "tape_momentum_5s",
    "tape_momentum_15s",
    "order_flow_pressure",
    "volume_spike_intensity",
    "momentum_acceleration",
    "l2_wall_detection",
    "microstructure_edge",
    # Connor's conviction signal
    "connor_size_pct",
]


PRICE_BUCKETS = [
    (0.0, 5.0, "0.5-5"),
    (5.0, 10.0, "5-10"),
    (10.0, 20.1, "10-20"),
]

TIME_BUCKETS = [
    ((9, 30), (10, 30), "open"),
    ((10, 30), (14, 30), "mid"),
    ((14, 30), (16, 5), "close"),
]

# Enhanced regime classification for specialist models
TRADING_REGIMES = [
    ((4, 0), (9, 25), "premarket"),     # Pre-market gap scanning
    ((9, 25), (9, 45), "opening"),     # Opening burst/volatility  
    ((9, 45), (11, 0), "morning"),     # Morning momentum
    ((11, 0), (14, 0), "midday"),      # Midday consolidation
    ((14, 0), (15, 30), "afternoon"),  # Afternoon breakouts
    ((15, 30), (16, 5), "closing"),    # Power hour/closing
]

LearningLabel = Literal["suggested_taken", "suggested_ignored", "manual_no_alert", "copied_outcome", "paper_outcome"]

LABEL_SUGGESTED_TAKEN: LearningLabel = "suggested_taken"
LABEL_SUGGESTED_IGNORED: LearningLabel = "suggested_ignored"
LABEL_MANUAL_NO_ALERT: LearningLabel = "manual_no_alert"
LABEL_COPIED_OUTCOME: LearningLabel = "copied_outcome"
LABEL_PAPER_OUTCOME: LearningLabel = "paper_outcome"


@dataclass(frozen=True)
class HybridTargetWeights:
    behavior_weight: float = 0.35
    outcome_weight: float = 0.50
    manual_edge_weight: float = 0.15
    penalty_weight: float = 0.10
    min_realized_r: float = -2.0
    max_realized_r: float = 3.0


DEFAULT_HYBRID_WEIGHTS = HybridTargetWeights()


def bucket_price(price: float) -> str:
    """Bucket price into predefined ranges for learning."""
    for min_price, max_price, label in PRICE_BUCKETS:
        if min_price <= price < max_price:
            return label
    return "20+"


def bucket_time(dt: datetime) -> str:
    """Bucket time into trading session periods."""
    hour, minute = dt.hour, dt.minute
    for (start_h, start_m), (end_h, end_m), label in TIME_BUCKETS:
        start_time = start_h * 60 + start_m
        end_time = end_h * 60 + end_m
        current_time = hour * 60 + minute
        if start_time <= current_time < end_time:
            return label
    return "other"


def classify_trading_regime(dt: datetime) -> str:
    """Classify trading session into regime for specialist models."""
    hour, minute = dt.hour, dt.minute
    for (start_h, start_m), (end_h, end_m), label in TRADING_REGIMES:
        start_time = start_h * 60 + start_m
        end_time = end_h * 60 + end_m
        current_time = hour * 60 + minute
        if start_time <= current_time < end_time:
            return label
    return "after_hours"


def get_regime_features(regime: str, base_features: dict[str, float]) -> dict[str, float]:
    """Add regime-specific feature engineering with discretionary trading enhancements."""
    features = base_features.copy()
    
    # Add regime indicators
    for regime_name in ["premarket", "opening", "morning", "midday", "afternoon", "closing"]:
        features[f"regime_{regime_name}"] = 1.0 if regime == regime_name else 0.0
    
    # Enhanced L2 and momentum features for discretionary trading
    l2_mean = features.get("l2_mean", 0.0)
    l2_persist = features.get("l2_persist", 0.0)
    rvol = features.get("rvol_break", 0.0)
    spread = features.get("spread_cents", 0.0)
    
    # Universal discretionary features (Connor's "feel" patterns)
    features["l2_absorption_score"] = l2_mean * l2_persist  # Sustained L2 pressure
    features["bid_ask_imbalance"] = abs(l2_mean - 0.5) * 2.0  # How far from 50/50
    features["tape_momentum_5s"] = rvol * l2_mean if l2_mean > 0.6 else rvol * (1 - l2_mean)
    features["tape_momentum_15s"] = features["tape_momentum_5s"] * 0.8  # Smoothed version
    features["order_flow_pressure"] = l2_mean * (1.0 / max(spread, 0.01))  # L2 vs spread
    features["volume_spike_intensity"] = rvol * (1.0 if rvol > 2.0 else 0.5)  # Spike detection
    features["momentum_acceleration"] = rvol * l2_persist  # Sustained volume + L2
    features["l2_wall_detection"] = 1.0 if l2_mean > 0.85 or l2_mean < 0.15 else 0.0  # Extreme imbalance
    features["microstructure_edge"] = (l2_mean * rvol) / max(spread, 0.01)  # Connor's edge composite
    
    # Regime-specific feature weighting
    if regime == "premarket":
        # Emphasize gap and pre-market volume
        features["gap_strength"] = features.get("dist_gap", 0.0) * 2.0
        features["premarket_rvol"] = rvol * 1.5
        # Premarket L2 is often thinner, emphasize volume over L2
        features["microstructure_edge"] *= 0.7
    
    elif regime == "opening":
        # Emphasize volume burst and L2 strength  
        features["opening_volume"] = rvol * 2.0
        features["l2_opening_pressure"] = l2_mean * 1.5
        # Opening chaos - prioritize tape reading
        features["tape_momentum_5s"] *= 1.3
        features["order_flow_pressure"] *= 1.2
    
    elif regime == "midday":
        # Emphasize consolidation quality and HTF levels
        features["consolidation_quality"] = features.get("box_bars", 0.0) / max(features.get("box_height", 1.0), 0.01)
        features["htf_support"] = features.get("dist_htf", 0.0) * 1.5
        # Midday - L2 more reliable, emphasize absorption
        features["l2_absorption_score"] *= 1.4
        features["l2_wall_detection"] *= 1.2
    
    elif regime in ["afternoon", "closing"]:
        # Emphasize momentum and trend continuation
        features["momentum_strength"] = rvol * l2_mean
        # Power hour - momentum + acceleration key
        features["momentum_acceleration"] *= 1.3
        features["volume_spike_intensity"] *= 1.2
    
    return features


@dataclass(frozen=True)
class LearningRow:
    label: LearningLabel
    symbol: str | None
    detected_ts: datetime | None
    features: dict[str, float]
    taken_by_master: bool = False
    realized_r: float | None = None
    pnl: float | None = None
    manual_no_alert: bool = False
    reject_or_bad_slippage: bool = False
    setup_id: int | None = None
    alert_id: int | None = None
    trade_id: int | None = None
    fill_id: int | None = None
    source: str | None = None
    setup_match_confidence: str | None = None
    setup_match_score: float | None = None
    setup_match_reason: dict[str, Any] | None = None
    sample_weight: float = 1.0

    def as_training_dict(self, weights: HybridTargetWeights = DEFAULT_HYBRID_WEIGHTS) -> dict[str, Any]:
        row: dict[str, Any] = {
            "label_class": self.label,
            "symbol": self.symbol,
            "detected_ts": self.detected_ts,
            "taken_by_master": self.taken_by_master,
            "realized_r": self.realized_r,
            "pnl": self.pnl,
            "manual_no_alert": self.manual_no_alert,
            "reject_or_bad_slippage": self.reject_or_bad_slippage,
            "setup_id": self.setup_id,
            "alert_id": self.alert_id,
            "trade_id": self.trade_id,
            "fill_id": self.fill_id,
            "source": self.source,
            "setup_match_confidence": self.setup_match_confidence,
            "setup_match_score": self.setup_match_score,
            "setup_match_reason": self.setup_match_reason,
            "sample_weight": self.sample_weight,
            "hybrid_target": calculate_hybrid_target(
                label=self.label,
                taken_by_master=self.taken_by_master,
                realized_r=self.realized_r,
                reject_or_bad_slippage=self.reject_or_bad_slippage,
                weights=weights,
            ),
        }
        row.update(self.features)
        return row


def bucket_price(price: float | None) -> str:
    if price is None:
        return "unknown"
    for lower, upper, name in PRICE_BUCKETS:
        if lower <= price < upper:
            return name
    return "unknown"


def bucket_time(ts: datetime | None) -> str:
    if ts is None:
        return "unknown"
    time_tuple = (ts.hour, ts.minute)
    for start, end, name in TIME_BUCKETS:
        if start <= time_tuple < end:
            return name
    return "unknown"


def _master_equity() -> float | None:
    account_ref = os.getenv("WEBULL_MASTER_ACCOUNT_ID")
    if account_ref:
        try:
            from sqlalchemy import desc
            with get_session() as session:
                row = (
                    session.execute(
                        select(AccountSnapshot)
                        .where(AccountSnapshot.account_ref == account_ref)
                        .order_by(desc(AccountSnapshot.snapshot_time))
                        .limit(1)
                    )
                    .scalars()
                    .first()
                )
            if row is not None:
                value = row.equity_value if row.equity_value is not None else row.total_value
                if value and float(value) > 0:
                    return float(value)
        except Exception:
            pass
    # Fallback to static env var (useful before the live account is connected)
    val = os.getenv("WEBULL_MASTER_ACCOUNT_EQUITY")
    if not val:
        return None
    try:
        equity = float(val)
        return equity if equity > 0 else None
    except (TypeError, ValueError):
        return None


def _fills_connor_size_pct(fills: list, equity: float | None) -> float:
    """Buy-side fill notional as a fraction of Connor's account equity."""
    if not fills or not equity:
        return 0.0
    notional = sum(
        float(f.qty or 0) * float(f.price or 0)
        for f in fills
        if str(getattr(f, "side", "") or "").upper().startswith("B")
    )
    return round(notional / equity, 6) if notional > 0 else 0.0


def _encode_price_bucket(value) -> float:
    mapping = {"unknown": 0.0, "0.5-5": 1.0, "5-10": 2.0, "10-20": 3.0, "20+": 4.0}
    if value in mapping:
        return mapping[value]
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _encode_time_bucket(value) -> float:
    mapping = {"unknown": 0.0, "other": 0.0, "open": 1.0, "mid": 2.0, "close": 3.0}
    if value in mapping:
        return mapping[value]
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def normalize_realized_r(realized_r: float | None, weights: HybridTargetWeights = DEFAULT_HYBRID_WEIGHTS) -> float:
    if realized_r is None:
        return 0.0
    if weights.max_realized_r <= weights.min_realized_r:
        return 0.0
    clipped = min(max(float(realized_r), weights.min_realized_r), weights.max_realized_r)
    return (clipped - weights.min_realized_r) / (weights.max_realized_r - weights.min_realized_r)


def calculate_hybrid_target(
    label: LearningLabel,
    taken_by_master: bool,
    realized_r: float | None,
    reject_or_bad_slippage: bool = False,
    weights: HybridTargetWeights = DEFAULT_HYBRID_WEIGHTS,
) -> float:
    taken_signal = 1.0 if taken_by_master or label in {LABEL_SUGGESTED_TAKEN, LABEL_MANUAL_NO_ALERT} else 0.0
    manual_trade_signal = 1.0 if label == LABEL_MANUAL_NO_ALERT else 0.0
    penalty_signal = 1.0 if reject_or_bad_slippage else 0.0

    score = (
        weights.behavior_weight * taken_signal
        + weights.outcome_weight * normalize_realized_r(realized_r, weights)
        + weights.manual_edge_weight * manual_trade_signal
        - weights.penalty_weight * penalty_signal
    )
    return round(max(0.0, min(1.0, score)), 6)


@dataclass
class LearningArtifacts:
    model_path: Path
    scaler_path: Path
    xgb_model_path: Path
    xgb_pickle_path: Path
    feature_importance_path: Path
    thresholds_path: Path
    canary_path: Path
    report_path: Path


class LearningService:
    def __init__(self) -> None:
        self.lookback_days = int(os.getenv("LEARNING_LOOKBACK_DAYS", "30"))
        self.manual_match_window_minutes = int(os.getenv("LEARNING_MANUAL_MATCH_WINDOW_MINUTES", "30"))
        self.match_pre_alert_seconds = int(os.getenv("LEARNING_MATCH_PRE_ALERT_SECONDS", "60"))
        self.learning_match_min_confidence = os.getenv("LEARNING_MATCH_MIN_CONFIDENCE", "likely").lower()
        self.xgb_min_rows = int(os.getenv("LEARNING_XGB_MIN_ROWS", "20"))
        self.sandbox_enabled = os.getenv("LEARNING_SANDBOX_ENABLED", "0").lower() in {"1", "true", "yes", "on"}
        self.synthetic_max_ratio = max(0.0, float(os.getenv("LEARNING_SYNTHETIC_MAX_RATIO", "3.0")))
        self.synthetic_sample_weight = max(0.0, float(os.getenv("LEARNING_SYNTHETIC_SAMPLE_WEIGHT", "0.15")))
        self.paper_sample_weight = max(0.0, float(os.getenv("LEARNING_PAPER_SAMPLE_WEIGHT", "0.25")))
        self.hybrid_weights = HybridTargetWeights(
            behavior_weight=float(os.getenv("LEARNING_BEHAVIOR_WEIGHT", "0.35")),
            outcome_weight=float(os.getenv("LEARNING_OUTCOME_WEIGHT", "0.50")),
            manual_edge_weight=float(os.getenv("LEARNING_MANUAL_EDGE_WEIGHT", "0.15")),
            penalty_weight=float(os.getenv("LEARNING_PENALTY_WEIGHT", "0.10")),
            min_realized_r=float(os.getenv("LEARNING_MIN_REALIZED_R", "-2.0")),
            max_realized_r=float(os.getenv("LEARNING_MAX_REALIZED_R", "3.0")),
        )
        self.artifacts = LearningArtifacts(
            model_path=ARTIFACT_DIR / "alert_ranker.pkl",
            scaler_path=ARTIFACT_DIR / "alert_ranker_scaler.pkl",
            xgb_model_path=ARTIFACT_DIR / "alert_ranker_xgb.json",
            xgb_pickle_path=ARTIFACT_DIR / "alert_ranker_xgb.pkl",
            feature_importance_path=ARTIFACT_DIR / "feature_importance.json",
            thresholds_path=ARTIFACT_DIR / "thresholds.json",
            canary_path=ARTIFACT_DIR / "thresholds_canary.json",
            report_path=ARTIFACT_DIR / "learning_report.json",
        )

    # ----------------- Data building -------------------------------------------------
    def _load_setups(self, start_ts: datetime) -> List[Setup]:
        with get_session() as session:
            stmt = select(Setup).where(Setup.detected_ts >= start_ts).order_by(Setup.detected_ts.asc())
            return list(session.execute(stmt).scalars())

    def _alerts_for_setups(self, setup_ids: Iterable[int]) -> Dict[int, List[Alert]]:
        ids = [sid for sid in set(setup_ids) if sid]
        if not ids:
            return {}
        with get_session() as session:
            stmt = select(Alert).where(Alert.setup_id.in_(ids))
            alerts: Dict[int, List[Alert]] = {sid: [] for sid in ids}
            for alert in session.execute(stmt).scalars():
                alerts.setdefault(alert.setup_id, []).append(alert)
            return alerts

    def _load_fills(self, start_ts: datetime, end_ts: datetime) -> List[Fill]:
        with get_session() as session:
            stmt = select(Fill).where(Fill.ts >= start_ts, Fill.ts < end_ts).order_by(Fill.ts.asc())
            return list(session.execute(stmt).scalars())

    def _load_trades(self, start_ts: datetime, end_ts: datetime) -> List[Trade]:
        with get_session() as session:
            stmt = select(Trade).where(Trade.open_ts >= start_ts, Trade.open_ts < end_ts).order_by(Trade.open_ts.asc())
            return list(session.execute(stmt).scalars())

    def _load_closed_paper_trades(self, start_ts: datetime, end_ts: datetime) -> list[tuple[PaperTrade, ShadowDecision | None]]:
        with get_session() as session:
            stmt = (
                select(PaperTrade, ShadowDecision)
                .outerjoin(ShadowDecision, PaperTrade.shadow_decision_id == ShadowDecision.id)
                .where(PaperTrade.status == "closed", PaperTrade.closed_at >= start_ts, PaperTrade.closed_at < end_ts)
                .order_by(PaperTrade.closed_at.asc())
            )
            return list(session.execute(stmt).all())

    def _setup_features(self, setup: Setup) -> dict[str, float]:
        payload = setup.payload_json or {}
        features = dict(payload.get("features", {}) or {})
        entry_price = setup.entry_price or payload.get("entry_price") or payload.get("entry")
        try:
            price = float(entry_price) if entry_price not in (None, "n/a", "") else None
        except (TypeError, ValueError):
            price = None
        detected_ts = setup.detected_ts
        features.setdefault("price", price or 0.0)
        features.setdefault("direction_long", 1.0 if setup.direction == "long" else 0.0)
        if setup.rr_min is not None:
            features.setdefault("rr_min", float(setup.rr_min))
        if setup.score is not None:
            features.setdefault("score", float(setup.score))
        if detected_ts:
            features.setdefault("session_phase_open", 1.0 if bucket_time(detected_ts) == "open" else 0.0)
            features.setdefault("session_phase_mid", 1.0 if bucket_time(detected_ts) == "mid" else 0.0)
            features.setdefault("session_phase_close", 1.0 if bucket_time(detected_ts) == "close" else 0.0)
        return features

    def _manual_trade_features(self, fill: Fill | None = None, trade: Trade | None = None) -> dict[str, float]:
        price = None
        qty = None
        side = None
        if fill is not None:
            price = fill.price
            qty = fill.qty
            side = fill.side
        elif trade is not None:
            price = trade.basis
            qty = trade.qty
        equity = _master_equity()
        notional = float(qty or 0) * float(price or 0)
        connor_size_pct = round(notional / equity, 6) if (equity and notional > 0) else 0.0
        features = {
            "price": float(price or 0.0),
            "qty": float(qty or 0.0),
            "direction_long": 0.0 if side and str(side).upper().startswith("S") else 1.0,
            "manual_trade": 1.0,
            "box_height": 0.0,
            "box_bars": 0.0,
            "rvol_break": 0.0,
            "l2_mean": 0.0,
            "l2_persist": 0.0,
            "dist_htf": 0.0,
            "dist_gap": 0.0,
            "spread_cents": 0.0,
            "connor_size_pct": connor_size_pct,
        }
        return features

    def _paper_trade_features(self, paper_trade: PaperTrade, shadow_decision: ShadowDecision | None = None) -> dict[str, float]:
        payload = getattr(shadow_decision, "payload_json", None) or {}
        features = dict(payload.get("features", {}) or {})
        entry_price = paper_trade.entry_price or payload.get("entry_price") or payload.get("entry")
        try:
            price = float(entry_price) if entry_price not in (None, "n/a", "") else 0.0
        except (TypeError, ValueError):
            price = 0.0
        direction = str(paper_trade.direction or payload.get("direction") or "long").lower()
        features.setdefault("price", price)
        features.setdefault("direction_long", 1.0 if direction == "long" else 0.0)
        if paper_trade.realized_r is not None:
            features.setdefault("rr_min", float(paper_trade.realized_r))
        for key, default in {
            "box_height": 0.0,
            "box_bars": 0.0,
            "rvol_break": 0.0,
            "l2_mean": 0.0,
            "l2_persist": 0.0,
            "dist_htf": 0.0,
            "dist_gap": 0.0,
            "spread_cents": 0.0,
            "score": 0.0,
            "rr_min": 0.0,
            "connor_size_pct": 0.0,
        }.items():
            features.setdefault(key, default)
        return features

    def _is_near_setup(self, symbol: str | None, ts: datetime | None, setups: list[Setup]) -> bool:
        if not symbol or not ts:
            return False
        window_s = self.manual_match_window_minutes * 60
        symbol_upper = symbol.upper()
        for setup in setups:
            payload = setup.payload_json or {}
            setup_symbol = payload.get("symbol")
            if not setup_symbol:
                continue
            if str(setup_symbol).upper() != symbol_upper:
                continue
            if setup.detected_ts and abs((ts - setup.detected_ts).total_seconds()) <= window_s:
                return True
        return False

    def _best_fill_setup_match(self, fill: Fill, setups: list[Setup]) -> SetupMatch:
        return best_setup_match(
            symbol=getattr(fill, "symbol", None),
            side=getattr(fill, "side", None),
            ts=getattr(fill, "ts", None),
            price=getattr(fill, "price", None),
            setups=setups,
            window_seconds=self.manual_match_window_minutes * 60,
            pre_alert_seconds=self.match_pre_alert_seconds,
        )

    def _build_learning_rows(self, trade_date: date) -> list[LearningRow]:
        end = datetime.combine(trade_date, datetime.min.time(), tzinfo=timezone.utc)
        start = end - timedelta(days=self.lookback_days)
        master_equity = _master_equity()
        setups = self._load_setups(start)
        fills = self._load_fills(start, end)
        trades = self._load_trades(start, end)
        paper_trades = self._load_closed_paper_trades(start, end)
        alerts_map = self._alerts_for_setups([s.id for s in setups])

        fills_by_setup: dict[int, list[Fill]] = {}
        fill_matches: dict[int, SetupMatch] = {}
        unmatched_fills: list[Fill] = []
        for fill in fills:
            if fill.setup_id is not None:
                fills_by_setup.setdefault(int(fill.setup_id), []).append(fill)
                continue
            match = self._best_fill_setup_match(fill, setups)
            if match.matched and match.meets(self.learning_match_min_confidence):
                fills_by_setup.setdefault(int(match.setup.id), []).append(fill)
                fill_matches[getattr(fill, "id", id(fill))] = match
            else:
                if match.confidence != "none":
                    fill_matches[getattr(fill, "id", id(fill))] = match
                unmatched_fills.append(fill)

        trades_by_setup: dict[int, list[Trade]] = {}
        for trade in trades:
            if trade.setup_id is not None:
                trades_by_setup.setdefault(int(trade.setup_id), []).append(trade)

        rows: list[LearningRow] = []
        for setup in setups:
            payload = setup.payload_json or {}
            alerts = alerts_map.get(setup.id, [])
            setup_fills = fills_by_setup.get(setup.id, [])
            setup_trades = trades_by_setup.get(setup.id, [])
            taken = bool(setup_fills or setup_trades)

            realized_r = payload.get("realized_r")
            if realized_r is None and setup_trades:
                realized_r = setup_trades[-1].realized_r
            if realized_r is None and setup.rr_min is not None:
                realized_r = setup.rr_min
            try:
                realized_r_float = float(realized_r) if realized_r is not None else None
            except (TypeError, ValueError):
                realized_r_float = None

            pnl = payload.get("pnl")
            if pnl is None and setup_trades:
                pnl = setup_trades[-1].p_and_l
            try:
                pnl_float = float(pnl) if pnl is not None else None
            except (TypeError, ValueError):
                pnl_float = None

            alert_id = alerts[-1].id if alerts else None
            trade_id = setup_trades[-1].id if setup_trades else None
            fill_id = setup_fills[-1].id if setup_fills else None
            fill_match = fill_matches.get(fill_id) if fill_id is not None else None
            setup_match_confidence = (
                fill_match.confidence
                if fill_match
                else getattr(setup_fills[-1], "setup_match_confidence", None)
                if setup_fills
                else None
            )
            setup_match_score = (
                fill_match.score
                if fill_match
                else getattr(setup_fills[-1], "setup_match_score", None)
                if setup_fills
                else None
            )
            setup_match_reason = (
                fill_match.reason
                if fill_match
                else getattr(setup_fills[-1], "setup_match_reason", None)
                if setup_fills
                else None
            )
            label = LABEL_SUGGESTED_TAKEN if taken else LABEL_SUGGESTED_IGNORED
            setup_feats = self._setup_features(setup)
            setup_feats["connor_size_pct"] = _fills_connor_size_pct(setup_fills, master_equity)
            rows.append(
                LearningRow(
                    label=label,
                    symbol=payload.get("symbol"),
                    detected_ts=setup.detected_ts,
                    features=setup_feats,
                    taken_by_master=taken,
                    realized_r=realized_r_float,
                    pnl=pnl_float,
                    manual_no_alert=False,
                    setup_id=setup.id,
                    alert_id=alert_id,
                    trade_id=trade_id,
                    fill_id=fill_id,
                    source="scout",
                    setup_match_confidence=setup_match_confidence,
                    setup_match_score=setup_match_score,
                    setup_match_reason=setup_match_reason,
                )
            )

        for fill in unmatched_fills:
            fill_key = getattr(fill, "id", id(fill))
            weak_match = fill_matches.get(fill_key)
            rows.append(
                LearningRow(
                    label=LABEL_MANUAL_NO_ALERT,
                    symbol=fill.symbol,
                    detected_ts=fill.ts,
                    features=self._manual_trade_features(fill=fill),
                    taken_by_master=True,
                    realized_r=None,
                    pnl=None,
                    manual_no_alert=True,
                    fill_id=fill.id,
                    source="master_manual",
                    setup_match_confidence=weak_match.confidence if weak_match else None,
                    setup_match_score=weak_match.score if weak_match else None,
                    setup_match_reason=weak_match.reason if weak_match else None,
                )
            )

        for paper_trade, shadow_decision in paper_trades:
            rows.append(
                LearningRow(
                    label=LABEL_PAPER_OUTCOME,
                    symbol=paper_trade.symbol,
                    detected_ts=paper_trade.closed_at or paper_trade.opened_at,
                    features=self._paper_trade_features(paper_trade, shadow_decision),
                    taken_by_master=False,
                    realized_r=paper_trade.realized_r,
                    pnl=paper_trade.realized_pnl,
                    manual_no_alert=False,
                    setup_id=paper_trade.setup_id,
                    alert_id=paper_trade.alert_id,
                    source="paper_trader",
                    sample_weight=self.paper_sample_weight,
                )
            )

        return rows

    def _label_breakdown(self, df: pd.DataFrame) -> dict[str, int]:
        if df.empty or "label_class" not in df:
            return {}
        return {str(k): int(v) for k, v in df["label_class"].value_counts().to_dict().items()}

    def _build_rows(self, trade_date: date) -> pd.DataFrame:
        learning_rows = self._build_learning_rows(trade_date)
        if not learning_rows:
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        for learning_row in learning_rows:
            row = learning_row.as_training_dict(self.hybrid_weights)
            detected_ts = row.get("detected_ts")
            price = row.get("price")
            row["price_bucket"] = bucket_price(float(price)) if price not in (None, "") else "unknown"
            row["time_bucket"] = bucket_time(detected_ts.astimezone(timezone.utc) if detected_ts else None)
            row["label"] = 1 if (row.get("realized_r") is not None and row["realized_r"] >= 2.0) else 0
            rows.append(row)

        df = pd.DataFrame(rows)
        df.dropna(subset=["box_height", "box_bars", "rvol_break", "spread_cents"], inplace=True)
        if df.empty:
            return df
        df.sort_values("detected_ts", inplace=True)
        return df

    # ----------------- Training -----------------------------------------------------
    async def train_with_sandbox(self, trade_date: date) -> dict[str, Any]:
        """Enhanced training with sandbox simulation for accelerated learning."""
        if not self.sandbox_enabled:
            return self.train(trade_date)

        try:
            # Import here to avoid circular dependencies
            from app.learning.sandbox_simulator import enhance_learning_with_sandbox
            
            # Build real dataset
            df = self._build_rows(trade_date)
            if df.empty:
                logger.warning("learning.dataset.insufficient", rows=len(df))
                return {"status": "skipped", "reason": "insufficient_data", "rows": len(df)}
            
            # Convert to LearningRow objects for sandbox enhancement
            real_samples = [
                LearningRow(
                    label=row["label_class"],
                    symbol=row.get("symbol"),
                    detected_ts=row.get("detected_ts"),
                    features={k: v for k, v in row.items() if k in FEATURE_COLS},
                    taken_by_master=row.get("taken_by_master", False),
                    realized_r=row.get("realized_r"),
                    source=row.get("source") or "real_trading",
                    sample_weight=float(row.get("sample_weight", 1.0) or 1.0),
                )
                for _, row in df.iterrows()
            ]
            
            # Enhance with sandbox simulation
            enhanced_samples = await enhance_learning_with_sandbox(trade_date, real_samples)
            synthetic_samples = [sample for sample in enhanced_samples if sample.source == "sandbox_simulation"]
            max_synthetic = int(len(real_samples) * self.synthetic_max_ratio)
            synthetic_samples = synthetic_samples[:max_synthetic]
            capped_samples = real_samples + synthetic_samples
            
            # Convert back to DataFrame
            enhanced_rows = []
            for sample in capped_samples:
                row = sample.as_training_dict(self.hybrid_weights)
                row["price_bucket"] = bucket_price(float(row.get("price", 0.0)))
                row["time_bucket"] = bucket_time(row.get("detected_ts") or datetime.now())
                row["label"] = 1 if (row.get("realized_r") is not None and row["realized_r"] >= 2.0) else 0
                row["sample_weight"] = self.synthetic_sample_weight if sample.source == "sandbox_simulation" else sample.sample_weight
                enhanced_rows.append(row)
            
            enhanced_df = pd.DataFrame(enhanced_rows) if enhanced_rows else df
            
            # Train with enhanced dataset
            try:
                result = self._train_xgboost(trade_date, enhanced_df)
                result["sandbox_enhanced"] = {
                    "enabled": True,
                    "real_samples": len(real_samples),
                    "synthetic_samples": len(synthetic_samples),
                    "synthetic_max_ratio": self.synthetic_max_ratio,
                    "synthetic_sample_weight": self.synthetic_sample_weight,
                    "enhancement_ratio": len(capped_samples) / len(real_samples) if real_samples else 0,
                }
                return result
            except Exception as exc:
                logger.warning("learning.xgboost.sandbox.fallback", err=str(exc), rows=len(enhanced_df))
                return self._train_logistic(trade_date, enhanced_df, fallback_reason=f"sandbox_enhanced:{exc}")
                
        except Exception as exc:
            logger.error("learning.sandbox.failed", err=str(exc))
            # Fallback to regular training
            return self.train(trade_date)

    def train(self, trade_date: date) -> dict[str, Any]:
        df = self._build_rows(trade_date)
        if df.empty:
            logger.warning("learning.dataset.insufficient", rows=len(df))
            return {"status": "skipped", "reason": "insufficient_data", "rows": len(df)}

        try:
            return self._train_xgboost(trade_date, df)
        except Exception as exc:
            logger.warning("learning.xgboost.fallback", err=str(exc), rows=len(df))
            return self._train_logistic(trade_date, df, fallback_reason=str(exc))

    def _feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        for col in FEATURE_COLS:
            if col not in df:
                df[col] = 0.0
        frame = df[FEATURE_COLS].copy()
        frame["price_bucket"] = frame["price_bucket"].map(_encode_price_bucket)
        frame["time_bucket"] = frame["time_bucket"].map(_encode_time_bucket)
        for col in FEATURE_COLS:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        return frame.fillna(0.0).astype(float).values

    def _sample_weights(self, df: pd.DataFrame) -> np.ndarray | None:
        if "sample_weight" not in df:
            return None
        weights = pd.to_numeric(df["sample_weight"], errors="coerce").fillna(1.0)
        return weights.clip(lower=0.0, upper=10.0).astype(float).values

    def _train_xgboost(self, trade_date: date, df: pd.DataFrame) -> dict[str, Any]:
        if len(df) < self.xgb_min_rows:
            raise RuntimeError(f"insufficient_xgboost_rows:{len(df)}<{self.xgb_min_rows}")
        if "hybrid_target" not in df:
            raise RuntimeError("hybrid_target_missing")

        try:
            from xgboost import XGBRegressor
        except Exception as exc:
            raise RuntimeError("xgboost_unavailable") from exc

        X = self._feature_matrix(df)
        y = df["hybrid_target"].fillna(0.0).astype(float).values
        sample_weights = self._sample_weights(df)
        if len(np.unique(y)) < 2:
            raise RuntimeError("single_hybrid_target")

        model = XGBRegressor(
            n_estimators=160,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=2,
        )

        split_idx = max(1, int(len(df) * 0.8))
        if split_idx >= len(df):
            split_idx = len(df) - 1
        X_train, X_valid = X[:split_idx], X[split_idx:]
        y_train, y_valid = y[:split_idx], y[split_idx:]
        if sample_weights is not None:
            model.fit(X_train, y_train, sample_weight=sample_weights[:split_idx])
        else:
            model.fit(X_train, y_train)

        valid_preds = model.predict(X_valid) if len(X_valid) else np.array([])
        rmse = None
        if len(valid_preds):
            rmse = float(np.sqrt(np.mean((valid_preds - y_valid) ** 2)))

        if sample_weights is not None:
            model.fit(X, y, sample_weight=sample_weights)
        else:
            model.fit(X, y)
        preds = np.clip(model.predict(X), 0.0, 1.0)
        df["p2r"] = preds

        model_buf = io.BytesIO()
        joblib.dump(model, model_buf)
        set_bytes(_XGB_MODEL_KEY, model_buf.getvalue())
        try:
            joblib.dump(model, self.artifacts.xgb_pickle_path)
            model.save_model(self.artifacts.xgb_model_path)
        except Exception:
            pass

        feature_importance = self._feature_importance(model)
        set_json(_FEATURE_IMPORTANCE_KEY, feature_importance)
        try:
            self.artifacts.feature_importance_path.write_text(json.dumps(feature_importance, indent=2))
        except Exception:
            pass

        threshold_report = self._grid_search(df)
        self._persist_threshold_report(threshold_report)
        self._refresh_logistic_compatibility_model(df)

        report = self._build_training_report(
            trade_date=trade_date,
            df=df,
            threshold_report=threshold_report,
            model_type="xgboost",
            metrics={"rmse": round(rmse, 4) if rmse is not None else None},
            feature_importance=feature_importance,
            fallback_used=False,
            fallback_reason=None,
        )
        self._persist_report(report)
        
        # Try training regime specialists if we have enough data
        if len(df) >= 50:
            try:
                regime_result = self._train_regime_specialists(trade_date, df)
                if regime_result.get("status") == "regime_specialists_trained":
                    report["regime_specialists"] = regime_result
                    logger.info("learning.regime_specialists.success", regimes=regime_result.get("regimes_trained"))
            except Exception as exc:
                logger.warning("learning.regime_specialists.failed", err=str(exc))

        self._train_connor_sizer(df)
        return {"status": "trained", **report}

    def _train_regime_specialists(self, trade_date: date, df: pd.DataFrame) -> dict[str, Any]:
        """Train regime-specific specialist models for improved accuracy."""
        if len(df) < 50:  # Need sufficient data for regime splitting
            return {"status": "skipped", "reason": "insufficient_data_for_regimes"}

        try:
            from xgboost import XGBRegressor
        except Exception as exc:
            return {"status": "skipped", "reason": f"xgboost_unavailable: {exc}"}

        # Classify each training sample by regime
        df["regime"] = df.apply(
            lambda row: classify_trading_regime(row.get("detected_ts", datetime.now())), axis=1
        )

        regime_models = {}
        regime_metrics = {}
        total_samples = 0

        # Train specialist for each regime with sufficient data
        for regime in ["premarket", "opening", "morning", "midday", "afternoon", "closing"]:
            regime_df = df[df["regime"] == regime].copy()
            
            if len(regime_df) < 15:  # Skip regimes with too little data
                logger.info(f"learning.regime.skip", regime=regime, samples=len(regime_df))
                continue

            logger.info(f"learning.regime.training", regime=regime, samples=len(regime_df))
            
            X = self._feature_matrix(regime_df)
            y = regime_df["hybrid_target"].fillna(0.0).astype(float).values
            sample_weights = self._sample_weights(regime_df)
            
            if len(np.unique(y)) < 2:
                continue

            # Smaller, more specialized model for regime
            model = XGBRegressor(
                n_estimators=80,  # Fewer trees for specialization
                max_depth=4,      # Slightly deeper for regime patterns
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=1,
            )

            # Train specialist
            if sample_weights is not None:
                model.fit(X, y, sample_weight=sample_weights)
            else:
                model.fit(X, y)
            regime_models[regime] = model
            
            # Calculate regime-specific metrics
            preds = model.predict(X)
            rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
            regime_metrics[regime] = {
                "samples": len(regime_df),
                "rmse": round(rmse, 4),
                "feature_importance": [
                    {"feature": feature, "importance": float(importance)}
                    for feature, importance in zip(FEATURE_COLS, model.feature_importances_)
                ],
            }
            total_samples += len(regime_df)

        if not regime_models:
            return {"status": "skipped", "reason": "no_regime_models_trained"}

        # Persist regime models
        regime_models_buf = io.BytesIO()
        joblib.dump(regime_models, regime_models_buf)
        set_bytes(_REGIME_MODELS_KEY, regime_models_buf.getvalue())

        return {
            "status": "regime_specialists_trained",
            "regimes_trained": list(regime_models.keys()),
            "total_samples": total_samples,
            "regime_metrics": regime_metrics,
            "model_type": "regime_specialists",
        }

    def _train_connor_sizer(self, df: pd.DataFrame) -> None:
        """Train a model to predict Connor's expected position size from setup features."""
        if "connor_size_pct" not in df:
            return
        size_df = df[df["connor_size_pct"] > 0].copy()
        if len(size_df) < 10:
            return
        try:
            from xgboost import XGBRegressor
        except Exception:
            return
        try:
            X = self._feature_matrix(size_df)
            y = size_df["connor_size_pct"].astype(float).values
            model = XGBRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=2,
            )
            model.fit(X, y)
            model_buf = io.BytesIO()
            joblib.dump(model, model_buf)
            set_bytes(_CONNOR_SIZER_KEY, model_buf.getvalue())
            logger.info("learning.connor_sizer.trained", rows=len(size_df))
        except Exception as exc:
            logger.warning("learning.connor_sizer.failed", err=str(exc))

    def load_connor_sizer(self):
        model_bytes = get_bytes(_CONNOR_SIZER_KEY)
        if model_bytes:
            return joblib.load(io.BytesIO(model_bytes))
        raise FileNotFoundError("Connor sizer model not found")

    def _train_logistic(self, trade_date: date, df: pd.DataFrame, fallback_reason: str | None = None) -> dict[str, Any]:
        if df.empty or df["label"].sum() < 5:
            logger.warning("learning.dataset.insufficient", rows=len(df))
            return {"status": "skipped", "reason": "insufficient_data", "rows": len(df), "fallback_reason": fallback_reason}

        X = self._feature_matrix(df)
        y = df["label"].values
        sample_weights = self._sample_weights(df)

        if len(np.unique(y)) < 2:
            logger.warning("learning.dataset.single_class", rows=len(df))
            return {"status": "skipped", "reason": "single_class", "rows": len(df), "fallback_reason": fallback_reason}

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        tscv = TimeSeriesSplit(n_splits=min(5, max(2, len(df) // 50)))
        preds = np.zeros_like(y, dtype=float)
        model = LogisticRegression(max_iter=200, solver="liblinear")
        for train_idx, test_idx in tscv.split(X_scaled):
            if len(np.unique(y[train_idx])) < 2:
                continue
            if sample_weights is not None:
                model.fit(X_scaled[train_idx], y[train_idx], sample_weight=sample_weights[train_idx])
            else:
                model.fit(X_scaled[train_idx], y[train_idx])
            preds[test_idx] = model.predict_proba(X_scaled[test_idx])[:, 1]

        if sample_weights is not None:
            model.fit(X_scaled, y, sample_weight=sample_weights)
        else:
            model.fit(X_scaled, y)
        # Persist model/scaler to DB so worker + API share artifacts across services.
        model_buf = io.BytesIO()
        scaler_buf = io.BytesIO()
        joblib.dump(model, model_buf)
        joblib.dump(scaler, scaler_buf)
        set_bytes(_MODEL_KEY, model_buf.getvalue())
        set_bytes(_SCALER_KEY, scaler_buf.getvalue())

        # Keep local copies for optional debugging in local dev.
        try:
            joblib.dump(model, self.artifacts.model_path)
            joblib.dump(scaler, self.artifacts.scaler_path)
        except Exception:
            pass

        df["p2r"] = preds

        threshold_report = self._grid_search(df)
        self._persist_threshold_report(threshold_report)

        report = self._build_training_report(
            trade_date=trade_date,
            df=df,
            threshold_report=threshold_report,
            model_type="logistic_regression",
            metrics=threshold_report["metrics"],
            feature_importance=[],
            fallback_used=bool(fallback_reason),
            fallback_reason=fallback_reason,
        )
        self._persist_report(report)
        self._train_connor_sizer(df)
        return {"status": "trained", **report}

    def _feature_importance(self, model) -> list[dict[str, float | str]]:
        values = getattr(model, "feature_importances_", None)
        if values is None:
            return []
        pairs = [
            {"feature": feature, "importance": round(float(importance), 6)}
            for feature, importance in zip(FEATURE_COLS, values)
            if float(importance) > 0
        ]
        pairs.sort(key=lambda item: item["importance"], reverse=True)
        return pairs

    def _persist_threshold_report(self, threshold_report: dict[str, Any]) -> None:
        set_json(_THRESHOLDS_KEY, threshold_report["thresholds"])
        set_json(_CANARY_KEY, threshold_report["thresholds_canary"])
        try:
            (self.artifacts.thresholds_path).write_text(json.dumps(threshold_report["thresholds"], indent=2))
            (self.artifacts.canary_path).write_text(json.dumps(threshold_report["thresholds_canary"], indent=2))
        except Exception:
            pass

    def _persist_report(self, report: dict[str, Any]) -> None:
        set_json(_REPORT_KEY, report)
        try:
            self.artifacts.report_path.write_text(json.dumps(report, indent=2, default=str))
        except Exception:
            pass

    def _build_training_report(
        self,
        trade_date: date,
        df: pd.DataFrame,
        threshold_report: dict[str, Any],
        model_type: str,
        metrics: dict[str, Any],
        feature_importance: list[dict[str, Any]],
        fallback_used: bool,
        fallback_reason: str | None,
    ) -> dict[str, Any]:
        return {
            "date": trade_date.isoformat(),
            "feature_version": FEATURE_VERSION,
            "rows": len(df),
            "positives": int(df["label"].sum()) if "label" in df else 0,
            "label_breakdown": self._label_breakdown(df),
            "source_breakdown": self._source_breakdown(df),
            "model_type": model_type,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "feature_importance": feature_importance,
            "precision@50": threshold_report.get("precision_at_k"),
            "thresholds": threshold_report["thresholds"],
            "thresholds_canary": threshold_report["thresholds_canary"],
            "metrics": metrics,
            "ranking_metrics": self._ranking_metrics(df),
        }

    def _source_breakdown(self, df: pd.DataFrame) -> dict[str, int]:
        if df.empty or "source" not in df:
            return {}
        sources = df["source"].fillna("unknown").astype(str)
        return {str(k): int(v) for k, v in sources.value_counts().to_dict().items()}

    def _ranking_metrics(self, df: pd.DataFrame) -> dict[str, Any]:
        if df.empty or "p2r" not in df:
            return {}
        frame = df.copy()
        frame["p2r"] = pd.to_numeric(frame["p2r"], errors="coerce")
        frame = frame.dropna(subset=["p2r"])
        if frame.empty:
            return {}

        taken_mask = pd.Series(False, index=frame.index)
        if "taken_by_master" in frame:
            taken_mask = taken_mask | frame["taken_by_master"].fillna(False).astype(bool)
        if "label_class" in frame:
            taken_mask = taken_mask | frame["label_class"].isin({LABEL_SUGGESTED_TAKEN, LABEL_MANUAL_NO_ALERT})
        frame["_taken_for_rank"] = taken_mask
        if not bool(frame["_taken_for_rank"].any()):
            return {"taken_rows": 0}

        if "detected_ts" in frame:
            ts = pd.to_datetime(frame["detected_ts"], errors="coerce", utc=True)
            frame["_rank_group"] = ts.dt.date.astype(str).replace("NaT", "unknown")
        else:
            frame["_rank_group"] = "all"

        ranks: list[int] = []
        groups = 0
        for _, group in frame.groupby("_rank_group", dropna=False):
            if group.empty:
                continue
            groups += 1
            ordered = group.sort_values("p2r", ascending=False)
            rank_by_index = {idx: rank for rank, idx in enumerate(ordered.index, start=1)}
            ranks.extend(rank_by_index[idx] for idx in group[group["_taken_for_rank"]].index)

        if not ranks:
            return {"taken_rows": 0, "evaluated_groups": groups}

        rank_values = np.array(ranks, dtype=float)
        return {
            "taken_rows": int(len(ranks)),
            "evaluated_groups": int(groups),
            "top_5_taken_hit_rate": round(float(np.mean(rank_values <= 5)), 4),
            "top_10_taken_hit_rate": round(float(np.mean(rank_values <= 10)), 4),
            "median_taken_rank": round(float(np.median(rank_values)), 2),
            "best_taken_rank": int(np.min(rank_values)),
        }

    def _refresh_logistic_compatibility_model(self, df: pd.DataFrame) -> None:
        try:
            if df.empty or df["label"].sum() < 5 or len(np.unique(df["label"].values)) < 2:
                return
            X = self._feature_matrix(df)
            y = df["label"].values
            sample_weights = self._sample_weights(df)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model = LogisticRegression(max_iter=200, solver="liblinear")
            if sample_weights is not None:
                model.fit(X_scaled, y, sample_weight=sample_weights)
            else:
                model.fit(X_scaled, y)
            model_buf = io.BytesIO()
            scaler_buf = io.BytesIO()
            joblib.dump(model, model_buf)
            joblib.dump(scaler, scaler_buf)
            set_bytes(_MODEL_KEY, model_buf.getvalue())
            set_bytes(_SCALER_KEY, scaler_buf.getvalue())
            joblib.dump(model, self.artifacts.model_path)
            joblib.dump(scaler, self.artifacts.scaler_path)
        except Exception as exc:
            logger.warning("learning.logistic_compat_failed", err=str(exc))

    def _grid_search(self, df: pd.DataFrame) -> dict[str, Any]:
        if df.empty:
            return {"thresholds": {}, "thresholds_canary": {}, "metrics": {}, "precision_at_k": None}

        combos = [(0.70, 1.3), (0.7, 1.5), (0.75, 1.5), (0.8, 1.5), (0.8, 1.8)]
        results: dict[str, dict[str, Any]] = {}
        metrics = {}

        for price_bucket in df["price_bucket"].unique():
            for time_bucket in df["time_bucket"].unique():
                mask = (df["price_bucket"] == price_bucket) & (df["time_bucket"] == time_bucket)
                if mask.sum() < 10:
                    continue
                best_combo = None
                best_precision = -1.0
                for l2_thresh, rvol_thresh in combos:
                    gated = df[mask & (df["l2_mean"].fillna(0) >= l2_thresh) & (df["rvol_break"].fillna(0) >= rvol_thresh)]
                    if gated.empty:
                        continue
                    k = max(1, int(0.1 * len(gated)))
                    topk = gated.nlargest(k, "p2r")
                    precision = topk["label"].mean()
                    if precision > best_precision:
                        best_precision = precision
                        best_combo = {"l2": l2_thresh, "rvol": rvol_thresh}
                if best_combo:
                    key = f"{price_bucket}|{time_bucket}"
                    results[key] = best_combo
                    metrics[key] = {"precision": round(best_precision, 3)}

        thresholds = {"version": datetime.utcnow().isoformat() + "Z", "buckets": results}
        thresholds["valid_from"] = (datetime.utcnow() + timedelta(days=1)).date().isoformat()

        canary_symbols = self._canary_symbols(df)
        thresholds_canary = {
            "version": thresholds["version"],
            "valid_from": thresholds["valid_from"],
            "symbols": canary_symbols,
            "buckets": results,
        }

        return {
            "thresholds": thresholds,
            "thresholds_canary": thresholds_canary,
            "metrics": metrics,
            "precision_at_k": {k: v["precision"] for k, v in metrics.items()},
        }

    def _canary_symbols(self, df: pd.DataFrame) -> List[str]:
        symbols = df["symbol"].dropna().unique()
        if len(symbols) <= 5:
            return symbols.tolist()
        rng = np.random.default_rng(seed=42)
        sample_size = max(1, int(0.2 * len(symbols)))
        idx = rng.choice(len(symbols), size=sample_size, replace=False)
        return symbols[idx].tolist()

    # ----------------- Runtime scoring ---------------------------------------------
    def load_xgb_model(self) -> Any:
        model_bytes = get_bytes(_XGB_MODEL_KEY)
        if model_bytes:
            return joblib.load(io.BytesIO(model_bytes))
        if self.artifacts.xgb_pickle_path.exists():
            return joblib.load(self.artifacts.xgb_pickle_path)
        raise FileNotFoundError("XGBoost model artifact not found")

    def load_model(self) -> Tuple[Any, Any]:
        model_bytes = get_bytes(_MODEL_KEY)
        scaler_bytes = get_bytes(_SCALER_KEY)
        if model_bytes and scaler_bytes:
            model = joblib.load(io.BytesIO(model_bytes))
            scaler = joblib.load(io.BytesIO(scaler_bytes))
            return model, scaler
        if self.artifacts.model_path.exists() and self.artifacts.scaler_path.exists():
            model = joblib.load(self.artifacts.model_path)
            scaler = joblib.load(self.artifacts.scaler_path)
            return model, scaler
        raise FileNotFoundError("Model artifacts not found")

    def load_thresholds(self) -> dict[str, Any]:
        stored = get_json(_THRESHOLDS_KEY)
        if stored:
            return stored
        if self.artifacts.thresholds_path.exists():
            return json.loads(self.artifacts.thresholds_path.read_text())
        return {"buckets": {}}

    def load_canary(self) -> dict[str, Any]:
        stored = get_json(_CANARY_KEY)
        if stored:
            return stored
        if self.artifacts.canary_path.exists():
            return json.loads(self.artifacts.canary_path.read_text())
        return {"symbols": [], "buckets": {}}

    def load_report(self) -> dict[str, Any] | None:
        stored = get_json(_REPORT_KEY)
        if stored:
            return stored
        if self.artifacts.report_path.exists():
            return json.loads(self.artifacts.report_path.read_text())
        return None

    def score(self, features: dict[str, float]) -> float | None:
        """Enhanced scoring with regime-aware specialists."""
        vector = self._feature_matrix(pd.DataFrame([features]))
        
        # Try regime specialists first (most accurate)
        try:
            regime_models = self.load_regime_models()
            if regime_models:
                # Detect current regime from features
                current_regime = self._detect_regime_from_features(features)
                
                if current_regime in regime_models:
                    specialist_model = regime_models[current_regime]
                    prediction = float(specialist_model.predict(vector)[0])
                    return max(0.0, min(1.0, prediction))
        except Exception as exc:
            logger.debug("learning.regime_scoring.failed", err=str(exc))
        
        # Fallback to main XGBoost model
        try:
            model = self.load_xgb_model()
            prediction = float(model.predict(vector)[0])
            return max(0.0, min(1.0, prediction))
        except Exception:
            pass

        # Final fallback to logistic regression
        try:
            model, scaler = self.load_model()
        except FileNotFoundError:
            return None
        vector = scaler.transform(vector)
        return float(model.predict_proba(vector)[0, 1])

    def load_regime_models(self) -> dict[str, Any] | None:
        """Load regime-specific specialist models."""
        try:
            model_bytes = get_bytes(_REGIME_MODELS_KEY)
            if model_bytes:
                return joblib.load(io.BytesIO(model_bytes))
        except Exception:
            pass
        return None

    def _detect_regime_from_features(self, features: dict[str, float]) -> str:
        """Detect trading regime from feature indicators."""
        for regime in ["premarket", "opening", "morning", "midday", "afternoon", "closing"]:
            if features.get(f"regime_{regime}", 0.0) > 0.5:
                return regime
        
        # Fallback to time-based detection
        from datetime import datetime
        return classify_trading_regime(datetime.now())


_LEARNING: LearningService | None = None


def get_learning_service() -> LearningService:
    global _LEARNING
    if _LEARNING is None:
        _LEARNING = LearningService()
    return _LEARNING


def predict_connor_size_pct(features: dict[str, float]) -> float | None:
    """Return the model's predicted position size for a setup, or None if no model exists yet."""
    try:
        svc = get_learning_service()
        model = svc.load_connor_sizer()
        vector = svc._feature_matrix(pd.DataFrame([features]))
        prediction = float(model.predict(vector)[0])
        max_size = float(os.getenv("AI_LAB_BUYING_POWER_MULTIPLIER", "4.00"))
        return min(max(0.0, prediction), max(0.0, max_size))
    except Exception:
        return None
