from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from sqlalchemy import select

from app.db.models import Alert, Setup
from app.db.session import get_session
from app.observability.logging import get_logger


logger = get_logger("learning")

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


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


@dataclass
class LearningArtifacts:
    model_path: Path
    scaler_path: Path
    thresholds_path: Path
    canary_path: Path
    report_path: Path


class LearningService:
    def __init__(self) -> None:
        self.lookback_days = int(os.getenv("LEARNING_LOOKBACK_DAYS", "30"))
        self.artifacts = LearningArtifacts(
            model_path=ARTIFACT_DIR / "alert_ranker.pkl",
            scaler_path=ARTIFACT_DIR / "alert_ranker_scaler.pkl",
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

    def _build_rows(self, trade_date: date) -> pd.DataFrame:
        end = datetime.combine(trade_date, datetime.min.time(), tzinfo=timezone.utc)
        start = end - timedelta(days=self.lookback_days)
        setups = self._load_setups(start)
        if not setups:
            return pd.DataFrame()
        alerts_map = self._alerts_for_setups([s.id for s in setups])

        rows: list[dict[str, Any]] = []
        for setup in setups:
            payload = setup.payload_json or {}
            features = payload.get("features", {})
            if not features:
                continue
            rr = payload.get("rr")
            price = payload.get("entry_price")
            detected_ts = setup.detected_ts
            trade_bucket = bucket_price(price)
            time_bucket = bucket_time(detected_ts.astimezone(timezone.utc) if detected_ts else None)
            alerts = alerts_map.get(setup.id, [])
            p2r = payload.get("p2r")

            realized_r = payload.get("realized_r")
            if realized_r is None and setup.rr_min is not None:
                realized_r = setup.rr_min

            label = 1 if realized_r is not None and realized_r >= 2.0 else 0
            rows.append(
                {
                    "setup_id": setup.id,
                    "detected_ts": detected_ts,
                    "symbol": payload.get("symbol"),
                    "direction": setup.direction,
                    "box_height": features.get("box_height"),
                    "box_bars": features.get("box_bars"),
                    "rvol_break": features.get("rvol_break"),
                    "l2_mean": features.get("l2_mean"),
                    "l2_persist": features.get("l2_persist"),
                    "dist_htf": features.get("dist_htf"),
                    "dist_gap": features.get("dist_gap"),
                    "spread_cents": features.get("spread_cents"),
                    "price_bucket": trade_bucket,
                    "time_bucket": time_bucket,
                    "p2r_prev": p2r,
                    "label": label,
                }
            )

        df = pd.DataFrame(rows)
        df.dropna(subset=["box_height", "box_bars", "rvol_break", "spread_cents"], inplace=True)
        if df.empty:
            return df
        df.sort_values("detected_ts", inplace=True)
        return df

    # ----------------- Training -----------------------------------------------------
    def train(self, trade_date: date) -> dict[str, Any]:
        df = self._build_rows(trade_date)
        if df.empty or df["label"].sum() < 5:
            logger.warning("learning.dataset.insufficient", rows=len(df))
            return {"status": "skipped", "reason": "insufficient_data", "rows": len(df)}

        feature_cols = [
            "box_height",
            "box_bars",
            "rvol_break",
            "l2_mean",
            "l2_persist",
            "dist_htf",
            "dist_gap",
            "spread_cents",
        ]
        X = df[feature_cols].fillna(0.0).values
        y = df["label"].values

        if len(np.unique(y)) < 2:
            logger.warning("learning.dataset.single_class", rows=len(df))
            return {"status": "skipped", "reason": "single_class", "rows": len(df)}

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        tscv = TimeSeriesSplit(n_splits=min(5, max(2, len(df) // 50)))
        preds = np.zeros_like(y, dtype=float)
        model = LogisticRegression(max_iter=200, solver="liblinear")
        for train_idx, test_idx in tscv.split(X_scaled):
            if len(np.unique(y[train_idx])) < 2:
                continue
            model.fit(X_scaled[train_idx], y[train_idx])
            preds[test_idx] = model.predict_proba(X_scaled[test_idx])[:, 1]

        model.fit(X_scaled, y)
        joblib.dump(model, self.artifacts.model_path)
        joblib.dump(scaler, self.artifacts.scaler_path)

        df["p2r"] = preds

        threshold_report = self._grid_search(df)
        (self.artifacts.thresholds_path).write_text(json.dumps(threshold_report["thresholds"], indent=2))
        (self.artifacts.canary_path).write_text(json.dumps(threshold_report["thresholds_canary"], indent=2))

        report = {
            "date": trade_date.isoformat(),
            "rows": len(df),
            "positives": int(df["label"].sum()),
            "precision@50": threshold_report.get("precision_at_k"),
            "thresholds": threshold_report["thresholds"],
            "thresholds_canary": threshold_report["thresholds_canary"],
            "metrics": threshold_report["metrics"],
        }
        self.artifacts.report_path.write_text(json.dumps(report, indent=2, default=str))
        return {"status": "trained", **report}

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
    def load_model(self) -> Tuple[Any, Any]:
        if not (self.artifacts.model_path.exists() and self.artifacts.scaler_path.exists()):
            raise FileNotFoundError("Model artifacts not found")
        model = joblib.load(self.artifacts.model_path)
        scaler = joblib.load(self.artifacts.scaler_path)
        return model, scaler

    def load_thresholds(self) -> dict[str, Any]:
        if not self.artifacts.thresholds_path.exists():
            return {"buckets": {}}
        return json.loads(self.artifacts.thresholds_path.read_text())

    def load_canary(self) -> dict[str, Any]:
        if not self.artifacts.canary_path.exists():
            return {"symbols": [], "buckets": {}}
        return json.loads(self.artifacts.canary_path.read_text())

    def score(self, features: dict[str, float]) -> float | None:
        try:
            model, scaler = self.load_model()
        except FileNotFoundError:
            return None
        feature_cols = [
            "box_height",
            "box_bars",
            "rvol_break",
            "l2_mean",
            "l2_persist",
            "dist_htf",
            "dist_gap",
            "spread_cents",
        ]
        vector = np.array([[features.get(col, 0.0) for col in feature_cols]])
        vector = scaler.transform(vector)
        return float(model.predict_proba(vector)[0, 1])


_LEARNING: LearningService | None = None


def get_learning_service() -> LearningService:
    global _LEARNING
    if _LEARNING is None:
        _LEARNING = LearningService()
    return _LEARNING
