import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from app.services.learning import (
    FEATURE_COLS,
    LABEL_MANUAL_NO_ALERT,
    LABEL_PAPER_OUTCOME,
    LABEL_SUGGESTED_IGNORED,
    LABEL_SUGGESTED_TAKEN,
    LearningRow,
    LearningService,
    _fills_connor_size_pct,
    _master_equity,
    calculate_hybrid_target,
    normalize_realized_r,
)
from app.services.trade_matcher import best_setup_match


@pytest.fixture(autouse=True)
def no_paper_trades_by_default(monkeypatch):
    monkeypatch.setattr(LearningService, "_load_closed_paper_trades", lambda self, start_ts, end_ts: [])


def test_normalize_realized_r_clips_and_scales():
    assert normalize_realized_r(None) == 0.0
    assert normalize_realized_r(-5.0) == 0.0
    assert normalize_realized_r(3.0) == 1.0
    assert normalize_realized_r(10.0) == 1.0
    assert normalize_realized_r(0.5) == pytest.approx(0.5)


def test_hybrid_target_rewards_taken_profitable_setup_more_than_ignored_alert():
    taken = calculate_hybrid_target(
        label=LABEL_SUGGESTED_TAKEN,
        taken_by_master=True,
        realized_r=2.0,
    )
    ignored = calculate_hybrid_target(
        label=LABEL_SUGGESTED_IGNORED,
        taken_by_master=False,
        realized_r=None,
    )

    assert taken > ignored
    assert ignored == 0.0


def test_hybrid_target_penalizes_bad_slippage_or_rejects():
    clean = calculate_hybrid_target(
        label=LABEL_SUGGESTED_TAKEN,
        taken_by_master=True,
        realized_r=1.0,
        reject_or_bad_slippage=False,
    )
    penalized = calculate_hybrid_target(
        label=LABEL_SUGGESTED_TAKEN,
        taken_by_master=True,
        realized_r=1.0,
        reject_or_bad_slippage=True,
    )

    assert penalized < clean


def test_manual_no_alert_trade_has_learning_signal_without_setup_id():
    row = LearningRow(
        label=LABEL_MANUAL_NO_ALERT,
        symbol="XYZ",
        detected_ts=datetime(2026, 5, 16, 14, 30, tzinfo=timezone.utc),
        features={"box_height": 0.06, "rvol_break": 2.1},
        taken_by_master=True,
        realized_r=1.5,
        manual_no_alert=True,
        setup_id=None,
        trade_id=123,
        source="master_manual",
    )

    payload = row.as_training_dict()

    assert payload["label_class"] == LABEL_MANUAL_NO_ALERT
    assert payload["setup_id"] is None
    assert payload["trade_id"] == 123
    assert payload["manual_no_alert"] is True
    assert payload["hybrid_target"] > 0.0
    assert payload["box_height"] == 0.06
    assert payload["rvol_break"] == 2.1


def test_learning_row_handles_missing_outcome_values():
    row = LearningRow(
        label=LABEL_SUGGESTED_TAKEN,
        symbol="ABC",
        detected_ts=None,
        features={},
        taken_by_master=True,
        realized_r=None,
        pnl=None,
    )

    payload = row.as_training_dict()

    assert payload["hybrid_target"] == 0.35
    assert payload["realized_r"] is None
    assert payload["pnl"] is None


def test_dataset_builder_labels_suggested_taken(monkeypatch):
    service = LearningService()
    detected_ts = datetime(2026, 5, 16, 14, 30, tzinfo=timezone.utc)
    setup = SimpleNamespace(
        id=1,
        payload_json={
            "symbol": "XYZ",
            "features": {
                "box_height": 0.08,
                "box_bars": 12,
                "rvol_break": 1.8,
                "l2_mean": 0.74,
                "l2_persist": 3.0,
                "dist_htf": 0.2,
                "dist_gap": 0.1,
                "spread_cents": 0.5,
                "realized_r": 2.4,
            },
        },
        direction="long",
        detected_ts=detected_ts,
        entry_price=4.25,
        rr_min=2.4,
        score=82,
    )
    alert = SimpleNamespace(id=10)
    fill = SimpleNamespace(id=20, setup_id=1, symbol="XYZ", ts=detected_ts, side="BUY", qty=100, price=4.25)

    monkeypatch.setattr(service, "_load_setups", lambda start_ts: [setup])
    monkeypatch.setattr(service, "_load_fills", lambda start_ts, end_ts: [fill])
    monkeypatch.setattr(service, "_load_trades", lambda start_ts, end_ts: [])
    monkeypatch.setattr(service, "_alerts_for_setups", lambda setup_ids: {1: [alert]})

    df = service._build_rows(detected_ts.date())

    assert len(df) == 1
    assert df.iloc[0]["label_class"] == LABEL_SUGGESTED_TAKEN
    assert bool(df.iloc[0]["taken_by_master"]) is True
    assert df.iloc[0]["label"] == 1
    assert df.iloc[0]["hybrid_target"] > 0.5


def test_dataset_builder_labels_manual_no_alert(monkeypatch):
    service = LearningService()
    fill_ts = datetime(2026, 5, 16, 15, 0, tzinfo=timezone.utc)
    fill = SimpleNamespace(id=30, setup_id=None, symbol="ABC", ts=fill_ts, side="BUY", qty=50, price=7.5)

    monkeypatch.setattr(service, "_load_setups", lambda start_ts: [])
    monkeypatch.setattr(service, "_load_fills", lambda start_ts, end_ts: [fill])
    monkeypatch.setattr(service, "_load_trades", lambda start_ts, end_ts: [])
    monkeypatch.setattr(service, "_alerts_for_setups", lambda setup_ids: {})

    df = service._build_rows(fill_ts.date())

    assert len(df) == 1
    assert df.iloc[0]["label_class"] == LABEL_MANUAL_NO_ALERT
    assert bool(df.iloc[0]["manual_no_alert"]) is True
    assert df.iloc[0]["setup_id"] is None
    assert service._label_breakdown(df) == {LABEL_MANUAL_NO_ALERT: 1}


def test_delayed_manual_fill_matches_scout_setup_for_learning(monkeypatch):
    service = LearningService()
    detected_ts = datetime(2026, 5, 16, 14, 30, tzinfo=timezone.utc)
    fill_ts = datetime(2026, 5, 16, 14, 35, tzinfo=timezone.utc)
    setup = SimpleNamespace(
        id=41,
        payload_json={
            "symbol": "MNY",
            "features": {
                "box_height": 0.08,
                "box_bars": 12,
                "rvol_break": 2.2,
                "l2_mean": 0.71,
                "l2_persist": 4.0,
                "dist_htf": 0.2,
                "dist_gap": 0.1,
                "spread_cents": 0.8,
            },
        },
        direction="long",
        detected_ts=detected_ts,
        entry_price=4.25,
        rr_min=2.2,
        score=86,
    )
    alert = SimpleNamespace(id=42)
    fill = SimpleNamespace(id=43, setup_id=None, symbol="MNY", ts=fill_ts, side="BUY", qty=100, price=4.30)

    monkeypatch.setattr(service, "_load_setups", lambda start_ts: [setup])
    monkeypatch.setattr(service, "_load_fills", lambda start_ts, end_ts: [fill])
    monkeypatch.setattr(service, "_load_trades", lambda start_ts, end_ts: [])
    monkeypatch.setattr(service, "_alerts_for_setups", lambda setup_ids: {41: [alert]})

    df = service._build_rows(detected_ts.date())

    assert len(df) == 1
    assert df.iloc[0]["label_class"] == LABEL_SUGGESTED_TAKEN
    assert bool(df.iloc[0]["taken_by_master"]) is True
    assert df.iloc[0]["setup_id"] == 41
    assert df.iloc[0]["fill_id"] == 43
    assert df.iloc[0]["setup_match_confidence"] in {"likely", "exact"}
    assert df.iloc[0]["setup_match_score"] > 0.0


def test_direction_mismatch_does_not_mark_setup_taken(monkeypatch):
    service = LearningService()
    detected_ts = datetime(2026, 5, 16, 14, 30, tzinfo=timezone.utc)
    fill_ts = datetime(2026, 5, 16, 14, 35, tzinfo=timezone.utc)
    setup = SimpleNamespace(
        id=51,
        payload_json={
            "symbol": "MNY",
            "features": {
                "box_height": 0.08,
                "box_bars": 12,
                "rvol_break": 2.2,
                "l2_mean": 0.71,
                "l2_persist": 4.0,
                "dist_htf": 0.2,
                "dist_gap": 0.1,
                "spread_cents": 0.8,
            },
        },
        direction="long",
        detected_ts=detected_ts,
        entry_price=4.25,
        rr_min=2.2,
        score=86,
    )
    fill = SimpleNamespace(id=53, setup_id=None, symbol="MNY", ts=fill_ts, side="SELL", qty=100, price=4.30)

    monkeypatch.setattr(service, "_load_setups", lambda start_ts: [setup])
    monkeypatch.setattr(service, "_load_fills", lambda start_ts, end_ts: [fill])
    monkeypatch.setattr(service, "_load_trades", lambda start_ts, end_ts: [])
    monkeypatch.setattr(service, "_alerts_for_setups", lambda setup_ids: {})

    df = service._build_rows(detected_ts.date())

    assert df["label_class"].tolist() == [LABEL_SUGGESTED_IGNORED, LABEL_MANUAL_NO_ALERT]
    assert bool(df.iloc[0]["taken_by_master"]) is False
    assert bool(df.iloc[1]["manual_no_alert"]) is True


def test_closed_paper_trades_enter_learning_as_low_weight_outcomes(monkeypatch):
    service = LearningService()
    detected_ts = datetime(2026, 5, 16, 14, 30, tzinfo=timezone.utc)
    closed_ts = datetime(2026, 5, 16, 14, 45, tzinfo=timezone.utc)
    paper_trade = SimpleNamespace(
        id=70,
        setup_id=71,
        alert_id=72,
        symbol="MNY",
        direction="long",
        opened_at=detected_ts,
        closed_at=closed_ts,
        entry_price=4.25,
        realized_r=2.3,
        realized_pnl=115.0,
    )
    shadow_decision = SimpleNamespace(
        payload_json={
            "features": {
                "box_height": 0.08,
                "box_bars": 12,
                "rvol_break": 2.2,
                "l2_mean": 0.71,
                "l2_persist": 4.0,
                "dist_htf": 0.2,
                "dist_gap": 0.1,
                "spread_cents": 0.8,
            }
        }
    )

    monkeypatch.setattr(service, "_load_setups", lambda start_ts: [])
    monkeypatch.setattr(service, "_load_fills", lambda start_ts, end_ts: [])
    monkeypatch.setattr(service, "_load_trades", lambda start_ts, end_ts: [])
    monkeypatch.setattr(service, "_load_closed_paper_trades", lambda start_ts, end_ts: [(paper_trade, shadow_decision)])
    monkeypatch.setattr(service, "_alerts_for_setups", lambda setup_ids: {})

    df = service._build_rows(closed_ts.date())

    assert len(df) == 1
    assert df.iloc[0]["label_class"] == LABEL_PAPER_OUTCOME
    assert df.iloc[0]["source"] == "paper_trader"
    assert df.iloc[0]["sample_weight"] == service.paper_sample_weight
    assert df.iloc[0]["realized_r"] == 2.3
    assert df.iloc[0]["label"] == 1


def test_best_setup_match_scores_time_direction_and_price():
    detected_ts = datetime(2026, 5, 16, 14, 30, tzinfo=timezone.utc)
    setup = SimpleNamespace(
        id=61,
        payload_json={"symbol": "MNY"},
        direction="long",
        detected_ts=detected_ts,
        entry_price=4.25,
    )

    match = best_setup_match(
        symbol="MNY",
        side="BUY",
        ts=datetime(2026, 5, 16, 14, 34, tzinfo=timezone.utc),
        price=4.27,
        setups=[setup],
        window_seconds=1800,
    )
    mismatch = best_setup_match(
        symbol="MNY",
        side="SELL",
        ts=datetime(2026, 5, 16, 14, 34, tzinfo=timezone.utc),
        price=4.27,
        setups=[setup],
        window_seconds=1800,
    )

    assert match.matched is True
    assert match.confidence in {"likely", "exact"}
    assert mismatch.matched is False
    assert mismatch.reason["reject"] == "direction_mismatch"


def _training_df(rows: int = 12):
    data = []
    for idx in range(rows):
        row = {col: float((idx % 4) + 1) for col in FEATURE_COLS}
        row.update(
            {
                "detected_ts": datetime(2026, 5, 16, 13, idx % 60, tzinfo=timezone.utc),
                "symbol": f"T{idx}",
                "price_bucket": "0.5-5",
                "time_bucket": "open",
                "label": 1 if idx % 2 == 0 else 0,
                "label_class": LABEL_SUGGESTED_TAKEN if idx % 2 == 0 else LABEL_SUGGESTED_IGNORED,
                "hybrid_target": 0.8 if idx % 2 == 0 else 0.1,
            }
        )
        data.append(row)
    import pandas as pd

    return pd.DataFrame(data)


def test_train_falls_back_to_logistic_when_xgboost_is_not_usable(monkeypatch):
    service = LearningService()
    service.xgb_min_rows = 999
    monkeypatch.setattr(service, "_build_rows", lambda trade_date: _training_df())

    result = service.train(datetime(2026, 5, 16, tzinfo=timezone.utc).date())

    assert result["status"] == "trained"
    assert result["model_type"] == "logistic_regression"
    assert result["fallback_used"] is True
    assert "insufficient_xgboost_rows" in result["fallback_reason"]
    assert result["label_breakdown"][LABEL_SUGGESTED_TAKEN] == 6


def test_train_uses_xgboost_when_enough_data_exists(monkeypatch):
    pytest.importorskip("xgboost")
    service = LearningService()
    service.xgb_min_rows = 8
    monkeypatch.setattr(service, "_build_rows", lambda trade_date: _training_df(rows=24))

    result = service.train(datetime(2026, 5, 16, tzinfo=timezone.utc).date())

    assert result["status"] == "trained"
    assert result["model_type"] == "xgboost"
    assert result["fallback_used"] is False
    assert "feature_importance" in result
    assert "rmse" in result["metrics"]
    assert "source_breakdown" in result
    assert "ranking_metrics" in result


def test_feature_matrix_encodes_bucket_strings_as_numeric():
    service = LearningService()
    matrix = service._feature_matrix(_training_df(rows=2))

    assert matrix.dtype.kind == "f"
    assert matrix.shape == (2, len(FEATURE_COLS))


def test_train_with_sandbox_defaults_to_regular_training(monkeypatch):
    monkeypatch.delenv("LEARNING_SANDBOX_ENABLED", raising=False)
    service = LearningService()
    monkeypatch.setattr(service, "train", lambda trade_date: {"status": "trained", "regular_training": True})

    result = asyncio.run(service.train_with_sandbox(datetime(2026, 5, 16, tzinfo=timezone.utc).date()))

    assert result == {"status": "trained", "regular_training": True}


def test_train_with_sandbox_caps_and_downweights_synthetic_samples(monkeypatch):
    monkeypatch.setenv("LEARNING_SANDBOX_ENABLED", "1")
    monkeypatch.setenv("LEARNING_SYNTHETIC_MAX_RATIO", "1.0")
    monkeypatch.setenv("LEARNING_SYNTHETIC_SAMPLE_WEIGHT", "0.2")
    service = LearningService()
    base_df = _training_df(rows=4)
    base_df["source"] = "scout"
    base_df["taken_by_master"] = base_df["label"] == 1
    monkeypatch.setattr(service, "_build_rows", lambda trade_date: base_df)

    async def fake_enhance_learning_with_sandbox(trade_date, real_samples):
        synthetic = []
        for idx in range(10):
            synthetic.append(
                LearningRow(
                    label=LABEL_SUGGESTED_TAKEN,
                    symbol=f"S{idx}",
                    detected_ts=datetime(2026, 5, 16, 14, idx, tzinfo=timezone.utc),
                    features={col: 1.0 for col in FEATURE_COLS},
                    taken_by_master=True,
                    realized_r=2.5,
                    source="sandbox_simulation",
                )
            )
        return real_samples + synthetic

    import app.learning.sandbox_simulator as sandbox_simulator

    monkeypatch.setattr(sandbox_simulator, "enhance_learning_with_sandbox", fake_enhance_learning_with_sandbox)
    captured = {}

    def fake_train_xgboost(trade_date, df):
        captured["df"] = df.copy()
        return {
            "status": "trained",
            "rows": len(df),
            "source_breakdown": service._source_breakdown(df),
        }

    monkeypatch.setattr(service, "_train_xgboost", fake_train_xgboost)

    result = asyncio.run(service.train_with_sandbox(datetime(2026, 5, 16, tzinfo=timezone.utc).date()))

    trained_df = captured["df"]
    assert result["sandbox_enhanced"]["synthetic_samples"] == 4
    assert result["sandbox_enhanced"]["synthetic_sample_weight"] == 0.2
    assert result["sandbox_enhanced"]["enhancement_ratio"] == 2.0
    assert service._source_breakdown(trained_df) == {"scout": 4, "sandbox_simulation": 4}
    assert trained_df.loc[trained_df["source"] == "scout", "sample_weight"].unique().tolist() == [1.0]
    assert trained_df.loc[trained_df["source"] == "sandbox_simulation", "sample_weight"].unique().tolist() == [0.2]


def test_hybrid_weights_can_be_configured_from_environment(monkeypatch):
    monkeypatch.setenv("LEARNING_BEHAVIOR_WEIGHT", "0.20")
    monkeypatch.setenv("LEARNING_OUTCOME_WEIGHT", "0.60")
    monkeypatch.setenv("LEARNING_MANUAL_EDGE_WEIGHT", "0.10")
    monkeypatch.setenv("LEARNING_PENALTY_WEIGHT", "0.05")

    service = LearningService()

    assert service.hybrid_weights.behavior_weight == 0.20
    assert service.hybrid_weights.outcome_weight == 0.60
    assert service.hybrid_weights.manual_edge_weight == 0.10
    assert service.hybrid_weights.penalty_weight == 0.05


# ---- connor_size_pct feature tests ----------------------------------------

def test_master_equity_reads_from_env(monkeypatch):
    monkeypatch.setenv("WEBULL_MASTER_ACCOUNT_EQUITY", "75000")
    assert _master_equity() == 75000.0


def test_master_equity_returns_none_when_unset(monkeypatch):
    monkeypatch.delenv("WEBULL_MASTER_ACCOUNT_EQUITY", raising=False)
    assert _master_equity() is None


def test_master_equity_returns_none_for_zero(monkeypatch):
    monkeypatch.setenv("WEBULL_MASTER_ACCOUNT_EQUITY", "0")
    assert _master_equity() is None


def test_fills_connor_size_pct_computes_buy_notional_over_equity():
    fill = SimpleNamespace(side="BUY", qty=200, price=5.0)
    result = _fills_connor_size_pct([fill], equity=50000.0)
    # 200 * 5.0 = 1000; 1000 / 50000 = 0.02
    assert result == pytest.approx(0.02, rel=1e-5)


def test_fills_connor_size_pct_ignores_sell_fills():
    buy_fill = SimpleNamespace(side="BUY", qty=100, price=10.0)
    sell_fill = SimpleNamespace(side="SELL", qty=100, price=11.0)
    result = _fills_connor_size_pct([buy_fill, sell_fill], equity=50000.0)
    # only buy: 100 * 10 = 1000; 1000 / 50000 = 0.02
    assert result == pytest.approx(0.02, rel=1e-5)


def test_fills_connor_size_pct_returns_zero_when_no_equity():
    fill = SimpleNamespace(side="BUY", qty=100, price=10.0)
    assert _fills_connor_size_pct([fill], equity=None) == 0.0


def test_fills_connor_size_pct_returns_zero_when_no_fills():
    assert _fills_connor_size_pct([], equity=50000.0) == 0.0


def test_manual_trade_features_includes_connor_size_pct(monkeypatch):
    monkeypatch.setenv("WEBULL_MASTER_ACCOUNT_EQUITY", "50000")
    service = LearningService()
    fill = SimpleNamespace(qty=200, price=5.0, side="BUY")
    feats = service._manual_trade_features(fill=fill)
    assert feats["connor_size_pct"] == pytest.approx(0.02, rel=1e-5)


def test_manual_trade_features_connor_size_pct_zero_without_equity(monkeypatch):
    monkeypatch.delenv("WEBULL_MASTER_ACCOUNT_EQUITY", raising=False)
    service = LearningService()
    fill = SimpleNamespace(qty=200, price=5.0, side="BUY")
    feats = service._manual_trade_features(fill=fill)
    assert feats["connor_size_pct"] == 0.0


def test_paper_trade_features_connor_size_pct_is_zero():
    service = LearningService()
    paper_trade = SimpleNamespace(
        entry_price=4.0, direction="long", realized_r=1.5
    )
    feats = service._paper_trade_features(paper_trade, shadow_decision=None)
    assert feats["connor_size_pct"] == 0.0


def test_connor_size_pct_in_feature_cols():
    assert "connor_size_pct" in FEATURE_COLS


def test_setup_row_includes_connor_size_pct_from_fills(monkeypatch):
    monkeypatch.setenv("WEBULL_MASTER_ACCOUNT_EQUITY", "50000")
    service = LearningService()
    detected_ts = datetime(2026, 5, 24, 10, 0, tzinfo=timezone.utc)
    setup = SimpleNamespace(
        id=99,
        payload_json={
            "symbol": "TST",
            "features": {
                "box_height": 0.05,
                "box_bars": 10,
                "rvol_break": 2.0,
                "l2_mean": 0.7,
                "l2_persist": 3.0,
                "dist_htf": 0.1,
                "dist_gap": 0.05,
                "spread_cents": 0.3,
            },
        },
        direction="long",
        detected_ts=detected_ts,
        entry_price=5.0,
        rr_min=2.0,
        score=80,
    )
    fill = SimpleNamespace(id=99, setup_id=99, symbol="TST", ts=detected_ts, side="BUY", qty=100, price=5.0)

    monkeypatch.setattr(service, "_load_setups", lambda start_ts: [setup])
    monkeypatch.setattr(service, "_load_fills", lambda start_ts, end_ts: [fill])
    monkeypatch.setattr(service, "_load_trades", lambda start_ts, end_ts: [])
    monkeypatch.setattr(service, "_alerts_for_setups", lambda setup_ids: {})

    df = service._build_rows(detected_ts.date())

    assert len(df) == 1
    # 100 * 5.0 = 500; 500 / 50000 = 0.01
    assert df.iloc[0]["connor_size_pct"] == pytest.approx(0.01, rel=1e-5)
