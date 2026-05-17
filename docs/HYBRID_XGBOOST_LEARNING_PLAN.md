# Hybrid XGBoost Learning Upgrade Plan

## Goal

Upgrade MooneyCapitol's nightly learning system from a logistic-regression-first setup ranker into a hybrid XGBoost-first learning engine that learns from:

1. Scout alerts the trader took or ignored.
2. Trades the trader actually took and their realized outcomes.
3. Manual trades the trader took without a prior scout alert.

The first production version must remain explainable, auditable, and backward-compatible with the current logistic regression fallback.

## Core Principle

V1 learning should improve alert quality and ranking. It should not auto-trade, auto-size, or override the trader.

The model output should be treated as:

- alert score adjustment,
- setup rank,
- learning report insight,
- feature-importance feedback,
- future input to copy-ratio suggestions.

PPO/RL is a later v2 layer after enough labeled data exists.

## Data Sources

Use Postgres as the source of truth.

Required v1 tables:

- `setups`
- `alerts`
- `fills`
- `trades`

Future copier tables:

- `master_executions`
- `copy_orders`
- `copy_order_events`
- `copy_reconciliations`

## Labels

The dataset builder should produce rows with one of these event classes:

- `suggested_taken`: scout generated an alert/setup and the master trader traded it.
- `suggested_ignored`: scout generated an alert/setup and the master trader did not trade it.
- `manual_no_alert`: master trader traded a symbol without a matching scout alert/setup.
- `copied_outcome`: copied account outcome/slippage once Webull copier is live.

## Hybrid Target

The first target should be deterministic and explainable:

```text
hybrid_score =
  behavior_weight * taken_signal
  + outcome_weight * realized_r_clipped
  + manual_edge_weight * manual_trade_signal
  - penalty_weight * reject_or_bad_slippage_signal
```

Initial recommended weights:

- `behavior_weight = 0.35`
- `outcome_weight = 0.50`
- `manual_edge_weight = 0.15`
- `penalty_weight = 0.10`

Initial signals:

- `taken_signal`: 1 if master traded a scout setup, 0 if ignored.
- `realized_r_clipped`: realized R clipped to `[-2.0, 3.0]` and normalized.
- `manual_trade_signal`: 1 for manual trades without prior alert, 0 otherwise.
- `reject_or_bad_slippage_signal`: reserved for copier outcomes, default 0 until copier is live.

This target can be revised after the first stable set of nightly reports.

## Model Strategy

Primary model:

- XGBoost regressor or classifier, selected after dataset shape is known.
- Prefer regressor for `hybrid_score`.
- Keep classification metrics for secondary reporting:
  - taken vs ignored,
  - positive realized R,
  - 2R+ outcome.

Fallback model:

- Existing logistic regression path.
- Must remain available if:
  - `xgboost` is not installed,
  - training data is too small,
  - only one class/label exists,
  - XGBoost training fails,
  - XGBoost validation performs materially worse than fallback.

Artifacts:

- `artifacts/alert_ranker_xgb.json`
- `artifacts/alert_ranker_xgb.pkl` if needed for sklearn wrapper compatibility.
- `artifacts/alert_ranker.pkl` remains compatible with current fallback.
- `artifacts/learning_report.json`
- `artifacts/feature_importance.json`
- DB/shared-state copies where required for API/worker coordination.

## Feature Set

Start with features already present or derivable:

- `box_height`
- `box_bars`
- `rvol_break`
- `l2_mean`
- `l2_persist`
- `dist_htf`
- `dist_gap`
- `spread_cents`
- `price`
- `price_bucket`
- `time_bucket`
- `direction_long`
- `alert_type`
- `alert_state`
- `score`
- `rr_min`
- `session_phase`
- `gap_pct` if available from watchlist/context
- `premkt_volume` if available from watchlist/context

Derived outcome features for labels/reporting, not model leakage:

- `taken_by_master`
- `realized_r`
- `pnl`
- `hold_time_seconds`
- `slippage_bps`
- `manual_no_alert`
- `copied_fill_delta_bps`

Avoid label leakage: do not train on realized outcome values as input features for the same row.

## Implementation Phases

### Phase 1: Learning Data Contract

Deliverables:

- Define internal `LearningRow` schema.
- Define label classes.
- Define target calculation function.
- Add tests for hybrid target calculation.

Files likely touched:

- `app/services/learning.py`
- `tests/test_learning_hybrid.py`

Acceptance criteria:

- Target calculation is deterministic.
- Ignored alerts produce lower target values than taken profitable setups.
- Manual no-alert trades are represented without requiring a setup ID.
- Missing PnL/R values do not crash dataset generation.

### Phase 2: Postgres Dataset Builder

Deliverables:

- Query `setups`, `alerts`, `fills`, and `trades`.
- Join alerts to setups.
- Link fills/trades to setups where possible.
- Detect manual trades without a matching setup/alert within a configurable window.
- Produce a pandas DataFrame with features, label metadata, and target.

Acceptance criteria:

- Builder handles empty DB.
- Builder handles setups with no alerts.
- Builder handles alerts with no fills.
- Builder handles fills/trades with no setup ID.
- Dataset includes label breakdown counts.

### Phase 3: XGBoost Training Path

Deliverables:

- Add `xgboost` dependency.
- Train XGBoost when enough valid data exists.
- Use time-aware validation split.
- Save best XGBoost model artifact to disk.
- Save feature importance artifact.
- Include `model_type = "xgboost"` in report.

Acceptance criteria:

- XGBoost trains on a valid synthetic test dataset.
- Feature importance is non-empty after training.
- Model artifact can be reloaded.
- Logistic fallback still works without XGBoost.

### Phase 4: Slack Feature-Importance Report

Deliverables:

- Nightly Slack message includes:
  - model type,
  - row count,
  - label breakdown,
  - validation metric,
  - top 10 features,
  - fallback reason if used.

Acceptance criteria:

- Slack adapter logs message in demo mode.
- Message does not exceed Slack practical readability.
- Failures in Slack reporting do not fail model training.

### Phase 5: Runtime Scoring Compatibility

Deliverables:

- Worker can load XGBoost model.
- Runtime scoring returns comparable `p2r`/quality score.
- Logistic fallback remains supported.
- Existing threshold/canary logic remains compatible.

Acceptance criteria:

- Existing scanner code can call scoring without knowing model type.
- Missing XGBoost artifact falls back cleanly.
- Existing tests still pass.

### Phase 6: `/learning/report` Upgrade

Deliverables:

- Include:
  - `model_type`,
  - `fallback_used`,
  - `fallback_reason`,
  - `feature_importance`,
  - `label_breakdown`,
  - `source_breakdown`,
  - `feature_version`,
  - `ranking_metrics`,
  - `dataset_window`,
  - `metrics`.

Acceptance criteria:

- Existing `/learning/report` clients still receive old fields.
- New fields are additive.
- Report loads from DB/shared state or disk.
- Ranking metrics show whether Connor's actual trades are appearing near the model's top-ranked suggestions.

### Phase 6B: Sandbox Guardrails

Deliverables:

- Keep sandbox-generated learning disabled by default with `LEARNING_SANDBOX_ENABLED=0`.
- When enabled, cap synthetic rows with `LEARNING_SYNTHETIC_MAX_RATIO`.
- Downweight synthetic rows with `LEARNING_SYNTHETIC_SAMPLE_WEIGHT`.
- Train XGBoost, regime specialists, and logistic fallback with those sample weights.

Acceptance criteria:

- Real master/scout data remains the primary training signal.
- Synthetic rows cannot silently outnumber real rows beyond the configured cap.
- Training reports show real-vs-synthetic source breakdown.

### Phase 7: Backfill/Replay Tooling

Deliverables:

- Add local command to build learning dataset and optionally train manually.
- Useful command shape:

```text
python -m app.tools.learning_train --date YYYY-MM-DD --dry-run
python -m app.tools.learning_train --date YYYY-MM-DD --train
```

Acceptance criteria:

- Dry run reports row count and label breakdown.
- Train mode writes artifacts.
- Tool works without scheduler.

### Phase 8: Tests And Guardrails

Deliverables:

- Unit tests for:
  - target calculation,
  - manual trade detection,
  - fallback path,
  - report shape,
  - feature importance formatting.

Acceptance criteria:

- Tests pass without live broker credentials.
- Tests pass without Postgres by using isolated/mocked data where practical.

### Phase 9: Calibration

Deliverables:

- Add config for:
  - behavior/outcome/manual weights,
  - manual trade matching window,
  - minimum training rows,
  - fallback threshold.
- Add report warnings for insufficient data or weak validation.

Acceptance criteria:

- Weights can be tuned without code changes.
- Bad/insufficient data triggers fallback or warning.

### Phase 10: Alert Ranking Integration

Deliverables:

- Use XGBoost score to rank/annotate alerts.
- Preserve existing alert gates.
- Do not let model score bypass hard risk controls.

Acceptance criteria:

- Alert payload can include model score.
- Dashboard/Slack can show p2r/model score.
- Low score can de-prioritize but not silently hide critical events until validated.

### Phase 11: V2 PPO Preparation

Deliverables:

- Document offline RL state/action/reward design.
- Store enough nightly data for future offline RL experiments.
- Add no live PPO execution path in v1.

Planned v2 PPO design:

- State: setup features, market context, trader behavior, copied account slippage.
- Action: boost/de-boost alert score or suggest copy ratio.
- Reward: realized R from master plus copied account outcome, penalized for slippage/rejects.
- Constraint: PPO only suggests; it does not auto-trade or override risk controls.

Acceptance criteria:

- V2 PPO is documented but not wired into live trading.
- XGBoost remains the active production learning layer.

## Completion Criteria

By the end of this phase sequence, the project should have:

- XGBoost-first nightly learning.
- Logistic regression fallback.
- Hybrid labels for taken/ignored/manual trades.
- Feature importance logged to Slack nightly.
- Best model saved to disk.
- `/learning/report` upgraded.
- Runtime scanner able to use the new score.
- V2 PPO design documented but not live.

## Explicit Non-Goals For This Learning Push

- No live PPO.
- No autonomous trading decisions.
- No model-driven order sizing in live accounts.
- No bypassing hard risk controls.
- No removal of logistic regression fallback.
- No dependency on Webull credentials for learning work.
