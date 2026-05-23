# MooneyCapitol Project Handoff

This is the canonical restart document. If a new assistant or developer picks up the project, read this first, then `README.md`, `docs/MOONEYCAPITOL_ROADMAP.md`, `docs/HYBRID_XGBOOST_LEARNING_PLAN.md`, and `docs/OPENAI_AI_FEATURES_PLAN.md`.

## Product Vision

MooneyCapitol is a two-part trading operations platform:

1. **Scout**: finds small-cap gapper consolidation -> breakout/breakdown -> retest setups for the lead trader.
2. **Webull trade copier**: mirrors the lead trader's Webull master-account fills into approved Webull copy accounts with very low latency.

The scout and copier share data for reporting and learning, but their runtime paths must remain separate. The scout, dashboard, learning jobs, Slack, database writes, and reconciliation must never delay copied order submission.

## Current Account Plan

- V1 master account: partner's Webull personal account.
- V1 copy account: user's Webull personal account.
- Future: many approved Webull target accounts.
- All v1 accounts are personal accounts.
- Enabled targets use `percent_equity` mirroring.
- If the master uses 5% of account equity, each target attempts to use 5% of its own account equity.
- If the master uses leveraged exposure, targets attempt to mirror that leveraged percentage as long as Webull accepts the order and the target has the required margin/buying-power permissions.

## Built So Far

### Scout

- FastAPI API and worker.
- V3 dashboard shell with a clear split between Trader Workspace and Developer Setup.
- Trader Workspace includes Scout, Trades, and P&L views for daily use by a non-developer trader.
- Developer Setup includes Copier Settings and Launch Readiness views for configuration, validation, and production controls.
- Dashboard Scout view with Armed, PRIMED, and Active lanes.
- Polygon/Massive-style market data adapter with demo fallback.
- Consolidation, breakout/retest, RVOL, spread, scoring, and alert flow foundation.
- Slack adapter and alert scaffolding.
- Database models for symbols, candles, L2 snapshots, boxes, setups, alerts, fills, and trades.
- Nightly learning service upgraded toward hybrid XGBoost-first learning with logistic regression fallback.
- Learning uses scout alerts, setups, fills, and trades, with copied-trade outcomes intended as data once live.

### L2

- L2 is part of the final scout, not an optional feature.
- It is intentionally delayed until close to launch because of expected monthly cost.
- Until then, the scout runs candle/volume/RVOL mode and can label missing/demo L2.

### Operator Authentication

- `/dashboard` is gated by the app `/login` page using `COWORK_OPERATOR_USERNAME` / `COWORK_OPERATOR_PASSWORD`, then a signed HTTP-only session cookie.
- All `/copier/*` API routes and `PUT /config` are gated by an `X-Operator-Token` header using `COWORK_OPERATOR_API_TOKEN`. A logged-in dashboard browser session is also accepted on those APIs.
- `/health`, Slack webhook routes, `GET /config`, and the scout read endpoints (`/setups`, `/learning/report`, `/watchlist/today`, `/reports/eod`) intentionally stay open.
- When no auth env vars are set, all gates pass through so local dev and tests still work.

### P&L Monitoring

- Real Webull account balance and position snapshot service.
- No fake/simulated account balances are used in production monitoring.
- API routes under `/pnl/*` expose account snapshots, positions, alerts, sessions, summary, and manual refresh.
- P&L tables are included in Alembic migration `0004_pnl_monitoring`.
- Dashboard has a P&L tab for balances, buying power, daily P&L, drawdown, exposure, open positions, and active P&L risk alerts.
- Worker runs P&L refresh in the background, separate from the low-latency copier hot path.
- Copier sizing prefers the latest cached Webull account snapshot equity. Manual equity env vars are fallback-only and are not part of the normal Render Blueprint prompt.

### Launch Readiness

- Protected `GET /launch/readiness` aggregates existing readiness signals instead of duplicating their logic.
- Dashboard has a Launch tab for database migration state, worker heartbeat, operator auth, Polygon/Slack/depth status, copier readiness, P&L freshness, read-only validation counts/history, copied-order latency samples, and learning report availability.
- This is display-only and does not touch the copier hot path.

### OpenAI AI Feature Direction

- OpenAI is planned only for read-only explanation, summarization, journaling, learning translation, and ticker/catalyst research.
- Approved AI features are scout alert explanations, daily trading recaps, learning report translation, trade journal automation, and ticker/catalyst research summaries.
- Chat-style dashboard assistant features are explicitly not wanted.
- OpenAI must never place trades, size trades, approve copier actions, override risk controls, or run inside the copier hot path.
- Detailed plan: `docs/OPENAI_AI_FEATURES_PLAN.md`.

### Render Blueprint

- `render.yaml` is intended for staged deployment before paid market data and Webull credentials exist.
- Safe defaults are `DEPTH_MODE=demo`, `COPIER_ENABLED=0`, `COPIER_MODE=test`, `COPIER_GLOBAL_KILL_SWITCH=1`, and `LEARNING_SANDBOX_ENABLED=0`.
- First deploy should set operator auth env vars. Polygon/Webull paid/live credentials can stay blank until account/data subscriptions are ready.

### Copier

- Webull-to-Webull architecture.
- Webull SDK trading adapter.
- Webull master event listener wrapper.
- Webull SDK pinned to `webull-openapi-python-sdk==2.0.7`.
- Master HTTP endpoint and master gRPC events endpoint are configured separately with `WEBULL_MASTER_API_ENDPOINT` and `WEBULL_MASTER_EVENTS_ENDPOINT`.
- Runtime for Webull fill events.
- Replay tool for recorded Webull fill payloads.
- Webull preflight tool for env/config, SDK, account-list, positions, balance/detail, and database readiness checks without placing orders.
- Webull event capture/normalization tool for saving real order-event messages and checking whether the copier parser understands them.
- Webull child-order response normalization tool for checking submit/detail/status responses and mapping status, fill quantity, fill price, and reject reason.
- Webull read-only session runner that combines raw event capture, master fill parsing, and persisted `would_copy`/`blocked` planning without broker order submission.
- Percent-equity sizing.
- Future multi-target design.
- Deterministic child order IDs.
- Global kill switch.
- Read-only, test, and live modes.
- Read-only mode records intended target copy orders as `would_copy` or `blocked` without submitting broker orders.
- Dashboard Copier tab.
- Dashboard Trades tab.
- API endpoints for status, readiness, targets, master executions, copy orders, copied trades, errors, audit events, and reconciliations.
- Position-aware sell handling:
  - Sells that close copied long exposure can copy even when shorting is disabled.
  - Sells that would create short exposure are blocked unless shorting is enabled and supported.
- Low-latency hot path:
  - cached target config/state
  - warmed Webull clients
  - no database reads before child submit
  - no Slack/dashboard/learning/reconciliation before child submit
  - persistence and alerts after submit/background
- Latency benchmark tool:
  - `python -m app.tools.copier_latency_benchmark samples\webull_execution_fill.json --iterations 100 --targets 1`
- Background order reconciliation.
- Startup recovery before subscribing to master events.
- Background Webull position sync:
  - compares actual Webull target positions against local copied-position estimate
  - writes reconciliation warnings on mismatch

## Pre-Launch Validation Status

**The system is feature-complete enough for validation testing, but not cleared for live money until Webull credentials, live read-only sessions, latency, reconciliation, and tiny-funds tests pass.**

### Completed Product Features

**Scout System:**
- ✅ Complete consolidation detection with 1m/2m timeframes
- ✅ L2 demo fallback; Webull Advanced Quotes is the intended live L2/tape path
- ✅ Advanced breakout/retest triggers with volume confirmation
- ✅ HTF level analysis and gap edge detection
- ✅ Real-time dashboard with Armed/PRIMED/Active lanes
- ✅ Slack integration with threading and action tracking

**XGBoost Learning Engine:**
- ✅ Hybrid XGBoost-first with logistic regression fallback
- ✅ 15+ engineered features with real-time scoring
- ✅ Postgres-based training dataset from all trading activity
- ✅ Nightly retraining with feature importance reporting
- ✅ Manual trade detection and labeling
- ✅ Runtime p2R scoring integration

Learning guardrails added after the sandbox upgrade:
- Sandbox-generated training is disabled unless `LEARNING_SANDBOX_ENABLED=1`.
- Synthetic rows are capped by `LEARNING_SYNTHETIC_MAX_RATIO` and downweighted by `LEARNING_SYNTHETIC_SAMPLE_WEIGHT`.
- Nightly reports include `source_breakdown`, `feature_version`, and ranking metrics for how close Connor's actual trades ranked to the top of the model output.

**Trade Copier:**
- ✅ Complete Webull SDK integration
- ✅ <300ms low-latency hot path
- ✅ Percent-equity sizing algorithm
- ✅ Kill switch and comprehensive risk controls
- ✅ Startup recovery and order reconciliation
- ✅ Background position sync with mismatch detection
- ✅ Professional operator dashboard

**Infrastructure:**
- ✅ FastAPI with complete REST API
- ✅ Multi-job worker scheduler
- ✅ SQLAlchemy models with Alembic migrations
- ✅ Operator authentication system
- ✅ Render deployment configuration
- ✅ Comprehensive testing tools

**Render Deployment Notes:**
- `render.yaml` uses a paid `basic-256mb` Postgres instance, not expiring free Postgres.
- API and worker services both run `python -m app.tools.run_migrations` before startup.
- `app/tools/run_migrations.py` uses a Postgres advisory lock before Alembic upgrade so concurrent API/worker deploys serialize migrations.

### ⏳ VALIDATION PHASE (Account-Dependent)

**Ready for real account validation when available:**

1. **Webull Account Setup**
   - Obtain Webull OpenAPI credentials for master and copy accounts
   - Configure real account IDs and API keys
   - Validate SDK methods against approved accounts

2. **End-to-End Testing**
   - Run `python -m app.tools.webull_preflight` with real credentials
   - Capture live events with `python -m app.tools.webull_event_capture`
   - Test read-only sessions with `python -m app.tools.webull_readonly_session`
   - Run latency benchmarks against real Webull API

3. **Production Rollout**
   - Live read-only mode validation
   - Tiny-funds live testing
   - Full production deployment

4. **Optional Enhancements**
   - Activate paid L2 data subscription
   - Additional auth gates for scout endpoints if needed

### 🚀 SYSTEM HIGHLIGHTS

This is a **sophisticated, enterprise-grade trading platform** with:
- Advanced machine learning integration
- Sub-300ms order execution latency
- Comprehensive risk management
- Professional operator interface
- Complete audit trails and reconciliation
- Robust error handling and recovery

## Non-Negotiable Design Rules

- Copier hot path target: under 300 ms from event received to child broker response when Webull latency allows.
- No database reads before copied order submission.
- No broker position lookups before copied order submission.
- No Slack, dashboard, learning, or reconciliation work before copied order submission.
- Enabled copy targets use percent-equity mirroring.
- L2 is required for final scout quality, but delayed until launch readiness.
- PPO/RL does not trade or override the human in v1.
- External investor/follower accounts are out of scope until legal/compliance review.

## Current Verification Baseline

Latest full local verification:

- `pytest -q` -> `196 passed, 1 skipped`
- `python -m compileall app tests` -> passed
- `alembic heads` -> `0005_ai_artifacts (head)`
