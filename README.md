# MooneyCapitol

MooneyCapitol is a trading operations platform with two connected systems:

1. A small-cap setup scout for consolidation -> breakout/breakdown -> retest opportunities.
2. A low-latency Webull-to-Webull trade copier that mirrors the lead trader's master-account fills into approved Webull copy accounts.

The scout and copier share data for reports and learning, but their runtime paths are separate so scanning, dashboards, Slack, learning, database writes, and reconciliation cannot delay copied order submission.

## Start Here

- Canonical handoff/status: `docs/PROJECT_HANDOFF.md`
- Full roadmap: `docs/MOONEYCAPITOL_ROADMAP.md`
- Hybrid learning plan: `docs/HYBRID_XGBOOST_LEARNING_PLAN.md`

## Current State: Feature-Complete for Pre-Launch Validation

**Implemented features:**

### Core Infrastructure
- ✅ **FastAPI API**: Complete REST API with health/config/watchlist/reports/learning/copier endpoints
- ✅ **Worker System**: Multi-job scheduler with premarket watchlist, real-time scanning, dashboard updates, Slack alerts
- ✅ **Database**: Complete SQLAlchemy models with Alembic migrations for all data entities
- ✅ **Authentication**: App login page, signed session cookie, and API token system for operator controls
- ✅ **Deployment**: Full Render configuration for API, worker, and PostgreSQL

### Scout System 
- ✅ **Market Data**: Polygon client with demo fallback, 1m/2m/5m/15m aggregates
- ✅ **L2 Depth**: Demo mode now, Webull Advanced Quotes planned for live L2/tape
- ✅ **Pattern Detection**: Advanced consolidation box detection with breakout/retest triggers
- ✅ **Risk/Reward**: HTF level analysis, gap edge detection, dynamic R:R calculation
- ✅ **Dashboard**: Live 3-lane system (Armed/PRIMED/Active) with real-time WebSocket updates
- ✅ **Alerts**: Slack integration with threading, mentions, action tracking

### Learning System
- ✅ **XGBoost ML**: Complete hybrid XGBoost-first implementation with logistic regression fallback
- ✅ **Feature Engineering**: 15+ features including L2, RVOL, spread, price buckets, time buckets
- ✅ **Training Data**: Postgres-based dataset from setups/alerts/fills/trades with manual trade detection
- ✅ **Model Artifacts**: Automatic model saving/loading with feature importance analysis
- ✅ **Runtime Scoring**: Real-time p2R scoring for live setups
- ✅ **Nightly Jobs**: Automated retraining with Slack reporting of feature importance

Additional learning guardrails now in place:
- 35+ feature set with regime and microstructure signals.
- Sandbox-generated training is off by default and, when enabled, capped and downweighted so real Connor trades stay primary.
- Nightly reports include source breakdown and master-trade ranking metrics.

### P&L Monitoring
- Real Webull balance/position snapshot service with no simulated account values
- P&L API routes for account snapshots, positions, alerts, sessions, summary, and manual refresh
- Daily P&L, drawdown, exposure, and risk-level tracking backed by Postgres
- Dashboard P&L tab for account value, buying power, daily P&L, positions, and active risk alerts
- Worker background refresh keeps P&L snapshots current without touching the copier hot path

### Trade Copier
- ✅ **Webull Integration**: SDK wrapper with master event listening, pending real-account validation
- ✅ **Low-Latency Engine**: <300ms hot path with cached config, warmed clients
- ✅ **Percent-Equity Sizing**: Production-ready mirroring algorithm
- ✅ **Risk Controls**: Kill switch, position tracking, order reconciliation
- ✅ **Startup Recovery**: Handles restarts without duplicate orders
- ✅ **Comprehensive Tooling**: Preflight, event capture, replay, benchmarking, read-only testing
- ✅ **Dashboard Controls**: V3 operator interface with separated trader and developer setup areas
- ✅ **Background Jobs**: Order reconciliation, position sync, readiness monitoring

### V3 Operator Dashboard
- ✅ **Separated Navigation**: Trader Workspace for daily use, Developer Setup for configuration and launch controls
- ✅ **Trader Workspace**: Scout, Trades, and P&L
- ✅ **Developer Setup**: Copier Settings and Launch Readiness
- ✅ **Real-time Updates**: WebSocket integration for live data
- ✅ **Mobile Responsive**: Works on all device sizes
- ✅ **Operator Controls**: Target account management, kill switch, readiness monitoring
- ✅ **Trade Analytics**: Latency tracking, slippage analysis, fill quality metrics

**⏳ VALIDATION REQUIRED (Not Missing Features):**
- Webull OpenAPI credentials and live account validation
- End-to-end testing with real accounts
- L2 data activation (feature complete, just needs paid subscription)

## Copier Direction

- V1 master account: partner's Webull personal account.
- V1 copy account: user's Webull personal account.
- Future copy accounts: many approved Webull accounts.
- All enabled copy targets use `percent_equity` mirroring.
- If the master uses 5% of its account, each target attempts to use 5% of its account.
- If the master uses leveraged exposure, targets attempt to mirror that leveraged percentage as long as Webull accepts the order and the target account has the required buying-power/margin permissions.
- Do not enable live copy trading until Webull OpenAPI access, account IDs, credentials, live-read-only testing, kill switch, and reconciliation are confirmed.

## Webull OpenAPI Notes

- Python dependency is pinned to `webull-openapi-python-sdk==2.0.7`.
- Webull HTTP trading/account host and gRPC trading-events host are separate.
- Test HTTP host: `us-openapi-alb.uat.webullbroker.com`
- Test events host: `us-openapi-events.uat.webullbroker.com`
- Production HTTP host: `api.webull.com`
- Production events host: `events-api.webull.com`
- Set `WEBULL_MASTER_API_ENDPOINT` for HTTP account/trading calls and `WEBULL_MASTER_EVENTS_ENDPOINT` for master fill event streaming.
- Target copy accounts use their own `WEBULL_*_API_ENDPOINT` for HTTP order placement.
- Copied Webull equity orders use `support_trading_session="ALL"` by default so they are not intentionally limited to core regular-hours trading.

## Low-Latency Copier Rules

- Target: under 300 ms from master event received to child broker response when Webull latency allows.
- No database reads before copied order submission.
- No broker position lookup before copied order submission.
- No Slack/dashboard/learning/reconciliation work before copied order submission.
- Use cached target config/state.
- Use warmed Webull clients.
- Submit child orders first; persist/audit/reconcile after submit or in background.

## Local Setup

1. Python 3.11+
2. Create a virtualenv and install deps:
   - `python -m venv .venv`
   - Windows: `.venv\Scripts\activate`
   - Unix/macOS: `source .venv/bin/activate`
   - `pip install -r requirements.txt`
3. Copy `.env.sample` or `.env.example` to `.env` and fill values.
   - For local dev without Postgres, set `STATE_STORE=mem`.
   - For production/Render, set `STATE_STORE=db`.
4. Ensure Postgres is available and `DATABASE_URL` points to it.
5. Run migrations:
   - `alembic upgrade head`
6. Run API:
   - `uvicorn app.api.main:app --reload`
7. Run worker:
   - `python -m app.workers.runner`

## Copier Tools

Validate Webull copier configuration and read-only account connectivity:

```powershell
python -m app.tools.webull_preflight
```

Check only local config/env values without calling Webull:

```powershell
python -m app.tools.webull_preflight --skip-network
```

Analyze a saved Webull event or sample payload:

```powershell
python -m app.tools.webull_event_capture --input samples\webull_execution_fill.json
```

Capture live Webull master order events without placing trades:

```powershell
python -m app.tools.webull_event_capture --out captures\webull_events.jsonl --seconds 300 --max-events 25
```

Analyze Webull child-order submit/detail/status responses:

```powershell
python -m app.tools.webull_order_response samples\webull_child_order_submit_response.json
```

Run a full no-trade read-only copier session from a sample payload:

```powershell
python -m app.tools.webull_readonly_session --input samples\webull_execution_fill.json --no-persist
```

Run a live no-trade read-only session after Webull credentials exist:

```powershell
python -m app.tools.webull_readonly_session --out captures\webull_readonly_session.jsonl --seconds 300 --max-events 25
```

Replay a recorded Webull fill without broker submission:

```powershell
python -m app.tools.copier_replay samples\webull_execution_fill.json
```

Record master executions and intended copy decisions without broker submission:

```powershell
python -m app.tools.copier_replay samples\webull_execution_fill.json --read-only
```

Submit through configured Webull targets:

```powershell
python -m app.tools.copier_replay samples\webull_execution_fill.json --submit
```

Benchmark the copier hot path with fake broker responses:

```powershell
python -m app.tools.copier_latency_benchmark samples\webull_execution_fill.json --iterations 100 --targets 1
```

Simulate broker delay:

```powershell
python -m app.tools.copier_latency_benchmark samples\webull_execution_fill.json --iterations 100 --targets 1 --broker-delay-ms 50
```

## Copier API

- `GET /copier/status`
- `GET /copier/readiness`
- `GET /copier/targets`
- `PATCH /copier/settings`
- `PATCH /copier/targets/{target_name}`
- `GET /copier/master-executions`
- `GET /copier/copy-orders`
- `GET /copier/trades`
- `GET /copier/audit-events`
- `GET /copier/errors`
- `GET /copier/reconciliations`
- `POST /copier/kill-switch/enable`
- `POST /copier/kill-switch/disable`

## Launch API

- `GET /launch/readiness`

This protected endpoint combines database migration state, worker heartbeat, operator auth, Polygon/Slack/depth envs, copier readiness, P&L freshness, read-only validation history, copied-order latency samples, and learning report availability.

## P&L API

- `POST /pnl/refresh`
- `GET /pnl/accounts`
- `GET /pnl/accounts/{account_ref}`
- `GET /pnl/accounts/{account_ref}/positions`
- `GET /pnl/accounts/{account_ref}/alerts`
- `POST /pnl/alerts/{alert_id}/acknowledge`
- `GET /pnl/sessions`
- `GET /pnl/summary`

Dangerous copier actions require confirmation through dashboard prompts or `X-Confirm` headers:

- `ENABLE_COPIER`
- `ENABLE_TARGET:{target_name}`
- `DISABLE_KILL_SWITCH`
- `SET_LIVE_MODE`

## Dashboard

- `/dashboard` uses a V3 left-side navigation layout.
- Trader Workspace:
  - Scout: Armed, PRIMED, and Active lanes.
  - Trades: copied trade results, read-only `would_copy` decisions, blocked decisions, latency, fill status, fill price, slippage, and reject reason.
  - P&L: account values, cash, buying power, daily P&L, drawdown, exposure, positions, and P&L risk alerts.
- Developer Setup:
  - Copier Settings: readiness, global controls, target enabled/equity controls, kill switch, recent orders, and reconciliations.
  - Launch Readiness: unified production readiness checks, launch blockers, and recent read-only validation decisions.

## Operator Authentication

The dashboard and copier control endpoints are gated by a single shared operator credential pair plus a separate API token. Both come from environment variables, so they never touch git and can be rotated by re-deploying.

Env vars:

- `COWORK_OPERATOR_USERNAME` and `COWORK_OPERATOR_PASSWORD` lock `/dashboard` behind the app login page. Set both.
- `COWORK_OPERATOR_API_TOKEN` locks the `/copier/*` APIs and `/config` PUT behind an `X-Operator-Token` request header for external callers. A logged-in dashboard browser session is also accepted on those APIs, so the dashboard works without a separate token.

Gated routes:

- `/login`, `/logout`, and `/dashboard` (login required when `COWORK_OPERATOR_USERNAME` / `COWORK_OPERATOR_PASSWORD` are set).
- All `/copier/*` routes (dashboard session cookie, Basic Auth, OR `X-Operator-Token` required when any operator credential is set).
- `PUT /config` (same as `/copier/*`; it can mutate copier config, so it shares the gate).

Open routes (intentional):

- `/health` so Render and uptime checks keep working.
- `/webhooks/slack/actions` and `/slack/commands` so Slack's own signature verification stays the auth boundary.
- `GET /config` and the scout read endpoints (`/setups`, `/learning/report`, `/watchlist/today`, `/reports/eod`) are not gated yet. Plan a follow-up if those need protection too.

Behavior when nothing is configured:

- If none of the auth env vars are set, all auth dependencies pass through. Local dev and the existing test suite work unchanged.
- The first env var you set turns the corresponding gate on. The other gate stays open until its env vars are set.

Example (PowerShell):

```powershell
$env:COWORK_OPERATOR_USERNAME = "operator"
$env:COWORK_OPERATOR_PASSWORD = "<long-random-password>"
$env:COWORK_OPERATOR_API_TOKEN = "<long-random-token>"
uvicorn app.api.main:app
```

Calling a gated API from outside the browser:

```powershell
curl -H "X-Operator-Token: <long-random-token>" http://localhost:8000/copier/status
```

## L2 Plan

L2 is part of the final intended scout, not a nice-to-have. It is delayed until launch readiness because of cost.

Current mode:

- Keep `DEPTH_MODE=demo`.
- Scout can run candle/volume/RVOL-only.
- Alerts can show demo/missing L2.

When ready:

- Subscribe to Webull OpenAPI Advanced Quotes.
- Build/enable the Webull depth adapter.
- Configure `DEPTH_MODE=webull` only after that adapter is implemented and tested.
- Do not use any IBKR credentials for the normal deployment path.

## Deployment

- `render.yaml` defines the API service, worker service, and managed Postgres database.
- `runtime.txt`, `.python-version`, and the Blueprint `PYTHON_VERSION=3.11.11` env vars pin Render to Python 3.11 so NumPy, pandas, scikit-learn, and XGBoost install from stable wheels instead of slow Python 3.14 source builds.
- Render runs `python -m app.tools.run_migrations` as a pre-deploy command for both API and worker services.
- The migration runner uses a Postgres advisory lock so concurrent API/worker deploys do not run Alembic at the same time.
- The managed Postgres plan is set to `basic-256mb`; do not use expiring free Postgres for a real-money launch.
- `STATE_STORE=db` is required in deployed environments so API and worker share config, dashboard state, worker tick, and learning artifacts.
- Keep broker credentials in environment variables or a secret manager, not database rows or git.

### Render Blueprint Staged Launch

The blueprint is safe to deploy before paid market data and Webull credentials exist.

Deploy-now defaults:

- `DEPTH_MODE=demo`
- `COPIER_ENABLED=0`
- `COPIER_MODE=test`
- `COPIER_GLOBAL_KILL_SWITCH=1`
- `LEARNING_SANDBOX_ENABLED=0`
- `OPENAI_AI_FEATURES_ENABLED=0`
- `OPENAI_RESEARCH_ENABLED=0`

Set these during the first Render Blueprint deploy:

- `COWORK_OPERATOR_USERNAME`
- `COWORK_OPERATOR_PASSWORD`
- `COWORK_OPERATOR_API_TOKEN`
- `SLACK_CHANNEL` if different from `all-trading`
- `SLACK_BOT_TOKEN` and `SLACK_SIGNING_SECRET` if Slack alerts/buttons should work now
- `SENTRY_DSN` if error monitoring is enabled
- `OPENAI_API_KEY` can be set now, but OpenAI features stay inactive until `OPENAI_AI_FEATURES_ENABLED=1`

Leave these blank until accounts/subscriptions are ready:

- `POLYGON_API_KEY`
- `WEBULL_MASTER_API_ENDPOINT`
- `WEBULL_MASTER_EVENTS_ENDPOINT`
- `WEBULL_MASTER_APP_KEY`
- `WEBULL_MASTER_APP_SECRET`
- `WEBULL_MASTER_ACCOUNT_ID`
- `WEBULL_PERSONAL_API_ENDPOINT`
- `WEBULL_PERSONAL_APP_KEY`
- `WEBULL_PERSONAL_APP_SECRET`
- `WEBULL_PERSONAL_ACCOUNT_ID`

Before live validation:

- Add the Polygon Advanced key for real-time broad scanning.
- Add Webull OpenAPI credentials, account IDs, HTTP endpoint, and events endpoint.
- Let the P&L monitor pull live Webull account equity before enabling copy trading. `WEBULL_MASTER_ACCOUNT_EQUITY` and `WEBULL_PERSONAL_ACCOUNT_EQUITY` remain optional emergency fallback env vars, but they are not required for the normal launch path.
- Keep `COPIER_GLOBAL_KILL_SWITCH=1` until read-only validation passes.
- Use the Launch tab to confirm remaining blockers are expected credential/data blockers.

## Current Verification Baseline

Latest full local verification:

- `pytest -q` -> `158 passed, 1 skipped`
- `python -m compileall app tests` -> passed
- `alembic heads` -> `0005_ai_artifacts (head)`
