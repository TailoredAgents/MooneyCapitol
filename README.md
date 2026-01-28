Conner's Trading Agent — V3 Skeleton

Purpose: 24/7 scout for consolidation → breakout/breakdown → retest opportunities on small-cap gappers. This repo provides the initial skeleton for the Web API + Worker + DB, mapped to your V3 design and ready to deploy on Render.

What’s included now
- Web API (FastAPI): health, config, watchlist, setups, reports, Slack actions webhook, basic dashboard stub, and /ws/live.
- Worker: rolling longlist (pre-market), 09:10 freeze, 09:25 Watchlist 100 finalize, consolidation scan every 10s (demo candles if POLYGON_API_KEY missing).
- DB models (SQLAlchemy): symbols, candles, l2_snapshots, boxes, setups, alerts, fills, trades.
- Adapters (stubs): Polygon (HTTP), Slack (SDK client), IBKR depth provider interface placeholder.
- Depth provider flag: `DEPTH_MODE=demo|ibkr` with demo fallback + health reporting.
- Ledger + reports: demo TradeStation fills ingestion, EOD summary, `/reports/eod` JSON + CSV.
- Nightly learning loop: offline Logistic Regression ranker + thresholds with canary rollout and `/learning/report` endpoint.
- Observability: structured logs + optional Sentry init.
- Render config: `render.yaml` with API (web), Worker, and Postgres.
- Config defaults matching V3 in `app/core/config.py`.

What comes next (high-level milestones)
1) Data ingestion:
   - Polygon: gappers (pre-market), aggregates 1m/2m/5m/15m.
   - L2: plug IBKR adapter; provide candle-only fallback when unavailable.
2) Detection logic:
   - Consolidation boxes (1–2m), compression checks, quality gates (spread/RVOL/range caps).
   - Break + retest trigger and R:R gate; L2 confirm with rolling 3s imbalance.
3) Alert pipeline:
   - Alert state machine (Armed → PRIMED → Active → Resolved), TTL, noise controls.
   - Slack threads: heads-up (quiet), PRIMED ping, Trigger ping; threaded updates.
4) Dashboard:
   - Stream Armed/PRIMED/Active lanes, L2 mini, TTL, controls.
5) Reports & ledger:
   - TradeStation fills ingestion, join to setups, realized R, EOD summary to Slack.
6) Learning (safe):
   - Nightly ranker + threshold grid with canary and revert.

Local setup
1) Python 3.11+
2) Create a virtualenv and install deps:
   - `python -m venv .venv && source .venv/bin/activate`
   - `pip install -r requirements.txt`
3) Copy `.env.sample` (or `.env.example`) to `.env` and fill values.
   - For local dev without Postgres, set `STATE_STORE=mem` (dashboard/config will be process-local).
   - For Render/production, set `STATE_STORE=db` so API + worker share state via Postgres.
4) Ensure Postgres is available and `DATABASE_URL` points to it.
5) Initialize DB tables (temporary, until Alembic is wired):
   - `python -m app.db.init_db`
6) Run API: `uvicorn app.api.main:app --reload`
7) Run worker: `python -m app.workers.runner`

Demo mode
- If `POLYGON_API_KEY` is not set (or `DEMO_MODE=1`), the Polygon adapter serves deterministic mock data:
  - Watchlist 100 always returns 100 synthetic gappers.
  - Candle fetches provide synthetic compression boxes so Heads-Up threads + dashboard tiles can be validated without market data.
- Slack messages fall back to structured logs when `SLACK_BOT_TOKEN` is absent.
- Slack slash commands (`/watch`, `/alerts`, `/headsup`) let traders adjust pinned symbols and alert budgets on the fly (after wiring the slash command in Slack pointing to `/slack/commands`).
- Nightly learning runs automatically at 02:00 ET (or via `/learning/report` to inspect the last run). Artifacts land in `artifacts/alert_ranker.pkl`, `thresholds.json`, and `thresholds_canary.json`.

Enabling IBKR depth later
- Leave `DEPTH_MODE=demo` (default) while developing. PRIMED/Trigger flows will label L2 as demo.
- When ready for live depth:
  1. Set `DEPTH_MODE=ibkr` in your environment.
  2. Configure `IBKR_HOST`, `IBKR_PORT`, and `IBKR_CLIENT_ID` to point at your IB Gateway/TWS instance.
  3. (Recommended) Enable SMART depth aggregation (`IBKR_SMART_DEPTH=1`) and set `IBKR_DEPTH_LEVELS=5`.
  4. Ensure the IBKR API is enabled with trusted IPs and the gateway is reachable from the worker container.
  5. Restart the worker; `/health` will report an `ibkr` mode and the worker will mark snapshots once depth streams in.

Deployment note: IBKR market depth requires an always-on IB Gateway/TWS process. Render-hosted containers typically cannot reach your local gateway unless you expose it securely. In practice, run the worker where it can reliably connect to the gateway (same machine/VPC/VPS), or use demo depth until an IBKR connectivity plan is in place.

Deploy on Render
- `render.yaml` defines:
  - Web Service `api`: FastAPI app.
  - Worker `worker`: scheduled scans and jobs.
  - Managed Postgres database.
- `STATE_STORE=db` is required on Render so the API and worker share config, live lanes, worker tick, and learning artifacts via Postgres.
- Fill required env vars in the Render dashboard or `render.yaml` envVars.

Notes
- Many components are stubs with clear TODOs; they map exactly to your design so we can fill in detectors, scoring, adapters, and scheduling logic incrementally.
- The worker currently uses simple asyncio schedules. You can swap to Prefect orchestration once tokens/projects are configured.
- Replay tapes locally: `python -m app.tools.replay samples/tapes/XYZ.json`.
