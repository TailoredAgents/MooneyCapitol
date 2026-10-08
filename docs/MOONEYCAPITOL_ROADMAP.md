# MooneyCapitol Roadmap

> Legacy V1 history. This document is retained for reference and does not define V2 futures architecture, readiness, strategy, features, labels, or provider choices. See `docs/V2_FUTURES_ARCHITECTURE.md`.

## 1. Product Vision

MooneyCapitol is intended to become a two-part trading operations platform:

1. A scout that helps the lead trader find high-quality small-cap gapper setups faster.
2. A trade copier that mirrors the lead trader's Webull master-account fills into approved Webull copy accounts with low latency, strong controls, and complete auditability.

The final system should support discovery, alerting, execution copying, reporting, and learning feedback in one platform. The scout and copier should feed each other through shared data, but they should remain separate runtime paths so a slow scan, report, or dashboard update cannot delay copied orders.

OpenAI is planned as a read-only intelligence layer for explanations, recaps, learning-report translation, journals, and ticker/catalyst research. It is not part of trade execution or copier sizing. Detailed plan: `docs/OPENAI_AI_FEATURES_PLAN.md`.

## 2. Pre-Launch Validation System

**Current status:** The system is feature-complete enough for validation testing, but it is not cleared for live money until Webull credentials, live read-only sessions, latency, reconciliation, and tiny-funds tests pass.

### Implemented Product Areas:

**Complete Scout System:**
- ✅ Advanced pattern detection with 1m/2m consolidation boxes
- ✅ Demo depth mode now; Webull Advanced Quotes is the intended live L2/tape provider
- ✅ HTF level analysis and gap edge detection  
- ✅ Breakout/retest triggers with volume confirmation
- ✅ Live 3-lane dashboard (Armed/PRIMED/Active)
- ✅ Slack integration with threading and action tracking

**Complete XGBoost Learning Engine:**
- ✅ Hybrid XGBoost-first with logistic regression fallback
- ✅ 15+ engineered features with Postgres-based training
- ✅ Real-time p2R scoring integrated into live alerts
- ✅ Nightly retraining with feature importance reporting
- ✅ Manual trade detection and comprehensive labeling

**Complete Trade Copier:**
- ✅ Webull SDK wrapper with event listening, pending real-account validation
- ✅ Separate Webull HTTP and trading-events endpoint configuration
- ✅ Sub-300ms low-latency execution hot path
- ✅ Production-grade percent-equity sizing
- ✅ Kill switch, risk controls, startup recovery
- ✅ Order reconciliation and background position sync
- ✅ V3 operator dashboard with separated Trader Workspace and Developer Setup navigation
- ✅ Comprehensive testing tooling suite

**Complete Infrastructure:**
- ✅ FastAPI REST API with full endpoint coverage
- ✅ Multi-job worker scheduler with error handling
- ✅ Complete SQLAlchemy models with Alembic migrations
- ✅ Operator authentication (login page, session cookies, and API tokens)
- ✅ Render deployment configuration
- ✅ Real-time WebSocket updates

### Validation Phase Required:

**Ready for live validation after credentials/accounts exist:**
- Real Webull OpenAPI credentials and account setup
- End-to-end testing with approved accounts  
- Live-read-only validation sessions
- Tiny-funds live testing
- Optional paid L2 data activation

### 🚀 ACHIEVEMENT HIGHLIGHTS

This section records the intended legacy V1 design scope; it is not a production-readiness claim. It includes:
- Advanced machine learning with real-time scoring
- Sub-300ms order execution architecture  
- Comprehensive risk management and audit trails
- Professional operator interface
- Sophisticated pattern detection algorithms
- Complete testing and validation tooling

**The roadmap phases below are now complete - this serves as historical reference.**

Operator authentication for `/dashboard`, `/copier/*`, and `PUT /config` is implemented via the `COWORK_OPERATOR_USERNAME`/`COWORK_OPERATOR_PASSWORD` app login page, signed HTTP-only dashboard session cookies, and `COWORK_OPERATOR_API_TOKEN` (X-Operator-Token header) env var. Gates pass through when no auth env vars are set so local dev and tests are unchanged.

## 2.1 Webull Direction Change

The copier direction changed from a mixed-broker model to a Webull-to-Webull model:

- V1 master account: partner's Webull personal account.
- V1 copy account: your Webull personal account.
- Future target state: one master account can copy to many approved Webull accounts, each with its own credentials, equity value, enable flag, reconciliation state, and audit trail. All enabled targets use percent-equity mirroring.

Planning assumptions from official Webull documentation as of May 16, 2026:

- Webull OpenAPI supports HTTP for trading operations, account management, and market data queries.
- Webull OpenAPI supports gRPC for real-time order-status event push.
- Webull OpenAPI offers official Python and Java SDKs.
- Webull Trading API supports account lookup, balance/positions, order preview, order placement, batch placement, replace/cancel, order history, open orders, order detail, and trade event subscription.
- Webull's published stock feature matrix includes market orders, limit orders, fractional shares, short selling, extended-hours trading, and overnight session support.
- Webull OpenAPI uses App Key/App Secret signature authentication, with an optional token flow when 2FA is enabled.
- Webull lists production and test API environments.
- Entity accounts are not part of v1; all v1 accounts are planned as personal Webull accounts.

Primary docs:

- Webull OpenAPI overview: https://developer.webull.com/apis/docs/about-open-api/
- Webull Trading API overview: https://developer.webull.com/apis/docs/trade-api/overview/
- Webull Orders docs: https://developer.webull.com/apis/docs/trade-api/trade/
- Webull Authentication overview: https://developer.webull.com/apis/docs/authentication/overview/


Critical feasibility checkpoint:

- Do not enable live copy trading until the two Webull personal accounts can access OpenAPI or an approved equivalent integration path. If Webull does not approve API access for one of the accounts, the architecture must change before live trading.

## 3. Final System Architecture

The final system should run as separate services/processes:

- `api`: FastAPI routes, dashboard, websocket state, operator controls.
- `scanner_worker`: existing scout/watchlist/setup detection worker.
- `copier_worker`: low-latency Webull-to-Webull trade copier.
- `reconcile_worker`: optional later worker for slower position/order reconciliation.
- `db`: Postgres for durable state, audit trail, reports, learning data.

High-level flow:

```text
Polygon / market data
        |
        v
Scout worker -> alerts/dashboard -> trader decision

Partner Webull master account
        |
        | real-time fill/execution events
        v
Copier worker
        |
        | concurrent target orders
        v
Personal Webull copy account (v1)
Additional approved Webull accounts (future)

Copied fills + outcomes
        |
        v
Ledger, reports, learning engine
```

The copier hot path should be intentionally small:

1. Receive master execution event.
2. Deduplicate it.
3. Validate symbol, side, account, session, and copy eligibility from cached state.
4. Calculate target quantities.
5. Submit target Webull orders concurrently.
6. Persist master execution and child order records after submit/background.
7. Reconcile child order events and target positions in background jobs.

## 4. Phase 0: Research, Compliance, and Account Setup

Goal: define the legal, broker, and operational boundaries before live order-copying exists.

Scope:

- Confirm that v1 is only for accounts owned or controlled by the team:
  - Partner Webull master account.
  - Personal Webull account.
- Confirm account-opening requirements for Webull personal accounts.
- Confirm whether each Webull account can be approved for OpenAPI access and receive App Key/App Secret credentials.
- Confirm Webull master account OpenAPI access, account ID retrieval, data permissions, and test/live environment availability.
- Confirm whether copied trading will be equities only for v1.
- Decide whether short selling is allowed in v1.
- Decide whether premarket/after-hours copying is allowed in v1.
- Decide whether copied orders should be market orders during regular hours only in v1.
- Document compliance boundary: internal accounts first, no investor/follower product until reviewed separately.

Acceptance criteria:

- Account ownership and authorization model is documented.
- Webull test or live-read-only credentials are available for the Webull copy target.
- Webull test or live-read-only master environment is available.
- A live-trading kill switch policy is documented.
- Investor/follower functionality is explicitly out of scope for v1.

## 5. Phase 1: Stabilize Current Scout

Goal: make the current scout reliable enough to coexist with the copier.

Scope:

- Add a real migrations path with Alembic.
- Make `/setups` read from the database instead of the placeholder memory list.
- Add scanner worker health heartbeat details.
- Improve dashboard status for scanner worker, data mode, Slack mode, depth mode, and state store mode.
- Review encoding issues in visible text and Slack messages.
- Add tests around watchlist, config persistence, consolidation detection, trigger detection, and Slack action status updates.
- Confirm Render deployment works with `STATE_STORE=db`.
- Render deploys must run `python -m app.tools.run_migrations` before API/worker startup so a fresh Postgres has all Alembic tables before the KV-store startup probe.
- Use a paid Render Postgres plan for launch; free Postgres is not appropriate for real-money operation.
- The Render Blueprint is staged for safe early deployment with `DEPTH_MODE=demo`, `COPIER_ENABLED=0`, `COPIER_MODE=test`, `COPIER_GLOBAL_KILL_SWITCH=1`, and `LEARNING_SANDBOX_ENABLED=0`; add paid Polygon/Webull credentials only when validation starts.

Acceptance criteria:

- API and scanner worker can run locally in demo mode.
- API and scanner worker can share state through Postgres.
- Existing tests pass.
- Dashboard displays current `armed`, `primed`, and `active` lanes from shared state.
- Scout behavior is documented as separate from copier behavior.

## 6. Phase 2: Webull Trading Adapter

Goal: build a robust Webull order adapter before wiring master events to copy orders.

New modules:

- `app/copier/webull_client.py`
- `app/copier/models.py`
- `app/copier/config.py`
- `app/copier/errors.py`

Scope:

- Support one Webull target account initially.
- Keep the adapter, config, database, and engine designed for many Webull target accounts later, each with separate API key, secret, base URL, account ID, target equity, and enabled flag.
- Support test and production base URLs, with test as the default.
- Implement account lookup.
- Implement preflight checks for configured credentials, account lookup, account ID matching, positions, balance/detail availability, and database readiness.
- Implement order submission.
- Implement order lookup by Webull order ID.
- Implement order lookup by `client_order_id`.
- Implement cancel order.
- Implement positions lookup.
- Implement gRPC order-status events if credentials and environment support it cleanly.
- Use connection pooling and one reusable client per target account.
- Require deterministic `client_order_id` for every copied order.

Target account config should include:

- Internal name, such as `personal`.
- Broker type: `webull`.
- Environment: `test` or `live`.
- Enabled flag.
- API key and secret, loaded from environment or encrypted secret store.
- Target equity for percent-equity mirroring.
- Optional backend emergency-brake values.
- Regular-hours-only flag.
- Short-selling enabled flag.
- Optional backend symbol allowlist/blocklist.

Acceptance criteria:

- A test order can be submitted to the configured Webull target account.
- The same code path can add more Webull target accounts later without schema redesign.
- The same `client_order_id` cannot create duplicate orders.
- Disabled target accounts receive no orders.
- Adapter handles reject responses without crashing the worker.
- Unit tests cover request building, idempotency behavior, and error handling with injected SDK/test clients.
- `python -m app.tools.webull_preflight` can validate the configured Webull accounts without placing trades.

## 7. Phase 3: Webull Master Event Listener

Goal: listen to master account executions without placing copied trades yet.

New modules:

- `app/copier/webull_master.py`
- `app/copier/runtime.py`
- `app/workers/runner.py`
- `app/tools/copier_replay.py`

Scope:

- Connect to Webull OpenAPI using the official Python SDK and approved app credentials.
- Subscribe to real-time order-status events and capture fills/executions.
- Filter events to the configured master account.
- Normalize Webull order/fill events into internal `MasterExecution` records.
- Persist master executions to Postgres.
- Deduplicate by Webull order/fill event ID and client order ID.
- Track connection state and heartbeat.
- Support replaying recorded execution events from a local sample file.
- Support capturing real Webull order-event messages to JSONL and checking parser normalization before live read-only sessions.
- Ignore non-fill order events so submitted/cancelled master orders are not copied.
- Support read-only mode for capturing master executions and intended target copy decisions without child broker orders.

Important design choice:

- Copy from fills/executions first, not submitted master orders.
- This avoids copying orders that the master account never actually fills.

Acceptance criteria:

- Webull test/live-read-only fills are captured in real time.
- Duplicate execution events do not create duplicate records.
- Master executions are visible through an API endpoint.
- Copier worker heartbeat is visible in `/health` or a copier status endpoint.
- Recorded execution replay works without Webull connected.
- Captured Webull order-event files can be re-analyzed locally without Webull connected.
- Captured Webull child-order submit/detail/status responses can be normalized locally without Webull connected.
- Read-only sessions can capture raw events and persist `would_copy`/`blocked` decisions without sending broker orders.

## 8. Phase 4: Internal Trade Copier MVP

Goal: copy Webull master fills into the Webull personal target account.

New modules:

- `app/copier/engine.py`
- `app/copier/sizing.py`
- `app/copier/risk.py`
- `app/api/routes/copier.py`

Scope:

- On each valid master fill, build one child order per enabled Webull target.
- Submit child orders concurrently.
- Use deterministic `client_order_id`.
- Keep the hot path lean: cached target config, cached target state, warmed Webull clients, child order submit before database persistence, Slack, or reconciliation.
- Support buy and sell for long equity positions.
- Estimate each target's active copied position from submitted/accepted/filled copied orders so exits can be copied without a hot-path broker position lookup.
- Block sells that would create a target short position unless shorting is explicitly enabled and supported by the account.
- Start with regular-hours-only copying unless explicitly changed.
- Start with market orders for regular-hours fills.
- Reject unsupported symbols, sessions, asset classes, or sides.
- Store copy order status for each target account.
- Expose copier status, recent master fills, and recent child orders through the API.
- Preserve submit timestamps, latency, request payloads, response payloads, and target-level errors.
- Keep processing other targets if one target account rejects or fails.

Production sizing rule:

- `percent_equity`: target account uses the same account percentage as the master trade. This is the intended production default.
- `equity_ratio`: backward-compatible alias for `percent_equity`.
- `disabled`: account is configured but never receives orders.
- Legacy/testing modes still exist in backend code, but enabled live copy targets must use `percent_equity` so every target mirrors the master account's percent exposure.

Percent-equity sizing formula:

- `master_trade_pct = master_fill_notional / master_account_equity`
- `target_notional = target_account_equity * master_trade_pct`
- `target_qty = target_notional / master_fill_price`

Example: if a $30,000 master account uses 5% of equity, then a $2,000 target account uses about $100, a $10,000 target uses about $500, and a $100,000 target uses about $5,000. If the master uses leveraged exposure, targets attempt to mirror that same leveraged percentage as long as Webull accepts the child order for the target account.

Acceptance criteria:

- One Webull master fill creates one Webull child order per enabled target account.
- The v1 Webull personal target account receives a child order.
- Multiple enabled Webull target accounts can be submitted concurrently once more targets are configured.
- Replaying the same master fill does not create duplicate child orders.
- Each enabled target account uses `percent_equity` sizing; different account values are handled by the target equity value.
- `percent_equity` sizing preserves the master trade's account percentage across $2k, $10k, $30k, $100k, and larger target accounts.
- A sell that closes existing copied long exposure is copied even when shorting is disabled.
- A sell that would create short exposure is blocked unless shorting is enabled.
- Each target account can be disabled independently.
- Rejected child orders are persisted and visible.
- Hot-path latency is measured and stored.
- Operators can inspect recent master executions, child orders, audit events, and copier errors through the API.

Inspection endpoints:

- `GET /copier/status`
- `GET /copier/master-executions`
- `GET /copier/copy-orders`
- `GET /copier/trades`
- `GET /copier/audit-events`
- `GET /copier/errors`
- `GET /copier/reconciliations`

Replay commands:

- Webull preflight without placing trades: `python -m app.tools.webull_preflight`
- Capture/analyze Webull order events without placing trades: `python -m app.tools.webull_event_capture --out captures\webull_events.jsonl --seconds 300 --max-events 25`
- Analyze Webull child-order responses: `python -m app.tools.webull_order_response path\to\child_order_response.json`
- Run no-trade read-only session: `python -m app.tools.webull_readonly_session --out captures\webull_readonly_session.jsonl --seconds 300 --max-events 25`
- Dry run without broker submission: `python -m app.tools.copier_replay path\to\webull_event.json`
- Read-only master persistence and intended copy-decision logging: `python -m app.tools.copier_replay path\to\webull_event.json --read-only`
- Real configured target submission: `python -m app.tools.copier_replay path\to\webull_event.json --submit`
- Hot-path latency benchmark with fake broker responses: `python -m app.tools.copier_latency_benchmark samples\webull_execution_fill.json --iterations 100 --targets 1`

## 9. Phase 5: Reconciliation, Recovery, and Risk Controls

Goal: make the copier safe enough for controlled live testing.

Scope:

- Add global kill switch.
- Add per-account kill switch.
- Keep optional backend emergency brakes for max notional, daily copied notional, order rate, and symbol filters.
- Add session controls.
- Add shorting controls.
- Add position reconciliation against Webull.
- Run Webull position reconciliation as a background job because Webull account position queries are rate-limited and must never block copied order submission.
- Add order-status reconciliation by `client_order_id`.
- Poll submitted/open Webull child orders by deterministic `client_order_id`.
- Normalize child order status, fill quantity, average fill price, and reject reason.
- Add startup recovery:
  - Load recent master executions.
  - Load recent child orders.
  - Check whether any pending child order was already submitted.
  - Resume without duplicate orders.
  - Reconcile open child orders before subscribing to new master execution events.
  - Mark stale unresolved child orders for operator review.
- Add alerting for mismatches:
  - Master filled, child rejected.
  - Child partially filled.
  - Child filled at materially worse price.
  - Target account missing position.
  - Webull order-event stream disconnected.
  - Webull master connection disconnected.

Acceptance criteria:

- Kill switch prevents new copied orders immediately.
- Process restart does not duplicate orders.
- Worker startup runs recovery before the master listener subscribes.
- Reconciliation can detect missing, rejected, and partially filled child orders.
- Background position sync can detect when Webull target holdings differ from the local copied-position estimate.
- Operator can see mismatches from the API/dashboard.
- Real Webull account balance and position snapshots are stored for P&L, drawdown, exposure, and risk alert monitoring.
- Operator can see P&L account snapshots, positions, buying power, daily P&L, drawdown, exposure, and active risk alerts in the dashboard P&L tab.
- Worker refreshes P&L snapshots in the background without adding work to the copied-order hot path.
- Percent-equity copier sizing uses the latest cached Webull account snapshot equity first, with manual equity env vars only as fallback.
- Copier can run for a full Webull test or live-read-only session without manual database cleanup.
- Reconciliation records are visible through `GET /copier/reconciliations`.

## 10. Phase 6: Dashboard and Operator Controls

Goal: make the system operable during live trading.

Scope:

- Add dashboard section for copier state.
- Show Webull master connection status.
- Show Webull target account status.
- Show last master execution time.
- Show last copied order time.
- Show recent copy latency metrics.
- Show copied trade results with master trade, target account, copy status, latency, fill price, slippage, and rejection reason.
- Show recent rejects/mismatches.
- Show active kill switch state.
- Add API controls:
  - Enable/disable target account.
  - Toggle global kill switch.
  - Toggle test/live/read-only mode only through guarded config.
  - Update account equity for percent-equity mirroring.
  - Keep advanced emergency-brake fields in the API/config for backend operators.

Implemented dashboard/API controls:

- `PATCH /copier/settings` updates global copier enabled flag, mode, and master equity with validation.
- `PATCH /copier/targets/{target_name}` updates target enablement and target equity, and the dashboard saves targets as `percent_equity`.
- `GET /copier/readiness` returns blockers/warnings for credentials, equity, target mirror sizing, open reconciliation issues, and failed copied orders.
- `/dashboard` includes a Copier tab that exposes the simple live operator flow: global enabled/mode/master equity, target enabled/equity, kill switch, readiness checks, recent orders, and reconciliation warnings.
- `/dashboard` includes a Trades tab that shows copied trade results, including copy latency and fill/slippage fields.
- `/pnl/*` API routes expose account snapshots, positions, alerts, sessions, summary, and manual refresh for account balance/P&L monitoring.
- `/launch/readiness` and the dashboard Launch tab aggregate database, worker, auth, market-data, copier, P&L, read-only validation counts/history, latency, and learning readiness into one launch-blocker view.
- Operator changes are written to `copier_audit_events`.
- Worker Slack alerts are sent for readiness blockers, reconciliation mismatches/errors, startup recovery warnings, and Webull listener errors, with throttling to avoid repeated alerts.
- Worker Slack alerts are sent for Webull position sync mismatches/errors, with throttling to avoid repeated alerts.
- Optional emergency brakes still exist in backend config/API for max trade notional, max position percent, max daily notional, max daily copied trades, and max orders per minute. They are not required for readiness because the product rule is to mirror master percent exposure.
- Dangerous actions require explicit confirmation in the dashboard and API `X-Confirm` headers:
  - `ENABLE_COPIER`
  - `ENABLE_TARGET:{target_name}`
  - `DISABLE_KILL_SWITCH`
  - `SET_LIVE_MODE`

Acceptance criteria:

- Operator can tell within seconds whether copier is healthy.
- Operator can disable copying globally without redeploying.
- Operator can disable one account without affecting the other.
- Controls are audited.
- Dangerous controls require explicit confirmation or environment gating.

## 11. Phase 7: Learning Feedback Loop

Goal: feed copied-trade outcomes back into the existing reporting and learning system.

Detailed implementation plan: `docs/HYBRID_XGBOOST_LEARNING_PLAN.md`.

Scope:

- Upgrade the existing nightly learning job in `app/workers/runner.py` and `app/services/learning.py` from the current logistic-regression-first implementation to an XGBoost-first implementation.
- Build the nightly training dataset from Postgres tables:
  - `setups`
  - `alerts`
  - `fills`
  - `trades`
- Keep the current logistic regression path as a fully backward-compatible fallback when XGBoost is unavailable, fails training, or does not have enough valid data.
- Save the best nightly model to disk, while preserving the existing artifact-loading behavior used by API/worker services.
- Store learning artifacts in shared state where needed so API and workers can still coordinate in Render/Postgres deployments.
- Log feature importance to Slack every night after successful XGBoost training.
- Include feature importance and selected model type in `/learning/report`.
- Keep threshold/canary behavior compatible with the current learning report structure.
- Tag fills and trades by source:
  - `scout_alert`
  - `master_copy`
  - `manual`
- Link copied trades to scout setups when symbol/time proximity makes sense.
- Store master execution ID and child order IDs on resulting ledger records.
- Include copied account outcomes in reports.
- Teach learning jobs to distinguish:
  - Alert generated but not traded.
  - Alert generated and master traded.
  - Master traded without prior alert.
  - Copied trade outcome by target account.
- Add reporting views for slippage between master and copied accounts.

Acceptance criteria:

- XGBoost is the preferred nightly model when enough valid training data exists.
- Logistic regression remains available as a fallback without breaking existing reports or runtime scoring.
- Nightly training reads joined setup/alert/fill/trade context from Postgres rather than relying only on setup payloads.
- The best model artifact is saved to disk after training.
- Slack receives a nightly feature-importance summary when XGBoost trains successfully.
- `/learning/report` identifies the active model type and includes feature-importance data when available.
- `/learning/report` includes source breakdown, feature version, and master-trade ranking metrics.
- Sandbox-generated learning rows remain disabled by default and must be capped/downweighted when enabled.
- EOD reports include copied trades.
- Learning dataset can include copied-trade outcomes without mixing them incorrectly with scout-only alerts.
- Slippage and rejection rates are visible by target account.
- Reports can separate master and personal copy-account performance.

## 11.1 Phase 7A: OpenAI Read-Only Intelligence Layer

Detailed implementation plan: `docs/OPENAI_AI_FEATURES_PLAN.md`.

Scope:

- Scout alert explanations.
- Daily trading recap.
- Learning report translation.
- Trade journal automation.
- Ticker/catalyst research summaries.

Out of scope:

- Chat-style dashboard assistant.
- Any AI involvement in order submission, copied-order sizing, risk-control decisions, or the sub-300ms copier hot path.

Acceptance criteria:

- AI features can be disabled globally by env/config.
- AI failures do not break scout alerts, copier execution, reports, learning jobs, or dashboard loading.
- AI outputs are stored with model name, prompt version, source data, status, and error metadata.
- Short scout explanations can appear in Slack/dashboard after the base alert is already emitted.
- Research summaries are timestamped and source-aware.

## 12. Phase 8: Controlled Live Trading Rollout

Goal: move from Webull test/live-read-only validation to live copy trading with tight limits.

Required prerequisites:

- At least several full Webull test or live-read-only sessions completed.
- No unexplained duplicate orders.
- No unresolved reconciliation bugs.
- Kill switch tested.
- Restart recovery tested during market hours in Webull test or live-read-only mode.
- Webull live OpenAPI credentials available for each account.
- Webull live master account connection tested.
- Compliance and authorization notes reviewed.

Rollout steps:

1. Live read-only mode:
   - Listen to Webull live master fills.
   - Do not place Webull live copy orders.
   - Persist what would have copied as `would_copy` rows and blocked decisions as `blocked` rows.
2. Live tiny-funds mode:
   - Enable the personal target account with minimal account funding.
   - Keep production `percent_equity` mirroring.
   - Keep kill switch ready and monitor latency, rejects, copied fills, and position sync.
3. Additional account mode:
   - Later, enable additional approved Webull target accounts one at a time.
   - Each target uses `percent_equity` mirroring with its own equity value.
4. Normal funding mode:
   - Increase funding only after stable live operation and clean reconciliation.

Acceptance criteria:

- Live read-only logs match expected copy decisions.
- Tiny-size live copy produces no duplicate orders.
- Reconciliation is clean across both accounts.
- Operator has confidence in dashboard status and kill switch behavior.

## 13. Phase 9: Multi-Account and Investor-Ready Future

Goal: evaluate whether this becomes an external follower/investor platform.

This phase is intentionally separate from internal copy trading.

Scope:

- Legal and compliance review.
- Decide whether investment adviser, broker, solicitor, or other registrations/relationships apply.
- Decide whether Webull Connect API, Broker API, or another account-management model is required.
- Build proper onboarding, disclosures, agreements, suitability/risk acknowledgements, and account-level authorization.
- Add account grouping, billing, permissions, and investor controls.
- Add stronger supervision, audit export, and immutable logs.

Acceptance criteria:

- No outside investor/follower account is connected until legal and compliance requirements are documented.
- Technical architecture for external accounts is reviewed separately from internal copier architecture.
- Internal copier remains usable even if investor platform is delayed or never built.

## 14. Data Model Plan

New tables likely needed:

### `copy_target_accounts`

- `id`
- `name`
- `broker`
- `environment`
- `enabled`
- `account_ref`
- `equity`
- `sizing_mode`
- `sizing_value`
- `min_notional`
- `max_notional_per_trade`
- `max_position_pct`
- `max_daily_notional`
- `regular_hours_only`
- `shorting_enabled`
- `allowlist`
- `blocklist`
- `created_at`
- `updated_at`

Secrets should not be stored directly in this table unless encrypted. Prefer environment variables or a secret manager.

### `master_executions`

- `id`
- `broker`
- `account_ref`
- `broker_execution_id`
- `broker_order_id`
- `symbol`
- `side`
- `qty`
- `price`
- `asset_class`
- `executed_at`
- `received_at`
- `raw_payload`
- unique constraint on broker/account/execution ID

### `copy_orders`

- `id`
- `master_execution_id`
- `target_account_id`
- `broker`
- `client_order_id`
- `broker_order_id`
- `symbol`
- `side`
- `qty`
- `order_type`
- `time_in_force`
- `status`
- `submitted_at`
- `accepted_at`
- `filled_at`
- `filled_qty`
- `avg_fill_price`
- `reject_reason`
- `latency_ms`
- `raw_submit_payload`
- `raw_response_payload`

### `copy_order_events`

- `id`
- `copy_order_id`
- `event_type`
- `status`
- `event_at`
- `received_at`
- `raw_payload`

### `copy_reconciliations`

- `id`
- `target_account_id`
- `symbol`
- `severity`
- `status`
- `message`
- `detected_at`
- `resolved_at`
- `raw_context`

### `copier_audit_events`

- `id`
- `event_type`
- `actor`
- `target_account_id`
- `message`
- `created_at`
- `payload`

## 15. API Plan

Potential routes:

- `GET /copier/status`
- `GET /copier/accounts`
- `POST /copier/accounts/{id}/enable`
- `POST /copier/accounts/{id}/disable`
- `PUT /copier/accounts/{id}/policy`
- `GET /copier/master-executions`
- `GET /copier/copy-orders`
- `GET /copier/reconciliations`
- `POST /copier/kill-switch/enable`
- `POST /copier/kill-switch/disable`
- `POST /copier/replay`

These routes should be protected before live trading. The current API has no authentication layer, so operator controls should not be exposed publicly until auth is added.

## 16. Configuration Plan

Environment variables for v1 may include:

- `COPIER_ENABLED`
- `COPIER_MODE=test|live|read_only`
- `COPIER_GLOBAL_KILL_SWITCH`
- `WEBULL_MASTER_API_ENDPOINT`
- `WEBULL_MASTER_EVENTS_ENDPOINT`
- `WEBULL_MASTER_APP_KEY`
- `WEBULL_MASTER_APP_SECRET`
- `WEBULL_MASTER_ACCOUNT_ID`
- `WEBULL_PERSONAL_API_ENDPOINT`
- `WEBULL_PERSONAL_APP_KEY`
- `WEBULL_PERSONAL_APP_SECRET`
- Future extra targets should use a repeatable naming pattern such as `WEBULL_TARGET_<NAME>_API_ENDPOINT`, `WEBULL_TARGET_<NAME>_APP_KEY`, `WEBULL_TARGET_<NAME>_APP_SECRET`, and `WEBULL_TARGET_<NAME>_ACCOUNT_ID`.

Database config should hold non-secret policy values. Secret values should stay in environment variables or a dedicated secret manager.

## 17. Latency Requirements

Initial targets:

- Master fill event received to child broker response: under 300 ms when Webull event delivery and order API latency allow it.
- Webull master fill received to Webull child submit started: under 50 ms in normal conditions.
- Webull child submits for all enabled targets launched concurrently.
- No blocking report/dashboard/learning work in the hot path.
- No database reads, broker position lookups, Slack calls, dashboard work, learning work, or reconciliation work before child order submission.
- Database persistence, operator alerts, and reconciliation run after submit/background.
- All broker clients should be initialized before market open.

Metrics to record:

- Master execution timestamp.
- Copier received timestamp.
- Copy decision timestamp.
- Webull child submit start timestamp.
- Webull child response timestamp.
- Webull child accepted timestamp.
- Webull child filled timestamp.
- Event-received to submit-start latency.
- Submit-start to broker-response latency.
- Event-received to broker-response latency.
- Benchmark p50/p95/p99 latency and sub-300ms pass rate.
- End-to-end master fill to child fill latency.

## 18. Security Requirements

Before live trading:

- Add API authentication for operator controls.
- Keep broker credentials out of git.
- Do not store API secrets in plaintext database rows.
- Add explicit live-trading guard.
- Add audit events for every config/control change.
- Restrict dashboard/control access by network or authentication.
- Ensure logs do not print secrets.
- Separate Webull test and live credentials clearly.

## 19. Testing Strategy

Unit tests:

- Sizing policies.
- Risk checks.
- Deterministic client order ID generation.
- Webull request building.
- Webull error parsing.
- Webull execution/order-event normalization.
- Deduplication.
- Kill switch behavior.

Integration tests:

- Replay recorded master execution events.
- Submit child orders through the same copier code path with injected Webull SDK/test clients for one target account.
- Submit child orders through the same copier code path with injected Webull SDK/test clients for multiple target accounts.
- Simulate Webull reject and partial fill.
- Simulate process restart and recovery.
- Simulate duplicate Webull order/fill event.

Webull test/live-read-only tests:

- Webull master fill to Webull child order.
- The Webull personal target account is copied reliably.
- Account disabled mid-session.
- Global kill switch mid-session.
- Network disconnect and reconnect.
- Full trading day Webull test or live-read-only run.

Live rollout tests:

- Live read-only mode.
- Live tiny-size copy.
- Restart recovery during live read-only.
- Kill switch during live tiny-size.

## 20. Open Questions

- Will the partner master account and personal copy account both be approved for Webull OpenAPI credentials?
- Will the personal and Partner Webull master accounts each receive OpenAPI access for test and production environments?
- Should v1 copy only equities, or also options/crypto later?
- Should v1 allow short selling?
- Should v1 copy premarket/after-hours trades?
- What account onboarding/config pattern should be used when additional Webull target accounts are added?
- Should child orders use market orders only during regular hours?
- How should the system behave when the master partially fills?
- How should the system behave when one future target account rejects and another fills?
- Should the copier ever attempt automatic repair trades, or only alert the operator?
- Where will the Webull copier worker run for lowest reliable latency to Webull OpenAPI?
- What authentication should protect dashboard/operator controls?

## 21. Near-Term Implementation Order

Recommended next steps from the current state:

1. Validate Webull SDK calls with approved test/live-read-only credentials.
2. Run `python -m app.tools.webull_preflight` after credentials/account IDs exist.
3. Capture real Webull order-event payloads with `python -m app.tools.webull_event_capture`.
4. Analyze real child order submit/detail/status responses with `python -m app.tools.webull_order_response`.
5. Run no-trade read-only sessions with `python -m app.tools.webull_readonly_session`.
6. Confirm master fill event payloads and child order response payloads from real Webull accounts.
7. Run copier latency benchmark against real Webull test/live-read-only conditions.
8. Run live read-only mode for full sessions and compare intended copy decisions against master fills.
9. Verify background order reconciliation and position sync with real Webull responses.
10. `GET /config` is now operator-protected because it can expose account configuration. As an optional V1 follow-up, assess protection for the remaining scout read endpoints (`/setups`, `/learning/report`, `/watchlist/today`, `/reports/eod`) and `/ws/live`.
