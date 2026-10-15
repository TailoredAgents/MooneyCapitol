# MooneyCapitol

MooneyCapitol is transitioning from a legacy Webull/equity research and copier application (V1) to a broker-neutral futures platform (V2). Both systems remain in this repository, but their runtime responsibilities, data, labels, and model artifacts are deliberately isolated.

The V2 execution path is not production-ready and cannot submit a broker order. A production-shaped direct R|Protocol read-only transport and capture service now exist for Rithmic Test observation. A one-master/four-follower NQ/MNQ Shadow planner now calculates master-relative percentage risk and exposure, but every broker mutation remains structurally disabled. The target topology is five operationally independent AMP accounts with separately configurable credentials: one master and four followers. The software does not depend on AMP subaccounts, a shared entity, or one Rithmic user. The `conner_nq_v1` code is still an observation and learning-data foundation only: it contains no trained model, live Scout service, complete Conner strategy, or execution connection.

## Start here

- [Project implementation plan](docs/PROJECT_IMPLEMENTATION_PLAN.md) — canonical direction, sequence, readiness gates, and progress checkpoint
- [V2 architecture](docs/V2_FUTURES_ARCHITECTURE.md) — canonical plane boundaries and safety rules
- [V2 copy trader foundation](docs/V2_COPY_TRADER_FOUNDATION.md) — one-master/four-follower parity sizing, safety gates, and activation path
- [Conner NQ Scout foundation](docs/CONNER_NQ_SCOUT_FOUNDATION.md) — implemented observation, labeling, ranking, replay, and provider contracts
- [Project handoff](docs/PROJECT_HANDOFF.md) — current status, limitations, and exact next step
- [Rithmic read-only runbook](docs/RITHMIC_READ_ONLY_RUNBOOK.md) — private binding generation, Render configuration, and Test validation
- [Legacy roadmap](docs/MOONEYCAPITOL_ROADMAP.md) — historical V1 plans, not V2 authority

## Current systems

### V1: legacy equity/reference system

The existing API, worker, small-cap scout, Webull copier, P&L monitor, Slack integration, AI Lab, equity learning pipeline, migrations `0001` through `0008`, and V1 tables remain available as legacy/reference code. The retired V1 operator console is not served by the application.

V1 is legacy/reference code. Its stock features, `hybrid_target`, thresholds, synthetic rows, XGBoost artifacts, and paper outcomes are not valid V2 futures inputs. The V2 registry and dataset validation quarantine those sources.

Universal V1 safety fixes include authenticated watchlist mutations, fail-closed Slack signatures with replay-age validation, execution transition checks on whole-config replacement, deny-only deployment gates, a mandatory positive live per-order notional ceiling, and recursive secret/account redaction. These fixes do not certify the legacy Webull path for live use.

### V2: futures execution and reference foundation

`app/v2` retains four one-way planes:

1. Execution: lifecycle normalization, deterministic risk, actors, journal queue, reconciliation contracts, and lease/fencing seams.
2. Market data: exact-expiry references, CME sessions, typed provider events, entitlements, availability lineage, and provider adapters.
3. Intelligence: synchronized observation, candidate research measurements, isolated learning data, ranking/Shadow contracts, and replay.
4. Analysis/operations: immutable plans, trades, lifecycle facts, outcome/evaluation records, and later reporting inputs.

Execution does not depend on OpenAI, V1 learning, XGBoost, scikit-learn, or Scout output. Its standalone command remains disabled:

```powershell
python -m app.v2.execution.main
```

It reports not-ready and submission-disabled. The execution-side Rithmic adapter remains disabled, and the in-memory lease is only a local/test seam. The separate V2 Shadow copier planner can calculate stop-defined risk and point-exposure parity for one master and four followers using integer same-expiry NQ/MNQ quantities. It has no multi-credential supervisor, event orchestrator, durable copy intent, follower control persistence, or broker submission method. Separately, `app/v2/brokers/rithmic_protocol` and `app/v2/capture` implement direct R|Protocol 0.90 observation with independent Order/PnL sessions, account allowlisting, replay/reconciliation, durable append-only journaling, and sanitized health/readiness. The planned write path will be a separate execution boundary and will not weaken this observer's structural read-only guard. R|API+/.NET is a fallback/conformance reference only.

The dedicated command is:

```powershell
python -m app.v2.capture.main
```

It is fail-closed unless connectivity is explicitly enabled for `TEST`, external checksum-verified bindings and credentials are supplied through secrets, and an exact account allowlist is configured. The observer exposes no mutation methods, and a default-deny outbound-template policy rejects every known order/bracket/OCO mutation immediately before transport.

### Operator dashboard

`/dashboard` is the authenticated V2 Rithmic observation console. It reads the append-only V2 PostgreSQL projections and displays redacted account aliases, capture counts, connection generations, reconciliation checkpoints, account/P&L state, positions, orders, executions, brackets, RMS state, contract-reference observations, and the fail-closed Shadow copier readiness state. It exposes no broker mutation control.

The API service can optionally read the capture process's sanitized `/health` endpoint when `RITHMIC_CAPTURE_BASE_URL` is configured. If that value is absent or unreachable, the dashboard reports live health as unavailable but continues to display durable observations from PostgreSQL. There is no active V1 equity/Webull dashboard route.

### `conner_nq_v1`: observation foundation

The implemented strategy-specific namespace observes exact-contract NQ as the traded/prediction instrument and exact-contract ES as synchronized context. ES is not an NQ/MNQ execution mapping. Same-expiry product equivalence remains NQ/MNQ only.

Implemented observation capabilities include:

- synchronized multi-timeframe NQ/ES state with backward/as-of cutoff enforcement;
- separate event, availability, received/retrieved, computation, definition, and feature-cutoff clocks;
- configurable versioned lead-up horizons and candidate New York windows;
- candle geometry, relative structure/SMT, Fibonacci, divergence, and candidate-level measurements;
- positive/unlabeled behavior labels, evidenced explicit passes, and pairwise-preference contracts;
- four isolated tasks for behavior, outcome, conviction, and copier execution quality;
- separate ranking components and Shadow-only, never-executable output contracts;
- deterministic exact-contract NQ/ES replay with distinct point-in-time and finalized-history modes;
- additive migration `0010_conner_nq_scout` for durable observation/learning records.

These measurements are research clues, not Conner rules. The repository does not encode `SMT + Fibonacci + rejection block = trade`, infer unspoken strategy logic, or treat every untraded state as a negative. Provider-finalized history is barred from Conner behavior labels.

### Massive Futures REST adapter

`app/v2/providers/massive_futures.py` implements the first replaceable historical research adapter. With explicit entitlements and exact provider bindings, it supports REST contract reference, schedules, fixed-resolution bars, trades, BBO, and deterministic historical replay.

Authentication, HTTP lifecycle, credentials, and retries are injected through `AsyncJsonTransport`; the repository does not yet contain a configured production transport. Entitlements default to disabled, there is no demo fallback, and the adapter does not implement live streaming. Its historical events are marked finalized history and are not eligible as point-in-time behavior evidence.

## Persistence

Migrations are additive:

- `0009_v2_futures_foundation` adds 23 V2 execution/reference tables.
- `0010_conner_nq_scout` adds 13 observation, candidate measurement, learning, ranking, Shadow, and evaluation tables.
- `0011_rithmic_read_capture` adds 11 append-only connection, event, replay, account, order, execution, bracket, reference, P&L/RMS, and reconciliation tables.

V1 records are neither altered nor converted. Monetary, price, tick, and R values use `NUMERIC(24,10)` where applicable; futures quantities remain integers.

The Rithmic capture service writes immutable native events and append-only normalized projections through a durable PostgreSQL journal. A durable high-volume strategy market-data capture service and production Scout repository/orchestrator are not implemented yet.

## Intentionally absent

- Rithmic broker mutation of any kind, including order, cancel/modify, flatten, bracket/OCO, or follower submission
- Completed Rithmic Test conformance and manual R|Trader-to-API visibility validation
- Live master-event orchestration and real follower order placement
- A complete or handcrafted Conner strategy
- A live Scout/candidate-generation service
- Trained futures models or a selected XGBoost/ranking objective
- Model-controlled risk, sizing, or autonomous trading
- Automatic rolling of open positions
- Live Massive streaming or permanent provider lock-in
- Synthetic Conner decisions or V1-equity-to-NQ label conversion

## Local development

Requirements: Python 3.11 and PostgreSQL for full migration/runtime testing.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.sample .env
alembic upgrade head
pytest -q
python -m compileall app tests migrations
```

For local tests that do not require PostgreSQL, `STATE_STORE=mem` can be used. Never commit broker or provider credentials. Database URLs, API keys, Slack secrets, and account identifiers belong in environment/secret storage and must be redacted from logs and API payloads.

## V1 safety configuration

The active Render blueprint no longer provisions Webull credentials, and its V1 shadow, AI Lab, and paper switches default to disabled. The remaining V1 copier variables are deny-only gates:

- `COPIER_ENABLED=0` blocks operation regardless of persisted config.
- `COPIER_MODE=test` blocks a persisted mode mismatch.
- `COPIER_GLOBAL_KILL_SWITCH=1` forces a deployment block.

Environment values cannot enable trading. Live V1 mode also requires persisted enablement, explicit transition confirmations, and a positive `live_max_notional_per_order`. V2 submission cannot be enabled by configuration.

## Exact next step

Follow [the implementation plan](docs/PROJECT_IMPLEMENTATION_PLAN.md) to make the copier credential-ready without waiting for account credentials. The next implementation task is independent secret-reference/account configuration for one master and four separately connected followers, followed by multi-account supervision, durable controls, cumulative-position sizing, lifecycle copying, recovery/fencing, a minimal operational UI, and the complete simulated acceptance suite.

When the five credential bundles become available, insert them through secret management and complete Rithmic Test, conformance, measured latency, and a staged one-follower live canary. The dashboard structure/aesthetic pass follows the working copier. Learning and Conner-focused read-only AI agents come later, while the execution foundation records trustworthy lifecycle and execution-quality facts from day one. Production readiness has not been declared.
