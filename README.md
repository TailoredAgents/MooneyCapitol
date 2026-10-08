# MooneyCapitol

MooneyCapitol is transitioning from a legacy Webull/equity research and copier application (V1) to a broker-neutral futures platform (V2). Both systems remain in this repository, but their runtime responsibilities, data, labels, and model artifacts are deliberately isolated.

The V2 execution path is not production-ready and cannot submit a broker order. A production-shaped direct R|Protocol read-only transport and capture service now exist for Rithmic Test observation, while every broker mutation remains structurally disabled. The `conner_nq_v1` code is still an observation and learning-data foundation only: it contains no trained model, live Scout service, complete Conner strategy, or execution connection.

## Start here

- [V2 architecture](docs/V2_FUTURES_ARCHITECTURE.md) — canonical plane boundaries and safety rules
- [Conner NQ Scout foundation](docs/CONNER_NQ_SCOUT_FOUNDATION.md) — implemented observation, labeling, ranking, replay, and provider contracts
- [Project handoff](docs/PROJECT_HANDOFF.md) — current status, limitations, and exact next step
- [Rithmic read-only runbook](docs/RITHMIC_READ_ONLY_RUNBOOK.md) — private binding generation, Render configuration, and Test validation
- [Legacy roadmap](docs/MOONEYCAPITOL_ROADMAP.md) — historical V1 plans, not V2 authority

## Current systems

### V1: legacy equity/reference system

The existing API, worker, small-cap scout, Webull copier, dashboard, P&L monitor, Slack integration, AI Lab, equity learning pipeline, migrations `0001` through `0008`, and V1 tables remain available.

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

It reports not-ready and submission-disabled. The execution-side Rithmic adapter remains disabled, and the in-memory lease is only a local/test seam. Separately, `app/v2/brokers/rithmic_protocol` and `app/v2/capture` implement direct R|Protocol 0.90 observation with independent Order/PnL sessions, account allowlisting, replay/reconciliation, durable append-only journaling, and sanitized health/readiness. R|API+/.NET is a fallback/conformance reference only.

The dedicated command is:

```powershell
python -m app.v2.capture.main
```

It is fail-closed unless connectivity is explicitly enabled for `TEST`, external checksum-verified bindings and credentials are supplied through secrets, and an exact account allowlist is configured. The observer exposes no mutation methods, and a default-deny outbound-template policy rejects every known order/bracket/OCO mutation immediately before transport.

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
- Real follower order placement
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

Render's V1 copier variables are deny-only gates:

- `COPIER_ENABLED=0` blocks operation regardless of persisted config.
- `COPIER_MODE=test` blocks a persisted mode mismatch.
- `COPIER_GLOBAL_KILL_SWITCH=1` forces a deployment block.

Environment values cannot enable trading. Live V1 mode also requires persisted enablement, explicit transition confirmations, and a positive `live_max_notional_per_order`. V2 submission cannot be enabled by configuration.

## Exact next step

Deploy `mooney-rithmic-capture` fail-closed, provide the privately generated licensed bindings and Rithmic Test secrets, then follow the read-only runbook to validate account discovery, manual R|Trader visibility, orders/fills, P&L/RMS, brackets, replay overlap, reconnect, redaction, and reconciliation. That validation must not enable or exercise any API-side broker mutation.

In parallel, configure and validate an authenticated market-data transport, exact NQ/ES provider bindings, and explicit entitlements so point-in-time strategy observations can accumulate. Run ranking contracts in Shadow and collect actual-trade matches, misses, ranks, lead times, and data-quality failures. Train nothing until enough honest chronological labels exist. Production readiness has not been declared; paper and live broker mutation remain separate future gates after read-only conformance.
