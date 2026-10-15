# V2 copy trader foundation

Status: five-account configuration/simulation, durable controls, cumulative sizing, lifecycle orchestration, fenced recovery/reconciliation, latency telemetry, and an isolated Test-only write boundary implemented; live driver/runtime attachment and production submission not implemented

Updated: 2026-10-15

## What this is

The V2 copy trader is intended to observe one Mooney Capital master futures account and reproduce eligible master trades across exactly four operationally independent follower accounts. All five are separate AMP accounts. The application does not assume AMP subaccounts, one legal entity, one portal, one Rithmic user, or shared credentials. Every account receives its own exact broker selector and independently configurable credential/connection bundle. The initial product set is NQ and MNQ. Account ownership does not replace AMP/Rithmic permissions, credentials, session limits, or conformance requirements.

The implemented sizing phase is deterministic and Shadow-only. It calculates each follower's desired total, evaluated whole-contract target, achievable total, required signed delta, and the reason it would be allowed or blocked. A separate lifecycle layer carries those immutable facts into committed per-follower intents, deterministic ordered commands, and durable outbox records before any adapter call. Every dispatch claim requires the current database-backed owner/fence. A transport-ambiguous or abandoned claim is never resent blindly: a fresh, complete broker snapshot must explain orders, executions, positions, and protection first, otherwise that follower is locked. The standalone execution process still cannot submit, modify, cancel, or flatten a broker order: requesting `TEST` or `LIVE` through its current configuration makes the effective mode `OFF`. An isolated Rithmic writer exists only as an unattached `OFF`/`SHADOW`/explicit-Test boundary; its enum has no live value.

The forward implementation order is defined in [PROJECT_IMPLEMENTATION_PLAN.md](PROJECT_IMPLEMENTATION_PLAN.md). The immediate target is a credential-ready copier built and simulated without waiting for the five credential bundles. Broker validation and live readiness remain later gates.

## Sizing objective

The planner targets the master's actual exposure as a percentage of account equity, not the same raw contract count.

For the master:

```text
risk dollars = (absolute(entry - initial stop) * point value + cost buffer) * contracts
risk fraction = risk dollars / master equity
point exposure fraction = (point value * contracts) / master equity
```

For each follower, the planner searches integer quantities of same-expiry NQ and MNQ. It compares both:

- stop-defined risk as a percentage of follower equity; and
- dollars-per-point exposure as a percentage of follower equity.

The larger relative mismatch is the parity error. The candidate must satisfy the configured parity tolerance and every independent account limit. This makes percentage return and percentage stop risk track the master as closely as contract granularity permits. It does not promise identical realized ROI: fill price, slippage, fees, partial fills, disconnects, and manual management can still create differences once execution exists.

Broker margin is not treated as trade risk. Current operator-supplied margin requirements only determine whether the account has enough capacity after the configured headroom.

## Whole-contract limitation

Futures contracts cannot be fractional. One MNQ has one tenth of the point value of one NQ. Therefore, for a master holding one NQ, one MNQ gives exact percentage point exposure when the follower equity is one tenth of master equity. With the same entry and stop, it also gives exact stop-risk percentage before differing costs.

If a master has $30,000 and a follower has $500, one NQ on the master implies a target of roughly 0.167 MNQ on the follower. The planner cannot place that fraction. Its default policy reports the nearest one-MNQ result as over target and blocks it. It never rounds an underfunded follower up silently. Over-target planning requires an explicit configuration opt-in and still must satisfy the parity tolerance and all hard risk limits.

## Cumulative targets and funding adequacy

Production sizing starts from the master's authoritative total signed position rather than sizing each fill independently. The planner selects the follower's total whole-contract target and then subtracts the authoritative current copied position. The resulting delta explicitly classifies hold, open, add, reduce, close, or reverse. This prevents repeated scale-in or scale-out events from accumulating rounding error.

An open follower contract is pinned by exact identity until flat. If the best flat-account choice would switch NQ/MNQ, the existing contract is retained or safely closed; it is never converted implicitly. An expiry mismatch is blocked and reported rather than interpreted as permission to roll. Master-flat exposure produces a close delta even when entry gates are active. Other risk-reducing deltas may pass entry-only kill/loss gates, while adds and reversals still require full new-risk eligibility.

Every cumulative result records desired risk/exposure, evaluated whole-contract quantity, achievable risk/exposure, current and target signed quantity, delta, parity error, margin requirement, action, and reason. The Phase 5 bridge preserves those facts in durable copy intents; the preview endpoint itself remains non-persisting.

Each follower also receives a one-MNQ funding report. It calculates the stop-defined risk of one exact-expiry MNQ, minimum equity needed to avoid exceeding master-relative risk/exposure, per-trade and concurrent-risk capacity, minimum available margin after headroom, and the current shortfalls. Operational blockers are reported separately from whether the account is economically large enough.

## Eligibility and safety rules

A Shadow plan requires:

- the fixed structural topology of one master and exactly four followers;
- an initial protective stop on the master trade, on the protective side of entry;
- an exact NQ or MNQ contract and a same-exchange, same-expiry follower mapping;
- positive, current account equity and operator-supplied broker margin facts;
- positive fail-closed contract, per-trade risk, concurrent-risk, and daily-loss limits;
- global and follower kill switches to be clear;
- enough margin after headroom and enough remaining risk capacity; and
- an integer quantity within the configured parity tolerance.

The master trade's actual risk fraction must itself be within the follower policy ceiling. Scout rankings, AI output, predictions, and V1 equity state cannot alter these rules.

## Implemented surface

- `app/v2/execution/copier.py`: pure Decimal-based parity and cumulative-position planning, exact-contract pinning, signed deltas, and one-MNQ funding adequacy with no broker dependency or side effect.
- `app/v2/execution/account_config.py`: typed master/follower connection slots, per-account secret references, redacted resolution, and fail-closed Test-connectivity readiness.
- `app/v2/execution/connection_supervisor.py`: independent per-account discovery plus ten generation-fenced Order/PnL session states, heartbeat/staleness checks, broker-neutral fresh-data gates, failure isolation, and deterministic simulation fixtures.
- `app/v2/execution/copier_config.py`: fail-closed Shadow/risk configuration composed with the independent account topology.
- `app/v2/execution/copy_control.py` and `copy_control_store.py`: fixed copy-profile membership, explicit follower transitions, restart directives, optimistic revisions, row-locked serialization, and controlled reason codes.
- `app/v2/execution/lifecycle.py`, `lifecycle_bridge.py`, `lifecycle_store.py`, and `lifecycle_service.py`: deterministic master/follower correlation, committed revisioned intents, ordered commands, duplicate/stale-event protection, monotonic receipts, follower-isolated concurrent dispatch, and close-before-open reversal sequencing.
- `app/v2/execution/lease.py`, `recovery.py`, and `reconciliation.py`: database-time durable lease/fencing, transactional outbox claims, no-blind-retry crash recovery, fresh/complete broker snapshot comparison, explained-state adoption, follower-isolated divergence locking, and p50/p95/p99 lifecycle telemetry.
- `app/v2/brokers/rithmic_write.py`: official R|Protocol 0.90 request mapping for market/limit/stop, bracket/OCO, modify/cancel, and position exit, with no live mode and explicit Test authorization, external-binding, and injected-transport gates.
- `app/tools/generate_rithmic_write_bindings.py`: separate external-only generation for the official write request/response schemas; licensed schemas and derivatives remain outside Git.
- migration `0012_v2_copy_control`: globally paused/follower-disabled defaults plus database-enforced append-only audit history.
- migration `0013_v2_copy_lifecycle`: per-follower intent, command, broker-correlation, and append-only lifecycle-event persistence.
- migration `0014_v2_execution_recovery`: durable command outbox, append-only delivery/reconciliation evidence, database-fenced recovery state, and lifecycle timing projections.
- `python -m app.tools.v2_copy_latency_report`: redacted aggregate or per-follower persisted latency report.
- `GET /v2/copier/status`: authenticated, redacted readiness and blocker status.
- `POST /v2/copier/preview`: authenticated, non-persisting Shadow calculation for supplied hypothetical state.
- `POST /v2/copier/cumulative-preview`: authenticated, non-persisting total-position/delta and funding-adequacy calculation for supplied authoritative state.
- `GET /v2/copier/control` and `/control/audit`: authenticated state and audit reads.
- `/v2/copier/control/*` mutation endpoints: authenticated, CSRF-protected for dashboard sessions, and revision-checked.
- `/dashboard`: read-only Shadow copier readiness alongside Rithmic observations.

The preview endpoint is a calculator. It does not read broker credentials, observe a live master event, persist copy intent, or call Rithmic. Control mutations change only durable internal intent. In particular, flatten-and-disable becomes `FLATTEN_REQUESTED`; only the future execution system may confirm broker-flat reconciliation and transition it to `DISABLED`.

Control restart behavior is explicit: disabled and locked followers remain held; enabled followers must reconcile before accepting new entries; paused followers resume management without new entries; and pending flatten requests reconcile and resume flattening. Global pause always removes every follower from new-entry eligibility. Audit and control reasons are fixed codes so free-form secrets cannot be written through these endpoints.

## Configuration

| Environment variable | Meaning | Safe committed value |
| --- | --- | --- |
| `V2_COPIER_MODE` | `OFF` or `SHADOW` in this phase | `SHADOW` |
| `V2_COPIER_CONNECTIVITY_ENABLED` | Explicit external-connectivity gate | `0` / disabled |
| `V2_COPIER_ENVIRONMENT` | Shared default for the five connection environments | `DISABLED` |
| `V2_COPIER_{SLOT}_ENVIRONMENT` | Optional per-account override; all five must match | unset / inherit |
| `V2_COPIER_{SLOT}_ACCOUNT_ID_SECRET_REF` | Name of the environment secret containing the exact account selector | fixed non-secret reference |
| `V2_COPIER_{SLOT}_USERNAME_SECRET_REF` | Name of the environment secret containing that connection's username | fixed non-secret reference |
| `V2_COPIER_{SLOT}_PASSWORD_SECRET_REF` | Name of the environment secret containing that connection's password | fixed non-secret reference |
| `V2_COPIER_{SLOT}_DISCOVERY_URI` | Account connection's secure Rithmic discovery endpoint | unset / blocked |
| `V2_COPIER_{SLOT}_SYSTEM_NAME` | Account connection's Rithmic system | unset / blocked |
| `V2_COPIER_{SLOT}_GATEWAY_NAME` | Optional explicit gateway selection | unset |
| `V2_RITHMIC_{SLOT}_{ACCOUNT_ID\|USERNAME\|PASSWORD}` | Default referenced secrets | unset secret |
| `V2_COPIER_MAX_RISK_FRACTION` | Hard per-trade equity-risk ceiling | `0` / blocked |
| `V2_COPIER_MAX_CONTRACTS` | Hard total quantity ceiling per follower plan | `0` / blocked |
| `V2_COPIER_MAX_CONCURRENT_RISK_FRACTION` | Maximum aggregate open stop risk | `0` / blocked |
| `V2_COPIER_DAILY_REALIZED_LOSS_FRACTION` | Daily realized-loss stop | `0` / blocked |
| `V2_COPIER_DAILY_TOTAL_LOSS_FRACTION` | Daily total-loss/drawdown stop | `0` / blocked |
| `V2_COPIER_MAX_RELATIVE_PARITY_ERROR` | Maximum relative sizing mismatch | `0` |
| `V2_COPIER_MARGIN_HEADROOM_FRACTION` | Available margin held back | `0.20` |
| `V2_COPIER_ALLOW_OVER_TARGET` | Permit consideration of quantities above parity | `0` |
| `V2_COPIER_NQ_INITIAL_MARGIN` | Current NQ margin per contract | `0` / blocked |
| `V2_COPIER_MNQ_INITIAL_MARGIN` | Current MNQ margin per contract | `0` / blocked |
| `V2_COPIER_NQ_COST_BUFFER` | Conservative NQ cost allowance per contract | `0` |
| `V2_COPIER_MNQ_COST_BUFFER` | Conservative MNQ cost allowance per contract | `0` |

`{SLOT}` is `MASTER`, `FOLLOWER_1`, `FOLLOWER_2`, `FOLLOWER_3`, or `FOLLOWER_4`. The status API reports only presence/readiness booleans and fixed connection labels; it never returns secret-reference names, selectors, usernames, passwords, or endpoint values. Shadow simulation does not require credentials. External connectivity requires all five complete, unique bundles, valid secret-free `wss://` discovery endpoints, matching `TEST` environments, and the explicit connectivity gate.

The repository deliberately does not hard-code website margin figures. The operator must verify and provide current account-specific values before each deployment or whenever the broker changes them. Account IDs and credentials must remain in secret management, not Git.

## Still required before copied orders

The repository does not yet contain:

- live master-order detection and attachment to the implemented lifecycle orchestration;
- the live Rithmic driver that feeds the implemented five-account supervisor and publishes its runtime health;
- configured credential-backed Rithmic Test order submission or production/live order authority;
- runtime feeding of real Rithmic reconnect snapshots into the implemented broker-authoritative recovery service;
- automatic protection repair after reconciliation (unexplained missing protection currently locks instead of guessing);
- actual-fill rebalancing or execution-quality reconciliation;
- real Rithmic Test/gateway end-to-end latency and deployment-region measurements; or
- completed Rithmic Test write conformance and production approval.

Low latency must come from a persistent event-driven Rithmic connection and immediate in-process planning, not polling the dashboard. Safety and lifecycle correctness remain release gates even when optimizing speed.

## Delivery and activation sequence

1. Completed: build the minimal operator UI over the implemented account, supervision, simulation, controls, cumulative-sizing, lifecycle, fenced recovery/reconciliation, latency, and Test-write foundation without waiting for credentials.
2. Next: complete the broader automated simulation and failure-injection acceptance suite for entries, stops, targets, partial fills, cancel/replace, scale changes, flatten, rejection, reconnect, follower controls, duplication, staleness, and crash recovery.
3. Obtain the five exact AMP/Rithmic credential bundles and confirm each account's API/session permissions and authorized operators.
4. Insert credentials only through secret management and complete the existing read-only Rithmic Test runbook for every independent connection.
5. Run the complete write-path scenario suite in Rithmic Test and verify actual lifecycle semantics, equity/margin facts, reconciliation, and latency.
6. Complete required Rithmic conformance and AMP approval.
7. Use one follower, MNQ, and the minimum accepted size for a supervised live canary before adding followers individually.

The official integration starting points are [Rithmic APIs and conformance](https://www.rithmic.com/apis), [AMP's Rithmic API page](https://www.ampfutures.com/trading-platform/rithmic-r-api), and the [read-only project runbook](RITHMIC_READ_ONLY_RUNBOOK.md).
