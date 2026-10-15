# MooneyCapitol implementation plan

Status: canonical forward plan

Updated: 2026-10-14

## Authority

This document records the agreed product direction and implementation order. It is the source of truth for what is built next. Detailed architecture and runbooks remain authoritative within their narrower technical scopes, but they must not contradict the topology, priorities, or release gates here.

Historical V1 documents are retained for reference only. They do not define V2 broker choices, account topology, execution behavior, learning targets, dashboard direction, or readiness.

## Locked product definition

MooneyCapitol will operate a custom, real-time futures copier across five operationally independent AMP accounts:

- one account is the master;
- four accounts are followers;
- every account has its own exact broker account identity and independently usable credential/connection configuration;
- the software does not require the accounts to be AMP subaccounts, share one legal entity, share one portal, or share one Rithmic user;
- the background service observes the master and submits separate orders to each enabled follower through the broker-authorized API path;
- the initial execution products are NQ and MNQ; and
- the objective is to make follower percentage exposure, stop risk, and resulting returns track the master as closely as integer futures contracts and real fills permit.

Rithmic is the currently selected connectivity and order-routing layer for AMP. It is not the account owner, copier product, or account-grouping mechanism. The application must isolate each account connection even if AMP later chooses to issue credentials that can see more than one account.

All accounts and operators must be authorized by AMP/Rithmic. The code must not infer authorization from ownership, a shared name, a credential, or account discovery.

## Meaning of ready

The project uses three distinct readiness terms:

1. **Credential-ready:** the complete copier is implemented, simulated, failure-tested, and ready to receive five secret credential bundles.
2. **Broker-validated:** the copier has passed the documented scenario suite against Rithmic Test using actual credentials and observed protocol behavior.
3. **Live-ready:** required conformance and permissions are complete, a minimum-size live canary has passed, and the operator has explicitly approved staged expansion.

The immediate objective is credential-ready. Account creation and credentials are parallel external work, not a reason to delay implementation.

## Current repository baseline

Implemented now:

- broker-neutral V2 futures domain, reference, calendar, Decimal math, and deterministic risk foundations;
- a direct R|Protocol 0.90 **read-only** observer with independent Order/PnL plants, recovery, journaling, reconciliation, and append-only projections;
- a pure one-master/four-follower Shadow planner for same-expiry NQ/MNQ stop-risk and point-exposure parity;
- typed, redacted secret-reference configuration for one master and four independent follower connections;
- authenticated `GET /v2/copier/status` and non-persisting `POST /v2/copier/preview` endpoints;
- a read-only operations dashboard that shows capture and Shadow readiness;
- fail-closed environment and Render defaults; and
- `conner_nq_v1` observation, labeling, replay, ranking-contract, and learning-data foundations with no trained or live model.

Not implemented yet:

- five independently supervised credential/connection sessions;
- persistent follower enable/pause/flatten controls;
- live master lifecycle orchestration;
- durable copy intents and master/follower order correlation;
- a Rithmic write adapter or any follower broker mutation;
- cumulative target-position sizing across adds and reductions;
- write-path crash recovery, production fencing, and execution reconciliation;
- measured copy latency;
- the functional copier control UI; or
- Rithmic Test, conformance, or live validation.

The current read-only observer must remain structurally read-only. Write capability will be added through a separate execution adapter/service boundary rather than weakening its outbound template denylist.

## Safety and behavior contract

### Follower controls

The user-facing control may be presented simply, but the internal states must distinguish:

- **Enabled:** accept new copy entries and manage existing copied exposure.
- **Paused:** accept no new entries but continue stop, target, reduction, and exit management until existing copied exposure is flat.
- **Flatten and disable:** cancel relevant working orders, request a close of copied exposure, reconcile broker state, and remain disabled.
- **Locked:** fail-closed state caused by stale data, risk breach, reconciliation mismatch, rejected protection, connection loss, or operator lock.

A separate global pause blocks all new follower entries. Emergency flatten is an explicit, confirmed, audited action and is never the implicit meaning of an ordinary off toggle.

### Sizing objective

For a new master plan, the starting point is:

```text
target follower contracts =
    master contracts
    * follower equity / master equity
    * master point value / follower point value
```

The executable quantity must be an integer. The engine evaluates same-expiry NQ/MNQ candidates and minimizes the larger mismatch between:

- stop-defined risk as a fraction of account equity; and
- dollars-per-point exposure as a fraction of account equity.

The production engine will size toward the desired **total follower position** and submit only the required delta. This avoids accumulating rounding error when the master scales in or out through multiple orders.

By default, the system does not round a follower above the master's percentage target. If no safe one-contract result exists, that follower is skipped and the reason is displayed. Broker margin is a feasibility constraint, not the risk definition. Exact realized daily ROI is not promised because contracts are indivisible and fills, slippage, fees, rejections, and connection timing can differ.

Daily comparison uses each account's CME trade-date starting equity. New trade plans use current authoritative equity. An open plan keeps an immutable initial risk denominator; later P&L does not rewrite that trade's original risk.

### Stops and protection

The default release rule is that follower exposure requires a known valid protective stop. If the trader's real workflow sometimes adds a stop after entry, that does not silently relax the rule. A provisional emergency-stop policy would require a separate explicit decision, implementation, and Test validation.

### Execution independence

Scout rankings, learned conviction, OpenAI output, and future agents cannot size, approve, modify, or submit follower orders. Execution consumes deterministic account, contract, lifecycle, risk, and operator-control facts only.

## Ordered implementation plan

### Phase 0 — Documentation and architecture correction

Status: **complete**

- Record the five-independent-account topology.
- Remove shared-entity, AMP-subaccount, and single-Rithmic-user assumptions from active documentation.
- Preserve a clear implemented-versus-planned boundary.
- Mark V1 roadmap, XGBoost, and prior AI decisions as historical where appropriate.

### Phase 1 — Independent account and secret configuration

Status: **complete**

- Replace the interim account-ID list with one master and four typed account connection records.
- Give every account an opaque internal ID, exact broker account selector, environment, role, and secret reference.
- Support distinct Rithmic username/password/system/gateway settings per credential bundle.
- Keep raw credentials out of PostgreSQL, logs, API responses, browser storage, and Git.
- Validate uniqueness, exact role count, environment consistency, and account allowlists fail-closed.
- Make incomplete credentials a readiness blocker without preventing simulation.

### Phase 2 — Multi-account connection supervision

Status: **planned**

- Supervise independent Order and PnL sessions per credential bundle.
- Track authentication, heartbeat, reconnect generation, discovery, readiness, and staleness independently.
- Normalize master events and follower account state behind broker-neutral interfaces.
- Prevent one failed follower from blocking healthy followers.
- Block all new copying when the master or required master state is stale.
- Provide deterministic simulated connections and protocol fixtures for credential-free development.

### Phase 3 — Durable copy control plane

Status: **planned**

- Add copy profile, account membership, follower state, control revision, and immutable audit records.
- Implement authenticated, CSRF-protected, revision-checked control APIs.
- Add enabled, paused, flatten-and-disable, locked, and global-pause transitions.
- Serialize commands per account and reject stale or conflicting operator actions.
- Define restart behavior for every control state and open-position condition.

### Phase 4 — Production sizing and funding adequacy

Status: **planned**

- Extend the current pure Shadow planner rather than duplicating its formulas.
- Use authoritative current equity, available margin, open stop risk, daily P&L/drawdown, and exact contract facts.
- Plan cumulative target positions and follower deltas across adds, reductions, and reversals.
- Preserve same-expiry NQ/MNQ mapping and never roll an open copied position automatically.
- Produce a funding-adequacy report showing when a follower cannot safely trade one MNQ.
- Store target versus achievable exposure and the reason for every skip or block.

### Phase 5 — Durable lifecycle copier and Rithmic write boundary

Status: **planned**

- Create durable, revisioned copy intents before follower actions.
- Correlate each master lifecycle with each independent follower lifecycle.
- Add idempotent command handling and duplicate-event protection without treating Rithmic tags as guaranteed broker idempotency.
- Implement supported market, limit, stop, bracket/OCO, cancel/replace, partial-fill, scale, exit, rejection, and reversal behavior.
- Keep follower failures isolated and report exact divergence.
- Build the write adapter from the official external Rithmic materials behind `OFF`, `SHADOW`, and Test-only gates.
- Preserve a compile/runtime boundary that makes live activation impossible until later authorization gates exist.

### Phase 6 — Reconciliation, recovery, fencing, and latency

Status: **planned**

- Treat broker state as authoritative after reconnect or uncertainty.
- Reconcile positions, working orders, protection, executions, and copy-intent revisions.
- Lock a follower on unexplained divergence rather than guessing.
- Add a durable outbox, crash recovery, and a production distributed lease/fencing token so only one active instance can submit.
- Keep persistent event-driven connections and four-way concurrent follower dispatch off the dashboard/HTTP path.
- Capture event, decision, submission, acknowledgement, and fill timestamps and report p50/p95/p99 latency.
- Benchmark deployment regions and keep the execution service and durable store appropriately close to the selected Rithmic gateway.

### Phase 7 — Minimal functional operator UI

Status: **planned**

Before the aesthetic redesign, add only the controls and evidence needed to operate and test safely:

- master and four follower connection/readiness cards;
- individual follower controls and global pause;
- explicitly separated flatten controls with confirmation;
- current equity, margin, daily return, position, and protection state;
- target versus actual quantity, exposure, stop risk, and parity error;
- latest copy lifecycle, latency, rejection, block, and reconciliation status; and
- redacted immutable control/audit history.

### Phase 8 — Credential-ready acceptance suite

Status: **planned**

The system is credential-ready only after automated simulation and failure injection cover:

- all supported entry and lifecycle types;
- independent follower enable/pause/flatten behavior;
- stop and bracket protection;
- cumulative rounding across scale-ins and scale-outs;
- underfunded followers and insufficient margin;
- master and follower rejection;
- partial fills and out-of-order/duplicate events;
- credential/account mismatch;
- disconnect, reconnect, service crash, and stale equity/P&L;
- active-instance takeover/fencing;
- contract-expiry mismatch and no automatic roll;
- secret and account redaction;
- authorization, CSRF, revision conflict, and audit behavior; and
- latency instrumentation under concurrent follower load.

The repository must remain unable to activate live order submission at this milestone.

### Phase 9 — Credential insertion and broker validation

Status: **blocked only on future external credentials/approval**

- Insert secrets through deployment secret management, never source or chat.
- Confirm the five exact AMP account selectors and actual credential topology.
- Validate independent Test authentication, discovery, session limits, equity, margin, order lifecycle, brackets, RMS/P&L, reconnect, and reconciliation.
- Compare observed protocol behavior with the official materials and correct assumptions.
- Complete required Rithmic conformance and AMP authorization.
- Measure real event-to-submit, acknowledgement, and fill latency.

### Phase 10 — Staged live canary

Status: **future gate**

- Require explicit owner authorization separate from configuration.
- Begin with one follower, MNQ, minimum accepted size, narrow hours, and an operator present.
- Verify entry, protection, modification, reduction, exit, reconciliation, and statement results.
- Add followers one at a time only after clean review.
- Retain immediate pause and emergency controls plus a documented rollback procedure.

### Phase 11 — Dashboard structure and aesthetics

Status: **after the copier is operationally proven**

- Rework information architecture around real operator workflows.
- Separate trading operations, risk, accounts, performance, history/audit, research/learning, settings, and administration.
- Establish a consistent visual system, responsive behavior, accessible controls, and clear healthy/warning/blocked states.
- Keep the execution hot path independent of dashboard availability.

### Phase 12 — Learning and Conner-focused AI agents

Status: **later**

Model development waits, but trustworthy data collection begins with the copier. Preserve master decisions, pre-trade state, plan versions, lifecycle timing, fills, follower variance, slippage, latency, outcomes, skips, and data-quality failures from day one.

After sufficient prospective data exists:

- build chronological behavior and outcome baselines;
- evaluate candidate/ranking features without importing V1 labels or synthetic decisions;
- expose learning results as read-only research and review tools; and
- add read-only Conner-focused agents for status explanation, trade review, journaling, historical retrieval, execution discrepancy analysis, and performance summaries.

Agents do not enter the execution dependency graph. Any future action capability requires a new explicit scope, permission model, audit design, and safety review.

## Information needed when credentials arrive

For each of the five independent accounts, collect through a secure channel:

- AMP account identifier and exact Rithmic account selector;
- Rithmic username and secret reference;
- Test and live environment/system names;
- authorized gateway/endpoint discovery settings;
- API enablement and applicable session limits;
- current margin requirements and account-specific restrictions; and
- confirmation of authorized operators and intended trading frontend/workflow.

Also finalize owner-approved per-trade risk, concurrent-risk, daily realized-loss, daily total-loss, maximum-contract, margin-headroom, parity-tolerance, and over-target policies. The recommended over-target default remains disabled.

## Progress checkpoint

Current checkpoint: Phase 1 independent account and secret-reference configuration complete. The code now has five fixed typed connection slots, per-account environment/endpoint/system/gateway settings, indirect secret resolution, uniqueness validation, redacted status output, and separate simulation versus Test-connectivity readiness.

Next implementation task: Phase 2, deterministic multi-account connection supervision and simulated connections without broker credentials.

Do not skip directly to UI polish, learning, AI agents, Rithmic Test, or live activation while an earlier execution milestone remains incomplete.
