# MooneyCapitol implementation plan

Status: canonical forward plan

Updated: 2026-10-15

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

The credential-ready objective is complete in deterministic simulation. Account creation, secure credential insertion, and Rithmic Test validation are the next external gate.

## Current repository baseline

Implemented now:

- broker-neutral V2 futures domain, reference, calendar, Decimal math, and deterministic risk foundations;
- a direct R|Protocol 0.90 **read-only** observer with independent Order/PnL plants, recovery, journaling, reconciliation, and append-only projections;
- a pure one-master/four-follower Shadow planner for same-expiry NQ/MNQ stop-risk and point-exposure parity;
- typed, redacted secret-reference configuration for one master and four independent follower connections;
- deterministic supervision for five independent discoveries and ten Order/PnL sessions, with generation fencing, heartbeat/staleness tracking, failure isolation, and credential-free simulation;
- a durable fail-closed copy profile with four follower state machines, global pause, optimistic revisions, CSRF-protected control APIs, restart directives, and append-only audit history;
- cumulative total-position sizing across adds, reductions, closes, and reversals, with exact-contract pinning and per-follower one-MNQ funding-adequacy reports;
- durable per-master-event/per-follower lifecycle intents, deterministic command identities, monotonic partial-fill/terminal transitions, and close-before-open reversal ordering;
- an atomically enqueued durable command outbox, database-backed monotonic execution fence, abandoned-claim recovery, and append-only delivery evidence;
- broker-authoritative follower reconciliation contracts for positions, working orders, executions, and protection, with automatic follower locking on unexplained divergence;
- persisted event/decision/submission/response/acknowledgement/fill timing and aggregate or per-follower p50/p95/p99 reporting;
- a separate official-schema R|Protocol write boundary for market, limit, stop, bracket/OCO, modify, cancel, and position-exit requests, limited in code to `OFF`, `SHADOW`, and explicitly authorized Test use;
- authenticated `GET /v2/copier/status` and non-persisting `POST /v2/copier/preview` endpoints;
- an authenticated premium operator command center with five-account readiness, revision-checked copy controls, typed flatten confirmation, parity/lifecycle/recovery evidence, redacted immutable audit history, responsive layouts, and read-only broker observations;
- a credential-ready release gate with a fixed requirement-to-test manifest, four-follower concurrent dispatch evidence, rejection failure injection, and local JUnit output;
- fail-closed environment and Render defaults; and
- `conner_nq_v1` observation, labeling, replay, ranking-contract, and learning-data foundations with no trained or live model.

Not implemented yet:

- the live Rithmic connection driver and runtime-health attachment for the five supervised account connections;
- live master-event ingestion and runtime attachment of lifecycle orchestration;
- configured external write bindings/transport and credential-backed Rithmic Test submission;
- runtime attachment of the implemented recovery/reconciliation worker to real Rithmic snapshots and Test writes;
- real gateway/region copy-latency measurements;
- Rithmic Test, conformance, or live validation.

The current read-only observer remains structurally read-only. The new write capability is a separate execution adapter/service boundary and does not weaken the observer's outbound template denylist. The normal execution process still rejects submission configuration and does not attach the Test writer.

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

Status: **complete**

- Supervise independent Order and PnL sessions per credential bundle.
- Track authentication, heartbeat, reconnect generation, discovery, readiness, and staleness independently.
- Normalize master events and follower account state behind broker-neutral interfaces.
- Prevent one failed follower from blocking healthy followers.
- Block all new copying when the master or required master state is stale.
- Provide deterministic simulated connections and protocol fixtures for credential-free development.

### Phase 3 — Durable copy control plane

Status: **complete**

- Add copy profile, account membership, follower state, control revision, and immutable audit records.
- Implement authenticated, CSRF-protected, revision-checked control APIs.
- Add enabled, paused, flatten-and-disable, locked, and global-pause transitions.
- Serialize commands per account and reject stale or conflicting operator actions.
- Define restart behavior for every control state and open-position condition.

### Phase 4 — Production sizing and funding adequacy

Status: **complete**

- Extend the current pure Shadow planner rather than duplicating its formulas.
- Use authoritative current equity, available margin, open stop risk, daily P&L/drawdown, and exact contract facts.
- Plan cumulative target positions and follower deltas across adds, reductions, and reversals.
- Preserve same-expiry NQ/MNQ mapping and never roll an open copied position automatically.
- Produce a funding-adequacy report showing when a follower cannot safely trade one MNQ.
- Record target versus achievable exposure and the reason for every skip or block in each immutable sizing result; Phase 5 persists those facts with durable copy intents.

### Phase 5 — Durable lifecycle copier and Rithmic write boundary

Status: **complete**

- Create durable, revisioned copy intents before follower actions.
- Correlate each master lifecycle with each independent follower lifecycle.
- Add idempotent command handling and duplicate-event protection without treating Rithmic tags as guaranteed broker idempotency.
- Implement supported market, limit, stop, bracket/OCO, cancel/replace, partial-fill, scale, exit, rejection, and reversal behavior.
- Keep follower failures isolated and report exact divergence.
- Build the write adapter from the official external Rithmic materials behind `OFF`, `SHADOW`, and Test-only gates.
- Preserve a compile/runtime boundary that makes live activation impossible until later authorization gates exist.

Implementation note: migration `0013_v2_copy_lifecycle` stores revisioned intents, ordered commands, all broker IDs returned for multi-order requests, and append-only lifecycle events. Deterministic local IDs prevent command recreation on duplicate source events; Rithmic `user_tag`/`user_msg` remain correlation evidence only. A separate external-binding generator covers only the official write request/response schemas. The adapter contains no live mode, Test construction requires an explicit authorization gate plus injected codec/transport, and the production execution entry point remains submission-disabled.

### Phase 6 — Reconciliation, recovery, fencing, and latency

Status: **complete in credential-free code and deterministic simulation; real gateway/region measurement remains a Phase 9 validation**

- Treat broker state as authoritative after reconnect or uncertainty.
- Reconcile positions, working orders, protection, executions, and copy-intent revisions.
- Lock a follower on unexplained divergence rather than guessing.
- Add a durable outbox, crash recovery, and a production distributed lease/fencing token so only one active instance can submit.
- Keep persistent event-driven connections and four-way concurrent follower dispatch off the dashboard/HTTP path.
- Capture event, decision, submission, acknowledgement, and fill timestamps and report p50/p95/p99 latency.
- Benchmark deployment regions and keep the execution service and durable store appropriately close to the selected Rithmic gateway.

Implementation note: migration `0014_v2_execution_recovery` atomically pairs every new lifecycle command with a durable outbox record and timing projection, backfills pre-existing commands fail-closed, and protects delivery/reconciliation evidence with append-only PostgreSQL triggers. Claims require the current database-backed lease owner and monotonically increasing fence. A stale owner cannot complete a write after takeover; abandoned or transport-ambiguous commands become `UNCERTAIN` and cannot be resent. The submission-start milestone is committed with the fenced claim immediately before the adapter call, avoiding a second pre-send database round trip; actual socket/write timing remains a Rithmic Test driver measurement. A complete, fresh broker snapshot must explain the order, execution, position, and protective-stop state. Explained broker progress is adopted into the lifecycle; missing, unknown, or conflicting state marks the command divergent and locks only that follower. `python -m app.tools.v2_copy_latency_report` reports persisted p50/p95/p99 milestones without exposing broker account identifiers. Actual gateway/region comparison requires the future five credential bundles and Rithmic Test connectivity and is therefore an external Phase 9 measurement, not a simulated claim.

### Phase 7 — Minimal functional operator UI

Status: **complete**

Before the aesthetic redesign, add only the controls and evidence needed to operate and test safely:

- master and four follower connection/readiness cards;
- individual follower controls and global pause;
- explicitly separated flatten controls with confirmation;
- current equity, margin, daily return, position, and protection state;
- target versus actual quantity, exposure, stop risk, and parity error;
- latest copy lifecycle, latency, rejection, block, and reconciliation status; and
- redacted immutable control/audit history.

Implementation note: the authenticated `/dashboard` merges the fixed five-slot readiness topology with durable broker observations and per-follower copy execution projections. It shows configuration separately from actual Order/PnL plant readiness, leaves unavailable broker facts explicitly unavailable, and links accounts to fixed internal connection labels without exposing broker selectors. Global pause and follower enable/pause/unlock requests use the existing optimistic revisions, session CSRF proof, reason codes, and append-only audit store. Flatten-and-disable remains a separate typed-confirmation flow and results in `FLATTEN_REQUESTED`; the UI never claims an account is flat before the execution system confirms it. The dashboard reads lifecycle, parity, protection-policy, outbox, reconciliation, finding, and latency projections, but it cannot construct an execution adapter or submit to Rithmic. Audit actor/request identifiers and broker object identifiers are deterministic aliases in browser/API output.

### Phase 8 — Credential-ready acceptance suite

Status: **complete in credential-free simulation; Rithmic Test remains Phase 9**

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

Implementation note: `python -m app.tools.v2_credential_ready_acceptance` runs the fixed acceptance manifest and writes an ignored local JUnit report. The gate covers all 15 requirements above through 37 production-component tests, including all four follower adapters entering dispatch concurrently with persisted timing evidence. It also adds a lifecycle invariant that prevents a rejected master event from creating or increasing follower exposure. See [V2_CREDENTIAL_READY_ACCEPTANCE.md](V2_CREDENTIAL_READY_ACCEPTANCE.md) for the matrix and the boundary between simulated credential readiness and broker validation.

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

Status: **credential-free visual/hierarchy pass complete early; populated workflow refinement follows Phase 9 validation**

- Rework information architecture around real operator workflows.
- Separate trading operations, risk, accounts, performance, history/audit, research/learning, settings, and administration.
- Establish a consistent visual system, responsive behavior, accessible controls, and clear healthy/warning/blocked states.
- Keep the execution hot path independent of dashboard availability.

Implementation note: while credentials were pending, the dashboard and login were rebuilt around a consistent dark institutional command-center system. The opening view now prioritizes system posture, copy gate, account network, capital/readiness facts, follower controls, and exceptions. All five accounts share a deliberate responsive grid; follower actions are presented as an immediate control deck; dense capture, configuration, account, activity, risk, contract, and boundary evidence remains available in compact technical drawers. Empty and credential-pending states stay explicit rather than displaying fabricated balances, performance, or activity. The authenticated APIs, CSRF/revision checks, typed flatten confirmation, audit behavior, and isolated execution boundary are unchanged. Real Rithmic Test data will drive the later populated-state usability refinement and any deeper page separation.

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

Current checkpoint: Phase 8 credential-ready acceptance complete in deterministic simulation. The fixed release gate passed all 37 selected tests, including four-way concurrent dispatch/timing, independent follower controls, rejection isolation, recovery/fencing, stale-state handling, security/redaction, and proof that live submission remains impossible. The separate official-schema adapter still has no live mode and is not attached to the dashboard or production execution runtime.

Next gated task: Phase 9 credential insertion and Rithmic Test validation after the five credential bundles, exact account selectors, licensed bindings, and broker approval are available through secure channels.

Do not treat simulated credential readiness as broker validation or live readiness. Rithmic Test, conformance, measured gateway latency, and a later explicitly authorized live canary remain separate gates.
