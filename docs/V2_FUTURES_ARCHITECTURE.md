# MooneyCapitol V2 futures architecture

Status: execution submission disabled; Rithmic read-only capture and `conner_nq_v1` observation foundations implemented
Canonical as of: 2026-10-07

## Purpose and boundary

V2 supplies broker-neutral futures execution boundaries and provider-neutral market-data/intelligence boundaries without inheriting V1 equity assumptions. V1 and V2 coexist, but V1 stock features, labels, synthetic rows, `hybrid_target`, paper outcomes, artifacts, provider symbols, and calendar logic cannot enter V2 futures learning.

The strategy-specific work implemented so far is an observational namespace named `conner_nq_v1`. It represents synchronized NQ/ES state, research measurements, honest learning labels, Shadow rankings, and evaluation. It is not a hand-coded implementation of Conner's discretionary strategy and does not affect execution.

See [CONNER_NQ_SCOUT_FOUNDATION.md](CONNER_NQ_SCOUT_FOUNDATION.md) for the detailed strategy-observation contract and [RITHMIC_READ_ONLY_RUNBOOK.md](RITHMIC_READ_ONLY_RUNBOOK.md) for capture deployment and Test validation.

## Dependency direction

```text
Market-data plane ---> Intelligence plane ---> Analysis/operations plane
        |                                          ^
        +------------> Execution plane ------------+
```

Execution may consume validated contract/session facts and write immutable operational facts. It must not consume Scout predictions, OpenAI output, V1 learning state, or analysis summaries. Intelligence is read-only with respect to execution. Analysis may read journaled facts but cannot mutate broker/order state.

## Execution plane

The execution plane owns broker connections/accounts, normalized lifecycle events, deterministic sizing, copy intent, follower lifecycle, positions, account risk, reconciliation, journal enqueueing, account serialization, and active-instance fencing.

Hard constraints include:

- integer contract quantities;
- `Decimal` at domain boundaries and `NUMERIC(24,10)` in storage;
- zero or missing hard ceilings mean no authority;
- economic stop risk and broker margin remain separate constraints;
- global/account switches and daily loss ceilings are evaluated before sizing;
- model output can never weaken a hard risk limit;
- lifecycle revisions are monotonic and terminal order states cannot roll backward.

The execution process remains a boundary skeleton. Immutable configuration rejects submission enablement, readiness remains false, and the execution-side Rithmic placeholder's submission methods raise. No work in `conner_nq_v1` changes that limitation.

### Rithmic observation boundary

The separate `app/v2/brokers/rithmic_protocol` adapter and `app/v2/capture` service implement production-shaped read-only R|Protocol 0.90 capture. R|API+/.NET is a fallback/conformance reference only. The adapter uses external checksum/manifest-verified protobuf bindings and secure binary WSS with TLS certificate and hostname verification.

Order Plant and PnL Plant have independent sockets, authentication, heartbeat, generation, timestamps, reconnect, and readiness state. Account discovery preserves FCM, IB, account, user/access/status/currency, and session-limit facts, but subscription is permitted only for exact configured account IDs. Opaque broker IDs are never reinterpreted as Mooney ownership. Origin metadata and tags remain evidence; they are not guaranteed ownership or idempotency keys.

For each new connection generation, the service subscribes to live Order/bracket/RMS/PnL streams for every allowlisted account before requesting current orders, execution replay, order/fill history, brackets/stops, RMS, and PnL snapshots. Live events are buffered during recovery, all inputs are durably journaled and deduplicated, and readiness requires clean checkpoints for every configured account on both required plants. Failed recovery is not repeatedly polled in the same generation.

The observer has no order, modify, cancel, flatten, exit, bracket/OCO send, link, target/stop change, or follower method. A default-deny outbound template guard runs at the final transport boundary and rejects all known broker-mutation templates. This is a structural boundary, not an operator convention or environment flag.

Optional Ticker Plant request/response support provides exact symbol, exchange/product, expiration, tick-table, point-value, and tradability facts. Normal capture requires only Order and PnL once exact contract mappings are validated. Continuous Rithmic market-data streaming, History Plant strategy data, and Scout integration are outside this phase.

## Contract reference and instrument roles

An internal futures identity includes exchange, product, and expiration. Bare product codes are not live contract identities.

Static product economics currently registered are:

| Product | Research/execution role | Point value | Tick size | Tick value |
| --- | --- | ---: | ---: | ---: |
| NQ | Conner traded instrument; copier source candidate | $20 | 0.25 | $5 |
| MNQ | Explicit follower/equivalence candidate | $2 | 0.25 | $0.50 |
| ES | `conner_nq_v1` reference/context only | $50 | 0.25 | $12.50 |

NQ/MNQ equivalence is an explicit same-expiry mapping and never rolls an open trade. ES is not added to that mapping. A research observation instead stores one exact NQ contract and one exact ES contract as a versioned contextual pair with independent provider bindings and lineage.

Expiry symbols, first/last trade dates, holiday schedules, early closes, and availability remain dynamic reference/provider facts. CME operational accounting retains its trade-date calendar and maintenance behavior; configurable strategy-time windows do not replace it.

## Market-data plane

The provider-neutral boundary exposes exact contract reference, schedules, bars, trades, BBO, depth, MBO, historical replay, and live-streaming capability names. Consumers must require only the capabilities they actually need. `conner_nq_v1` requires reference, schedules, bars, trades, BBO, and historical replay; depth and MBO are deliberately not required.

Typed market events preserve:

- exact internal contract and provider symbol binding;
- event/source and received timestamps;
- an explicit availability timestamp and availability mode;
- provider sequence, nanosecond source time, event ID, revision, and finality;
- endpoint/request/schema/retrieval lineage;
- gap, lateness, and correction quality flags.

### Massive Futures REST

`app/v2/providers/massive_futures.py` is the first historical research adapter. Behind explicit entitlements, it supports exact-contract reference, schedule events, fixed-resolution bars, trades, BBO, and multi-contract historical replay. It validates exact bindings, response identity, historical windows, pagination origin, payload types, cutoff-complete bars, and deterministic output ordering.

Authentication, HTTP lifecycle, secrets, and retry behavior are injected through `AsyncJsonTransport`; no credential-bearing production transport exists in the repository. The entitlement profile defaults to disabled. There is no demo fallback and the REST adapter neither advertises nor implements live streaming.

Massive historical REST events are tagged `FINALIZED_HISTORICAL`. This adapter can support reconstruction and outcome research, but those events cannot be silently relabeled as point-in-time Conner behavior evidence. Massive remains replaceable behind the market-data interface.

## Intelligence plane: `conner_nq_v1`

### Synchronized observation

Every observation requires exact NQ and ES contracts sharing one feature cutoff and purpose. Backward/as-of synchronization keeps requested timeframes separate, selects only eligible facts at or before the cutoff, enforces versioned event-age and cross-leg-skew limits, and reports missing, late, or stale series.

The system distinguishes event time, availability/knowledge time, received/retrieved time, feature cutoff, computation time, and feature-definition availability time. Point-in-time measurements require both maximum input event time and maximum input availability time to be no later than the cutoff. Later historical computation is allowed; future input knowledge is not.

Finalized history is allowed only for outcome research or explicitly finalized reconstruction. It is barred from behavior-training observations and cannot fabricate Conner actions.

### Research measurements, not rules

The candidate measurement foundation includes versioned candle geometry and multi-candle relationships, NQ/ES swing and relative-structure candidates, SMT/divergence candidates and persistence, Fibonacci candidates, candidate levels, volatility context, and time context.

Reversal levels, SMT, Fibonacci relationships, candle rejection structure, and possible New York windows are research clues. No detector is claimed to be Conner's rule, and no combination is treated as a complete strategy.

### Horizons and windows

Pre-decision horizons are stored as versioned sets of unique offsets. The code does not hard-code the suggested 30/15/10/5/3/1-minute sequence. Each materialized snapshot preserves target cutoff, actual cutoff, alignment error, origin, and lineage.

Possible New York windows are versioned research candidates with configurable timezone, anchor, duration, and parameters. There is no permanent assumption about what “first three hours” means and no reuse of V1 equity regimes.

### Learning tasks and labels

Four tasks remain isolated:

1. behavior/imitation;
2. setup/outcome quality;
3. conviction behavior;
4. copier execution quality.

Each task retains its own namespace, schema/dataset versions, target semantics, lineage, metrics, and artifact lifecycle. Follower execution quality cannot label strategy quality. Learned conviction cannot control follower risk.

Behavior data uses positive/unlabeled semantics. Opportunities default to unknown exposure, unknown consideration, and `UNLABELED`. A real master trade with lifecycle evidence can create a positive. A negative is accepted only as an evidenced explicit pass after confirmed exposure and consideration. An unmatched candidate is not an automatic negative. Query groups and evidenced pairwise preferences are available for later PU/ranking research.

Learning validation also rejects target-as-feature, outcome/future inputs at cutoff, follower fields in strategy tasks, implicit missing-to-zero conversion, projected R:R as realized R, invalid temporal lineage, finalized history as behavior, and V1/equity/sandbox/synthetic sources.

No learning task has a trained model in this phase.

### Ranking and Shadow

Ranking contracts store behavior, outcome, and optional later conviction outputs separately, with confidence, data quality, model versions, candidate-generator version, and ranking-policy version. Copier execution output cannot rank strategy opportunities. An optional display score is not a training target.

Rankings are constrained to Shadow-only and never execution-eligible. Shadow evaluation supports actual-trade rank and Top 1/3/5 presence, direction match, prospective lead time, R metrics when an original denominator exists, unlabeled unmatched candidates, explicit rejection evidence, data-quality failure tracking, and chronological walk-forward contracts.

Retrospective trade lead-up snapshots may support later training but cannot claim prospective detection or lead time.

## Replay

The strategy-specific replay request contains an exact NQ/ES pair, half-open interval, required event kinds, replay mode, and version. It deterministically merges the two event streams using knowledge/event time and stable provider ordering fields.

Blind replay accepts only point-in-time facts and is behavior-research eligible. Finalized reconstruction accepts only finalized history and is behavior-ineligible. The dispatcher has no strategy, label, fill, risk, sizing, or order-submission hook.

## Analysis/operations plane

Immutable plan versions preserve the original stop and risk denominator. Later adds, scale-outs, stop changes, target changes, and exits are separate lifecycle/management facts. If no original stop was actually observed, original risk and R-denominated metrics remain null.

Actual Conner trade labels state whether their anchor is a decision, first order submission, or first fill. They preserve exact NQ/ES context and lifecycle evidence rather than calling a fill timestamp a decision. Analysis/OpenAI may eventually summarize these stored facts but cannot control execution.

## Persistence

Migration `0009_v2_futures_foundation` adds the broker-neutral execution/reference journal tables.

Migration `0010_conner_nq_scout` adds 13 observation/learning tables for specifications, horizons, synchronized NQ/ES states, candidate feature definitions/values, opportunities, revisioned behavior evidence, lead-up snapshots, isolated learning examples, Scout runs/rankings, Shadow evaluation, and chronological evaluation runs.

Migration `0011_rithmic_read_capture` adds 11 tables for independent connection generations, replay batches, immutable broker events, account/order/execution/bracket/reference/P&L/RMS observations, and reconciliation checkpoints. BigInteger columns retain official 64-bit quantities; Numeric columns retain monetary facts without inventing a generic equity interpretation. Native IDs, unknown statuses, source/provenance, fill busts/corrections, replay terminals, and account metadata are append-only. PostgreSQL triggers reject update/delete operations on broker history.

All three migrations are additive. V1 records are not altered or converted. The `0010` constraints encode cutoff ordering, point-in-time availability, explicit missingness, unknown/unlabeled defaults, evidenced positive/pass labels, task isolation, Shadow-only outputs, lead-time rules, and walk-forward chronology.

A PostgreSQL 16 upgrade/downgrade/re-upgrade verified the full chain, confirmed that the `0010` downgrade removes only its 13 tables, and produced zero V2 metadata drift.

The Rithmic capture process has a durable journal/projection writer. The schema does not yet provide a durable high-volume strategy market-data ingestion process or production Scout repository/orchestrator; those remain next-phase components.

## Deterministic risk selection

Supported sizing modes are `FIXED_DOLLAR`, `PERCENT_EQUITY`, `MASTER_RISK_MULTIPLIER`, `FIXED_CONTRACTS`, and `DISABLED`. Quantity is constrained by sizing budget, maximum risk, remaining concurrent risk, global/product contract limits, margin capacity after headroom, and deterministic hard-ceiling rounding.

`AUTO_NQ_MNQ` accepts only exact same-expiry NQ/MNQ candidates. ES context and every Scout/model output are outside this sizing decision.

## Intentionally absent

The repository still has no:

- completed Rithmic Test conformance or any broker order submission/mutation;
- complete Conner strategy or handcrafted trade rule;
- live Scout service or live candidate-generation loop;
- trained futures model, selected ranking objective, or promoted artifact;
- model-controlled execution or follower sizing;
- live Massive stream or permanent market-data-provider lock-in;
- automatic contract rolling;
- durable distributed lease, journal worker, or high-volume market-data capture service.

## Exact next step

Deploy the dedicated Rithmic capture service offline, deliver the externally generated licensed bindings through secret management, and follow the read-only runbook against Rithmic Test. Validate independent Order/PnL health, account allowlisting, manual R|Trader-to-API visibility, lifecycle/fill/P&L/RMS/bracket facts, replay overlap, reconnect, redaction, and reconciliation before considering a separate paper-execution phase.

In parallel, configure and validate an authenticated strategy market-data transport, exact NQ/ES provider bindings, and explicit entitlements. Begin durable point-in-time NQ/ES capture; the historical Massive REST adapter must retain its finalized-history semantics rather than being presented as live behavior evidence. Once prospective data is accumulating, run Shadow ranking records and attach observed master actions, misses, lead times, ranks, and data-quality failures. Only after enough trustworthy labels exist should chronological behavior/outcome baselines be trained and evaluated. Production readiness has not been declared, and all Rithmic broker mutation remains disabled.
