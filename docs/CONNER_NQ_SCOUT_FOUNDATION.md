# Conner NQ Scout observation foundation

Status: observation and learning-data foundation implemented; no trained or live Scout
Namespace: `conner_nq_v1`
Updated: 2026-10-05

## Purpose

This phase creates the contracts, validation, provider boundary, replay behavior, and persistence needed to learn from Conner's future NQ decisions. It does not turn the partial strategy description into a hand-coded trading system.

The eventual research question is: which developing NQ opportunities most resemble situations Conner chooses to trade, and what outcomes have followed comparable market states? The current code can represent and evaluate the evidence needed to answer that question later. It does not yet produce a live ranking or train a model.

## Instrument roles

`conner_nq_v1` has two explicit research roles:

- NQ is the traded/prediction instrument.
- ES is a synchronized contextual instrument.

Both legs require expiration-aware internal identities such as `CME:NQ:2026-12-18` and `CME:ES:2026-12-18`. Bare `NQ` and `ES` are invalid observation identities.

ES is reference and research context. It is not an execution substitute for NQ and is not part of the NQ/MNQ copier mapping. The only product-equivalence mapping remains explicit, same-expiry NQ to MNQ or MNQ to NQ. NQ and ES contract selection is stored separately as a versioned research pair with lineage.

## What the known strategy description means

Reversal levels, NQ/ES relative structure, SMT candidates, Fibonacci candidates, rejection-candle geometry, and New York time windows are research clues. None is declared to be Conner's rule, and the code does not implement `SMT + Fibonacci + rejection block = trade`.

The implemented measurement foundation includes:

- raw candle body, range, wick, close-location, overlap, displacement, streak, and volatility-normalized geometry;
- versioned NQ/ES swing candidates, normalized relative changes, timing separation, and divergence persistence;
- candidate SMT-pattern descriptions derived from explicitly versioned swing selectors;
- candidate Fibonacci anchors, ratios, levels, and signed price distance;
- versioned candidate levels with availability and source lineage;
- versioned multi-timeframe and strategy-window configuration.

These are reproducible measurements and candidate definitions. Their names, parameters, definition versions, evidence status, and source events are retained so later out-of-sample evidence can determine whether they matter.

## Point-in-time clocks

The observation path distinguishes several clocks rather than treating one timestamp as proof that a feature was knowable:

- `event_at` or source time: when the market fact occurred;
- `available_at`: when that exact fact or revision could first be known;
- received/retrieved time: when the provider response or live event reached the system;
- `feature_cutoff_at`: the latest knowledge allowed in an observation;
- `computed_at`: when a derived measurement was materialized;
- `definition_available_at`: when the versioned feature definition itself existed.

Point-in-time features require their maximum input event time, maximum input availability time, and definition availability time to be no later than the cutoff. A feature may be computed later during historical materialization, but future inputs may not be hidden behind an old event timestamp. Missing values require an explicit reason or declared imputation rule; they are not silently changed to zero.

Bars are eligible only after their complete window ends. Synchronization uses backward/as-of selection at the cutoff, keeps each requested timeframe distinct, enforces configurable event-age and NQ/ES skew tolerances, and records gaps or late data.

Provider-finalized historical data is tagged `FINALIZED_HISTORICAL`. It can support market reconstruction, outcome research, and candidate-definition evaluation. It cannot be treated as point-in-time behavior evidence or used to fabricate whether Conner would have traded.

## Configurable lead-up observations

`HorizonSet` stores versioned, unique offsets before a documented decision/action anchor. No permanent 30/15/10/5/3/1-minute schedule is hard-coded. A materialized record preserves the configured target cutoff, the actual available cutoff, alignment error, origin, and source lineage.

Observation origins remain distinct:

- `LIVE` for future captured observations;
- `BLIND_REPLAY` for point-in-time replay whose future facts remain hidden;
- `RETROSPECTIVE_TRADE_LEADUP` for trade-centered reconstruction.

Retrospective lead-up snapshots may become positive training examples after a real trade is linked, but they cannot claim prospective lead time or prove the Scout detected that trade.

The possible New York activity window is represented by versioned `StrategyWindowCandidate` records. The known three-hour description is therefore configurable research context, not a recreated equities regime or a permanent timestamp assumption. CME trade-date/session accounting remains separate and authoritative.

## Positive/unlabeled behavior semantics

An opportunity defaults to:

- `seen_by_conner = UNKNOWN`;
- consideration status `UNKNOWN`;
- behavior label `UNLABELED`.

A real linked master trade with lifecycle evidence can create `POSITIVE_TRADE`. A negative behavior label is allowed only as `EXPLICIT_PASS`, with evidence that Conner saw and considered the candidate. A candidate with no matched trade remains unlabeled; it is not automatically a rejection.

The contracts also support evidence-based pairwise preferences and query/ranking group IDs. This keeps the data suitable for positive/unlabeled research, pairwise ranking, or learning-to-rank later without assigning relevance zero to every untraded state.

## Isolated learning tasks

Four learning tasks remain physically and semantically separate:

1. `behavior_imitation`: similarity to states where Conner actually chose to trade.
2. `setup_outcome_quality`: what subsequently happened to comparable market states.
3. `conviction_behavior`: honestly observed initial Conner risk/conviction, only after enough live master evidence exists.
4. `copier_execution_quality`: follower latency, fills, rejection, bracket, and reconciliation facts.

Follower execution facts cannot label behavior or setup quality. Conviction output cannot control follower sizing; the deterministic V2 risk engine remains authoritative. Finalized history cannot create Conner behavior or conviction labels. V1 equity, randomized sandbox, synthetic, and `hybrid_target` lineage is quarantined.

No model has been trained, selected, or promoted for these tasks.

## Ranking and Shadow contracts

A Scout ranking snapshot can retain separate behavior, outcome, and later conviction components, along with confidence, data quality, exact contracts, cutoff, model versions, candidate-generator version, and ranking-policy version. Copier execution quality is excluded from strategy ranking.

An optional display score may combine separately stored components, but it is explicitly not a training target. Every `conner_nq_v1` ranking is constrained to `shadow_only=true` and `execution_eligible=false`. Nothing in the intelligence package submits orders or changes follower risk.

Shadow evaluation can record actual-trade rank, Top 1/3/5 presence, direction match, prospective lead time, realized R/MFE/MAE when an original risk denominator exists, data-quality failures, and chronological walk-forward metadata. Pending or unmatched candidates stay unlabeled unless explicit rejection evidence exists. Retrospective snapshots are excluded from prospective lead-time claims.

## Deterministic NQ/ES replay

`MultiContractReplayRequest` requires one exact NQ contract, one exact ES contract, a half-open time interval, requested event kinds, replay mode, and version. Events are deterministically merged using availability/event time, nanosecond source time where available, sequence, contract, kind, provider event ID, and revision.

Blind point-in-time replay accepts only point-in-time facts and is eligible for behavior research. Finalized market reconstruction accepts only explicitly finalized facts and is ineligible for behavior learning. The replay engine is read-only and has no fill, strategy, label, risk, or order hook.

## Massive Futures REST adapter

`app/v2/providers/massive_futures.py` is the first research-provider adapter behind the provider-neutral market-data interface. It currently supports entitlement-gated historical REST access for:

- exact-contract reference facts;
- schedule events;
- fixed-resolution intraday bars;
- trades;
- BBO quotes;
- deterministic multi-contract historical replay.

The adapter requires explicit internal-contract-to-provider-symbol bindings. It normalizes typed payloads, preserves nanosecond provider timestamps, records endpoint/request/retrieval/revision lineage, validates pagination stays on the configured HTTPS origin, sorts deterministically, and refuses incomplete bars that overlap the request cutoff.

Authentication, HTTP-client lifecycle, secrets, and retry policy are deliberately supplied through an injected `AsyncJsonTransport`. The repository does not currently provide a credential-bearing production transport. Entitlements default to disabled and must explicitly grant each capability and historical window. There is no demo fallback, and this REST adapter does not advertise or implement live streaming.

Events returned by the historical REST adapter are deliberately tagged finalized historical, not point-in-time behavior facts. Massive is the first adapter evaluated, not a permanent provider lock-in.

## Persistence

Migration `0010_conner_nq_scout` is additive on top of `0009_v2_futures_foundation`. It adds 13 `v2_*` tables for:

- observation specifications and horizons;
- synchronized NQ/ES market states;
- candidate feature definitions and values;
- Scout opportunities and revisioned/superseding evidenced behavior labels;
- trade/candidate lead-up snapshots;
- task-isolated learning examples;
- Scout runs and ranked candidates;
- Shadow evaluations and chronological evaluation runs.

Database constraints preserve exact contract links, cutoff ordering, point-in-time availability, explicit missingness, unknown/unlabeled defaults, evidenced positive/pass labels, task boundaries, shadow-only rankings, nonnegative lead time, and walk-forward chronology.

The complete migration chain and an `0010` downgrade/re-upgrade have been exercised against PostgreSQL 16. All 13 Scout tables are reversible without removing the 23-table `0009` foundation, and V2 metadata comparison is clean.

The schema is ready to receive durable observations. A durable market-data ingestion service, object/segment store for high-volume raw data, and production repository/orchestrator are not implemented yet.

## Explicit non-goals of this phase

This phase does not add:

- a complete or handcrafted Conner strategy;
- a trained model or XGBoost ranking objective;
- a live candidate detector or live Scout service;
- autonomous trading or model-controlled sizing;
- any import from intelligence into execution;
- a live Massive stream or permanent provider commitment;
- synthetic Conner decisions;
- any Rithmic broker mutation or coupling from Scout into the separate read-only capture service.

## Exact next step

Configure and validate an authenticated market-data transport, exact NQ/ES provider bindings, and explicit entitlements. Then begin durable point-in-time capture of synchronized NQ and ES market facts without reclassifying finalized REST history. In parallel, deploy the isolated direct R|Protocol observer fail-closed and complete the documented Rithmic Test validation so real master lifecycle facts can begin accumulating without enabling broker mutation.

After enough trustworthy prospective observations exist, run the ranking contracts in Shadow and collect actual-trade matches, misses, lead time, ranks, and data-quality failures. Only then define a chronological training dataset and evaluate behavior/outcome baselines. Do not train on finalized history as Conner behavior, and do not enable order submission.
