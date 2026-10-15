# MooneyCapitol project handoff

Updated: 2026-10-15

Read [PROJECT_IMPLEMENTATION_PLAN.md](PROJECT_IMPLEMENTATION_PLAN.md) first for the current product direction and work order, then [V2_FUTURES_ARCHITECTURE.md](V2_FUTURES_ARCHITECTURE.md) for plane boundaries. The implemented Shadow copy contract is in [V2_COPY_TRADER_FOUNDATION.md](V2_COPY_TRADER_FOUNDATION.md), the detailed strategy-observation contract is in [CONNER_NQ_SCOUT_FOUNDATION.md](CONNER_NQ_SCOUT_FOUNDATION.md), and current read-only deployment/validation is in [RITHMIC_READ_ONLY_RUNBOOK.md](RITHMIC_READ_ONLY_RUNBOOK.md). Older roadmap and learning documents describe V1 history and must not define V2 futures behavior, labels, or features.

## Direction

MooneyCapitol V2 combines a broker-neutral, strategy-neutral execution foundation with a provider-neutral futures intelligence boundary. V1's equity scout, Webull copier, AI Lab, learning pipeline, APIs, dashboard, and tables remain legacy/reference code.

The locked execution topology is one independent AMP master and four independent AMP followers. Every account receives an exact account selector and independently configurable credential/connection bundle. The system does not depend on shared AMP subaccounts, entity, portal, Rithmic user, or credentials. The immediate priority is a fully simulated credential-ready copier, followed by credential insertion and broker validation, the dashboard structural/aesthetic pass, learning, and later read-only Conner-focused AI agents.

V1 equity artifacts, randomized sandbox data, synthetic labels, and `hybrid_target` cannot enter V2 futures datasets or artifact selection. V2 execution cannot import intelligence, OpenAI, or ML packages.

The first strategy-specific observation namespace is `conner_nq_v1`. It learns nothing yet; it defines how trustworthy NQ/ES observations and Conner lifecycle labels can be collected for later learning.

## Foundation completed

### Execution/reference foundation

- `app/v2/domain`: immutable futures, lifecycle, plan, trade, follower, snapshot, and reconciliation contracts.
- `app/v2/reference.py`: NQ, MNQ, and ES product specifications plus dynamic exact-expiry registration. Only NQ/MNQ support explicit same-expiry execution/equivalence mapping.
- `app/v2/calendar.py`: CME trade-date sessions, maintenance windows, and supplied holiday/early-close/availability overrides.
- `app/v2/math.py`: Decimal-only risk, P&L, VWAP, R, excursions, fills, scale-out, reversal, costs, and NQ/MNQ equivalence math.
- `app/v2/risk.py`: deterministic fail-closed sizing and hard account/product/loss/margin constraints.
- `app/v2/brokers`: broker interface, disabled production placeholder, direct read-only R|Protocol 0.90 observer, and separate explicit-Test write boundary with no live mode.
- `app/v2/capture`: isolated TEST-only capture, subscribe-first recovery, durable journaling, and sanitized health/readiness/metrics.
- `app/v2/execution`: separate disabled process with immutable config, health, actor, journal queue, fencing seam, graceful shutdown, pure one-master/four-follower parity and cumulative-position sizing with funding reports, five typed independent secret-reference connection slots, deterministic discovery/Order/PnL supervision with simulation fixtures, a durable revisioned follower/global control plane, and durable per-follower lifecycle intents/commands/events.

The standalone execution process is intentionally not ready: its placeholder submission always raises and `submission_enabled=True` is rejected. The sizing engine calculates stop-risk and point-exposure percentage parity across integer same-expiry NQ/MNQ candidates, then plans cumulative follower totals and signed deltas across adds, reductions, closes, and reversals. It pins open exact contracts, blocks automatic rolls, retains desired/evaluated/achievable exposure facts, and quantifies one-MNQ equity/risk/margin shortfalls. The lifecycle bridge persists those facts before dispatch, uses deterministic intent/command IDs, blocks stale/conflicting master revisions, handles partial fill and terminal monotonicity, and orders reversal close before open. A separate adapter maps commands to the official external R|Protocol write schemas, but can transmit only when directly constructed for Rithmic Test with explicit authorization, external bindings, and an injected transport; it is not runtime-attached and has no live mode. Five fixed connection records resolve distinct account selectors and credentials indirectly from secret management, validate Test-only readiness, and expose only redacted status. The connection supervisor models independent discovery and Order/PnL sessions for every record, generation-fences reconnects, isolates follower failures, blocks globally on stale master state, and has deterministic credential-free fixtures. The durable control plane starts globally paused with every follower disabled; state changes are authenticated, revision-checked, row-serialized, CSRF-protected for cookie sessions, and append-only audited. A flatten request cannot claim success until the future execution system confirms a reconciled flat account. A live driver and runtime-health attachment are still absent. The separate read-only observer remains structurally read-only and production-shaped for the next Rithmic Test gate. Its mutation denylist is unchanged. R|API+/.NET is only a fallback/conformance reference. Production readiness has not been declared.

### Market-data foundation

- `app/v2/market_data.py`: typed bars/trades/BBO, exact provider bindings, schedules/reference, entitlement contracts, event/availability/retrieval/revision lineage, fixed-resolution validation, and multi-contract historical requests.
- `app/v2/providers/massive_futures.py`: entitlement-gated historical REST adapter for exact contract reference, schedules, bars, trades, BBO, safe pagination, and deterministic replay.

The Massive adapter requires an injected authenticated `AsyncJsonTransport`; authentication, secrets, HTTP lifecycle, and retry policy are not implemented in the adapter. Entitlements grant nothing by default. It has no demo fallback and no live-streaming capability. Historical REST results are explicitly finalized history.

Massive is the first adapter under evaluation, not a permanent dependency.

### `conner_nq_v1` intelligence foundation

- `app/v2/intelligence/observation.py`: exact NQ traded leg plus exact ES contextual leg, synchronized state, purpose, availability, and quality contracts.
- `app/v2/intelligence/specification.py`: provider-neutral observation specs, multi-timeframe requirements, versioned measurement families, and configurable New York-window candidates.
- `app/v2/intelligence/synchronization.py`: backward/as-of point-in-time selection with configurable age/skew tolerances and no future fill.
- `app/v2/intelligence/measurements.py`: versioned candle geometry/relationships, swing and relative-structure candidates, SMT/divergence persistence, Fibonacci candidates, and candidate levels.
- `app/v2/intelligence/horizons.py`: configurable versioned lead-up offsets and origin-aware snapshot records.
- `app/v2/intelligence/datasets.py`: positive/unlabeled opportunities, evidenced explicit passes, actual master labels, separate outcome/conviction/execution labels, pairwise preferences, and task-isolated examples.
- `app/v2/intelligence/ranking.py`: separately retained behavior, outcome, conviction, confidence, and data-quality outputs; rankings are Shadow-only and never execution-eligible.
- `app/v2/intelligence/evaluation.py`: trade/candidate matching status, Top 1/3/5 and lead-time records, unlabeled unmatched candidates, chronological folds, and metric contracts.
- `app/v2/intelligence/replay.py`: deterministic exact-contract NQ/ES replay with blind point-in-time and finalized-reconstruction modes.
- `app/v2/learning`: temporal-role, source-domain, event/availability/computation, task-boundary, artifact, and V1/synthetic quarantine validation.

Known strategy elements are represented only as candidate measurements and research clues. Nothing declares an SMT definition, Fibonacci anchoring method, candidate level, candle structure, or time window to be Conner's actual rule.

## Label and time semantics

Point-in-time eligibility uses both event time and availability/knowledge time. Derived measurements also retain computation time, feature-definition availability, maximum input clocks, source events, parameters, versions, and lineage. Finalized provider history cannot enter behavior training or fabricate Conner decisions.

Opportunities begin with unknown exposure/consideration and an unlabeled target. Actual master trades with lifecycle evidence are strong positives. Only confirmed seen-and-considered explicit passes can become negatives. An unmatched opportunity stays unlabeled.

Four tasks remain isolated:

- behavior imitation;
- setup/outcome quality;
- conviction behavior;
- copier execution quality.

Execution-quality facts never become strategy labels. Conviction output is prohibited from controlling follower risk.

## Persistence and migration status

V1 migrations `0001` through `0008` and V1 tables remain unchanged.

`0009_v2_futures_foundation` adds 23 V2 execution/reference tables using integer quantities and NUMERIC economics.

`0010_conner_nq_scout` adds 13 tables:

- `v2_observation_specs`
- `v2_observation_horizons`
- `v2_synchronized_market_states`
- `v2_candidate_feature_definitions`
- `v2_candidate_feature_values`
- `v2_scout_opportunities`
- `v2_opportunity_behavior_labels`
- `v2_trade_leadup_snapshots`
- `v2_learning_examples`
- `v2_scout_runs`
- `v2_scout_rankings`
- `v2_shadow_evaluations`
- `v2_scout_evaluation_runs`

The migration is additive and its downgrade targets only those 13 tables. Constraints encode exact contract links, cutoff/availability ordering, explicit missing values, PU semantics, evidenced labels, task boundaries, Shadow-only outputs, lead-time rules, and chronological evaluation.

`0011_rithmic_read_capture` adds 11 read-only capture tables:

- `v2_rithmic_connection_generations`
- `v2_rithmic_replay_batches`
- `v2_rithmic_broker_events`
- `v2_rithmic_account_observations`
- `v2_rithmic_order_observations`
- `v2_rithmic_execution_observations`
- `v2_rithmic_bracket_observations`
- `v2_rithmic_reference_observations`
- `v2_rithmic_pnl_observations`
- `v2_rithmic_rms_observations`
- `v2_rithmic_reconciliation_checkpoints`

The migration is additive, uses BigInteger for official 64-bit quantities, preserves opaque native identities and unknown/correction/bust facts, and installs PostgreSQL append-only triggers on immutable history. Broker account references are preserved in full while deterministic internal hashes are used as relational keys. Generic V2 broker connection/account execution flags remain false.

`0012_v2_copy_control` adds the fail-closed five-account copy profile, four fixed follower control rows, and database-protected append-only control audit history.

`0013_v2_copy_lifecycle` adds revisioned per-follower intents, deterministic ordered commands, all returned broker-order IDs, and database-protected append-only lifecycle history.

The Rithmic capture journal durably writes immutable native events and append-only observation projections. No production ingestion/repository service is currently writing a durable point-in-time NQ/ES strategy-market history.

The authenticated `/dashboard` route now exposes a read-only V2 operations console over those durable projections and the Shadow copier's redacted readiness state. Account and broker object identifiers are represented as deterministic aliases rather than raw values. Optional sanitized live process health is read from `RITHMIC_CAPTURE_BASE_URL`. The retired V1 equity/Webull console route and template have been removed; remaining V1 backend code is not linked or served through the operator UI.

The earlier PostgreSQL 16 validation upgraded `0001` through `0010`, downgraded/re-upgraded `0010`, and reported zero V2 metadata drift through that revision. `0011` through `0013` have automated additive schema/model tests and are wired into the same migration runner; a live PostgreSQL upgrade plus Rithmic Test capture is the next deployment validation. Existing V1 drift observations remain a separate legacy-schema audit item.

## Tests in place

V2 tests cover:

- exact NQ/ES identity and separation from NQ/MNQ mapping;
- point-in-time cutoffs, late revisions, finalized-history restrictions, and no-future synchronization;
- candidate measurement lineage and raw candle geometry;
- configurable horizons and retrospective lead-time exclusion;
- positive/unlabeled labels and evidenced negatives;
- task isolation and follower-quality quarantine;
- ranking/Shadow execution ineligibility;
- deterministic NQ/ES replay and chronological evaluation;
- Massive entitlement, binding, typed payload, precision, pagination, and cutoff behavior;
- R|Protocol WSS/TLS, protobuf framing, external binding integrity, discovery, login deadlines, heartbeats, forced logout/rejects, multipart responses, and independent plant generations;
- account allowlisting, lifecycle/64-bit/fill-bust-correction/origin normalization, P&L/RMS/brackets/reference data, subscribe-first buffering, replay overlap, deduplication, and reconnect recovery;
- transport-level rejection of all broker-mutation template IDs and structural absence of capture mutation methods;
- durable projections, append-only migration/model alignment, redaction, and fail-closed Render configuration.
- exact and underfunded NQ/MNQ master-relative sizing, cumulative add/reduce/close/reversal deltas, contract/expiry pinning, one-MNQ funding shortfalls, configuration redaction, and authenticated status/preview APIs.
- durable lifecycle idempotency, stale/duplicate protection, partial and terminal fills, ordered reversals, follower-failure isolation, official new/modify/cancel/bracket/OCO/exit field maps, external write-binding integrity, and Test-only/no-live gates.

Current verification is 534 passing repository tests with one skipped test. Compilation of `app`, `tests`, and `migrations`, the single Alembic head at `0013`, automated migration/model checks, and `git diff --check` pass. Seven existing `datetime.utcnow()` deprecation warnings remain. Whole-chain offline Alembic SQL is still limited by `0001` inspecting a mock connection; deployment uses the existing live PostgreSQL migration runner, and a live PostgreSQL upgrade through `0013` remains a deployment-validation gate.

## Not implemented

- An authenticated production market-data transport or configured entitlements
- A durable point-in-time NQ/ES capture pipeline or high-volume raw-data store
- A live candidate detector, observation scheduler, or live Scout service
- Any trained behavior, outcome, conviction, or execution-quality model
- A selected XGBoost classification/ranking objective or promoted artifact
- A complete/handcrafted Conner strategy
- Automatic trading from Scout output or any model-controlled sizing
- Rithmic Test conformance, including manual R|Trader-to-API visibility and observed edge cases
- Live-driver/runtime attachment for the implemented five-account connection supervisor
- Live master-event ingestion, runtime lifecycle attachment, or credential-backed Rithmic follower submission
- Automatic contract rolling
- A production distributed lease or durable execution journal worker

## Operational cautions

- Do not call Massive finalized history point-in-time behavior data.
- Do not mark unmatched opportunities as Conner passes.
- Do not count retrospective trade-centered snapshots as prospective Scout lead time.
- Do not infer a decision timestamp from a fill without recording the anchor semantics.
- Do not infer original R when no valid original stop was observed.
- Do not allow later stop changes to replace the original risk denominator.
- Do not add ES to NQ/MNQ execution mappings; ES is contextual research reference.
- Do not let ranking/display scores become a blended training target.
- Do not let Scout, conviction, or outcome output enter execution or weaken deterministic risk.
- Do not add unofficial Rithmic URLs, vendor schemas/bindings, credentials, certificates, or guessed lifecycle mappings to Git.
- Do not interpret generic `COMPLETE` as filled without the native completion reason.
- Do not use `user_tag`, `user_msg`, or `manual_or_auto` as guaranteed ownership/idempotency proof.

## Exact next step

Follow [PROJECT_IMPLEMENTATION_PLAN.md](PROJECT_IMPLEMENTATION_PLAN.md). Typed account configuration, deterministic multi-account supervision/simulation, durable controls, cumulative target-position sizing, follower funding-adequacy reporting, durable lifecycle orchestration, and the separate Test-gated write boundary are complete. The next code task is Phase 6: broker-authoritative reconciliation, crash recovery/outbox/fencing, and latency telemetry. Continue through the minimal operational UI and failure-injected acceptance testing without waiting for credentials.

When the credential bundles arrive, privately generate/deliver the licensed read and write bindings, insert secrets through secret management, and follow [RITHMIC_READ_ONLY_RUNBOOK.md](RITHMIC_READ_ONLY_RUNBOOK.md) for every independent connection before attaching and exercising the Test write path. Rithmic Test, conformance, actual latency validation, and a one-follower minimum-size live canary remain distinct later gates. The broader dashboard redesign follows the proven copier; learning and Conner-focused read-only AI agents follow later. No environment variable in the current code can attach the writer or enable live broker mutation.
