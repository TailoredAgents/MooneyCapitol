# V2 copy trader foundation

Status: one-master/four-follower Shadow planner implemented; broker submission not implemented

Updated: 2026-10-14

## What this is

The V2 copy trader is intended to observe one Mooney Capital master futures account and reproduce eligible master trades across exactly four follower accounts owned by the same operator. The initial product set is NQ and MNQ. All accounts are expected to be AMP accounts connected through an authorized Rithmic API setup, but account ownership does not replace AMP/Rithmic permissions, credentials, session limits, or conformance requirements.

The implemented phase is a deterministic Shadow planner. It calculates what each follower trade would be and why it would be allowed or blocked. It cannot submit, modify, cancel, or flatten a broker order. Requesting `TEST` or `LIVE` mode makes the effective mode `OFF` and reports `broker_submission_not_implemented`.

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

## Eligibility and safety rules

A Shadow plan requires:

- one configured master and exactly four unique followers;
- an initial protective stop on the master trade, on the protective side of entry;
- an exact NQ or MNQ contract and a same-exchange, same-expiry follower mapping;
- positive, current account equity and operator-supplied broker margin facts;
- positive fail-closed contract, per-trade risk, concurrent-risk, and daily-loss limits;
- global and follower kill switches to be clear;
- enough margin after headroom and enough remaining risk capacity; and
- an integer quantity within the configured parity tolerance.

The master trade's actual risk fraction must itself be within the follower policy ceiling. Scout rankings, AI output, predictions, and V1 equity state cannot alter these rules.

## Implemented surface

- `app/v2/execution/copier.py`: pure Decimal-based parity planning with no broker dependency or side effect.
- `app/v2/execution/copier_config.py`: fail-closed environment configuration for the fixed topology and hard limits.
- `GET /v2/copier/status`: authenticated, redacted readiness and blocker status.
- `POST /v2/copier/preview`: authenticated, non-persisting Shadow calculation for supplied hypothetical state.
- `/dashboard`: read-only Shadow copier readiness alongside Rithmic observations.

The preview endpoint is a calculator. It does not read broker credentials, observe a live master event, persist copy intent, or call Rithmic.

## Configuration

| Environment variable | Meaning | Safe committed value |
| --- | --- | --- |
| `V2_COPIER_MODE` | `OFF` or `SHADOW` in this phase | `SHADOW` |
| `V2_COPIER_MASTER_ACCOUNT_ID` | Exact opaque Rithmic master account ID | unset secret |
| `V2_COPIER_FOLLOWER_ACCOUNT_IDS` | Four comma-separated exact follower IDs | unset secret |
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

The repository deliberately does not hard-code website margin figures. The operator must verify and provide current account-specific values before each deployment or whenever the broker changes them. Account IDs and credentials must remain in secret management, not Git.

## Still required before copied orders

The repository does not yet contain:

- live master-order detection and lifecycle orchestration;
- a Rithmic write adapter or any follower order submission;
- durable copy intents, an outbox, active-instance lease, or crash recovery for writes;
- parent/child, partial-fill, replace, cancel, bracket/OCO, flatten, or disconnect handling for copied orders;
- actual-fill rebalancing or execution-quality reconciliation;
- measured end-to-end copy latency; or
- completed Rithmic Test write conformance and production approval.

Low latency must come from a persistent event-driven Rithmic connection and immediate in-process planning, not polling the dashboard. Safety and lifecycle correctness remain release gates even when optimizing speed.

## Activation sequence

1. Obtain the five exact account IDs under the approved Rithmic user and confirm API/session permissions with AMP/Rithmic.
2. Complete the existing read-only Rithmic Test runbook and prove master orders, fills, brackets, RMS, P&L, reconnect, and reconciliation are observed correctly.
3. Record owner-approved hard risk/loss limits, current margin requirements, cost buffers, and acceptable parity error; keep over-target copying disabled unless explicitly accepted.
4. Run the Shadow planner against representative master/follower balances and reject funding combinations that cannot fit one MNQ safely.
5. Build the separate master-event orchestrator, durable intent/outbox/lease layer, and write adapter behind a new fail-closed Test gate.
6. Validate entry, stop, target, partial-fill, cancel/replace, flatten, rejection, reconnect, and kill-switch scenarios in Rithmic Test.
7. Complete any required Rithmic conformance and AMP approval, then use a deliberately small live canary before broader operation.

The official integration starting points are [Rithmic APIs and conformance](https://www.rithmic.com/apis), [AMP's Rithmic API page](https://www.ampfutures.com/trading-platform/rithmic-r-api), and the [read-only project runbook](RITHMIC_READ_ONLY_RUNBOOK.md).
