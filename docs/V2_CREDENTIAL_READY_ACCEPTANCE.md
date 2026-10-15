# V2 credential-ready acceptance

Status: **passed in credential-free simulation**

Updated: 2026-10-15

This is the release gate between local implementation and credential-backed Rithmic Test validation. Passing it means the repository is ready to receive the five independent account credential bundles and begin broker validation. It does **not** mean the system is broker-validated or live-ready.

Run the gate from the repository root:

```powershell
python -m app.tools.v2_credential_ready_acceptance
```

The command runs the fixed requirement-to-test manifest and writes `artifacts/v2-credential-ready-junit.xml`. The artifact is intentionally ignored by Git because each run is local evidence. To inspect the machine-readable manifest without running tests:

```powershell
python -m app.tools.v2_credential_ready_acceptance --list
```

## Acceptance matrix

| Requirement | Automated evidence |
| --- | --- |
| Supported entry and lifecycle types | Position add/reduce/exit/reversal; market/limit/stop/stop-limit; bracket/OCO; replace/cancel; official write field maps |
| Independent follower controls | Four followers independently enabled, paused, or flatten-requested; global pause; execution-confirmed flatten completion |
| Stop and bracket protection | Exposure increases require an attached stop policy; missing broker protection locks the affected follower |
| Cumulative sizing | Total-position deltas cover scale-in, scale-out, close, and reversal without fill-by-fill rounding drift |
| Funding and margin | Underfunded followers, whole-contract granularity, minimum-equity shortfall, and margin shortfall are explicit blocks |
| Master and follower rejection | A rejected master event can only cancel an existing follower order; a follower broker rejection is terminal and isolated |
| Event ordering | Partial fills are monotonic; duplicate receipts are idempotent; stale revisions and late terminal events are rejected |
| Credential/account mismatch | Duplicate credential/account identity and adapter-to-follower slot mismatch fail closed |
| Connection and recovery | Independent failure, reconnect generation fencing, stale equity/P&L rejection, and broker-authoritative crash recovery |
| Active-instance fencing | A newer lease fences the old owner; ambiguous abandoned writes are never retried blindly |
| Contract expiry | Same-expiry mapping is mandatory and open positions are never rolled or converted automatically |
| Redaction | Credentials, exact broker account selectors, broker object identifiers, and protocol payloads are excluded from operator output |
| Operator security | Authentication, CSRF, optimistic revisions, controlled reasons, and append-only audit behavior |
| Concurrent latency | All four follower adapters must enter dispatch concurrently and persist submission/response timing evidence |
| Live submission impossible | The adapter has no live mode, the normal execution service remains non-submitting, and the status API reports live submission impossible |

The canonical node IDs for every row live in `app.tools.v2_credential_ready_acceptance.ACCEPTANCE_REQUIREMENTS`. A normal full test run also validates that every manifest entry still points to a real test.

## Current result

The credential-ready gate passed on 2026-10-15 with **37 tests passed**. The full repository suite remains the broader regression gate and must also pass before credential insertion or deployment.

## What remains external

Phase 9 still requires the real AMP/Rithmic Test account selectors, five credential bundles, licensed generated bindings, session-limit confirmation, and broker permission/conformance work. Those inputs must enter through deployment secret management, not source, chat, logs, browser storage, or test fixtures. Real gateway latency and observed broker lifecycle behavior cannot be certified by simulation.
