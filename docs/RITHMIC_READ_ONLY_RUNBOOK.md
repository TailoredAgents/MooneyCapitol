# Rithmic read-only capture runbook

## Status and boundary

MooneyCapitol implements a direct R|Protocol 0.90 read-only capture adapter. R|API+/.NET is retained only as a fallback and conformance reference. The next gate is Rithmic Test validation; production readiness has not been declared.

The deployed observer has no order-submission, modification, cancellation, flatten, exit, bracket-send, OCO, link, target-change, stop-change, or follower-execution method. A default-deny transport guard checks every outbound protobuf template immediately before WebSocket send and rejects all known mutation templates. This runbook never enables broker mutation.

The capture process is isolated from the API, workers, Scout, ML, OpenAI, Shadow Trader, and follower execution. It starts offline unless every TEST-only preflight condition is satisfied.

## Licensed binding preparation

The official SDK, `.proto` files, generated Python bindings, DLLs, PDFs, samples, credentials, and certificates must remain outside this repository. Before using the official materials or deploying generated derivatives, confirm that the applicable Rithmic agreement permits the intended private local and hosted use.

Use a private working directory outside the Git checkout. The generator includes only schemas required by the read-only runtime, emits a per-file manifest for RProtocolAPI 0.90.0.0/template 5.55, and refuses repository-local outputs.

```powershell
python -m pip install -r requirements-rithmic-build.txt

python -m app.tools.generate_rithmic_bindings `
  --sdk-path "C:\Users\xenon\OneDrive\Desktop\Rithmic official docs\RProtocolAPI.0.90.0.0\0.90.0.0" `
  --output "C:\private\mooney-rithmic\generated" `
  --archive "C:\private\mooney-rithmic\rithmic-bindings.zip" `
  --archive-base64 "C:\private\mooney-rithmic\rithmic-bindings.zip.b64"
```

The command prints the binary ZIP SHA-256 and base64 file size. Record the checksum in secret management; do not paste it into tracked files. The runtime verifies the archive checksum, the manifest version, the exact module set, and every generated file checksum before import. Archives reject traversal paths, symlinks, unrelated members, and excessive size.

For local operation, choose exactly one source:

- `RITHMIC_GENERATED_BINDINGS_PATH`: external generated directory containing `rithmic_bindings_manifest.json`;
- `RITHMIC_GENERATED_BINDINGS_ARCHIVE`: external binary ZIP plus `RITHMIC_GENERATED_BINDINGS_SHA256`; or
- `RITHMIC_GENERATED_BINDINGS_ARCHIVE_B64_FILE`: external base64 file plus the SHA-256 of the decoded binary ZIP.

Do not set more than one source. `RITHMIC_PROTOCOL_SDK_PATH` is not a runtime dependency; use the generator's `--sdk-path` argument at private build time.

## Render deployment

`render.yaml` defines a single-instance `mooney-rithmic-capture` web service. It runs the additive migrations before starting, exposes `/health`, `/ready`, and aggregate-only `/metrics`, and allows 60 seconds for graceful shutdown. Its committed defaults are deliberately offline:

```text
RITHMIC_CAPTURE_CONNECTIVITY_ENABLED=0
RITHMIC_ENVIRONMENT=DISABLED
RITHMIC_ENABLED_PLANTS=ORDER,PNL
```

Deploy those defaults first. `/health` is liveness and remains available while configuration is blocked; `/ready` must return 503 until both required plants are authenticated, every allowlisted account is discovered, and clean per-generation reconciliation is durably checkpointed. Health output never includes account IDs, credentials, protocol payloads, or order IDs.

### Render Secret File

Create a plaintext Render Secret File named exactly:

```text
rithmic-bindings.zip.b64
```

Its contents are the single-line base64 file produced above. The Blueprint points the process at `/etc/secrets/rithmic-bindings.zip.b64`. Confirm that the file fits Render's current combined Secret File size limit before activation. If it does not, stop and choose a license-approved private artifact-delivery mechanism; never commit the bindings as a workaround.

### Required Render values

Enter these through Render environment/secret management:

| Variable | Value or purpose |
| --- | --- |
| `DATABASE_URL` | Supplied from `trading-agent-db` by the Blueprint. |
| `RITHMIC_GENERATED_BINDINGS_SHA256` | SHA-256 of the decoded binary bindings ZIP. |
| `RITHMIC_DISCOVERY_URI` | Authorized Rithmic Test `wss://` discovery endpoint. |
| `RITHMIC_SYSTEM_NAME` | Exact Test system returned by discovery. |
| `RITHMIC_GATEWAY_NAME` | Required if discovery returns multiple gateways and no per-plant selection is used. |
| `RITHMIC_USERNAME` | Test username secret. |
| `RITHMIC_PASSWORD` | Test password secret. |
| `RITHMIC_ACCOUNT_ALLOWLIST` | Comma-separated exact opaque account IDs authorized for capture. |
| `RITHMIC_ENVIRONMENT` | Change from `DISABLED` to exactly `TEST` only at the activation gate. |
| `RITHMIC_CAPTURE_CONNECTIVITY_ENABLED` | Change from `0` to `1` last, after every other check. |

The Blueprint supplies these non-secret values:

```text
RITHMIC_APPLICATION_NAME=MooneyCapitol
RITHMIC_APPLICATION_VERSION=v2-read-only
RITHMIC_TEMPLATE_VERSION=5.55
RITHMIC_CAPTURE_OBSERVER_FACTORY=app.v2.brokers.rithmic_protocol.adapter:create_observer
RITHMIC_CAPTURE_JOURNAL_FACTORY=app.v2.capture.persistence:create_journal
RITHMIC_CAPTURE_POLL_SECONDS=1
RITHMIC_LOGIN_TIMEOUT_SECONDS=30
RITHMIC_REQUEST_TIMEOUT_SECONDS=30
RITHMIC_RECONCILIATION_TIMEOUT_SECONDS=120
RITHMIC_MAX_BUFFERED_EVENTS=50000
RITHMIC_ACCOUNT_METADATA_START_SSBOE=0
```

`RITHMIC_ACCOUNT_METADATA_START_SSBOE=0` makes the official account/user playback establish account access and status before discovery can complete. A later non-zero epoch may be used only after Test proves that the bounded window still includes authoritative current metadata; missing access/status fails discovery closed.

Optional controls are `RITHMIC_CA_FILE`, `RITHMIC_ORDER_GATEWAY_NAME`, `RITHMIC_PNL_GATEWAY_NAME`, `RITHMIC_TICKER_GATEWAY_NAME`, explicit `RITHMIC_ORDER_URI`, `RITHMIC_PNL_URI`, or `RITHMIC_TICKER_URI` overrides, and the YYYYMMDD recovery overrides `RITHMIC_RECOVERY_TRADE_DATE`, `RITHMIC_FILL_HISTORY_START_INDEX`, and `RITHMIC_FILL_HISTORY_FINISH_INDEX`. Every endpoint must use `wss://`, certificate and hostname verification remain mandatory, and endpoint URLs may not contain credentials, query strings, or fragments.

Normal capture uses `ORDER,PNL`. Add `TICKER` only for deliberate contract/reference queries; continuous Rithmic market-data streaming and History Plant strategy data are out of scope.

### Optional operational reference queries

The capture process exposes three strictly read-only Ticker Plant query routes:

- `GET /reference/symbols?q=MNQ&limit=25`
- `GET /reference/contracts/MNQZ6?exchange=CME`
- `GET /reference/tick-sizes/{tick_size_type}`

They remain unavailable unless all of the following are true:

1. `TICKER` is included in `RITHMIC_ENABLED_PLANTS`;
2. the optional Ticker Plant is connected and authenticated;
3. `RITHMIC_REFERENCE_QUERY_ENABLED=1`; and
4. `RITHMIC_REFERENCE_QUERY_TOKEN` is a random 32-512 character secret.

Send the token only in the `X-Rithmic-Reference-Token` header. Comparison uses a fixed-size digest and constant-time comparison. Inputs, result counts, concurrency, and timeouts are bounded; responses expose only an explicit reference-data field allowlist. Account identity, credentials, protocol correlation fields, and raw payloads are never returned. Query access logging is disabled, errors are generic, and every outbound request still passes the transport's read-only template allowlist.

`RITHMIC_REFERENCE_QUERY_TIMEOUT_SECONDS` defaults to 15 and is restricted to 1-30 seconds. `RITHMIC_REFERENCE_QUERY_MAX_RESULTS` defaults to 50 and is restricted to 1-100 rows. Symbol searches require at least two constrained ASCII characters; exchange, product, contract, and tick-size identifiers accept only bounded symbol-safe characters.

The committed Render defaults keep this surface disabled. To perform an approved reference lookup, add `TICKER`, enter the query token as a Render secret, enable the query surface, redeploy, perform the bounded lookup, then return to `ORDER,PNL` and disable the surface.

### Command and endpoints

Render must run:

```text
python -m app.v2.capture.main
```

The process binds `$PORT` and responds at:

- `GET /health` — liveness and sanitized state;
- `GET /ready` — 200 only after clean required-plant/account reconciliation;
- `GET /metrics` — aggregate gauges only, including a constant submission-enabled value of zero.
- the authenticated `/reference/*` routes documented above, only while their explicit Ticker/query gates are enabled.

## Recovery model

Order and PnL maintain independent sockets, authentication state, heartbeat state, generation IDs, timestamps, reconnect state, and readiness. On a new generation, the coordinator:

1. authenticates both required plants;
2. discovers and exact-matches allowlisted accounts;
3. installs all account Order, bracket, RMS, and PnL live subscriptions first;
4. buffers live events while recovery is active;
5. requests current orders, current-day executions, order/fill history, brackets/stops, account/product RMS, and PnL/position snapshots;
6. journals and deduplicates replay/snapshot/live overlap;
7. applies the buffered live tail;
8. folds snapshot/history plus the buffered tail into a deterministic account-state digest and rejects generation, identity, or state discrepancies; and
9. writes the zero-discrepancy reconciliation checkpoint, closes the recovery batch, and marks the generation ready.

A failed recovery is not repeatedly snapshot-polled in the same generation. A new connection generation or operator restart is the next retry boundary. Unknown broker values remain unknown, and ambiguous request outcomes are designed to pass through query/replay reconciliation rather than blind retry.

## Rithmic Test validation sequence

Perform this only after the fail-closed deployment is healthy:

1. Confirm the service starts with connectivity `0`, `/health` is sanitized, `/ready` is 503, and `submission_enabled` is false.
2. Upload the private base64 bindings Secret File, enter its binary ZIP checksum, and confirm no vendor file appears in the repository or build logs.
3. Enter the authorized Test discovery/system/gateway settings, exact account allowlist, and Test credentials. Keep connectivity `0`.
4. Confirm database migration `0011_rithmic_read_capture` is at head and generic V2 broker connection/account execution flags remain disabled.
5. Set `RITHMIC_ENVIRONMENT=TEST`; then set `RITHMIC_CAPTURE_CONNECTIVITY_ENABLED=1` as the final switch and redeploy.
6. Confirm independent Order and PnL connections authenticate with distinct generation/heartbeat health and that only allowlisted accounts are discovered and subscribed.
7. Wait for subscribe-before-snapshot recovery to complete. Verify immutable events, replay batch completion, projections, and reconciliation checkpoints; only then expect `/ready` to return 200.
8. In R|Trader Test, manually create an order. Verify it appears as externally originated unless a verified Mooney-owned tag exists, with basket/exchange IDs, native status, origin metadata, 64-bit quantities, and timestamps intact.
9. Modify and cancel Test orders manually in R|Trader. Verify pending, working, modification, cancellation, rejection/command-failure, completion-reason, and unknown-state handling without any API mutation request.
10. Exercise partial and full Test fills where the environment permits. Verify `fill_id` identity, cumulative/unfilled quantities, prices, and deduplication. Validate bust/correction facts if the Test environment can produce them.
11. Create/manage bracket or OCO behavior manually in R|Trader Test. Verify parent/linked baskets, target/stop tiers, released quantities, and trailing facts are observed read-only.
12. Compare instrument/account positions, open/closed and daily P&L, cash/balance/margin/buying-power/commission fields, RMS limits, product margins, currency, and auto-liquidation facts against R|Trader.
13. Interrupt/restart the capture connection. Verify independent reconnect generations, live-first buffering, replay/live overlap deduplication, no repeated same-generation polling, and clean readiness recovery.
14. If needed, enable Ticker Plant temporarily and validate exact NQ/MNQ symbol, exchange, expiration, tick-table, point-value, and tradability mappings. Disable it for ordinary capture after mappings are persisted.
15. Review redaction and append-only behavior, record all conformance differences, then return connectivity to `0` until the separate paper-order phase is explicitly authorized.

Rithmic Test success is evidence for the observation adapter only. It does not authorize paper or live order submission and does not declare the complete system production-ready.
