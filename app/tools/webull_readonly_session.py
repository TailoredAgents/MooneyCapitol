from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.copier.models import WebullCredentials
from app.copier.readonly_session import WebullReadOnlySessionRunner, read_only_session_config
from app.copier.service import CopyOrchestrator
from app.copier.webull_master import WebullMasterEventListener
from app.core.config_store import CONFIG, ensure_config_initialized, refresh_config
from app.services.kv_store import StateStoreError


class _NoopSlack:
    def post(self, text: str) -> None:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a no-trade Webull read-only copier validation session.")
    parser.add_argument("--input", help="Replay a local Webull event JSON/JSONL file instead of connecting to Webull.")
    parser.add_argument("--out", help="Raw capture JSONL output path.")
    parser.add_argument("--seconds", type=float, default=300.0, help="Maximum live session duration.")
    parser.add_argument("--max-events", type=int, default=25, help="Stop after this many callback events.")
    parser.add_argument("--no-persist", action="store_true", help="Do not write master/would-copy rows to Postgres.")
    parser.add_argument(
        "--simulate-copying",
        action="store_true",
        help="In read-only only, force global copier enabled and kill switch off for would-copy simulation.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    _init_config()
    refresh_config()

    config = read_only_session_config(CONFIG.copier, simulate_copying=args.simulate_copying)
    runner = WebullReadOnlySessionRunner(
        config,
        output_path=Path(args.out) if args.out else _default_output_path(),
        orchestrator=CopyOrchestrator(
            slack=_NoopSlack(),
            require_persistence=not args.no_persist,
            persist_results=not args.no_persist,
            background_persistence=False,
            background_alerts=False,
        ),
    )

    reports: list[dict[str, Any]]
    if args.input:
        reports = _run_input_session(runner, Path(args.input))
    else:
        reports = _run_live_session(runner, seconds=args.seconds, max_events=args.max_events, machine_json=args.json)

    result = {
        "mode": "read_only",
        "persisted": not args.no_persist,
        "simulated_copying": args.simulate_copying,
        "capture_path": str(runner.capture.output_path),
        "summary": runner.summary.to_dict(),
        "events": reports,
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
    else:
        _print_text(result)


def _run_input_session(runner: WebullReadOnlySessionRunner, path: Path) -> list[dict[str, Any]]:
    payloads = _load_payloads(path)
    return [runner.handle_event(payload) for payload in payloads]


def _run_live_session(
    runner: WebullReadOnlySessionRunner,
    *,
    seconds: float,
    max_events: int,
    machine_json: bool,
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    credentials = _master_credentials(runner.config)
    if not credentials.account_id:
        raise SystemExit(f"Missing Webull master account id: {runner.config.master_account_env}")

    def on_event(*event_args):
        report = runner.handle_event(*event_args)
        reports.append(report)
        if not machine_json:
            print(
                f"event #{report['sequence']} capture={report['capture_status']} "
                f"payloads={len(report['payloads'])}",
                flush=True,
            )

    listener = WebullMasterEventListener(
        credentials=credentials,
        account_ids=[credentials.account_id],
        on_event=on_event,
    )
    listener.subscribe()
    deadline = time.monotonic() + max(seconds, 0.0)
    while runner.summary.events < max_events and time.monotonic() < deadline:
        time.sleep(0.25)
    return reports


def _load_payloads(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        rows = []
        for line in text.splitlines():
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
        return rows
    if isinstance(parsed, list):
        return [row for row in parsed if isinstance(row, dict)]
    if isinstance(parsed, dict):
        data = parsed.get("data")
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
        return [parsed]
    raise ValueError("Input must be a JSON object, JSON array, or JSONL file")


def _print_text(result: dict[str, Any]) -> None:
    summary = result["summary"]
    print(f"Webull read-only session: {result['capture_path']}")
    print(f"persisted={result['persisted']} simulated_copying={result['simulated_copying']}")
    print(
        f"events={summary['events']} payloads={summary['payloads']} parsed_fills={summary['parsed_fills']} "
        f"would_copy={summary['would_copy']} blocked={summary['blocked']} ignored={summary['ignored']} "
        f"parse_errors={summary['parse_errors']} duplicates={summary['duplicates']}"
    )
    if summary["errors"]:
        print("errors:")
        for error in summary["errors"][:10]:
            print(f"  - {error}")


def _default_output_path() -> Path:
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("captures") / f"webull_readonly_session_{stamp}.jsonl"


def _master_credentials(config) -> WebullCredentials:
    app_key = os.getenv(config.master_app_key_env)
    app_secret = os.getenv(config.master_app_secret_env)
    endpoint = os.getenv(config.master_endpoint_env)
    missing = [
        name
        for name, value in [
            (config.master_app_key_env, app_key),
            (config.master_app_secret_env, app_secret),
            (config.master_endpoint_env, endpoint),
        ]
        if not value
    ]
    if missing:
        raise SystemExit(f"Missing Webull master environment values: {', '.join(missing)}")
    return WebullCredentials(
        app_key=app_key or "",
        app_secret=app_secret or "",
        endpoint=endpoint or "",
        account_id=config.master_account or os.getenv(config.master_account_env),
        environment=config.mode,
    )


def _init_config() -> None:
    try:
        ensure_config_initialized()
    except StateStoreError:
        refresh_config()


if __name__ == "__main__":
    main()
