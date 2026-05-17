from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from app.copier.event_capture import WebullEventCapture, analyze_capture_file, summarize_capture
from app.copier.models import WebullCredentials
from app.copier.webull_master import WebullMasterEventListener
from app.core.config_store import CONFIG, ensure_config_initialized, refresh_config
from app.services.kv_store import StateStoreError


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture and analyze raw Webull order-event messages.")
    parser.add_argument("--out", help="JSONL output path for live captured Webull events.")
    parser.add_argument("--seconds", type=float, default=300.0, help="Maximum live capture duration.")
    parser.add_argument("--max-events", type=int, default=25, help="Stop after this many captured callback events.")
    parser.add_argument("--input", help="Analyze an existing JSON/JSONL capture file instead of connecting to Webull.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    _init_config()
    refresh_config()

    if args.input:
        records = analyze_capture_file(Path(args.input), CONFIG.copier)
        _print_report(records, path=Path(args.input), machine_json=args.json)
        return

    output_path = Path(args.out) if args.out else _default_output_path()
    capture = WebullEventCapture(output_path, CONFIG.copier)
    credentials = _master_credentials()
    account_id = credentials.account_id
    if not account_id:
        raise SystemExit(f"Missing Webull master account id: {CONFIG.copier.master_account_env}")

    listener = WebullMasterEventListener(
        credentials=credentials,
        account_ids=[account_id],
        on_event=lambda *event_args: _on_event(capture, event_args, machine_json=args.json),
    )
    listener.subscribe()
    deadline = time.monotonic() + max(args.seconds, 0.0)
    while capture.count < args.max_events and time.monotonic() < deadline:
        time.sleep(0.25)
    records = analyze_capture_file(output_path, CONFIG.copier)
    _print_report(records, path=output_path, machine_json=args.json)


def _on_event(capture: WebullEventCapture, event_args: tuple, *, machine_json: bool) -> None:
    record = capture.handle_event(*event_args)
    if not machine_json:
        print(
            f"captured #{record['sequence']} status={record['status']} "
            f"payloads={record['payload_count']} at {record['received_at']}",
            flush=True,
        )


def _print_report(records: list[dict], *, path: Path, machine_json: bool) -> None:
    summary = summarize_capture(records)
    if machine_json:
        print(json.dumps({"path": str(path), "summary": summary, "records": records}, indent=2, sort_keys=True))
        return
    print(f"Webull event capture: {path}")
    print(
        f"total={summary['total']} parsed={summary['parsed']} ignored={summary['ignored']} "
        f"parse_errors={summary['parse_errors']} no_payload={summary['no_payload']}"
    )
    if summary["symbols"]:
        print("symbols:")
        for symbol, count in sorted(summary["symbols"].items()):
            print(f"  - {symbol}: {count}")
    if summary["statuses"]:
        print("statuses:")
        for status, count in sorted(summary["statuses"].items()):
            print(f"  - {status}: {count}")


def _default_output_path() -> Path:
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("captures") / f"webull_events_{stamp}.jsonl"


def _master_credentials() -> WebullCredentials:
    config = CONFIG.copier
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
