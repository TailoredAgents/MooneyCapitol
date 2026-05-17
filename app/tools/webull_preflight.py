from __future__ import annotations

import argparse
import json
import sys

from app.copier.preflight import run_webull_preflight
from app.core.config_store import CONFIG, ensure_config_initialized, refresh_config
from app.db.session import get_session
from app.services.kv_store import StateStoreError


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Webull copier configuration and account connectivity.")
    parser.add_argument("--skip-network", action="store_true", help="Only validate local config/env values.")
    parser.add_argument("--include-disabled-targets", action="store_true", help="Also report disabled target accounts.")
    parser.add_argument("--skip-db", action="store_true", help="Skip database readiness checks.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("--no-fail", action="store_true", help="Always exit 0 even when blockers exist.")
    args = parser.parse_args()

    _init_config()
    refresh_config()

    session = None
    if not args.skip_db:
        try:
            with get_session() as db_session:
                report = run_webull_preflight(
                    CONFIG.copier,
                    session=db_session,
                    include_network=not args.skip_network,
                    include_disabled_targets=args.include_disabled_targets,
                )
        except Exception as exc:
            report = run_webull_preflight(
                CONFIG.copier,
                include_network=not args.skip_network,
                include_disabled_targets=args.include_disabled_targets,
            )
            report["checks"].append(
                {
                    "key": "database.connection",
                    "ok": False,
                    "severity": "blocker",
                    "label": "Database connection is available",
                    "context": {"error": str(exc)},
                }
            )
            report["blockers"].append(report["checks"][-1])
            report["ready"] = False
    else:
        report = run_webull_preflight(
            CONFIG.copier,
            session=session,
            include_network=not args.skip_network,
            include_disabled_targets=args.include_disabled_targets,
        )

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
    else:
        _print_text(report)

    if report["blockers"] and not args.no_fail:
        raise SystemExit(1)


def _init_config() -> None:
    try:
        ensure_config_initialized()
    except StateStoreError:
        refresh_config()


def _print_text(report: dict) -> None:
    status = "READY" if report.get("ready") else "NOT READY"
    print(f"Webull copier preflight: {status}")
    print(f"Network checks: {'on' if report.get('network_checked') else 'off'}")
    print("")
    for account in report.get("accounts", []):
        print(f"{account['role']}:{account['name']} account={account.get('account') or 'unset'} env={account.get('environment')}")
        for check in account.get("checks", []):
            print(f"  {_mark(check)} {check['key']} - {check['label']}")
            error = (check.get("context") or {}).get("error")
            if error:
                print(f"      error: {error}")
        print("")

    blockers = report.get("blockers") or []
    warnings = report.get("warnings") or []
    print(f"Blockers: {len(blockers)}")
    for check in blockers[:12]:
        print(f"  - {check['key']}: {check['label']}")
    if len(blockers) > 12:
        print(f"  - plus {len(blockers) - 12} more")
    print(f"Warnings: {len(warnings)}")
    for check in warnings[:12]:
        print(f"  - {check['key']}: {check['label']}")
    if len(warnings) > 12:
        print(f"  - plus {len(warnings) - 12} more")


def _mark(check: dict) -> str:
    if check.get("ok"):
        return "OK"
    if check.get("severity") == "warning":
        return "WARN"
    if check.get("severity") == "info":
        return "INFO"
    return "FAIL"


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
