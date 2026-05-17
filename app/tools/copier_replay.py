from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.copier.engine import CopyEngine
from app.copier.factory import build_copy_targets
from app.copier.models import MasterExecutionEvent, WebullEquityOrder
from app.copier.runtime import _is_fill_payload
from app.copier.service import CopyOrchestrator
from app.core.config_store import CONFIG, ensure_config_initialized, refresh_config
from app.services.kv_store import StateStoreError


class DryRunTradingClient:
    def place_equity_order(self, account_id: str, order: WebullEquityOrder) -> dict:
        return {
            "dry_run": True,
            "account_id": account_id,
            "client_order_id": order.client_order_id,
            "symbol": order.symbol,
            "side": order.side,
            "quantity": order.quantity,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay recorded Webull master execution payloads through copier logic.")
    parser.add_argument("path", help="Path to a JSON object or JSON array of Webull event payloads.")
    parser.add_argument("--submit", action="store_true", help="Submit child orders through the configured Webull targets.")
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="Persist master executions and intended copy decisions; no broker orders.",
    )
    args = parser.parse_args()

    if args.submit and args.read_only:
        raise SystemExit("--submit and --read-only cannot be used together")

    _init_config()
    payloads = _load_payloads(Path(args.path))
    refresh_config()

    summaries = []
    for payload in payloads:
        if not _is_fill_payload(payload, CONFIG.copier):
            summaries.append({"status": "ignored", "reason": "not_fill_payload", "payload": payload})
            continue
        event = MasterExecutionEvent.from_webull_payload(payload)
        if args.read_only:
            targets = build_copy_targets(
                CONFIG.copier,
                client_factory=lambda target_cfg: DryRunTradingClient(),
                allow_missing_accounts=True,
            )
            results = CopyOrchestrator().plan_execution(event, targets)
            summaries.append(_result_summary(event, results, mode="read_only"))
            continue
        if args.submit:
            targets = build_copy_targets(CONFIG.copier)
            results = CopyOrchestrator().copy_execution(event, targets)
            summaries.append(_result_summary(event, results, mode="submit"))
            continue

        targets = build_copy_targets(
            CONFIG.copier,
            client_factory=lambda target_cfg: DryRunTradingClient(),
            allow_missing_accounts=True,
        )
        results = CopyEngine().copy_execution(event, targets)
        summaries.append(_result_summary(event, results, mode="dry_run"))

    print(json.dumps(summaries, indent=2, sort_keys=True, default=str))


def _init_config() -> None:
    try:
        ensure_config_initialized()
    except StateStoreError:
        # Local replay can run with STATE_STORE=mem or without Postgres. Submit/read-only
        # modes still need a working DB because the orchestrator requires persistence.
        refresh_config()


def _load_payloads(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, dict):
        data = raw.get("data")
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        return [raw]
    raise ValueError("Replay input must be a JSON object or JSON array")


def _result_summary(event: MasterExecutionEvent, results, mode: str) -> dict:
    return {
        "mode": mode,
        "execution_id": event.execution_id,
        "symbol": event.symbol,
        "side": event.side,
        "quantity": event.quantity,
        "price": event.price,
        "results": [
            {
                "target": result.target,
                "allowed": result.allowed,
                "submitted": result.submitted,
                "reason": result.reason,
                "client_order_id": result.client_order_id,
                "quantity": result.quantity,
                "error": result.error,
            }
            for result in results
        ],
    }


if __name__ == "__main__":
    main()
