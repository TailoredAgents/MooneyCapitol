from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from time import perf_counter, sleep

from app.copier.engine import CopyTarget
from app.copier.models import MasterExecutionEvent, WebullEquityOrder
from app.copier.risk import RiskPolicy
from app.copier.runtime import _is_fill_payload
from app.copier.service import CopyOrchestrator
from app.copier.sizing import SizingPolicy
from app.core.config import CopierConfig


class BenchmarkTradingClient:
    def __init__(self, broker_delay_ms: float = 0.0) -> None:
        self.broker_delay_ms = broker_delay_ms
        self.orders: list[tuple[str, WebullEquityOrder]] = []

    def place_equity_order(self, account_id: str, order: WebullEquityOrder) -> dict:
        if self.broker_delay_ms > 0:
            sleep(self.broker_delay_ms / 1000)
        self.orders.append((account_id, order))
        return {
            "benchmark": True,
            "account_id": account_id,
            "client_order_id": order.client_order_id,
            "order_id": f"bench-{len(self.orders)}",
        }


class NoopSlack:
    def post(self, text: str) -> None:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark copier hot-path latency with recorded Webull fill payloads.")
    parser.add_argument("path", help="Path to a JSON object or JSON array of Webull fill payloads.")
    parser.add_argument("--iterations", type=int, default=100, help="Number of benchmark iterations.")
    parser.add_argument("--targets", type=int, default=1, help="Number of fake copy targets.")
    parser.add_argument("--broker-delay-ms", type=float, default=0.0, help="Artificial delay inside the fake broker client.")
    parser.add_argument("--master-equity", type=float, default=30_000.0, help="Benchmark master account equity.")
    parser.add_argument("--target-equity", type=float, default=10_000.0, help="Benchmark target account equity.")
    parser.add_argument("--json-out", help="Optional path to write the benchmark summary JSON.")
    args = parser.parse_args()

    if args.iterations <= 0:
        raise SystemExit("--iterations must be greater than 0")
    if args.targets <= 0:
        raise SystemExit("--targets must be greater than 0")

    payloads = _load_payloads(Path(args.path))
    config = CopierConfig(enabled=True, mode="test", global_kill_switch=False, master_equity=args.master_equity)
    valid_payloads = [payload for payload in payloads if _is_fill_payload(payload, config)]
    if not valid_payloads:
        raise SystemExit("No valid fill payloads found in benchmark input")

    clients = [BenchmarkTradingClient(broker_delay_ms=args.broker_delay_ms) for _ in range(args.targets)]
    targets = _benchmark_targets(clients, master_equity=args.master_equity, target_equity=args.target_equity)
    orchestrator = CopyOrchestrator(
        require_persistence=False,
        slack=NoopSlack(),
        background_alerts=False,
        persist_results=False,
        log_latency=False,
    )

    rows = []
    for idx in range(args.iterations):
        payload = dict(valid_payloads[idx % len(valid_payloads)])
        payload["order_id"] = f"{payload.get('order_id', 'bench-order')}-{idx}"
        payload["execution_id"] = f"{payload.get('execution_id') or payload['order_id']}-{idx}"
        master = MasterExecutionEvent.from_webull_payload(payload)
        started = datetime.now(tz=timezone.utc)
        timer_started = perf_counter()
        results = orchestrator.copy_execution(master, targets)
        elapsed_ms = (perf_counter() - timer_started) * 1000
        submitted = [result for result in results if result.submitted]
        first_submit_ms = min(
            ((result.submitted_at - started).total_seconds() * 1000 for result in submitted if result.submitted_at),
            default=None,
        )
        broker_response_ms = max((result.latency_ms for result in submitted if result.latency_ms is not None), default=None)
        rows.append(
            {
                "iteration": idx + 1,
                "submitted": len(submitted),
                "blocked": sum(1 for result in results if not result.submitted),
                "event_to_submit_started_ms": first_submit_ms,
                "broker_submit_ms": broker_response_ms,
                "event_to_broker_response_ms": elapsed_ms,
            }
        )

    summary = _summary(rows, args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def _benchmark_targets(
    clients: list[BenchmarkTradingClient],
    master_equity: float,
    target_equity: float,
) -> list[CopyTarget]:
    targets = []
    for idx, client in enumerate(clients, start=1):
        targets.append(
            CopyTarget(
                name=f"bench_{idx}",
                account_id=f"bench-account-{idx}",
                client=client,
                sizing=SizingPolicy(mode="percent_equity"),
                risk=RiskPolicy(enabled=True, global_kill_switch=False),
                master_equity=master_equity,
                target_equity=target_equity,
            )
        )
    return targets


def _summary(rows: list[dict], args: argparse.Namespace) -> dict:
    event_to_submit = _series(rows, "event_to_submit_started_ms")
    broker_submit = _series(rows, "broker_submit_ms")
    event_to_response = _series(rows, "event_to_broker_response_ms")
    return {
        "iterations": args.iterations,
        "targets": args.targets,
        "broker_delay_ms": args.broker_delay_ms,
        "submitted": sum(row["submitted"] for row in rows),
        "blocked": sum(row["blocked"] for row in rows),
        "latency": {
            "event_to_submit_started_ms": _stats(event_to_submit),
            "broker_submit_ms": _stats(broker_submit),
            "event_to_broker_response_ms": _stats(event_to_response),
        },
        "sub_300ms_pass_rate": _pass_rate(event_to_response, threshold_ms=300),
        "rows": rows,
    }


def _series(rows: list[dict], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) is not None]


def _stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "min": None, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "mean": mean(ordered),
        "p50": median(ordered),
        "p95": _percentile(ordered, 95),
        "p99": _percentile(ordered, 99),
        "max": ordered[-1],
    }


def _percentile(ordered: list[float], percentile: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (percentile / 100)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _pass_rate(values: list[float], threshold_ms: float) -> float | None:
    if not values:
        return None
    return sum(1 for value in values if value <= threshold_ms) / len(values)


def _load_payloads(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, dict):
        data = raw.get("data")
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        return [raw]
    raise ValueError("Benchmark input must be a JSON object or JSON array")


if __name__ == "__main__":
    main()
