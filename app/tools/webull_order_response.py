from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.copier.order_response import normalize_copy_order_responses


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Webull child-order submit/detail/status response payloads.")
    parser.add_argument("path", help="Path to a JSON object, JSON array, or JSONL file.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    payloads = _load_payloads(Path(args.path))
    records = []
    for idx, payload in enumerate(payloads, start=1):
        details = normalize_copy_order_responses(payload)
        records.append(
            {
                "sequence": idx,
                "normalized": [
                    {
                        "status": detail.status,
                        "broker_order_id": detail.broker_order_id,
                        "client_order_id": detail.client_order_id,
                        "filled_qty": detail.filled_qty,
                        "avg_fill_price": detail.avg_fill_price,
                        "reject_reason": detail.reject_reason,
                        "accepted_at": detail.accepted_at.isoformat() if detail.accepted_at else None,
                        "filled_at": detail.filled_at.isoformat() if detail.filled_at else None,
                    }
                    for detail in details
                ],
            }
        )

    summary = _summary(records)
    if args.json:
        print(json.dumps({"summary": summary, "records": records}, indent=2, sort_keys=True))
        return
    print(f"Webull child-order response analysis: {args.path}")
    print(f"payloads={summary['payloads']} orders={summary['orders']}")
    for status, count in sorted(summary["statuses"].items()):
        print(f"  - {status}: {count}")


def _load_payloads(path: Path) -> list[dict]:
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
        return [parsed]
    raise ValueError("Input must be a JSON object, JSON array, or JSONL file")


def _summary(records: list[dict]) -> dict:
    statuses: dict[str, int] = {}
    orders = 0
    for record in records:
        for detail in record["normalized"]:
            orders += 1
            status = detail["status"]
            statuses[status] = statuses.get(status, 0) + 1
    return {"payloads": len(records), "orders": orders, "statuses": statuses}


if __name__ == "__main__":
    main()
