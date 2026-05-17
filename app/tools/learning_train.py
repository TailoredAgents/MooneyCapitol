from __future__ import annotations

import argparse
import json
from datetime import date

from app.services.learning import get_learning_service


def _parse_date(value: str | None) -> date:
    if not value:
        return date.today()
    return date.fromisoformat(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or inspect MooneyCapitol learning training.")
    parser.add_argument("--date", help="Trade date in YYYY-MM-DD format. Defaults to today.")
    parser.add_argument("--dry-run", action="store_true", help="Build dataset and print counts without training.")
    parser.add_argument("--train", action="store_true", help="Train and persist learning artifacts.")
    args = parser.parse_args()

    if args.dry_run == args.train:
        parser.error("choose exactly one of --dry-run or --train")

    trade_date = _parse_date(args.date)
    service = get_learning_service()

    if args.dry_run:
        df = service._build_rows(trade_date)
        payload = {
            "date": trade_date.isoformat(),
            "rows": int(len(df)),
            "label_breakdown": service._label_breakdown(df),
            "columns": list(df.columns) if not df.empty else [],
        }
    else:
        payload = service.train(trade_date)

    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()

