from __future__ import annotations

import argparse
import json
from decimal import Decimal

from app.db.session import get_session
from app.v2.execution.copy_control import FOLLOWER_CONNECTION_IDS
from app.v2.execution.lifecycle_store import SqlAlchemyLifecycleStore
from app.v2.execution.recovery import LatencyStage, LatencySummary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Report persisted V2 copier p50/p95/p99 lifecycle latency. "
            "The output contains fixed internal follower labels only."
        )
    )
    parser.add_argument(
        "--follower",
        choices=FOLLOWER_CONNECTION_IDS,
        help="Limit the report to one internal follower slot.",
    )
    args = parser.parse_args()

    with get_session() as session:
        store = SqlAlchemyLifecycleStore(session)
        report = {
            stage.value: _row(
                store.latency_summary(
                    stage,
                    follower_connection_id=args.follower,
                )
            )
            for stage in LatencyStage
        }
    print(json.dumps(report, indent=2, sort_keys=True))


def _row(summary: LatencySummary) -> dict[str, object]:
    return {
        "sample_count": summary.sample_count,
        "p50_ms": _decimal_text(summary.p50_ms),
        "p95_ms": _decimal_text(summary.p95_ms),
        "p99_ms": _decimal_text(summary.p99_ms),
    }


def _decimal_text(value: Decimal | None) -> str | None:
    return None if value is None else format(value, "f")


if __name__ == "__main__":
    main()
