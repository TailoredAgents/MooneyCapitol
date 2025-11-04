from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo


ET = ZoneInfo("America/New_York")


def now_et() -> datetime:
    return datetime.now(tz=ET)


def is_rth(dt: datetime | None = None) -> bool:
    dt = dt or now_et()
    start = time(9, 30, tzinfo=ET)
    end = time(16, 0, tzinfo=ET)
    return start <= dt.timetz() <= end


def seconds_until(target_h: int, target_m: int) -> float:
    now = now_et()
    target = now.replace(hour=target_h, minute=target_m, second=0, microsecond=0)
    if target <= now:
        target = target + timedelta(days=1)
    return (target - now).total_seconds()

