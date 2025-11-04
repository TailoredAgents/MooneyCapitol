from __future__ import annotations


def project_rr(direction: str, entry: float, stop: float, target: float | None) -> float | None:
    if target is None:
        return None
    if direction == "long":
        risk = entry - stop
        reward = target - entry
        if risk <= 0 or reward <= 0:
            return None
        return reward / risk
    if direction == "short":
        risk = stop - entry
        reward = entry - target
        if risk <= 0 or reward <= 0:
            return None
        return reward / risk
    return None

