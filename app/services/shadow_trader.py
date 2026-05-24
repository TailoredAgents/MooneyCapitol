from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import func, select

from app.db.models import Fill, ShadowDecision
from app.db.session import get_session
from app.observability.logging import get_logger


logger = get_logger("shadow_trader")
MODEL_VERSION = "shadow-v1"


def _to_float(value: Any) -> float | None:
    if value in (None, "", "n/a", "-"):
        return None
    if isinstance(value, (list, tuple)):
        return _to_float(value[0]) if value else None
    if isinstance(value, str):
        value = value.replace("$", "").replace(",", "").replace("x", "").replace("R", "").strip()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _confidence(p2r: float | None) -> str:
    if p2r is None:
        return "low"
    if p2r >= 0.85:
        return "high"
    if p2r >= 0.70:
        return "medium"
    return "low"


def evaluate_shadow_decision(payload: dict[str, Any], alert_type: str) -> dict[str, Any]:
    min_p2r = float(os.getenv("SHADOW_TRADER_MIN_P2R", "0.70"))
    min_rr = float(os.getenv("SHADOW_TRADER_MIN_RR", "2.00"))
    size_pct = float(os.getenv("SHADOW_TRADER_SIZE_PCT", "0.05"))

    p2r = _to_float(payload.get("p2r"))
    rr = _to_float(payload.get("rr") or payload.get("rr_value"))
    entry = _to_float(payload.get("entry") or payload.get("entry_price"))
    stop = _to_float(payload.get("stop"))
    target = _to_float(payload.get("target") or payload.get("targets") or payload.get("primary_target"))
    note = str(payload.get("note") or "").strip().lower()

    blockers: list[str] = []
    positives: list[str] = []
    if p2r is None:
        blockers.append("missing p2R")
    elif p2r >= min_p2r:
        positives.append(f"p2R {p2r:.2f} met threshold {min_p2r:.2f}")
    else:
        blockers.append(f"p2R {p2r:.2f} below threshold {min_p2r:.2f}")

    if rr is None:
        blockers.append("missing RR")
    elif rr >= min_rr:
        positives.append(f"RR {rr:.2f} met threshold {min_rr:.2f}")
    else:
        blockers.append(f"RR {rr:.2f} below threshold {min_rr:.2f}")

    if "below learned threshold" in note:
        blockers.append("below learned threshold")
    if alert_type == "trigger":
        positives.append("active trigger")
    elif alert_type == "primed":
        positives.append("primed setup only")

    would_take = not blockers and alert_type == "trigger"
    decision = "would_take" if would_take else ("watch" if not blockers else "skip")
    reason = "; ".join(positives + blockers) or "no decision context"

    return {
        "p2r": p2r,
        "rr": rr,
        "entry_price": entry,
        "stop_price": stop,
        "target_price": target,
        "suggested_size_pct": size_pct if would_take else None,
        "would_take": would_take,
        "decision": decision,
        "confidence": _confidence(p2r),
        "reason": reason,
        "reason_json": {
            "min_p2r": min_p2r,
            "min_rr": min_rr,
            "positives": positives,
            "blockers": blockers,
        },
    }


def record_shadow_decision(
    *,
    alert_id: int | None,
    setup_id: int | None,
    symbol: str,
    direction: str | None,
    alert_type: str,
    payload: dict[str, Any],
) -> int | None:
    if os.getenv("SHADOW_TRADER_ENABLED", "1").lower() not in {"1", "true", "yes", "on"}:
        return None

    evaluation = evaluate_shadow_decision(payload, alert_type)
    now = datetime.now(timezone.utc)
    with get_session() as session:
        existing = None
        if alert_id is not None:
            existing = session.execute(
                select(ShadowDecision).where(ShadowDecision.alert_id == alert_id)
            ).scalar_one_or_none()
        decision = existing or ShadowDecision(alert_id=alert_id)
        decision.setup_id = setup_id
        decision.symbol = symbol
        decision.direction = direction
        decision.alert_type = alert_type
        decision.observed_at = now
        decision.model_version = MODEL_VERSION
        decision.p2r = evaluation["p2r"]
        decision.rr = evaluation["rr"]
        decision.entry_price = evaluation["entry_price"]
        decision.stop_price = evaluation["stop_price"]
        decision.target_price = evaluation["target_price"]
        decision.suggested_size_pct = evaluation["suggested_size_pct"]
        decision.would_take = evaluation["would_take"]
        decision.decision = evaluation["decision"]
        decision.confidence = evaluation["confidence"]
        decision.reason = evaluation["reason"]
        decision.reason_json = evaluation["reason_json"]
        decision.payload_json = payload
        decision.status = "observed"
        if existing is None:
            decision.created_at = now
            session.add(decision)
        session.flush()
        return int(decision.id)


def _master_taken_map(session, setup_ids: list[int]) -> dict[int, bool]:
    if not setup_ids:
        return {}
    rows = session.execute(
        select(Fill.setup_id, func.count(Fill.id)).where(Fill.setup_id.in_(setup_ids)).group_by(Fill.setup_id)
    ).all()
    return {int(setup_id): bool(count) for setup_id, count in rows if setup_id is not None}


def list_shadow_decisions(limit: int = 100) -> dict[str, Any]:
    limit = max(1, min(int(limit), 250))
    with get_session() as session:
        decisions = list(
            session.execute(
                select(ShadowDecision).order_by(ShadowDecision.observed_at.desc()).limit(limit)
            ).scalars()
        )
        setup_ids = [int(row.setup_id) for row in decisions if row.setup_id is not None]
        taken_map = _master_taken_map(session, setup_ids)
        items = [_serialize(row, taken_map.get(int(row.setup_id), False) if row.setup_id is not None else False) for row in decisions]
        return {"items": items, "count": len(items), "summary": shadow_summary_from_session(session)}


def shadow_summary_from_session(session) -> dict[str, Any]:
    since = datetime.now(timezone.utc) - timedelta(days=1)
    rows = list(session.execute(select(ShadowDecision).where(ShadowDecision.observed_at >= since)).scalars())
    would_take = [row for row in rows if row.would_take]
    setup_ids = [int(row.setup_id) for row in rows if row.setup_id is not None]
    taken_map = _master_taken_map(session, setup_ids)
    matched = [row for row in would_take if row.setup_id is not None and taken_map.get(int(row.setup_id), False)]
    return {
        "window": "24h",
        "total": len(rows),
        "would_take": len(would_take),
        "watch": sum(1 for row in rows if row.decision == "watch"),
        "skip": sum(1 for row in rows if row.decision == "skip"),
        "matched_master": len(matched),
        "match_rate": round(len(matched) / len(would_take), 4) if would_take else None,
    }


def _serialize(row: ShadowDecision, master_taken: bool) -> dict[str, Any]:
    return {
        "id": row.id,
        "alert_id": row.alert_id,
        "setup_id": row.setup_id,
        "symbol": row.symbol,
        "direction": row.direction,
        "alert_type": row.alert_type,
        "observed_at": row.observed_at.isoformat() if row.observed_at else None,
        "model_version": row.model_version,
        "p2r": row.p2r,
        "rr": row.rr,
        "entry_price": row.entry_price,
        "stop_price": row.stop_price,
        "target_price": row.target_price,
        "suggested_size_pct": row.suggested_size_pct,
        "would_take": row.would_take,
        "decision": row.decision,
        "confidence": row.confidence,
        "reason": row.reason,
        "status": row.status,
        "master_taken": master_taken,
    }
