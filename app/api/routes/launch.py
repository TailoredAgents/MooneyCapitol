from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alembic.config import Config
from alembic.script import ScriptDirectory
from fastapi import APIRouter, Depends
from sqlalchemy import desc, func, select, text
from sqlalchemy.orm import Session

from app.api.auth import require_operator
from app.copier.readiness import evaluate_copier_readiness
from app.copier.state import get_copier_status
from app.core.config_store import CONFIG, refresh_config
from app.db.models import AccountSnapshot, CopierAuditEvent, CopyOrder, CopyReconciliation, CopyTargetAccount, MasterExecution
from app.db.session import engine, get_session
from app.services.ai_artifacts import latest_ai_artifact
from app.services.kv_store import get_store_mode, get_updated_at
from app.services.learning import get_learning_service
from app.services.runtime import get_worker_tick


router = APIRouter(prefix="/launch", tags=["launch"], dependencies=[Depends(require_operator)])


def db_session():
    with get_session() as session:
        yield session


@router.get("/readiness")
def launch_readiness(session: Session = Depends(db_session)):
    refresh_config()
    checks: list[dict[str, Any]] = []
    sections: dict[str, list[dict[str, Any]]] = {}

    def add(section: str, key: str, ok: bool, severity: str, label: str, context: dict | None = None) -> None:
        item = {"section": section, "key": key, "ok": bool(ok), "severity": severity, "label": label, "context": context or {}}
        checks.append(item)
        sections.setdefault(section, []).append(item)

    db_ok, db_error = _database_ok()
    add("Database", "database_connection", db_ok, "blocker", "Database connection is working", {"error": db_error})
    version = _migration_version(session)
    head = _migration_head()
    add(
        "Database",
        "migrations_current",
        bool(version and head and version == head),
        "blocker",
        "Database migrations are at the current Alembic head",
        {"database_version": version, "expected_head": head},
    )
    add("Database", "state_store_db", get_store_mode() == "db", "blocker", "State store is using Postgres", {"mode": get_store_mode()})

    worker_tick = get_worker_tick() or {}
    worker_ts = _parse_ts(worker_tick.get("ts"))
    worker_age_s = (datetime.now(tz=timezone.utc) - worker_ts).total_seconds() if worker_ts else None
    add(
        "Runtime",
        "worker_tick_fresh",
        worker_age_s is not None and worker_age_s <= 180,
        "blocker",
        "Worker heartbeat is fresh",
        {"last_tick": worker_tick.get("ts"), "age_seconds": worker_age_s},
    )
    add("Runtime", "operator_auth", _operator_auth_configured(), "blocker", "Operator dashboard/API authentication is configured")
    add("Runtime", "polygon_key", bool(os.getenv("POLYGON_API_KEY")), "blocker", "Polygon API key is configured for live scout data")
    add("Runtime", "slack_bot", bool(os.getenv("SLACK_BOT_TOKEN")), "warning", "Slack bot token is configured")
    add("Runtime", "slack_signing", bool(os.getenv("SLACK_SIGNING_SECRET")), "warning", "Slack signing secret is configured")
    add(
        "Runtime",
        "depth_mode",
        os.getenv("DEPTH_MODE", "demo").lower() != "demo",
        "warning",
        "Live L2 depth is enabled",
        {"depth_mode": os.getenv("DEPTH_MODE", "demo")},
    )

    copier_readiness = evaluate_copier_readiness(session)
    for item in copier_readiness.get("checks", []):
        add("Copier", item.get("key", "copier_check"), item.get("ok", False), item.get("severity", "warning"), item.get("label", ""), item.get("context") or {})
    copier_status = get_copier_status()
    add(
        "Copier",
        "runtime_no_error",
        not bool((copier_status.get("runtime") or {}).get("last_error")),
        "blocker",
        "Copier runtime has no current error",
        {"runtime": copier_status.get("runtime") or {}},
    )

    latest_snapshots = _latest_account_snapshots(session)
    newest_snapshot = max((row.snapshot_time for row in latest_snapshots), default=None)
    snapshot_age_s = _age_seconds(newest_snapshot)
    add("P&L", "pnl_snapshots_exist", bool(latest_snapshots), "blocker", "P&L account snapshots exist")
    add(
        "P&L",
        "pnl_snapshots_fresh",
        snapshot_age_s is not None and snapshot_age_s <= 300,
        "blocker",
        "P&L snapshots are fresh",
        {"latest_snapshot": newest_snapshot.isoformat() if newest_snapshot else None, "age_seconds": snapshot_age_s},
    )
    active_pnl_alerts = session.execute(text("select count(*) from pnl_alerts where acknowledged = false")).scalar() or 0
    add("P&L", "no_active_pnl_alerts", int(active_pnl_alerts) == 0, "blocker", "No active P&L risk alerts", {"active_alerts": int(active_pnl_alerts)})

    readonly_counts = _read_only_counts(session)
    readonly_history = _read_only_history(session)
    add(
        "Validation",
        "read_only_session_seen",
        readonly_counts["total"] > 0,
        "blocker",
        "At least one read-only copy validation decision is recorded",
        readonly_counts,
    )
    latency = _latency_stats(session)
    add(
        "Validation",
        "latency_samples_seen",
        latency["count"] > 0,
        "warning",
        "Copied-order latency samples exist",
        latency,
    )
    add(
        "Validation",
        "latency_under_300ms",
        latency["count"] > 0 and (latency["under_300ms_rate"] or 0) >= 0.95,
        "warning",
        "At least 95% of copied-order latency samples are under 300 ms",
        latency,
    )

    learning_report = _learning_report()
    learning_translation = _learning_translation(session)
    add(
        "Learning",
        "learning_report_exists",
        learning_report is not None,
        "warning",
        "Latest learning report is available",
        {"model_type": (learning_report or {}).get("model_type"), "date": (learning_report or {}).get("date")},
    )

    open_reconciliations = session.execute(select(func.count(CopyReconciliation.id)).where(CopyReconciliation.status == "open")).scalar() or 0
    recent_audit = session.execute(select(CopierAuditEvent).order_by(desc(CopierAuditEvent.created_at)).limit(10)).scalars().all()

    blockers = [item for item in checks if item["severity"] == "blocker" and not item["ok"]]
    warnings = [item for item in checks if item["severity"] == "warning" and not item["ok"]]
    return {
        "ready": not blockers,
        "blocker_count": len(blockers),
        "warning_count": len(warnings),
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "sections": sections,
        "blockers": blockers,
        "warnings": warnings,
        "summary": {
            "database_version": version,
            "expected_migration_head": head,
            "worker_tick": worker_tick,
            "copier": copier_status,
            "pnl_accounts": len(latest_snapshots),
            "open_reconciliations": int(open_reconciliations),
            "read_only_validation": readonly_counts,
            "read_only_history": readonly_history,
            "latency": latency,
            "learning_report": learning_report,
            "learning_translation": learning_translation,
            "kv_updated_at": {
                "app_config": _updated_at_iso("app_config"),
                "worker_tick": _updated_at_iso("worker_tick"),
                "lanes": _updated_at_iso("lanes"),
                "learning_report": _updated_at_iso("learning_report"),
            },
            "recent_audit_events": [
                {
                    "event_type": row.event_type,
                    "actor": row.actor,
                    "message": row.message,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }
                for row in recent_audit
            ],
        },
    }


def _database_ok() -> tuple[bool, str | None]:
    try:
        with engine.begin() as conn:
            conn.execute(text("select 1"))
        return True, None
    except Exception as exc:
        return False, str(exc)


def _migration_version(session: Session) -> str | None:
    try:
        return session.execute(text("select version_num from alembic_version limit 1")).scalar_one_or_none()
    except Exception:
        return None


def _migration_head() -> str | None:
    try:
        cfg = Config(str(Path("alembic.ini")))
        return ScriptDirectory.from_config(cfg).get_current_head()
    except Exception:
        return None


def _operator_auth_configured() -> bool:
    return all(os.getenv(name) for name in ("COWORK_OPERATOR_USERNAME", "COWORK_OPERATOR_PASSWORD", "COWORK_OPERATOR_API_TOKEN"))


def _latest_account_snapshots(session: Session) -> list[AccountSnapshot]:
    rows = session.execute(select(AccountSnapshot).order_by(desc(AccountSnapshot.snapshot_time))).scalars().all()
    latest: dict[str, AccountSnapshot] = {}
    for row in rows:
        latest.setdefault(row.account_ref, row)
    return list(latest.values())


def _read_only_counts(session: Session) -> dict[str, int]:
    would_copy = session.execute(select(func.count(CopyOrder.id)).where(CopyOrder.status == "would_copy")).scalar() or 0
    blocked = session.execute(select(func.count(CopyOrder.id)).where(CopyOrder.status == "blocked")).scalar() or 0
    return {"would_copy": int(would_copy), "blocked": int(blocked), "total": int(would_copy) + int(blocked)}


def _read_only_history(session: Session, limit: int = 25) -> list[dict[str, Any]]:
    rows = (
        session.execute(
            select(MasterExecution, CopyOrder, CopyTargetAccount.name)
            .join(CopyOrder, CopyOrder.master_execution_id == MasterExecution.id)
            .join(CopyTargetAccount, CopyTargetAccount.id == CopyOrder.target_account_id)
            .where(CopyOrder.status.in_(["would_copy", "blocked"]))
            .order_by(desc(MasterExecution.executed_at), desc(CopyOrder.id))
            .limit(limit)
        )
        .all()
    )
    return [_serialize_read_only_row(master, order, target_name) for master, order, target_name in rows]


def _serialize_read_only_row(master: MasterExecution, order: CopyOrder, target_name: str | None) -> dict[str, Any]:
    copy_notional = None
    if order.qty is not None and master.price is not None:
        copy_notional = float(order.qty) * float(master.price)
    return {
        "master_execution_id": master.id,
        "copy_order_id": order.id,
        "master_executed_at": _iso(master.executed_at),
        "master_received_at": _iso(master.received_at),
        "symbol": master.symbol,
        "side": master.side,
        "master_qty": master.qty,
        "master_price": master.price,
        "master_notional": float(master.qty) * float(master.price),
        "target": target_name,
        "client_order_id": order.client_order_id,
        "copy_qty": order.qty,
        "copy_notional": copy_notional,
        "copy_status": order.status,
        "copy_reject_reason": order.reject_reason,
        "copy_latency_ms": order.latency_ms,
    }


def _latency_stats(session: Session) -> dict[str, Any]:
    values = [
        float(row)
        for row in session.execute(select(CopyOrder.latency_ms).where(CopyOrder.latency_ms.is_not(None))).scalars().all()
        if row is not None
    ]
    if not values:
        return {"count": 0, "mean_ms": None, "max_ms": None, "under_300ms_rate": None}
    under = sum(1 for value in values if value <= 300)
    return {
        "count": len(values),
        "mean_ms": round(sum(values) / len(values), 2),
        "max_ms": round(max(values), 2),
        "under_300ms_rate": round(under / len(values), 4),
    }


def _learning_report() -> dict | None:
    try:
        return get_learning_service().load_report()
    except Exception:
        return None


def _learning_translation(session: Session) -> dict | None:
    try:
        artifact = latest_ai_artifact(session, artifact_type="learning_translation", source_type="learning_report")
    except Exception:
        return None
    if not artifact or artifact.status != "completed" or not artifact.output_text:
        return None
    return {
        "text": artifact.output_text,
        "model": artifact.model,
        "source_id": artifact.source_id,
        "created_at": artifact.created_at.isoformat() if artifact.created_at else None,
    }


def _updated_at_iso(key: str) -> str | None:
    try:
        value = get_updated_at(key)
        return value.isoformat() if value else None
    except Exception:
        return None


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _age_seconds(value: datetime | None) -> float | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return (datetime.now(tz=timezone.utc) - value.astimezone(timezone.utc)).total_seconds()


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None
