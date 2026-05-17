from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, time, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.copier.engine import CopyResult, CopyTarget
from app.copier.models import MasterExecutionEvent
from app.copier.order_response import normalize_copy_order_response
from app.copier.risk import RiskUsage
from app.db.models import CopierAuditEvent, CopyOrder, CopyOrderEvent, CopyTargetAccount, MasterExecution


def master_execution_exists(session: Session, event: MasterExecutionEvent) -> bool:
    return (
        session.execute(
            select(MasterExecution.id).where(
                MasterExecution.broker == event.broker,
                MasterExecution.account_ref == event.account_id,
                MasterExecution.broker_execution_id == event.execution_id,
            )
        ).scalar_one_or_none()
        is not None
    )


def persist_master_execution(session: Session, event: MasterExecutionEvent) -> MasterExecution:
    existing = session.execute(
        select(MasterExecution).where(
            MasterExecution.broker == event.broker,
            MasterExecution.account_ref == event.account_id,
            MasterExecution.broker_execution_id == event.execution_id,
        )
    ).scalar_one_or_none()
    if existing is not None:
        return existing

    row = MasterExecution(
        broker=event.broker,
        account_ref=event.account_id,
        broker_execution_id=event.execution_id,
        broker_order_id=event.order_id,
        symbol=event.symbol,
        side=event.side,
        qty=event.quantity,
        price=event.price,
        asset_class="equity",
        executed_at=event.executed_at,
        received_at=datetime.now(tz=timezone.utc),
        raw_payload=event.raw_payload,
    )
    session.add(row)
    session.flush()
    return row


def ensure_copy_target_account(session: Session, target: CopyTarget) -> CopyTargetAccount:
    row = session.execute(
        select(CopyTargetAccount).where(CopyTargetAccount.name == target.name)
    ).scalar_one_or_none()
    if row is None:
        row = CopyTargetAccount(name=target.name, broker="webull", environment="test")
        session.add(row)

    row.enabled = target.risk.enabled
    row.account_ref = target.account_id
    row.equity = target.target_equity
    row.sizing_mode = target.sizing.mode
    row.sizing_value = target.sizing.value
    row.min_notional = target.sizing.min_notional
    row.max_notional_per_trade = target.risk.max_notional_per_trade
    row.max_position_pct = target.risk.max_position_pct
    row.shorting_enabled = target.risk.shorting_enabled
    row.max_daily_notional = target.risk.max_daily_notional
    row.max_daily_trades = target.risk.max_daily_trades
    row.allowlist = sorted({item.upper() for item in target.risk.allowlist}) or None
    row.blocklist = sorted({item.upper() for item in target.risk.blocklist}) or None
    row.updated_at = datetime.now(tz=timezone.utc)
    session.flush()
    return row


def persist_copy_results(
    session: Session,
    master: MasterExecutionEvent,
    targets_by_name: Mapping[str, CopyTarget],
    results: list[CopyResult],
) -> MasterExecution:
    master_row = persist_master_execution(session, master)
    for result in results:
        target = targets_by_name.get(result.target)
        if target is None:
            _record_audit(session, "copy_target_missing", result, None)
            continue
        target_row = ensure_copy_target_account(session, target)

        if not result.client_order_id:
            _record_audit(session, "copy_blocked", result, target_row.id)
            continue

        copy_order = session.execute(
            select(CopyOrder).where(CopyOrder.client_order_id == result.client_order_id)
        ).scalar_one_or_none()
        if copy_order is None:
            copy_order = CopyOrder(
                master_execution_id=master_row.id,
                target_account_id=target_row.id,
                broker="webull",
                client_order_id=result.client_order_id,
                symbol=master.symbol,
                side=master.side,
                qty=result.quantity,
                order_type=(result.order.order_type if result.order else "MARKET").lower(),
                time_in_force=(result.order.time_in_force if result.order else "DAY").lower(),
            )
            session.add(copy_order)

        response_detail = normalize_copy_order_response(
            result.response,
            fallback_status=copy_order_status_for_result(result),
        )
        copy_order.status = response_detail.status
        copy_order.broker_order_id = response_detail.broker_order_id
        copy_order.submitted_at = result.submitted_at
        copy_order.accepted_at = response_detail.accepted_at
        copy_order.filled_at = response_detail.filled_at if response_detail.status == "filled" else None
        copy_order.filled_qty = response_detail.filled_qty
        copy_order.avg_fill_price = response_detail.avg_fill_price
        copy_order.reject_reason = result.error or response_detail.reject_reason or result.reason
        copy_order.latency_ms = result.latency_ms
        copy_order.raw_submit_payload = result.order.to_webull_payload() if result.order else None
        copy_order.raw_response_payload = result.response
        session.flush()

        if not result.allowed:
            _record_audit(session, "copy_blocked", result, target_row.id)

        session.add(
            CopyOrderEvent(
                copy_order_id=copy_order.id,
                event_type=copy_order.status,
                status=copy_order.status,
                event_at=result.submitted_at,
                received_at=datetime.now(tz=timezone.utc),
                raw_payload={
                    "reason": result.reason,
                    "error": result.error,
                    "response": result.response,
                    "latency_ms": result.latency_ms,
                },
            )
        )
    session.flush()
    return master_row


def load_copy_risk_usage(
    session: Session,
    target_names: list[str],
    now: datetime | None = None,
) -> dict[str, RiskUsage]:
    if not target_names:
        return {}
    now = now or datetime.now(tz=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    day_start = datetime.combine(now.date(), time.min, tzinfo=timezone.utc)
    minute_start = now - timedelta(minutes=1)
    active_statuses = ["submitted", "accepted", "partially_filled", "filled"]

    daily_rows = session.execute(
        select(
            CopyTargetAccount.name,
            func.coalesce(func.sum(CopyOrder.qty * MasterExecution.price), 0.0),
            func.count(CopyOrder.id),
        )
        .join(CopyOrder, CopyOrder.target_account_id == CopyTargetAccount.id)
        .join(MasterExecution, CopyOrder.master_execution_id == MasterExecution.id)
        .where(CopyTargetAccount.name.in_(target_names))
        .where(CopyOrder.status.in_(active_statuses))
        .where(CopyOrder.submitted_at >= day_start)
        .group_by(CopyTargetAccount.name)
    ).all()
    minute_rows = session.execute(
        select(CopyTargetAccount.name, func.count(CopyOrder.id))
        .join(CopyOrder, CopyOrder.target_account_id == CopyTargetAccount.id)
        .where(CopyTargetAccount.name.in_(target_names))
        .where(CopyOrder.status.in_(active_statuses))
        .where(CopyOrder.submitted_at >= minute_start)
        .group_by(CopyTargetAccount.name)
    ).all()

    usage = {name: RiskUsage() for name in target_names}
    for name, daily_notional, daily_orders in daily_rows:
        usage[name] = RiskUsage(daily_notional=float(daily_notional or 0.0), daily_orders=int(daily_orders or 0))
    for name, minute_orders in minute_rows:
        current = usage.get(name, RiskUsage())
        usage[name] = RiskUsage(
            daily_notional=current.daily_notional,
            daily_orders=current.daily_orders,
            minute_orders=int(minute_orders or 0),
        )
    return usage


def load_copy_positions(session: Session, target_names: list[str]) -> dict[str, dict[str, float]]:
    if not target_names:
        return {}
    active_statuses = ["submitted", "accepted", "partially_filled", "filled"]
    rows = session.execute(
        select(
            CopyTargetAccount.name,
            CopyOrder.symbol,
            CopyOrder.side,
            CopyOrder.qty,
            CopyOrder.filled_qty,
            CopyOrder.status,
        )
        .join(CopyOrder, CopyOrder.target_account_id == CopyTargetAccount.id)
        .where(CopyTargetAccount.name.in_(target_names))
        .where(CopyOrder.status.in_(active_statuses))
    ).all()
    positions: dict[str, dict[str, float]] = {name: {} for name in target_names}
    for target_name, symbol, side, qty, filled_qty, status in rows:
        symbol_key = str(symbol or "").upper()
        if not symbol_key:
            continue
        effective_qty = _effective_position_qty(qty, filled_qty, status)
        if effective_qty <= 0:
            continue
        signed_qty = effective_qty if str(side).upper() == "BUY" else -effective_qty
        target_positions = positions.setdefault(target_name, {})
        target_positions[symbol_key] = target_positions.get(symbol_key, 0.0) + signed_qty
    return positions


def _record_audit(
    session: Session,
    event_type: str,
    result: CopyResult,
    target_account_id: int | None,
) -> None:
    session.add(
        CopierAuditEvent(
            event_type=event_type,
            target_account_id=target_account_id,
            message=f"Copy attempt for target {result.target}: {result.reason or result.error or 'unknown'}",
            payload={
                "target": result.target,
                "allowed": result.allowed,
                "submitted": result.submitted,
                "reason": result.reason,
                "error": result.error,
                "quantity": result.quantity,
                "client_order_id": result.client_order_id,
            },
        )
    )


def copy_order_status_for_result(result: CopyResult) -> str:
    if result.submitted:
        return "submitted"
    if result.allowed and result.reason == "read_only":
        return "would_copy"
    if not result.allowed:
        return "blocked"
    return "submit_failed"


def _effective_position_qty(qty, filled_qty, status: str | None) -> float:
    normalized_status = str(status or "").lower()
    if normalized_status in {"filled", "partially_filled"} and filled_qty not in (None, ""):
        try:
            return max(float(filled_qty), 0.0)
        except (TypeError, ValueError):
            return 0.0
    try:
        return max(float(qty or 0.0), 0.0)
    except (TypeError, ValueError):
        return 0.0

