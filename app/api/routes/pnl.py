from __future__ import annotations

import os
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.api.auth import require_operator
from app.core.config_store import CONFIG
from app.db.models import AccountSnapshot, PnlAlert, PositionSnapshot, TradingSession
from app.db.session import get_session
from app.services.pnl_monitor import get_pnl_monitor, latest_account_snapshots, latest_position_snapshots, target_display_name


router = APIRouter(prefix="/pnl", tags=["pnl"], dependencies=[Depends(require_operator)])


def db_session():
    with get_session() as session:
        yield session


class AccountStatusResponse(BaseModel):
    account_ref: str
    account_name: str
    account_type: str
    timestamp: datetime
    cash_balance: float | None
    equity_value: float | None
    total_value: float
    buying_power: float | None
    unrealized_pnl: float
    realized_pnl_today: float
    total_pnl_today: float
    max_drawdown_today: float
    max_profit_today: float
    position_count: int
    total_exposure: float
    risk_level: str
    is_healthy: bool


@router.post("/refresh")
async def refresh_pnl():
    statuses = await get_pnl_monitor().collect_once_async()
    return {"updated": len(statuses), "accounts": [_status_response(status).model_dump() for status in statuses]}


@router.get("/accounts")
def get_accounts(session: Session = Depends(db_session)):
    rows = latest_account_snapshots(session)
    return {"count": len(rows), "items": [_snapshot_response(row).model_dump() for row in rows]}


@router.get("/accounts/{account_ref}")
def get_account(account_ref: str, session: Session = Depends(db_session)):
    row = (
        session.execute(
            select(AccountSnapshot)
            .where(AccountSnapshot.account_ref == account_ref)
            .order_by(desc(AccountSnapshot.snapshot_time))
            .limit(1)
        )
        .scalars()
        .first()
    )
    if row is None:
        raise HTTPException(status_code=404, detail="P&L account snapshot not found")
    return _snapshot_response(row)


@router.get("/accounts/{account_ref}/positions")
def get_positions(account_ref: str, session: Session = Depends(db_session)):
    rows = latest_position_snapshots(session, account_ref)
    return {"count": len(rows), "items": [_position_response(row) for row in rows]}


@router.get("/accounts/{account_ref}/alerts")
def get_alerts(
    account_ref: str,
    include_acknowledged: bool = False,
    limit: int = Query(default=50, ge=1, le=250),
    session: Session = Depends(db_session),
):
    stmt = (
        select(PnlAlert)
        .where(PnlAlert.account_ref == account_ref)
        .order_by(desc(PnlAlert.alert_time))
        .limit(limit)
    )
    if not include_acknowledged:
        stmt = stmt.where(PnlAlert.acknowledged.is_(False))
    rows = session.execute(stmt).scalars().all()
    return {"count": len(rows), "items": [_alert_response(row) for row in rows]}


@router.post("/alerts/{alert_id}/acknowledge")
def acknowledge_alert(alert_id: int, session: Session = Depends(db_session)):
    row = session.execute(select(PnlAlert).where(PnlAlert.id == alert_id)).scalars().first()
    if row is None:
        raise HTTPException(status_code=404, detail="P&L alert not found")
    row.acknowledged = True
    row.acknowledged_by = "operator"
    row.acknowledged_at = datetime.utcnow()
    return {"status": "acknowledged", "id": alert_id}


@router.get("/sessions")
def get_sessions(
    account_ref: str | None = None,
    limit: int = Query(default=50, ge=1, le=250),
    session: Session = Depends(db_session),
):
    stmt = select(TradingSession).order_by(desc(TradingSession.trade_date)).limit(limit)
    if account_ref:
        stmt = stmt.where(TradingSession.account_ref == account_ref)
    rows = session.execute(stmt).scalars().all()
    return {"count": len(rows), "items": [_session_response(row) for row in rows]}


@router.get("/summary")
def get_summary(session: Session = Depends(db_session)):
    rows = latest_account_snapshots(session)
    total_value = sum(float(row.total_value or 0.0) for row in rows)
    total_pnl = sum(float(row.total_pnl_today or 0.0) for row in rows)
    master_rows = [row for row in rows if row.account_type == "master"]
    copy_rows = [row for row in rows if row.account_type == "copy"]
    active_alerts = (
        session.execute(select(PnlAlert).where(PnlAlert.acknowledged.is_(False))).scalars().all()
    )
    return {
        "account_count": len(rows),
        "total_value": total_value,
        "total_pnl_today": total_pnl,
        "combined_summary": _summary_bucket(rows, active_alerts),
        "master_summary": _summary_bucket(master_rows, active_alerts),
        "copy_summary": _summary_bucket(copy_rows, active_alerts),
        "active_alerts": len(active_alerts),
        "accounts": [_snapshot_response(row).model_dump() for row in rows],
    }


def _summary_bucket(rows: list[AccountSnapshot], alerts: list[PnlAlert]) -> dict:
    account_refs = {row.account_ref for row in rows}
    matching_alerts = [alert for alert in alerts if alert.account_ref in account_refs]
    latest = max((row.snapshot_time for row in rows), default=None)
    return {
        "account_count": len(rows),
        "total_value": sum(float(row.total_value or 0.0) for row in rows),
        "total_pnl_today": sum(float(row.total_pnl_today or 0.0) for row in rows),
        "cash_balance": sum(float(row.cash_balance or 0.0) for row in rows),
        "buying_power": sum(float(row.buying_power or 0.0) for row in rows),
        "total_exposure": sum(float(row.total_exposure or 0.0) for row in rows),
        "position_count": sum(int(row.position_count or 0) for row in rows),
        "active_alerts": len(matching_alerts),
        "last_refresh": latest.isoformat() if latest else None,
    }


def _status_response(status) -> AccountStatusResponse:
    return AccountStatusResponse(
        account_ref=status.account_ref,
        account_name=status.account_name,
        account_type=status.account_type,
        timestamp=status.timestamp,
        cash_balance=status.cash_balance,
        equity_value=status.equity_value,
        total_value=status.total_value,
        buying_power=status.buying_power,
        unrealized_pnl=status.unrealized_pnl,
        realized_pnl_today=status.realized_pnl_today,
        total_pnl_today=status.total_pnl_today,
        max_drawdown_today=status.max_drawdown_today,
        max_profit_today=status.max_profit_today,
        position_count=status.position_count,
        total_exposure=status.total_exposure,
        risk_level=status.risk_level,
        is_healthy=status.is_healthy,
    )


def _snapshot_response(row: AccountSnapshot) -> AccountStatusResponse:
    return AccountStatusResponse(
        account_ref=row.account_ref,
        account_name=_friendly_account_name(row.account_ref, row.account_name, row.account_type),
        account_type=row.account_type,
        timestamp=row.snapshot_time,
        cash_balance=row.cash_balance,
        equity_value=row.equity_value,
        total_value=row.total_value,
        buying_power=row.buying_power,
        unrealized_pnl=row.unrealized_pnl,
        realized_pnl_today=row.realized_pnl_today,
        total_pnl_today=row.total_pnl_today,
        max_drawdown_today=row.max_drawdown_today,
        max_profit_today=row.max_profit_today,
        position_count=row.position_count,
        total_exposure=row.total_exposure,
        risk_level=row.risk_level,
        is_healthy=row.risk_level in {"normal", "warning"},
    )


def _position_response(row: PositionSnapshot) -> dict:
    return {
        "account_ref": row.account_ref,
        "account_name": _friendly_account_name(row.account_ref, row.account_name, row.account_type),
        "symbol": row.symbol,
        "snapshot_time": row.snapshot_time,
        "qty": row.qty,
        "avg_price": row.avg_price,
        "current_price": row.current_price,
        "market_value": row.market_value,
        "cost_basis": row.cost_basis,
        "unrealized_pnl": row.unrealized_pnl,
        "unrealized_pnl_pct": row.unrealized_pnl_pct,
        "side": row.side,
    }


def _alert_response(row: PnlAlert) -> dict:
    return {
        "id": row.id,
        "account_ref": row.account_ref,
        "account_name": _friendly_account_name(row.account_ref, row.account_name),
        "alert_time": row.alert_time,
        "alert_type": row.alert_type,
        "severity": row.severity,
        "message": row.message,
        "actual_value": row.actual_value,
        "threshold_value": row.threshold_value,
        "acknowledged": row.acknowledged,
    }


def _friendly_account_name(account_ref: str, stored_name: str | None, account_type: str | None = None) -> str:
    if account_type == "master":
        return stored_name or "master"
    for target in CONFIG.copier.targets:
        target_ref = target.account_ref or os.getenv(target.account_id_env)
        if account_ref == target_ref or stored_name == target.name:
            return target_display_name(target)
    return stored_name or account_ref


def _session_response(row: TradingSession) -> dict:
    return {
        "id": row.id,
        "account_ref": row.account_ref,
        "account_name": _friendly_account_name(row.account_ref, row.account_name),
        "trade_date": row.trade_date,
        "active": row.active,
        "starting_value": row.starting_value,
        "ending_value": row.ending_value,
        "realized_pnl": row.realized_pnl,
        "unrealized_pnl": row.unrealized_pnl,
        "total_pnl": row.total_pnl,
        "max_drawdown": row.max_drawdown,
        "max_profit": row.max_profit,
        "trades_count": row.trades_count,
    }
