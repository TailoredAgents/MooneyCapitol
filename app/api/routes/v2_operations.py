from __future__ import annotations

import hashlib
import os
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Iterable
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy import desc, func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.api.auth import require_operator
from app.db.session import get_session
from app.v2.db_models import (
    V2RithmicAccountObservation,
    V2RithmicBracketObservation,
    V2RithmicBrokerEvent,
    V2RithmicConnectionGeneration,
    V2RithmicExecutionObservation,
    V2RithmicOrderObservation,
    V2RithmicPnLObservation,
    V2RithmicReferenceObservation,
    V2RithmicReconciliationCheckpoint,
    V2RithmicRmsObservation,
)


router = APIRouter(
    prefix="/v2/operations",
    tags=["v2-operations"],
    dependencies=[Depends(require_operator)],
)

_CAPTURE_BASE_URL_ENV = "RITHMIC_CAPTURE_BASE_URL"
_CAPTURE_TIMEOUT_SECONDS_ENV = "RITHMIC_CAPTURE_HEALTH_TIMEOUT_SECONDS"
_MAX_CAPTURE_TIMEOUT_SECONDS = 5.0
_DEFAULT_CAPTURE_TIMEOUT_SECONDS = 2.0


def db_session():
    with get_session() as session:
        yield session


@router.get("/overview")
async def operations_overview(
    response: Response,
    limit: int = Query(default=40, ge=1, le=100),
    session: Session = Depends(db_session),
) -> dict[str, Any]:
    try:
        snapshot = build_operations_snapshot(session, limit=limit)
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=503,
            detail="V2 observation store is unavailable",
        ) from exc

    response.headers["Cache-Control"] = "no-store"
    snapshot["capture"] = await fetch_capture_health()
    snapshot["generated_at"] = _iso(datetime.now(tz=timezone.utc))
    return snapshot


def build_operations_snapshot(session: Session, *, limit: int = 40) -> dict[str, Any]:
    counts = {
        "events": _count(session, V2RithmicBrokerEvent),
        "account_observations": _count(session, V2RithmicAccountObservation),
        "orders": _count(session, V2RithmicOrderObservation),
        "executions": _count(session, V2RithmicExecutionObservation),
        "brackets": _count(session, V2RithmicBracketObservation),
        "pnl": _count(session, V2RithmicPnLObservation),
        "rms": _count(session, V2RithmicRmsObservation),
        "references": _count(session, V2RithmicReferenceObservation),
    }

    event_rows = _recent(session, V2RithmicBrokerEvent, "received_at", 1)
    generation_rows = _recent(session, V2RithmicConnectionGeneration, "connected_at", 30)
    account_rows = _recent(session, V2RithmicAccountObservation, "observed_at", 250)
    order_rows = _recent(session, V2RithmicOrderObservation, "received_at", limit)
    execution_rows = _recent(session, V2RithmicExecutionObservation, "received_at", limit)
    bracket_rows = _recent(session, V2RithmicBracketObservation, "received_at", limit)
    pnl_rows = _recent(session, V2RithmicPnLObservation, "received_at", 500)
    rms_rows = _recent(session, V2RithmicRmsObservation, "received_at", 250)
    reconciliation_rows = _recent(
        session,
        V2RithmicReconciliationCheckpoint,
        "recorded_at",
        250,
    )
    reference_rows = _recent(session, V2RithmicReferenceObservation, "received_at", limit)

    latest_accounts = _latest_by(
        account_rows,
        lambda row: row.account_id or row.broker_account_id,
    )
    latest_pnl = _latest_by(
        pnl_rows,
        lambda row: (
            row.account_id or row.broker_account_id,
            row.scope,
            row.symbol,
            row.exchange,
        ),
    )
    latest_rms = _latest_by(
        rms_rows,
        lambda row: (
            row.account_id or row.broker_account_id,
            row.scope,
            row.product_code,
        ),
    )
    latest_reconciliation = _latest_by(
        reconciliation_rows,
        lambda row: (row.account_id or row.broker_account_id, row.plant),
    )
    latest_plants = _latest_by(generation_rows, lambda row: row.plant)
    counts["accounts"] = len(latest_accounts)

    return {
        "mode": {
            "platform": "V2",
            "broker": "Rithmic",
            "environment": "TEST",
            "observation_only": True,
            "submission_enabled": False,
        },
        "storage": {
            **counts,
            "last_event_at": _iso(event_rows[0].received_at) if event_rows else None,
        },
        "plants": [_serialize_plant(row) for row in latest_plants],
        "accounts": [_serialize_account(row) for row in latest_accounts],
        "orders": [_serialize_order(row) for row in order_rows],
        "executions": [_serialize_execution(row) for row in execution_rows],
        "brackets": [_serialize_bracket(row) for row in bracket_rows],
        "account_pnl": [
            _serialize_pnl(row) for row in latest_pnl if row.scope == "ACCOUNT"
        ],
        "positions": [
            _serialize_pnl(row) for row in latest_pnl if row.scope == "INSTRUMENT"
        ],
        "risk": [_serialize_rms(row) for row in latest_rms],
        "reconciliation": [
            _serialize_reconciliation(row) for row in latest_reconciliation
        ],
        "contracts": [_serialize_reference(row) for row in reference_rows],
    }


async def fetch_capture_health() -> dict[str, Any]:
    base_url = os.getenv(_CAPTURE_BASE_URL_ENV, "").strip().rstrip("/")
    if not base_url:
        return {
            "configured": False,
            "reachable": False,
            "live": False,
            "ready": False,
            "submission_enabled": False,
            "status": "health_url_not_configured",
            "plants": [],
            "blockers": [],
        }

    parsed = urlparse(base_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        return {
            "configured": True,
            "reachable": False,
            "live": False,
            "ready": False,
            "submission_enabled": False,
            "status": "health_url_invalid",
            "plants": [],
            "blockers": [],
        }

    timeout = _capture_timeout_seconds()
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            response = await client.get(f"{base_url}/health")
        if response.status_code not in {200, 503}:
            raise ValueError("unexpected health status")
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("invalid health payload")
    except (httpx.HTTPError, ValueError):
        return {
            "configured": True,
            "reachable": False,
            "live": False,
            "ready": False,
            "submission_enabled": False,
            "status": "capture_unreachable",
            "plants": [],
            "blockers": [],
        }

    plants = payload.get("plants") if isinstance(payload.get("plants"), list) else []
    blockers = payload.get("blockers") if isinstance(payload.get("blockers"), list) else []
    return {
        "configured": True,
        "reachable": True,
        "live": bool(payload.get("live")),
        "ready": bool(payload.get("ready")),
        "connectivity_enabled": bool(payload.get("connectivity_enabled")),
        "submission_enabled": False,
        "lifecycle": _safe_text(payload.get("lifecycle"), 32),
        "configured_account_count": _safe_int(payload.get("configured_account_count")),
        "discovered_allowed_account_count": _safe_int(
            payload.get("discovered_allowed_account_count")
        ),
        "reconciled_account_count": _safe_int(payload.get("reconciled_account_count")),
        "buffered_event_count": _safe_int(payload.get("buffered_event_count")),
        "journal_depth": _safe_int(payload.get("journal_depth")),
        "status": "ok" if response.status_code < 500 else "capture_not_live",
        "plants": [_sanitize_health_plant(item) for item in plants[:3] if isinstance(item, dict)],
        "blockers": [
            text
            for item in blockers[:20]
            if (text := _safe_text(item, 128)) is not None
        ],
    }


def _capture_timeout_seconds() -> float:
    try:
        value = float(os.getenv(_CAPTURE_TIMEOUT_SECONDS_ENV, "") or _DEFAULT_CAPTURE_TIMEOUT_SECONDS)
    except ValueError:
        value = _DEFAULT_CAPTURE_TIMEOUT_SECONDS
    return min(max(value, 0.1), _MAX_CAPTURE_TIMEOUT_SECONDS)


def _sanitize_health_plant(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "plant": _safe_text(item.get("plant"), 16),
        "connected": bool(item.get("connected")),
        "authenticated": bool(item.get("authenticated")),
        "reconciled": bool(item.get("reconciled")),
        "reconnecting": bool(item.get("reconnecting")),
        "generation_present": bool(item.get("generation_present")),
        "last_message_at": _safe_text(item.get("last_message_at"), 64),
        "blocker": _safe_text(item.get("blocker"), 128),
    }


def _count(session: Session, model: Any) -> int:
    value = session.scalar(select(func.count()).select_from(model))
    return int(value or 0)


def _recent(session: Session, model: Any, timestamp_field: str, limit: int) -> list[Any]:
    timestamp = getattr(model, timestamp_field)
    return list(session.scalars(select(model).order_by(desc(timestamp)).limit(limit)).all())


def _latest_by(rows: Iterable[Any], key) -> list[Any]:
    seen: set[Any] = set()
    latest: list[Any] = []
    for row in rows:
        identity = key(row)
        if identity in seen:
            continue
        seen.add(identity)
        latest.append(row)
    return latest


def _serialize_plant(row: V2RithmicConnectionGeneration) -> dict[str, Any]:
    return {
        "plant": row.plant,
        "state": row.state,
        "connected": row.disconnected_at is None,
        "authenticated": row.authenticated_at is not None,
        "ready": bool(row.ready),
        "forced_logout": bool(row.forced_logout),
        "reconnect_attempt": row.reconnect_attempt,
        "connected_at": _iso(row.connected_at),
        "authenticated_at": _iso(row.authenticated_at),
        "reconciled_at": _iso(row.reconciled_at),
        "last_message_at": _iso(row.last_message_at),
        "disconnected_at": _iso(row.disconnected_at),
    }


def _serialize_account(row: V2RithmicAccountObservation) -> dict[str, Any]:
    return {
        "account": _opaque_alias(row.account_id or row.broker_account_id, "Account"),
        "currency": row.currency,
        "access_type": row.access_type,
        "status": row.account_status,
        "allowlisted": bool(row.allowlisted),
        "order_copy_status": row.order_copy_status,
        "observed_at": _iso(row.observed_at),
    }


def _serialize_order(row: V2RithmicOrderObservation) -> dict[str, Any]:
    return {
        "account": _opaque_alias(row.account_id or row.broker_account_id, "Account"),
        "order": _opaque_alias(row.basket_id, "Order"),
        "symbol": row.symbol,
        "exchange": row.exchange,
        "state": row.normalized_state,
        "broker_status": row.broker_status,
        "side": row.side,
        "order_type": row.order_type,
        "duration": row.duration,
        "quantity": row.quantity,
        "fill_size": row.fill_size,
        "total_fill_size": row.total_fill_size,
        "total_unfilled_size": row.total_unfilled_size,
        "limit_price": _number(row.limit_price),
        "trigger_price": _number(row.trigger_price),
        "fill_price": _number(row.fill_price),
        "average_fill_price": _number(row.average_fill_price),
        "terminal": bool(row.terminal),
        "unknown_state": bool(row.unknown_state),
        "source_kind": row.source_kind,
        "broker_at": _iso(row.broker_at),
        "received_at": _iso(row.received_at),
    }


def _serialize_execution(row: V2RithmicExecutionObservation) -> dict[str, Any]:
    return {
        "account": _opaque_alias(row.account_id or row.broker_account_id, "Account"),
        "order": _opaque_alias(row.basket_id, "Order"),
        "execution": _opaque_alias(row.fill_id, "Fill"),
        "kind": row.execution_kind,
        "side": row.side,
        "quantity": row.quantity,
        "effective_quantity_delta": row.effective_quantity_delta,
        "price": _number(row.price),
        "commission": _number(row.commission),
        "source_kind": row.source_kind,
        "executed_at": _iso(row.executed_at),
        "received_at": _iso(row.received_at),
    }


def _serialize_bracket(row: V2RithmicBracketObservation) -> dict[str, Any]:
    return {
        "account": _opaque_alias(row.account_id or row.broker_account_id, "Account"),
        "parent_order": _opaque_alias(row.parent_basket_id, "Order"),
        "bracket_type": row.bracket_type,
        "operation_type": row.operation_type,
        "status": row.status,
        "target_total_quantity": row.target_total_quantity,
        "target_released_quantity": row.target_released_quantity,
        "stop_total_quantity": row.stop_total_quantity,
        "stop_released_quantity": row.stop_released_quantity,
        "source_kind": row.source_kind,
        "observed_at": _iso(row.observed_at),
        "received_at": _iso(row.received_at),
    }


def _serialize_pnl(row: V2RithmicPnLObservation) -> dict[str, Any]:
    return {
        "account": _opaque_alias(row.account_id or row.broker_account_id, "Account"),
        "scope": row.scope,
        "symbol": row.symbol,
        "exchange": row.exchange,
        "currency": row.currency,
        "trade_date": _iso(row.trade_date),
        "net_quantity": row.net_quantity,
        "long_quantity": row.long_quantity,
        "short_quantity": row.short_quantity,
        "open_quantity": row.open_quantity,
        "working_buy_quantity": row.working_buy_quantity,
        "working_sell_quantity": row.working_sell_quantity,
        "average_open_fill_price": _number(row.average_open_fill_price),
        "open_position_pnl": _number(row.open_position_pnl),
        "closed_position_pnl": _number(row.closed_position_pnl),
        "day_total_pnl": _number(row.day_total_pnl),
        "account_balance": _number(row.account_balance),
        "cash_on_hand": _number(row.cash_on_hand),
        "margin_balance": _number(row.margin_balance),
        "available_buying_power": _number(row.available_buying_power),
        "used_buying_power": _number(row.used_buying_power),
        "commission": _number(row.commission),
        "freshness": row.freshness,
        "is_snapshot": bool(row.is_snapshot),
        "source_at": _iso(row.source_at),
        "received_at": _iso(row.received_at),
    }


def _serialize_rms(row: V2RithmicRmsObservation) -> dict[str, Any]:
    return {
        "account": _opaque_alias(row.account_id or row.broker_account_id, "Account"),
        "scope": row.scope,
        "product_code": row.product_code,
        "currency": row.currency,
        "status": row.status,
        "loss_limit": _number(row.loss_limit),
        "account_balance": _number(row.account_balance),
        "current_auto_liquidate_threshold": _number(
            row.current_auto_liquidate_threshold
        ),
        "auto_liquidate": row.auto_liquidate,
        "max_order_quantity": row.max_order_quantity,
        "buy_limit": row.buy_limit,
        "sell_limit": row.sell_limit,
        "source_at": _iso(row.source_at),
        "received_at": _iso(row.received_at),
    }


def _serialize_reconciliation(
    row: V2RithmicReconciliationCheckpoint,
) -> dict[str, Any]:
    return {
        "account": _opaque_alias(row.account_id or row.broker_account_id, "Account"),
        "plant": row.plant,
        "phase": row.phase,
        "status": row.status,
        "ready": bool(row.ready),
        "live_subscription_active": bool(row.live_subscription_active),
        "discrepancy_count": row.discrepancy_count,
        "snapshot_requested_at": _iso(row.snapshot_requested_at),
        "snapshot_completed_at": _iso(row.snapshot_completed_at),
        "buffered_events_applied_at": _iso(row.buffered_events_applied_at),
        "recorded_at": _iso(row.recorded_at),
        "has_failure": bool(row.failure_reason),
    }


def _serialize_reference(row: V2RithmicReferenceObservation) -> dict[str, Any]:
    return {
        "kind": row.observation_kind,
        "symbol": row.symbol,
        "exchange": row.exchange,
        "product_code": row.product_code,
        "instrument_type": row.instrument_type,
        "expiration_date": _iso(row.expiration_date),
        "currency": row.currency,
        "tradable": row.is_tradable,
        "minimum_quoted_price_change": _number(row.minimum_quoted_price_change),
        "single_point_value": _number(row.single_point_value),
        "source_kind": row.source_kind,
        "received_at": _iso(row.received_at),
    }


def _opaque_alias(value: object, prefix: str) -> str:
    material = str(value or "unavailable").encode("utf-8")
    digest = hashlib.sha256(material).hexdigest()[:8].upper()
    return f"{prefix} {digest}"


def _number(value: Decimal | int | float | None) -> str | None:
    return None if value is None else str(value)


def _iso(value: date | datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _safe_text(value: object, maximum: int) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text[:maximum] if text else None


def _safe_int(value: object) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0
