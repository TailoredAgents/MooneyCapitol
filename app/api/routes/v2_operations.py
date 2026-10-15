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
    V2BrokerAccount,
    V2CopyLatency,
    V2CopyLifecycleCommand,
    V2CopyLifecycleEvent,
    V2CopyLifecycleIntent,
    V2CopyOutbox,
    V2CopyReconciliationFinding,
    V2CopyReconciliationRun,
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
_CONNECTION_IDS = ("master", "follower-1", "follower-2", "follower-3", "follower-4")


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
        "copy_intents": _count(session, V2CopyLifecycleIntent),
        "copy_commands": _count(session, V2CopyLifecycleCommand),
        "copy_reconciliations": _count(session, V2CopyReconciliationRun),
    }

    event_rows = _recent(session, V2RithmicBrokerEvent, "received_at", 1)
    generation_rows = _recent(session, V2RithmicConnectionGeneration, "connected_at", 100)
    broker_account_rows = list(session.scalars(select(V2BrokerAccount)).all())
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
    copy_intent_rows = _recent(session, V2CopyLifecycleIntent, "updated_at", 100)
    copy_command_rows = _recent(session, V2CopyLifecycleCommand, "updated_at", 250)
    copy_event_rows = _recent(session, V2CopyLifecycleEvent, "recorded_at", limit)
    copy_outbox_rows = _recent(session, V2CopyOutbox, "updated_at", 100)
    copy_reconciliation_rows = _recent(
        session,
        V2CopyReconciliationRun,
        "completed_at",
        100,
    )
    copy_finding_rows = _recent(
        session,
        V2CopyReconciliationFinding,
        "detected_at",
        limit,
    )
    copy_latency_rows = _recent(session, V2CopyLatency, "updated_at", 100)

    account_connections = {
        row.account_id: connection_id
        for row in broker_account_rows
        if (connection_id := _safe_connection_id(row.connection_id)) is not None
    }

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
    latest_plants = _latest_by(
        generation_rows,
        lambda row: (row.connection_id, row.plant),
    )
    latest_copy_intents = _latest_by(
        copy_intent_rows,
        lambda row: row.follower_connection_id,
    )
    latest_copy_commands = _latest_by(
        copy_command_rows,
        lambda row: row.intent_id,
    )
    command_by_intent = {row.intent_id: row for row in latest_copy_commands}
    copy_intent_connections = {
        row.intent_id: row.follower_connection_id for row in copy_intent_rows
    }
    latest_copy_outbox = _latest_by(
        copy_outbox_rows,
        lambda row: row.follower_connection_id,
    )
    latest_copy_reconciliation = _latest_by(
        copy_reconciliation_rows,
        lambda row: row.follower_connection_id,
    )
    reconciliation_connections = {
        row.reconciliation_run_id: row.follower_connection_id
        for row in copy_reconciliation_rows
    }
    latest_copy_latency = _latest_by(
        copy_latency_rows,
        lambda row: row.follower_connection_id,
    )
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
        "accounts": [
            _serialize_account(row, account_connections) for row in latest_accounts
        ],
        "orders": [
            _serialize_order(row, account_connections) for row in order_rows
        ],
        "executions": [
            _serialize_execution(row, account_connections) for row in execution_rows
        ],
        "brackets": [
            _serialize_bracket(row, account_connections) for row in bracket_rows
        ],
        "account_pnl": [
            _serialize_pnl(row, account_connections)
            for row in latest_pnl
            if row.scope == "ACCOUNT"
        ],
        "positions": [
            _serialize_pnl(row, account_connections)
            for row in latest_pnl
            if row.scope == "INSTRUMENT"
        ],
        "risk": [_serialize_rms(row, account_connections) for row in latest_rms],
        "reconciliation": [
            _serialize_reconciliation(row, account_connections)
            for row in latest_reconciliation
        ],
        "contracts": [_serialize_reference(row) for row in reference_rows],
        "copy_execution": {
            "intents": [
                _serialize_copy_intent(row, command_by_intent.get(row.intent_id))
                for row in latest_copy_intents
            ],
            "lifecycle_events": [
                _serialize_copy_event(
                    row,
                    copy_intent_connections.get(row.intent_id),
                )
                for row in copy_event_rows
            ],
            "delivery": [
                _serialize_copy_outbox(row) for row in latest_copy_outbox
            ],
            "latency": [
                _serialize_copy_latency(row) for row in latest_copy_latency
            ],
            "reconciliation": [
                _serialize_copy_reconciliation(row)
                for row in latest_copy_reconciliation
            ],
            "findings": [
                _serialize_copy_finding(
                    row,
                    reconciliation_connections.get(row.reconciliation_run_id),
                )
                for row in copy_finding_rows
            ],
        },
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
        "connection_id": _safe_connection_id(row.connection_id),
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


def _serialize_account(
    row: V2RithmicAccountObservation,
    account_connections: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "connection_id": _connection_for(row, account_connections),
        "account": _opaque_alias(row.account_id or row.broker_account_id, "Account"),
        "currency": row.currency,
        "access_type": row.access_type,
        "status": row.account_status,
        "allowlisted": bool(row.allowlisted),
        "order_copy_status": row.order_copy_status,
        "observed_at": _iso(row.observed_at),
    }


def _serialize_order(
    row: V2RithmicOrderObservation,
    account_connections: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "connection_id": _connection_for(row, account_connections),
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


def _serialize_execution(
    row: V2RithmicExecutionObservation,
    account_connections: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "connection_id": _connection_for(row, account_connections),
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


def _serialize_bracket(
    row: V2RithmicBracketObservation,
    account_connections: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "connection_id": _connection_for(row, account_connections),
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


def _serialize_pnl(
    row: V2RithmicPnLObservation,
    account_connections: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "connection_id": _connection_for(row, account_connections),
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


def _serialize_rms(
    row: V2RithmicRmsObservation,
    account_connections: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "connection_id": _connection_for(row, account_connections),
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
    account_connections: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "connection_id": _connection_for(row, account_connections),
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


def _serialize_copy_intent(
    row: V2CopyLifecycleIntent,
    command: V2CopyLifecycleCommand | None,
) -> dict[str, Any]:
    return {
        "copy": _opaque_alias(row.intent_id, "Copy"),
        "connection_id": _safe_connection_id(row.follower_connection_id),
        "master_event_kind": row.master_event_kind,
        "directive": row.directive,
        "status": row.status,
        "allowed": bool(row.allowed),
        "reason": _safe_text(row.reason, 96),
        "symbol": row.symbol,
        "exchange": row.exchange,
        "current_signed_quantity": row.current_signed_quantity,
        "evaluated_target_signed_quantity": row.evaluated_target_signed_quantity,
        "achievable_target_signed_quantity": row.achievable_target_signed_quantity,
        "delta_signed_quantity": row.delta_signed_quantity,
        "desired_quantity": _number(row.desired_quantity),
        "desired_risk_dollars": _number(row.desired_risk_dollars),
        "achievable_risk_dollars": _number(row.achievable_risk_dollars),
        "desired_risk_fraction": _number(row.desired_risk_fraction),
        "achievable_risk_fraction": _number(row.achievable_risk_fraction),
        "desired_exposure_fraction": _number(row.desired_exposure_fraction),
        "achievable_exposure_fraction": _number(row.achievable_exposure_fraction),
        "risk_parity_error": _number(row.risk_parity_error),
        "exposure_parity_error": _number(row.exposure_parity_error),
        "parity_error": _number(row.parity_error),
        "target_margin_required": _number(row.target_margin_required),
        "updated_at": _iso(row.updated_at),
        "command": _serialize_copy_command(command) if command is not None else None,
    }


def _serialize_copy_command(row: V2CopyLifecycleCommand) -> dict[str, Any]:
    return {
        "command": _opaque_alias(row.command_id, "Command"),
        "action": row.action,
        "status": row.status,
        "side": row.side,
        "quantity": row.quantity,
        "filled_quantity": row.cumulative_filled_quantity,
        "order_type": row.order_type,
        "stop_ticks": row.stop_ticks,
        "target_ticks": row.target_ticks,
        "attempt_count": row.attempt_count,
        "updated_at": _iso(row.updated_at),
    }


def _serialize_copy_event(
    row: V2CopyLifecycleEvent,
    connection_id: str | None,
) -> dict[str, Any]:
    return {
        "event": _opaque_alias(row.lifecycle_event_id, "Event"),
        "connection_id": _safe_connection_id(connection_id),
        "kind": row.event_kind,
        "previous_status": row.previous_status,
        "resulting_status": row.resulting_status,
        "filled_quantity": row.cumulative_filled_quantity,
        "fill_price": _number(row.fill_price),
        "rejection_code": _safe_text(row.rejection_code, 128),
        "occurred_at": _iso(row.occurred_at),
        "recorded_at": _iso(row.recorded_at),
    }


def _serialize_copy_outbox(row: V2CopyOutbox) -> dict[str, Any]:
    return {
        "connection_id": _safe_connection_id(row.follower_connection_id),
        "status": row.status,
        "attempt_count": row.attempt_count,
        "last_error_code": _safe_text(row.last_error_code, 128),
        "submission_started_at": _iso(row.submission_started_at),
        "dispatch_response_at": _iso(row.dispatch_response_at),
        "confirmed_at": _iso(row.confirmed_at),
        "updated_at": _iso(row.updated_at),
    }


def _serialize_copy_latency(row: V2CopyLatency) -> dict[str, Any]:
    return {
        "connection_id": _safe_connection_id(row.follower_connection_id),
        "event_to_receive_ms": _duration_ms(row.event_received_at, row.event_occurred_at),
        "receive_to_decision_ms": _duration_ms(row.decision_at, row.event_received_at),
        "decision_to_submission_ms": _duration_ms(
            row.submission_started_at,
            row.decision_at,
        ),
        "submission_to_response_ms": _duration_ms(
            row.dispatch_response_at,
            row.submission_started_at,
        ),
        "event_to_ack_ms": _duration_ms(row.acknowledged_at, row.event_occurred_at),
        "event_to_first_fill_ms": _duration_ms(row.first_fill_at, row.event_occurred_at),
        "event_to_complete_ms": _duration_ms(row.completed_at, row.event_occurred_at),
        "updated_at": _iso(row.updated_at),
    }


def _serialize_copy_reconciliation(row: V2CopyReconciliationRun) -> dict[str, Any]:
    return {
        "connection_id": _safe_connection_id(row.follower_connection_id),
        "status": row.status,
        "adopted_receipt_count": row.adopted_receipt_count,
        "finding_count": row.finding_count,
        "snapshot_at": _iso(row.snapshot_at),
        "completed_at": _iso(row.completed_at),
    }


def _serialize_copy_finding(
    row: V2CopyReconciliationFinding,
    connection_id: str | None,
) -> dict[str, Any]:
    return {
        "finding": _opaque_alias(row.finding_id, "Finding"),
        "connection_id": _safe_connection_id(connection_id),
        "kind": _safe_text(row.finding_kind, 64),
        "severity": row.severity,
        "detected_at": _iso(row.detected_at),
    }


def _connection_for(
    row: object,
    account_connections: dict[str, str] | None,
) -> str | None:
    if not account_connections:
        return None
    return _safe_connection_id(account_connections.get(getattr(row, "account_id", None)))


def _safe_connection_id(value: object) -> str | None:
    text = str(value or "").strip()
    return text if text in _CONNECTION_IDS else None


def _duration_ms(end: datetime | None, start: datetime | None) -> int | None:
    if end is None or start is None:
        return None
    try:
        return max(0, round((end - start).total_seconds() * 1000))
    except (TypeError, ValueError):
        return None


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
