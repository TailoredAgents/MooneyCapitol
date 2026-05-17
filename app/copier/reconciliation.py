from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.copier.factory import build_copy_targets
from app.copier.order_response import normalize_copy_order_response
from app.copier.repository import load_copy_positions
from app.core.config import CopierConfig
from app.core.config_store import CONFIG, refresh_config
from app.db.models import CopyOrder, CopyOrderEvent, CopyReconciliation, CopyTargetAccount
from app.db.session import get_session
from app.observability.logging import get_logger


logger = get_logger("copier.reconciliation")

OPEN_COPY_STATUSES = {"created", "submitted", "accepted", "partially_filled"}
TERMINAL_COPY_STATUSES = {"filled", "rejected", "cancelled", "expired", "submit_failed"}


class OrderDetailClient(Protocol):
    def get_order_detail(self, account_id: str, client_order_id: str) -> dict[str, Any]: ...


class PositionClient(Protocol):
    def get_account_positions(self, account_id: str) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class NormalizedOrderDetail:
    status: str
    broker_order_id: str | None = None
    filled_qty: float | None = None
    avg_fill_price: float | None = None
    reject_reason: str | None = None
    accepted_at: datetime | None = None
    filled_at: datetime | None = None
    raw_payload: dict | None = None


@dataclass(frozen=True)
class NormalizedPosition:
    symbol: str
    qty: float
    avg_price: float | None = None
    market_value: float | None = None
    raw_payload: dict | None = None


@dataclass(frozen=True)
class ReconciliationResult:
    checked: int
    updated: int
    mismatches: int
    errors: int


@dataclass(frozen=True)
class StartupRecoveryResult:
    checked: int
    updated: int
    mismatches: int
    errors: int
    stale_open_orders: int


@dataclass(frozen=True)
class PositionSyncResult:
    targets_checked: int
    symbols_checked: int
    mismatches: int
    errors: int


class CopyOrderReconciler:
    def __init__(
        self,
        session_scope: Callable = get_session,
        config_provider: Callable[[], CopierConfig] | None = None,
        target_builder: Callable[[CopierConfig], list] | None = None,
    ) -> None:
        self.session_scope = session_scope
        self.config_provider = config_provider or (lambda: CONFIG.copier)
        self.target_builder = target_builder or (lambda config: build_copy_targets(config))

    def reconcile_open_orders(self, limit: int = 100) -> ReconciliationResult:
        refresh_config()
        config = self.config_provider()
        if not config.enabled:
            return ReconciliationResult(checked=0, updated=0, mismatches=0, errors=0)

        targets = {target.name: target for target in self.target_builder(config)}
        checked = updated = mismatches = errors = 0
        with self.session_scope() as session:
            rows = list_open_copy_orders(session, limit=limit)
            for order, target_row in rows:
                target = targets.get(target_row.name)
                if target is None:
                    create_reconciliation(
                        session,
                        order,
                        target_row.id,
                        severity="error",
                        message=f"No configured target client for {target_row.name}",
                        context={"target": target_row.name, "client_order_id": order.client_order_id},
                    )
                    errors += 1
                    continue
                checked += 1
                try:
                    payload = target.client.get_order_detail(target.account_id, order.client_order_id)
                except Exception as exc:
                    create_reconciliation(
                        session,
                        order,
                        target_row.id,
                        severity="warning",
                        message=f"Failed to reconcile child order {order.client_order_id}: {exc}",
                        context={"target": target_row.name, "client_order_id": order.client_order_id},
                    )
                    errors += 1
                    continue
                detail = normalize_order_detail(payload)
                if apply_order_detail(session, order, detail):
                    updated += 1
                if record_order_mismatches(session, order, target_row.id, detail):
                    mismatches += 1
        return ReconciliationResult(checked=checked, updated=updated, mismatches=mismatches, errors=errors)


class CopyPositionReconciler:
    def __init__(
        self,
        session_scope: Callable = get_session,
        config_provider: Callable[[], CopierConfig] | None = None,
        target_builder: Callable[[CopierConfig], list] | None = None,
        qty_tolerance: float = 0.0001,
    ) -> None:
        self.session_scope = session_scope
        self.config_provider = config_provider or (lambda: CONFIG.copier)
        self.target_builder = target_builder or (lambda config: build_copy_targets(config))
        self.qty_tolerance = qty_tolerance

    def sync_positions(self) -> PositionSyncResult:
        refresh_config()
        config = self.config_provider()
        if not config.enabled:
            return PositionSyncResult(targets_checked=0, symbols_checked=0, mismatches=0, errors=0)

        targets = [target for target in self.target_builder(config) if target.risk.enabled]
        if not targets:
            return PositionSyncResult(targets_checked=0, symbols_checked=0, mismatches=0, errors=0)

        target_names = [target.name for target in targets]
        targets_checked = symbols_checked = mismatches = errors = 0
        with self.session_scope() as session:
            local_positions = load_copy_positions(session, target_names)
            target_rows = {
                row.name: row
                for row in session.execute(
                    select(CopyTargetAccount).where(CopyTargetAccount.name.in_(target_names))
                ).scalars().all()
            }
            for target in targets:
                targets_checked += 1
                try:
                    broker_raw = target.client.get_account_positions(target.account_id)
                except Exception as exc:
                    create_position_reconciliation(
                        session,
                        target_account_id=getattr(target_rows.get(target.name), "id", None),
                        target_name=target.name,
                        symbol=None,
                        expected_qty=None,
                        actual_qty=None,
                        severity="warning",
                        message=f"Failed to sync Webull positions for {target.name}: {exc}",
                        context={"target": target.name, "account_id": target.account_id},
                    )
                    errors += 1
                    continue
                broker_positions = normalize_positions(broker_raw)
                local_by_symbol = local_positions.get(target.name, {})
                broker_by_symbol = {position.symbol: position for position in broker_positions}
                symbols = sorted(set(local_by_symbol) | set(broker_by_symbol))
                for symbol in symbols:
                    expected_qty = float(local_by_symbol.get(symbol, 0.0) or 0.0)
                    broker_position = broker_by_symbol.get(symbol)
                    actual_qty = broker_position.qty if broker_position else 0.0
                    symbols_checked += 1
                    if abs(expected_qty - actual_qty) <= self.qty_tolerance:
                        continue
                    created = create_position_reconciliation(
                        session,
                        target_account_id=getattr(target_rows.get(target.name), "id", None),
                        target_name=target.name,
                        symbol=symbol,
                        expected_qty=expected_qty,
                        actual_qty=actual_qty,
                        severity="warning",
                        message=f"Target {target.name} position mismatch for {symbol}",
                        context={
                            "target": target.name,
                            "symbol": symbol,
                            "local_qty": expected_qty,
                            "webull_qty": actual_qty,
                            "webull_position": broker_position.raw_payload if broker_position else None,
                        },
                    )
                    if created:
                        mismatches += 1
        return PositionSyncResult(
            targets_checked=targets_checked,
            symbols_checked=symbols_checked,
            mismatches=mismatches,
            errors=errors,
        )


class CopierStartupRecovery:
    def __init__(
        self,
        reconciler: CopyOrderReconciler | None = None,
        session_scope: Callable = get_session,
        stale_after_minutes: int = 10,
    ) -> None:
        self.reconciler = reconciler or CopyOrderReconciler(session_scope=session_scope)
        self.session_scope = session_scope
        self.stale_after_minutes = stale_after_minutes

    def recover(self, limit: int = 250) -> StartupRecoveryResult:
        result = self.reconciler.reconcile_open_orders(limit=limit)
        with self.session_scope() as session:
            stale = mark_stale_open_orders(
                session,
                stale_after=timedelta(minutes=self.stale_after_minutes),
                limit=limit,
            )
        return StartupRecoveryResult(
            checked=result.checked,
            updated=result.updated,
            mismatches=result.mismatches,
            errors=result.errors,
            stale_open_orders=stale,
        )


def list_open_copy_orders(session: Session, limit: int = 100):
    stmt = (
        select(CopyOrder, CopyTargetAccount)
        .join(CopyTargetAccount, CopyOrder.target_account_id == CopyTargetAccount.id)
        .where(CopyOrder.status.in_(sorted(OPEN_COPY_STATUSES)))
        .order_by(CopyOrder.submitted_at.asc().nullsfirst(), CopyOrder.id.asc())
        .limit(limit)
    )
    return session.execute(stmt).all()


def mark_stale_open_orders(
    session: Session,
    stale_after: timedelta,
    limit: int = 250,
    now: datetime | None = None,
) -> int:
    now = now or datetime.now(tz=timezone.utc)
    cutoff = now - stale_after
    stmt = (
        select(CopyOrder, CopyTargetAccount)
        .join(CopyTargetAccount, CopyOrder.target_account_id == CopyTargetAccount.id)
        .where(CopyOrder.status.in_(sorted(OPEN_COPY_STATUSES)))
        .where(CopyOrder.submitted_at.is_not(None))
        .where(CopyOrder.submitted_at < cutoff)
        .order_by(CopyOrder.submitted_at.asc(), CopyOrder.id.asc())
        .limit(limit)
    )
    count = 0
    for order, target_row in session.execute(stmt).all():
        created = create_reconciliation(
            session,
            order,
            target_row.id,
            severity="warning",
            message=f"Copied order {order.client_order_id} is still {order.status} after startup recovery",
            context={
                "target": target_row.name,
                "status": order.status,
                "submitted_at": order.submitted_at.isoformat() if order.submitted_at else None,
                "stale_after_seconds": stale_after.total_seconds(),
            },
        )
        if created:
            count += 1
    return count


def normalize_order_detail(payload: dict) -> NormalizedOrderDetail:
    normalized = normalize_copy_order_response(payload)
    return NormalizedOrderDetail(
        status=normalized.status,
        broker_order_id=normalized.broker_order_id,
        filled_qty=normalized.filled_qty,
        avg_fill_price=normalized.avg_fill_price,
        reject_reason=normalized.reject_reason,
        accepted_at=normalized.accepted_at,
        filled_at=normalized.filled_at,
        raw_payload=payload,
    )


def normalize_positions(payload: list[dict] | dict) -> list[NormalizedPosition]:
    rows = _position_rows(payload)
    positions: list[NormalizedPosition] = []
    for row in rows:
        symbol = _optional_str(_first(row, ["symbol", "ticker", "instrument_symbol", "instrumentSymbol"]))
        qty = _to_float(
            _first(row, ["qty", "quantity", "position_qty", "positionQty", "total_qty", "totalQty", "holding_qty", "holdingQty"])
        )
        if not symbol or qty is None:
            continue
        positions.append(
            NormalizedPosition(
                symbol=symbol.upper(),
                qty=qty,
                avg_price=_to_float(_first(row, ["avg_price", "avgPrice", "average_price", "averagePrice", "cost_price", "costPrice"])),
                market_value=_to_float(_first(row, ["market_value", "marketValue", "value"])),
                raw_payload=row,
            )
        )
    return positions


def apply_order_detail(session: Session, order: CopyOrder, detail: NormalizedOrderDetail) -> bool:
    changed = False
    for attr, value in [
        ("status", detail.status),
        ("broker_order_id", detail.broker_order_id),
        ("filled_qty", detail.filled_qty),
        ("avg_fill_price", detail.avg_fill_price),
        ("reject_reason", detail.reject_reason),
        ("accepted_at", detail.accepted_at),
        ("filled_at", detail.filled_at if detail.status == "filled" else None),
        ("raw_response_payload", detail.raw_payload),
    ]:
        if value is not None and getattr(order, attr) != value:
            setattr(order, attr, value)
            changed = True
    if changed:
        session.add(
            CopyOrderEvent(
                copy_order_id=order.id,
                event_type="reconciled",
                status=detail.status,
                event_at=detail.filled_at or detail.accepted_at,
                received_at=datetime.now(tz=timezone.utc),
                raw_payload=detail.raw_payload,
            )
        )
    return changed


def record_order_mismatches(
    session: Session,
    order: CopyOrder,
    target_account_id: int,
    detail: NormalizedOrderDetail,
) -> bool:
    if detail.status in {"rejected", "cancelled", "expired"}:
        return create_reconciliation(
            session,
            order,
            target_account_id,
            severity="error",
            message=f"Copied order {order.client_order_id} is {detail.status}",
            context={"status": detail.status, "reject_reason": detail.reject_reason, "raw": detail.raw_payload},
        )
    if detail.status == "filled" and detail.filled_qty is not None and detail.filled_qty < order.qty:
        return create_reconciliation(
            session,
            order,
            target_account_id,
            severity="warning",
            message=f"Copied order {order.client_order_id} filled {detail.filled_qty} of expected {order.qty}",
            context={"status": detail.status, "filled_qty": detail.filled_qty, "expected_qty": order.qty},
        )
    return False


def create_reconciliation(
    session: Session,
    order: CopyOrder,
    target_account_id: int | None,
    severity: str,
    message: str,
    context: dict | None = None,
) -> bool:
    raw_context = {
        "copy_order_id": order.id,
        "client_order_id": order.client_order_id,
        **(context or {}),
    }
    if _open_reconciliation_exists(session, order, message):
        return False
    session.add(
        CopyReconciliation(
            target_account_id=target_account_id,
            symbol=order.symbol,
            severity=severity,
            status="open",
            message=message,
            detected_at=datetime.now(tz=timezone.utc),
            raw_context=raw_context,
        )
    )
    return True


def create_position_reconciliation(
    session: Session,
    target_account_id: int | None,
    target_name: str,
    symbol: str | None,
    expected_qty: float | None,
    actual_qty: float | None,
    severity: str,
    message: str,
    context: dict | None = None,
) -> bool:
    if _open_position_reconciliation_exists(session, message, target_name, symbol):
        return False
    raw_context = {
        "target": target_name,
        "local_qty": expected_qty,
        "webull_qty": actual_qty,
        **(context or {}),
    }
    session.add(
        CopyReconciliation(
            target_account_id=target_account_id,
            symbol=symbol,
            severity=severity,
            status="open",
            message=message,
            detected_at=datetime.now(tz=timezone.utc),
            raw_context=raw_context,
        )
    )
    return True


def _open_reconciliation_exists(session: Session, order: CopyOrder, message: str) -> bool:
    if not hasattr(session, "execute"):
        return False
    try:
        rows = (
            session.execute(
                select(CopyReconciliation).where(
                    CopyReconciliation.status == "open",
                    CopyReconciliation.symbol == order.symbol,
                    CopyReconciliation.message == message,
                )
            )
            .scalars()
            .all()
        )
    except Exception:
        return False
    for row in rows:
        context = row.raw_context or {}
        if context.get("copy_order_id") == order.id or context.get("client_order_id") == order.client_order_id:
            return True
    return False


def _open_position_reconciliation_exists(session: Session, message: str, target_name: str, symbol: str | None) -> bool:
    if not hasattr(session, "execute"):
        return False
    try:
        stmt = select(CopyReconciliation).where(
            CopyReconciliation.status == "open",
            CopyReconciliation.message == message,
        )
        if symbol is None:
            stmt = stmt.where(CopyReconciliation.symbol.is_(None))
        else:
            stmt = stmt.where(CopyReconciliation.symbol == symbol)
        rows = session.execute(stmt).scalars().all()
    except Exception:
        return False
    for row in rows:
        context = row.raw_context or {}
        if context.get("target") == target_name:
            return True
    return False


def _payload_data(payload: dict) -> dict:
    data = payload.get("data")
    if isinstance(data, dict):
        return data
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0]
    return payload


def _position_rows(payload: list[dict] | dict) -> list[dict]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("positions", "items", "list"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    for key in ("positions", "items", "list"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _normalize_status(value) -> str:
    raw = str(value or "").strip().upper()
    mapping = {
        "NEW": "accepted",
        "PENDING": "submitted",
        "PENDING_NEW": "submitted",
        "SUBMITTED": "submitted",
        "ACCEPTED": "accepted",
        "PARTIALLY_FILLED": "partially_filled",
        "PARTIAL_FILLED": "partially_filled",
        "FILLED": "filled",
        "EXECUTED": "filled",
        "REJECTED": "rejected",
        "CANCELLED": "cancelled",
        "CANCELED": "cancelled",
        "EXPIRED": "expired",
    }
    return mapping.get(raw, raw.lower() if raw else "submitted")


def _first(payload: dict, keys: list[str]):
    for key in keys:
        value = payload.get(key)
        if value not in (None, ""):
            return value
    return None


def _optional_str(value) -> str | None:
    return str(value) if value not in (None, "") else None


def _to_float(value) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_ts(value) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        timestamp = float(value) / 1000 if float(value) > 10_000_000_000 else float(value)
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None
