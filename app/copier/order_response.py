from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class NormalizedCopyOrderResponse:
    status: str
    broker_order_id: str | None = None
    client_order_id: str | None = None
    filled_qty: float | None = None
    avg_fill_price: float | None = None
    reject_reason: str | None = None
    accepted_at: datetime | None = None
    filled_at: datetime | None = None
    raw_payload: dict | None = None


def normalize_copy_order_response(
    payload: dict | None,
    *,
    fallback_status: str = "submitted",
) -> NormalizedCopyOrderResponse:
    if not payload:
        return NormalizedCopyOrderResponse(status=fallback_status, raw_payload=payload)
    rows = _payload_rows(payload)
    data = rows[0] if rows else payload
    status = _normalize_status(_first(data, ["status", "order_status", "orderStatus", "state", "orderState"]))
    reject_reason = _optional_str(
        _first(
            data,
            [
                "reject_reason",
                "rejectReason",
                "error_message",
                "errorMessage",
                "error_msg",
                "errorMsg",
                "fail_reason",
                "failReason",
                "message",
                "msg",
                "reason",
            ],
        )
    )
    if status is None:
        status = _status_from_envelope(payload, fallback_status)
    if status == "rejected" and not reject_reason:
        reject_reason = _optional_str(_first(payload, ["error", "message", "msg", "reason"]))
    return NormalizedCopyOrderResponse(
        status=status,
        broker_order_id=_optional_str(
            _first(
                data,
                ["order_id", "orderId", "broker_order_id", "entrust_id", "entrustId", "entrust_no", "entrustNo"],
            )
        ),
        client_order_id=_optional_str(_first(data, ["client_order_id", "clientOrderId", "clientOrderID"])),
        filled_qty=_to_float(
            _first(data, ["filled_qty", "filled_quantity", "filledQuantity", "cum_qty", "cumQty", "totalFilledQty"])
        ),
        avg_fill_price=_to_float(
            _first(
                data,
                [
                    "avg_fill_price",
                    "avgFilledPrice",
                    "average_price",
                    "averagePrice",
                    "filled_price",
                    "filledPrice",
                ],
            )
        ),
        reject_reason=reject_reason,
        accepted_at=_parse_ts(_first(data, ["accepted_at", "acceptedAt", "created_at", "createdAt", "submittedAt"])),
        filled_at=_parse_ts(_first(data, ["filled_at", "filledAt", "updated_at", "updatedAt", "lastFilledAt"])),
        raw_payload=payload,
    )


def normalize_copy_order_responses(payload: dict | list | None, *, fallback_status: str = "submitted") -> list[NormalizedCopyOrderResponse]:
    if payload is None:
        return [normalize_copy_order_response(None, fallback_status=fallback_status)]
    if isinstance(payload, list):
        return [
            normalize_copy_order_response(row, fallback_status=fallback_status)
            for row in payload
            if isinstance(row, dict)
        ]
    rows = _payload_rows(payload)
    if len(rows) <= 1:
        return [normalize_copy_order_response(payload, fallback_status=fallback_status)]
    return [normalize_copy_order_response(row, fallback_status=fallback_status) for row in rows]


def _payload_rows(payload: dict) -> list[dict[str, Any]]:
    data = payload.get("data")
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("orders", "items", "list", "results"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [data]
    for key in ("orders", "items", "list", "results"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return [payload]


def _status_from_envelope(payload: dict, fallback_status: str) -> str:
    success = _first(payload, ["success", "ok", "succeed"])
    if success is False or str(success).lower() == "false":
        return "rejected"
    code = str(_first(payload, ["code", "status_code", "statusCode"]) or "").upper()
    if code and code not in {"0", "200", "OK", "SUCCESS"}:
        return "rejected"
    return fallback_status


def _normalize_status(value) -> str | None:
    raw = str(value or "").strip().upper()
    if not raw:
        return None
    mapping = {
        "NEW": "accepted",
        "OPEN": "accepted",
        "WORKING": "accepted",
        "QUEUED": "submitted",
        "PENDING": "submitted",
        "PENDING_NEW": "submitted",
        "PENDING_SUBMIT": "submitted",
        "SUBMITTED": "submitted",
        "ACCEPTED": "accepted",
        "UNFILLED": "accepted",
        "PARTIALLY_FILLED": "partially_filled",
        "PARTIAL_FILLED": "partially_filled",
        "PARTIAL_EXECUTED": "partially_filled",
        "FILLED": "filled",
        "EXECUTED": "filled",
        "TRADED": "filled",
        "REJECTED": "rejected",
        "FAILED": "rejected",
        "ERROR": "rejected",
        "CANCELLED": "cancelled",
        "CANCELED": "cancelled",
        "EXPIRED": "expired",
    }
    return mapping.get(raw, raw.lower())


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
