from __future__ import annotations

import asyncio
import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import fields, is_dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.security import redact_sensitive
from app.v2.capture.config import CaptureConfig
from app.v2.capture.contracts import (
    CaptureEvent,
    CapturePlant,
    CaptureSource,
    ReconciliationCheckpoint,
)
from app.v2.db_models import (
    V2BrokerAccount,
    V2BrokerConnection,
    V2RithmicAccountObservation,
    V2RithmicBracketObservation,
    V2RithmicBrokerEvent,
    V2RithmicConnectionGeneration,
    V2RithmicExecutionObservation,
    V2RithmicOrderObservation,
    V2RithmicPnLObservation,
    V2RithmicReferenceObservation,
    V2RithmicReconciliationCheckpoint,
    V2RithmicReplayBatch,
    V2RithmicRmsObservation,
)


_MAX_EVENT_FACT_BYTES = 256 * 1024


def _object_mapping(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return {str(key): child for key, child in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: getattr(value, item.name) for item in fields(value)}
    return {}


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, (bytes, bytearray, memoryview)):
        payload = bytes(value)
        return {
            "opaque_bytes_sha256": hashlib.sha256(payload).hexdigest(),
            "opaque_bytes_size": len(payload),
        }
    mapping = _object_mapping(value)
    if mapping:
        return {key: _json_safe(child) for key, child in mapping.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    return {"opaque_type": type(value).__name__}


def _canonical_payload(payload: Mapping[str, object]) -> tuple[str, dict[str, object]]:
    safe = _json_safe(payload)
    if not isinstance(safe, dict):
        safe = {"opaque_payload": safe}
    encoded = json.dumps(safe, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return encoded, safe


def _redacted_event_facts(
    payload: dict[str, object], *, payload_fingerprint: str
) -> dict[str, object]:
    redacted = redact_sensitive(payload)
    if not isinstance(redacted, dict):
        return {"payload_fingerprint": payload_fingerprint}
    encoded = json.dumps(redacted, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    if len(encoded.encode("utf-8")) <= _MAX_EVENT_FACT_BYTES:
        return redacted
    return {
        "payload_fingerprint": payload_fingerprint,
        "redacted_payload_truncated": True,
        "top_level_keys": sorted(redacted),
    }


def _flatten_payload(payload: Mapping[str, object]) -> tuple[str, dict[str, object]]:
    root = _object_mapping(payload)
    flattened = dict(root)
    nested_value: object | None = None
    for key in ("broker_identity", "native_identity", "observation", "data", "normalized"):
        value = root.get(key)
        nested = _object_mapping(value)
        if nested:
            flattened.update(nested)
        if key in {"observation", "data", "normalized"} and value is not None:
            nested_value = value
    # Account-update observations wrap their full account row once more under
    # ``normalized.account``.  Flatten it for the account projection while the
    # original nested structure remains in the immutable event facts.
    flattened.update(_object_mapping(flattened.get("account")))
    raw_kind = (
        root.get("kind")
        or root.get("observation_kind")
        or root.get("observation_type")
        or root.get("type")
    )
    if raw_kind is None and nested_value is not None:
        raw_kind = type(nested_value).__name__
    kind = _text(raw_kind) or "UNKNOWN"
    return kind.upper(), flattened


def _enum_value(value: object) -> object:
    return value.value if isinstance(value, Enum) else value


def _text(value: object, *, maximum: int | None = None) -> str | None:
    value = _enum_value(value)
    if value is None:
        return None
    if not isinstance(value, (str, int, float, bool, Decimal)):
        return None
    result = str(value)
    return result[:maximum] if maximum is not None else result


def _opaque_identifier(
    value: object,
    field_name: str,
    *,
    maximum: int = 256,
    required: bool = False,
) -> str | None:
    result = _text(value)
    if not result:
        if required:
            raise ValueError(f"{field_name} is required")
        return None
    if len(result) > maximum:
        # Never include the sensitive identifier itself in the exception.
        raise ValueError(f"{field_name} exceeds its {maximum}-character storage limit")
    return result


def _integer(value: object) -> int | None:
    value = _enum_value(value)
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _decimal(value: object) -> Decimal | None:
    value = _enum_value(value)
    if value is None or isinstance(value, bool):
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None


def _boolean(value: object, *, default: bool = False) -> bool:
    parsed = _optional_boolean(value)
    return default if parsed is None else parsed


def _optional_boolean(value: object) -> bool | None:
    value = _enum_value(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on", "enabled"}:
            return True
        if normalized in {"0", "false", "no", "off", "disabled"}:
            return False
        return None
    if isinstance(value, int):
        return value != 0
    return None


def _datetime(value: object) -> datetime | None:
    value = _enum_value(value)
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        try:
            result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return result if result.tzinfo is not None else result.replace(tzinfo=timezone.utc)
    return None


def _date(value: object) -> date | None:
    value = _enum_value(value)
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        normalized = value.strip()
        try:
            if len(normalized) == 8 and normalized.isdigit():
                return datetime.strptime(normalized, "%Y%m%d").date()
            return date.fromisoformat(normalized)
        except ValueError:
            return None
    return None


def _value(data: Mapping[str, object], *names: str) -> object | None:
    for name in names:
        if name in data:
            return data[name]
    return None


def _nested(data: Mapping[str, object], name: str) -> dict[str, object]:
    return _object_mapping(data.get(name))


def _broker_account(data: Mapping[str, object], fallback: str) -> tuple[str | None, str | None, str]:
    account = _nested(data, "account")
    fcm_id = _opaque_identifier(_value(data, "fcm_id"), "FCM ID") or _opaque_identifier(
        account.get("fcm_id"), "FCM ID"
    )
    ib_id = _opaque_identifier(_value(data, "ib_id"), "IB ID") or _opaque_identifier(
        account.get("ib_id"), "IB ID"
    )
    broker_account_id = _opaque_identifier(
        _value(data, "broker_account_id", "account_id"), "broker account ID"
    ) or _opaque_identifier(account.get("account_id"), "broker account ID")
    return (
        fcm_id,
        ib_id,
        broker_account_id
        or _opaque_identifier(fallback, "broker account ID", required=True),
    )


def _json_list(value: object) -> list[object]:
    safe = _json_safe(value)
    if safe is None:
        return []
    if isinstance(safe, list):
        return safe
    return [safe]


def _timestamp_evidence(data: Mapping[str, object], kind: str) -> datetime | None:
    direct = _datetime(_value(data, f"{kind}_at", f"{kind}_timestamp"))
    if direct is not None:
        return direct
    timestamps = _nested(data, "timestamps")
    if kind == "source":
        seconds = _integer(_value(timestamps, "source_ssboe", "ssboe"))
        micros = _integer(_value(timestamps, "source_usecs", "usecs")) or 0
        nanos = _integer(timestamps.get("source_nsecs")) or 0
    elif kind == "server":
        seconds = _integer(timestamps.get("server_received_ssboe"))
        micros = _integer(timestamps.get("server_received_usecs")) or 0
        nanos = 0
    else:
        seconds = _integer(timestamps.get("exchange_receipt_ssboe"))
        micros = 0
        nanos = _integer(timestamps.get("exchange_receipt_nsecs")) or 0
    if seconds is None:
        return None
    try:
        return datetime.fromtimestamp(
            seconds + (micros / 1_000_000) + (nanos / 1_000_000_000),
            tz=timezone.utc,
        )
    except (OverflowError, OSError, ValueError):
        return None


class SQLAlchemyCaptureJournal:
    """Durable append-only journal for the read-only capture service.

    Protocol payloads are fingerprinted in memory.  Only explicitly normalized
    columns and recursively redacted event facts are stored; this class never
    logs decoded messages, broker identifiers, or credentials.
    """

    durable = True

    def __init__(
        self,
        config: CaptureConfig,
        *,
        session_factory: Callable[[], Session] | None = None,
        system_name: str = "",
    ) -> None:
        self._session_factory = session_factory or SessionLocal
        self._environment = config.environment
        self._account_allowlist = frozenset(config.account_allowlist)
        self._system_name = system_name or "UNCONFIGURED"
        self._connection_id = f"v2-rithmic-{config.environment.lower()}"
        self._internal_account_allowlist = frozenset(
            self._internal_account_id(account_id) for account_id in self._account_allowlist
        )
        self._lock = asyncio.Lock()
        self._depth = 0

    async def initialize(self) -> None:
        async with self._lock:
            self._depth = await asyncio.to_thread(self._initialize_sync)

    def _initialize_sync(self) -> int:
        now = datetime.now(timezone.utc)
        with self._session_factory() as session:
            self._ensure_connection(session, now)
            depth = session.scalar(select(func.count()).select_from(V2RithmicBrokerEvent)) or 0
            session.commit()
            return int(depth)

    async def append(self, event: CaptureEvent) -> bool:
        async with self._lock:
            inserted = await asyncio.to_thread(self._append_sync, event)
            if inserted:
                self._depth += 1
            return inserted

    def _append_sync(self, event: CaptureEvent) -> bool:
        canonical, safe_payload = _canonical_payload(event.payload)
        fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        supplied_fingerprint = _text(event.payload.get("raw_frame_sha256"))
        if supplied_fingerprint and len(supplied_fingerprint) == 64 and all(
            character in "0123456789abcdefABCDEF" for character in supplied_fingerprint
        ):
            fingerprint = supplied_fingerprint.lower()
        kind, data = _flatten_payload(event.payload)
        event_id = _opaque_identifier(event.event_id, "capture event ID", maximum=96, required=True)
        if event.account_id is not None:
            _opaque_identifier(event.account_id, "broker account ID", required=True)
        _opaque_identifier(
            event.generation_id,
            "capture generation ID",
            maximum=96,
            required=True,
        )

        with self._session_factory() as session:
            if session.get(V2RithmicBrokerEvent, event_id) is not None:
                return False
            now = event.received_at
            self._ensure_connection(session, now)
            internal_account = (
                self._ensure_account(session, event.account_id)
                if event.account_id is not None
                else None
            )
            generation = self._ensure_generation(
                session,
                generation_id=event.generation_id,
                plant=event.plant,
                connected_at=now,
            )
            if generation.plant != event.plant.value:
                raise ValueError("capture generation plant mismatch")
            self._update_generation_from_event(generation, event, kind, data)

            replay_batch_id = _opaque_identifier(
                data.get("replay_batch_id"), "replay batch ID", maximum=96
            )
            if replay_batch_id:
                if event.account_id is None:
                    raise ValueError("accountless reference event cannot join a replay batch")
                self._ensure_replay_batch(session, event, replay_batch_id, data)

            deduplication_key = _opaque_identifier(
                _value(data, "deduplication_key", "dedupe_key"),
                "event deduplication key",
            ) or event_id
            if session.scalar(
                select(V2RithmicBrokerEvent.event_id).where(
                    V2RithmicBrokerEvent.deduplication_key == deduplication_key
                )
            ) is not None:
                return False

            if event.account_id is not None:
                fcm_id, ib_id, broker_account_id = _broker_account(
                    data, event.account_id
                )
            else:
                fcm_id = _opaque_identifier(data.get("fcm_id"), "FCM ID")
                ib_id = _opaque_identifier(data.get("ib_id"), "IB ID")
                broker_account_id = _opaque_identifier(
                    _value(data, "broker_account_id", "account_id"),
                    "broker account ID",
                )
            linked = _value(data, "linked_basket_ids")
            ingest_sequence = self._next_ingest_sequence(session, event.generation_id)
            unknown = _json_safe(data.get("unknown_fields", {}))
            if not isinstance(unknown, (dict, list)):
                unknown = {}
            unknown = redact_sensitive(unknown)

            row = V2RithmicBrokerEvent(
                event_id=event_id,
                generation_id=event.generation_id,
                replay_batch_id=replay_batch_id,
                account_id=(internal_account.account_id if internal_account else None),
                plant=event.plant.value,
                template_id=_integer(data.get("template_id")) or 0,
                template_name=_text(data.get("template_name"), maximum=128) or kind[:128],
                source_kind=event.source.value,
                local_ingest_sequence=ingest_sequence,
                request_key=_opaque_identifier(data.get("request_key"), "request key"),
                user_message=_opaque_identifier(
                    _value(data, "user_message", "user_msg"), "user message"
                ),
                fcm_id=fcm_id,
                ib_id=ib_id,
                broker_account_id=broker_account_id,
                basket_id=_opaque_identifier(data.get("basket_id"), "basket ID"),
                original_basket_id=_opaque_identifier(
                    data.get("original_basket_id"), "original basket ID"
                ),
                linked_basket_ids=_json_list(linked),
                exchange_order_id=_opaque_identifier(
                    data.get("exchange_order_id"), "exchange order ID"
                ),
                ticker_plant_exchange_order_id=_opaque_identifier(
                    _value(data, "ticker_plant_exchange_order_id", "tp_exchange_order_id"),
                    "ticker-plant exchange order ID",
                ),
                fill_id=_opaque_identifier(data.get("fill_id"), "fill ID"),
                sequence_number=_opaque_identifier(
                    data.get("sequence_number"), "sequence number"
                ),
                original_sequence_number=_opaque_identifier(
                    data.get("original_sequence_number"), "original sequence number"
                ),
                correlation_sequence_number=_opaque_identifier(
                    data.get("correlation_sequence_number"), "correlation sequence number"
                ),
                source_at=_timestamp_evidence(data, "source"),
                server_at=_timestamp_evidence(data, "server"),
                exchange_at=_timestamp_evidence(data, "exchange"),
                received_at=event.received_at,
                payload_fingerprint=fingerprint,
                deduplication_key=deduplication_key,
                event_facts=_redacted_event_facts(
                    safe_payload, payload_fingerprint=fingerprint
                ),
                unknown_fields=unknown,
            )
            try:
                session.add(row)
                session.flush()
                self._project_event(session, event, row, kind, data)
                session.commit()
                return True
            except IntegrityError:
                session.rollback()

        with self._session_factory() as session:
            duplicate = session.get(V2RithmicBrokerEvent, event_id)
            if duplicate is None:
                duplicate = session.scalar(
                    select(V2RithmicBrokerEvent).where(
                        V2RithmicBrokerEvent.deduplication_key == deduplication_key
                    )
                )
            if duplicate is not None:
                return False
        raise RuntimeError("capture journal transaction failed")

    async def begin_reconciliation(
        self,
        account_id: str,
        generations: Mapping[CapturePlant, str],
    ) -> None:
        async with self._lock:
            await asyncio.to_thread(
                self._begin_reconciliation_sync,
                account_id,
                dict(generations),
            )

    def _begin_reconciliation_sync(
        self,
        account_id: str,
        generations: Mapping[CapturePlant, str],
    ) -> None:
        now = datetime.now(timezone.utc)
        with self._session_factory() as session:
            self._ensure_connection(session, now)
            _opaque_identifier(account_id, "broker account ID", required=True)
            internal_account = self._ensure_account(session, account_id)
            for plant, generation_id in generations.items():
                generation = self._ensure_generation(
                    session,
                    generation_id=generation_id,
                    plant=plant,
                    connected_at=now,
                )
                generation.state = "RECONCILING"
                generation.ready = False
                session.add(
                    V2RithmicReconciliationCheckpoint(
                        checkpoint_id=uuid4().hex,
                        generation_id=generation_id,
                        account_id=internal_account.account_id,
                        broker_account_id=account_id,
                        plant=plant.value,
                        phase="SUBSCRIBE_BEFORE_SNAPSHOT",
                        status="STARTED",
                        live_subscription_active=False,
                        discrepancy_count=0,
                        ready=False,
                        recorded_at=now,
                        checkpoint_facts={},
                    )
                )
            session.commit()

    async def save_checkpoint(self, checkpoint: ReconciliationCheckpoint) -> None:
        async with self._lock:
            await asyncio.to_thread(self._save_checkpoint_sync, checkpoint)

    async def fail_reconciliation(
        self,
        account_id: str,
        generations: Mapping[CapturePlant, str],
        failure_reason: str,
    ) -> None:
        async with self._lock:
            await asyncio.to_thread(
                self._fail_reconciliation_sync,
                account_id,
                dict(generations),
                failure_reason,
            )

    def _fail_reconciliation_sync(
        self,
        account_id: str,
        generations: Mapping[CapturePlant, str],
        failure_reason: str,
    ) -> None:
        now = datetime.now(timezone.utc)
        with self._session_factory() as session:
            self._ensure_connection(session, now)
            _opaque_identifier(account_id, "broker account ID", required=True)
            internal_account = self._ensure_account(session, account_id)
            for plant, generation_id in generations.items():
                generation = self._ensure_generation(
                    session,
                    generation_id=generation_id,
                    plant=plant,
                    connected_at=now,
                )
                generation.state = "FAILED"
                generation.ready = False
                generation.reconciled_at = None
                session.add(
                    V2RithmicReconciliationCheckpoint(
                        checkpoint_id=uuid4().hex,
                        generation_id=generation_id,
                        account_id=internal_account.account_id,
                        broker_account_id=account_id,
                        plant=plant.value,
                        phase="RECONCILIATION",
                        status="FAILED",
                        live_subscription_active=False,
                        discrepancy_count=0,
                        ready=False,
                        recorded_at=now,
                        failure_reason=failure_reason,
                        checkpoint_facts={"failure_reason": failure_reason},
                    )
                )
            generation_ids = tuple(generations.values())
            if generation_ids:
                for replay in session.scalars(
                    select(V2RithmicReplayBatch).where(
                        V2RithmicReplayBatch.generation_id.in_(generation_ids),
                        V2RithmicReplayBatch.account_id
                        == internal_account.account_id,
                        V2RithmicReplayBatch.status == "IN_PROGRESS",
                    )
                ):
                    replay.status = "FAILED"
                    replay.completed_at = now
                    replay.failure_reason = failure_reason
                    replay.boundary_facts = {
                        **(replay.boundary_facts or {}),
                        "closed_by_failed_reconciliation": True,
                        "failure_reason": failure_reason,
                    }
            session.commit()

    def _save_checkpoint_sync(self, checkpoint: ReconciliationCheckpoint) -> None:
        if checkpoint.discrepancy_count != 0:
            raise ValueError("a reconciliation checkpoint cannot be READY with discrepancies")
        with self._session_factory() as session:
            self._ensure_connection(session, checkpoint.completed_at)
            _opaque_identifier(
                checkpoint.account_id,
                "broker account ID",
                required=True,
            )
            internal_account = self._ensure_account(session, checkpoint.account_id)
            checkpoint_generation_map = {
                plant.value: generation_id
                for plant, generation_id in checkpoint.generations.items()
            }
            for plant, generation_id in checkpoint.generations.items():
                generation = self._ensure_generation(
                    session,
                    generation_id=generation_id,
                    plant=plant,
                    connected_at=checkpoint.completed_at,
                )
                highest = session.scalar(
                    select(func.max(V2RithmicBrokerEvent.local_ingest_sequence)).where(
                        V2RithmicBrokerEvent.generation_id == generation_id,
                        V2RithmicBrokerEvent.account_id
                        == internal_account.account_id,
                    )
                )
                session.add(
                    V2RithmicReconciliationCheckpoint(
                        checkpoint_id=uuid4().hex,
                        generation_id=generation_id,
                        account_id=internal_account.account_id,
                        broker_account_id=checkpoint.account_id,
                        plant=plant.value,
                        phase="RECONCILIATION",
                        status="COMPLETE",
                        live_subscription_active=True,
                        snapshot_completed_at=checkpoint.completed_at,
                        buffered_events_applied_at=checkpoint.completed_at,
                        highest_ingest_sequence=highest,
                        discrepancy_count=checkpoint.discrepancy_count,
                        ready=True,
                        recorded_at=checkpoint.completed_at,
                        checkpoint_facts={
                            "observer_checkpoint_sha256": hashlib.sha256(
                                checkpoint.checkpoint.encode("utf-8")
                            ).hexdigest(),
                            "buffered_events_applied": checkpoint.buffered_events_applied,
                        },
                    )
                )
                session.flush()
                for replay in session.scalars(
                    select(V2RithmicReplayBatch).where(
                        V2RithmicReplayBatch.generation_id == generation_id,
                        V2RithmicReplayBatch.account_id == internal_account.account_id,
                        V2RithmicReplayBatch.status == "IN_PROGRESS",
                    )
                ):
                    replay.generation_map = dict(checkpoint_generation_map)
                    order_generation = checkpoint_generation_map.get("ORDER")
                    if order_generation:
                        replay.generation_id = order_generation
                    replay.batch_kind = "ACCOUNT_RECOVERY"
                    replay.source_kind = CaptureSource.REPLAY.value
                    replay.status = "COMPLETE"
                    replay.completed_at = checkpoint.completed_at
                    replay.boundary_facts = {
                        **(replay.boundary_facts or {}),
                        "closed_by_reconciliation_checkpoint": True,
                        "generations": dict(checkpoint_generation_map),
                    }
                completed_accounts = set(
                    session.scalars(
                        select(V2RithmicReconciliationCheckpoint.account_id).where(
                            V2RithmicReconciliationCheckpoint.generation_id == generation_id,
                            V2RithmicReconciliationCheckpoint.status == "COMPLETE",
                            V2RithmicReconciliationCheckpoint.ready.is_(True),
                        )
                    )
                )
                if self._internal_account_allowlist.issubset(completed_accounts):
                    generation.state = "READY"
                    generation.ready = True
                    generation.reconciled_at = checkpoint.completed_at
            session.commit()

    @property
    def depth(self) -> int:
        return self._depth

    def _ensure_connection(self, session: Session, created_at: datetime) -> V2BrokerConnection:
        connection = session.get(V2BrokerConnection, self._connection_id)
        if connection is None:
            connection = V2BrokerConnection(
                connection_id=self._connection_id,
                broker="rithmic",
                environment=self._environment,
                credential_ref=None,
                # Capture connectivity never enables this generic execution
                # record.  A later write phase must require separate approval.
                enabled=False,
                created_at=created_at,
            )
            session.add(connection)
            session.flush()
        return connection

    def _internal_account_id(self, broker_account_id: str) -> str:
        material = f"{self._connection_id}\x00{broker_account_id}".encode("utf-8")
        return f"rithmic-{hashlib.sha256(material).hexdigest()}"

    def _ensure_account(self, session: Session, broker_account_id: str) -> V2BrokerAccount:
        internal_account_id = self._internal_account_id(broker_account_id)
        account = session.get(V2BrokerAccount, internal_account_id)
        if account is None:
            account = V2BrokerAccount(
                account_id=internal_account_id,
                connection_id=self._connection_id,
                broker_account_ref=broker_account_id,
                display_name="",
                # An account can be allowlisted for observation without being
                # enabled for any broker mutation path.
                enabled=False,
            )
            session.add(account)
            session.flush()
        return account

    def _ensure_generation(
        self,
        session: Session,
        *,
        generation_id: str,
        plant: CapturePlant,
        connected_at: datetime,
    ) -> V2RithmicConnectionGeneration:
        _opaque_identifier(
            generation_id,
            "capture generation ID",
            maximum=96,
            required=True,
        )
        generation = session.get(V2RithmicConnectionGeneration, generation_id)
        if generation is not None:
            return generation
        latest = session.scalar(
            select(func.max(V2RithmicConnectionGeneration.generation_ordinal)).where(
                V2RithmicConnectionGeneration.connection_id == self._connection_id,
                V2RithmicConnectionGeneration.plant == plant.value,
            )
        )
        for prior in session.scalars(
            select(V2RithmicConnectionGeneration).where(
                V2RithmicConnectionGeneration.connection_id == self._connection_id,
                V2RithmicConnectionGeneration.plant == plant.value,
                V2RithmicConnectionGeneration.disconnected_at.is_(None),
                V2RithmicConnectionGeneration.generation_id != generation_id,
            )
        ):
            prior.state = "DISCONNECTED"
            prior.ready = False
            prior.disconnected_at = connected_at
            prior.disconnect_reason = "superseded_by_new_generation"
            prior.state_details = {
                **(prior.state_details or {}),
                "superseded_by_generation": generation_id,
            }
        generation = V2RithmicConnectionGeneration(
            generation_id=generation_id,
            connection_id=self._connection_id,
            generation_ordinal=0 if latest is None else int(latest) + 1,
            plant=plant.value,
            system_name=self._system_name,
            state="CONNECTED",
            reconnect_attempt=0 if latest is None else int(latest) + 1,
            connected_at=connected_at,
            forced_logout=False,
            ready=False,
            state_details={},
        )
        session.add(generation)
        session.flush()
        return generation

    def _ensure_replay_batch(
        self,
        session: Session,
        event: CaptureEvent,
        replay_batch_id: str,
        data: Mapping[str, object],
    ) -> V2RithmicReplayBatch:
        if event.account_id is None:
            raise ValueError("replay batches require an account-scoped event")
        incoming_generations = {
            str(plant).upper(): _opaque_identifier(
                generation_id,
                "replay generation ID",
                maximum=96,
                required=True,
            )
            for plant, generation_id in _object_mapping(data.get("generations")).items()
            if str(plant).upper() in {"ORDER", "PNL", "TICKER"}
        }
        expected_event_generation = incoming_generations.get(event.plant.value)
        if expected_event_generation and expected_event_generation != event.generation_id:
            raise ValueError("replay event generation does not match its generation map")
        replay = session.get(V2RithmicReplayBatch, replay_batch_id)
        if replay is None:
            anchor_generation = incoming_generations.get("ORDER") or event.generation_id
            if session.get(V2RithmicConnectionGeneration, anchor_generation) is None:
                anchor_generation = event.generation_id
            replay = V2RithmicReplayBatch(
                replay_batch_id=replay_batch_id,
                generation_id=anchor_generation,
                generation_map=incoming_generations
                or {event.plant.value: event.generation_id},
                account_id=self._internal_account_id(event.account_id),
                broker_account_id=event.account_id,
                batch_kind=_text(data.get("batch_kind"), maximum=48) or event.source.value,
                source_kind=event.source.value,
                request_key=_opaque_identifier(data.get("request_key"), "request key"),
                user_message=_opaque_identifier(
                    _value(data, "user_message", "user_msg"), "user message"
                ),
                status="IN_PROGRESS",
                requested_at=event.received_at,
                started_at=event.received_at,
                terminal_response_received=False,
                event_count=0,
                boundary_facts={},
            )
            session.add(replay)
            session.flush()
        else:
            merged_generations = dict(replay.generation_map or {})
            for plant, generation_id in incoming_generations.items():
                existing_generation = merged_generations.get(plant)
                if existing_generation and existing_generation != generation_id:
                    raise ValueError("replay batch generation map changed")
                merged_generations[plant] = generation_id
            existing_event_generation = merged_generations.get(event.plant.value)
            if existing_event_generation and existing_event_generation != event.generation_id:
                raise ValueError("replay event belongs to a different plant generation")
            merged_generations.setdefault(event.plant.value, event.generation_id)
            replay.generation_map = merged_generations
            order_generation = merged_generations.get("ORDER")
            if (
                order_generation
                and session.get(V2RithmicConnectionGeneration, order_generation) is not None
            ):
                replay.generation_id = order_generation
        replay.event_count += 1
        batch_status = (_text(data.get("batch_status"), maximum=32) or "").upper()
        terminal_boundary = bool(batch_status) or _boolean(
            data.get("terminal_response_received")
        )
        if terminal_boundary:
            replay.batch_kind = (
                _text(data.get("batch_kind"), maximum=48) or "ACCOUNT_RECOVERY"
            )
            replay.source_kind = event.source.value
            replay.boundary_facts = {
                **(replay.boundary_facts or {}),
                "generations": dict(replay.generation_map or {}),
                "clean": _boolean(data.get("clean")),
            }
        if batch_status == "FAILED" or data.get("clean") is False:
            replay.status = "FAILED"
            replay.completed_at = event.received_at
            replay.failure_reason = (
                _text(data.get("failure_reason"), maximum=256)
                or "reconciliation_failed"
            )
            replay.terminal_response_received = _boolean(
                data.get("terminal_response_received")
            )
        elif _boolean(data.get("terminal_response_received")):
            replay.terminal_response_received = True
            replay.status = "COMPLETE"
            replay.completed_at = event.received_at
        return replay

    @staticmethod
    def _next_ingest_sequence(session: Session, generation_id: str) -> int:
        latest = session.scalar(
            select(func.max(V2RithmicBrokerEvent.local_ingest_sequence)).where(
                V2RithmicBrokerEvent.generation_id == generation_id
            )
        )
        return 0 if latest is None else int(latest) + 1

    @staticmethod
    def _update_generation_from_event(
        generation: V2RithmicConnectionGeneration,
        event: CaptureEvent,
        kind: str,
        data: Mapping[str, object],
    ) -> None:
        generation.last_message_at = event.received_at
        template_name = (_text(data.get("template_name")) or "").upper()
        message_kind = f"{kind} {template_name}"
        compact_message_kind = message_kind.replace("_", "").replace(" ", "")
        if "LOGIN" in message_kind and _boolean(data.get("success")):
            generation.authenticated_at = event.received_at
            generation.state = "AUTHENTICATED"
            heartbeat_seconds = _decimal(
                _value(data, "heartbeat_interval_seconds", "heartbeat_interval")
            )
            if heartbeat_seconds is not None and heartbeat_seconds > 0:
                generation.heartbeat_interval_ms = int(heartbeat_seconds * 1000)
        if "FORCEDLOGOUT" in compact_message_kind:
            generation.forced_logout = True
            generation.state = "FORCED_LOGOUT"
            generation.ready = False
            generation.disconnected_at = event.received_at
        control_kind = (_text(data.get("control_kind")) or "").upper()
        if control_kind == "DISCONNECTED" or "LOCALDISCONNECTED" in compact_message_kind:
            generation.state = "DISCONNECTED"
            generation.ready = False
            generation.disconnected_at = event.received_at
            generation.disconnect_reason = (
                _text(data.get("disconnect_reason"), maximum=64)
                or "connection_interrupted"
            )

    def _project_event(
        self,
        session: Session,
        event: CaptureEvent,
        raw_event: V2RithmicBrokerEvent,
        kind: str,
        data: Mapping[str, object],
    ) -> None:
        if "ACCOUNT" in kind and "PNL" not in kind and "RMS" not in kind:
            self._project_account(session, event, raw_event, data)
        if "ORDER" in kind or data.get("basket_id") is not None:
            self._project_order(session, event, raw_event, data)
        execution_effect = (_text(data.get("execution_effect")) or "NONE").upper()
        if data.get("fill_id") is not None and (
            "EXECUTION" in kind or "FILL" in kind or execution_effect != "NONE"
        ):
            self._project_execution(session, event, raw_event, data, execution_effect)
        if "BRACKET" in kind:
            self._project_bracket(session, event, raw_event, data)
        if kind in {"REFERENCE", "REFERENCE_SEARCH", "REFERENCE_TICK_SIZE"}:
            self._project_reference(session, event, raw_event, kind, data)
        if "PNL" in kind or kind == "POSITION":
            self._project_pnl(session, event, raw_event, kind, data)
        if "RMS" in kind:
            self._project_rms(session, event, raw_event, kind, data)

    def _project_account(
        self,
        session: Session,
        event: CaptureEvent,
        raw_event: V2RithmicBrokerEvent,
        data: Mapping[str, object],
    ) -> None:
        fcm_id, ib_id, broker_account_id = _broker_account(data, event.account_id)
        if not fcm_id or not ib_id:
            return
        session.add(
            V2RithmicAccountObservation(
                observation_id=uuid4().hex,
                event_id=raw_event.event_id,
                generation_id=event.generation_id,
                account_id=self._internal_account_id(event.account_id),
                fcm_id=fcm_id,
                ib_id=ib_id,
                broker_account_id=broker_account_id,
                account_name=_text(data.get("account_name"), maximum=256),
                currency=_text(data.get("currency"), maximum=16),
                access_type=_text(data.get("access_type"), maximum=32),
                account_status=_text(_value(data, "account_status", "status"), maximum=64),
                user_id=_opaque_identifier(data.get("user_id"), "user ID"),
                user_type=_text(data.get("user_type"), maximum=64),
                user_status=_text(data.get("user_status"), maximum=64),
                order_copy_status=_text(data.get("order_copy_status"), maximum=64),
                country_code=_text(data.get("country_code"), maximum=16),
                state_code=_text(data.get("state_code"), maximum=32),
                max_order_sessions=_integer(
                    _value(data, "max_order_sessions", "order_session_max")
                ),
                max_ticker_sessions=_integer(
                    _value(data, "max_ticker_sessions", "ticker_session_max")
                ),
                allowlisted=event.account_id in self._account_allowlist,
                observed_at=event.received_at,
                account_facts=raw_event.event_facts,
            )
        )

    def _project_order(
        self,
        session: Session,
        event: CaptureEvent,
        raw_event: V2RithmicBrokerEvent,
        data: Mapping[str, object],
    ) -> None:
        basket_id = _opaque_identifier(data.get("basket_id"), "basket ID")
        if not basket_id:
            return
        fcm_id, ib_id, broker_account_id = _broker_account(data, event.account_id)
        origin = _nested(data, "origin")
        normalized_state = (_text(data.get("normalized_state"), maximum=48) or "UNKNOWN").upper()
        command_failure = _boolean(data.get("command_failure"))
        command_outcome = (
            "FAILED"
            if command_failure
            else "OUTCOME_UNKNOWN"
            if normalized_state == "OUTCOME_UNKNOWN"
            else "NOT_APPLICABLE"
        )
        session.add(
            V2RithmicOrderObservation(
                order_observation_id=uuid4().hex,
                event_id=raw_event.event_id,
                account_id=self._internal_account_id(event.account_id),
                fcm_id=fcm_id,
                ib_id=ib_id,
                broker_account_id=broker_account_id,
                basket_id=basket_id,
                original_basket_id=_opaque_identifier(
                    data.get("original_basket_id"), "original basket ID"
                ),
                linked_basket_ids=_json_list(data.get("linked_basket_ids")),
                exchange_order_id=_opaque_identifier(
                    data.get("exchange_order_id"), "exchange order ID"
                ),
                ticker_plant_exchange_order_id=_opaque_identifier(
                    _value(data, "ticker_plant_exchange_order_id", "tp_exchange_order_id"),
                    "ticker-plant exchange order ID",
                ),
                symbol=_text(data.get("symbol"), maximum=128),
                exchange=_text(data.get("exchange"), maximum=64),
                normalized_state=normalized_state,
                broker_status=_text(data.get("status"), maximum=128),
                notification_type=_text(
                    _value(data, "notification_type", "raw_notify_type"), maximum=128
                ),
                completion_reason=_text(data.get("completion_reason"), maximum=256),
                report_type=_text(data.get("report_type"), maximum=128),
                command_outcome=command_outcome,
                side=_text(_value(data, "side", "transaction_type"), maximum=32),
                order_type=_text(_value(data, "order_type", "price_type"), maximum=64),
                duration=_text(data.get("duration"), maximum=64),
                quantity=_integer(data.get("quantity")),
                fill_size=_integer(data.get("fill_size")),
                total_fill_size=_integer(
                    _value(data, "total_fill_size", "cumulative_fill_size")
                ),
                total_unfilled_size=_integer(
                    _value(data, "total_unfilled_size", "unfilled_size")
                ),
                limit_price=_decimal(_value(data, "limit_price", "price")),
                trigger_price=_decimal(data.get("trigger_price")),
                fill_price=_decimal(data.get("fill_price")),
                average_fill_price=_decimal(data.get("average_fill_price")),
                fill_id=_opaque_identifier(data.get("fill_id"), "fill ID"),
                sequence_number=_opaque_identifier(
                    data.get("sequence_number"), "sequence number"
                ),
                original_sequence_number=_opaque_identifier(
                    data.get("original_sequence_number"), "original sequence number"
                ),
                correlation_sequence_number=_opaque_identifier(
                    data.get("correlation_sequence_number"), "correlation sequence number"
                ),
                user_id=_opaque_identifier(
                    _value(data, "user_id") or origin.get("user_id"), "user ID"
                ),
                application=_text(
                    _value(data, "application") or origin.get("application"), maximum=256
                ),
                application_version=_text(
                    _value(data, "application_version", "version") or origin.get("version"),
                    maximum=128,
                ),
                originator_application=_text(
                    _value(data, "originator_application")
                    or origin.get("originator_application"),
                    maximum=256,
                ),
                originator_version=_text(
                    _value(data, "originator_version") or origin.get("originator_version"),
                    maximum=128,
                ),
                window_name=_text(
                    _value(data, "window_name") or origin.get("window_name"), maximum=256
                ),
                originator_window_name=_text(
                    _value(data, "originator_window_name")
                    or origin.get("originator_window_name"),
                    maximum=256,
                ),
                manual_or_auto=_text(
                    _value(data, "manual_or_auto") or origin.get("manual_or_auto"), maximum=32
                ),
                user_tag=_opaque_identifier(
                    _value(data, "user_tag") or origin.get("user_tag"), "user tag"
                ),
                mooney_owned=_boolean(data.get("mooney_owned")),
                unknown_state=normalized_state in {"UNKNOWN", "OUTCOME_UNKNOWN"},
                terminal=normalized_state in {"FILLED", "CANCELLED", "REJECTED"},
                source_kind=event.source.value,
                broker_at=_timestamp_evidence(data, "source"),
                server_at=_timestamp_evidence(data, "server"),
                exchange_at=_timestamp_evidence(data, "exchange"),
                received_at=event.received_at,
                order_facts=raw_event.event_facts,
            )
        )

    def _project_execution(
        self,
        session: Session,
        event: CaptureEvent,
        raw_event: V2RithmicBrokerEvent,
        data: Mapping[str, object],
        execution_effect: str,
        *,
        resolve_deferred: bool = True,
    ) -> None:
        basket_id = _opaque_identifier(data.get("basket_id"), "basket ID")
        fill_id = _opaque_identifier(data.get("fill_id"), "fill ID")
        quantity = _integer(_value(data, "fill_size", "quantity"))
        price = _decimal(_value(data, "fill_price", "price"))
        if not basket_id or not fill_id:
            return
        _, _, broker_account_id = _broker_account(data, event.account_id)
        key_material = "|".join(
            (
                broker_account_id,
                basket_id,
                fill_id,
                _text(data.get("report_type")) or "",
                _text(data.get("sequence_number")) or "",
                execution_effect,
            )
        )
        execution_key = hashlib.sha256(key_material.encode("utf-8")).hexdigest()
        if session.scalar(
            select(V2RithmicExecutionObservation.execution_observation_id).where(
                V2RithmicExecutionObservation.execution_key == execution_key
            )
        ) is not None:
            return
        internal_account_id = self._internal_account_id(event.account_id)
        corrected_execution: V2RithmicExecutionObservation | None = None
        if execution_effect in {"BUST_FILL", "CORRECT_FILL"}:
            referenced_sequences = tuple(
                value
                for value in (
                    _opaque_identifier(
                        data.get("original_sequence_number"),
                        "original sequence number",
                    ),
                    _opaque_identifier(
                        data.get("correlation_sequence_number"),
                        "correlation sequence number",
                    ),
                )
                if value
            )
            correction_query = select(V2RithmicExecutionObservation).where(
                V2RithmicExecutionObservation.account_id == internal_account_id,
                V2RithmicExecutionObservation.basket_id == basket_id,
            )
            if referenced_sequences:
                correction_query = correction_query.where(
                    V2RithmicExecutionObservation.sequence_number.in_(
                        referenced_sequences
                    )
                )
            else:
                # fill_id is the official execution identity and is the only
                # safe fallback when the correction sequence evidence is
                # absent.  Prefer the latest applied/corrected version.
                correction_query = correction_query.where(
                    V2RithmicExecutionObservation.fill_id == fill_id,
                    V2RithmicExecutionObservation.execution_kind.in_(
                        ("APPLY_FILL", "CORRECT_FILL")
                    ),
                )
            corrected_execution = session.scalar(
                correction_query.order_by(
                    V2RithmicExecutionObservation.received_at.desc()
                ).limit(1)
            )
            if corrected_execution is None:
                # Replay/live overlap can deliver the correction first. The
                # immutable raw event retains its relationship evidence; defer
                # this projection until the referenced execution arrives.
                return
        delta = _integer(data.get("effective_quantity_delta"))
        if delta is None:
            if execution_effect == "BUST_FILL" and quantity is not None:
                delta = -abs(quantity)
            elif execution_effect == "APPLY_FILL" and quantity is not None:
                delta = abs(quantity)
            elif (
                execution_effect == "CORRECT_FILL"
                and quantity is not None
                and corrected_execution is not None
                and corrected_execution.quantity is not None
            ):
                delta = quantity - corrected_execution.quantity
            elif (
                execution_effect == "BUST_FILL"
                and corrected_execution is not None
                and corrected_execution.quantity is not None
            ):
                delta = -abs(corrected_execution.quantity)
        execution = V2RithmicExecutionObservation(
            execution_observation_id=uuid4().hex,
            event_id=raw_event.event_id,
            account_id=internal_account_id,
            broker_account_id=broker_account_id,
            basket_id=basket_id,
            fill_id=fill_id,
            execution_key=execution_key,
            exchange_order_id=_opaque_identifier(
                data.get("exchange_order_id"), "exchange order ID"
            ),
            ticker_plant_exchange_order_id=_opaque_identifier(
                _value(data, "ticker_plant_exchange_order_id", "tp_exchange_order_id"),
                "ticker-plant exchange order ID",
            ),
            execution_kind=execution_effect[:32],
            corrects_execution_id=(
                corrected_execution.execution_observation_id
                if corrected_execution is not None
                else None
            ),
            side=_text(_value(data, "side", "transaction_type"), maximum=32),
            quantity=quantity,
            effective_quantity_delta=delta,
            price=price,
            commission=_decimal(_value(data, "commission", "fee")),
            sequence_number=_opaque_identifier(
                data.get("sequence_number"), "sequence number"
            ),
            source_kind=event.source.value,
            executed_at=_timestamp_evidence(data, "source"),
            server_at=_timestamp_evidence(data, "server"),
            received_at=event.received_at,
            execution_facts=raw_event.event_facts,
        )
        session.add(execution)
        session.flush()
        if resolve_deferred:
            self._project_deferred_execution_corrections(session, execution)

    def _project_deferred_execution_corrections(
        self,
        session: Session,
        execution: V2RithmicExecutionObservation,
    ) -> None:
        candidates = session.scalars(
            select(V2RithmicBrokerEvent)
            .where(
                V2RithmicBrokerEvent.account_id == execution.account_id,
                V2RithmicBrokerEvent.event_id.not_in(
                    select(V2RithmicExecutionObservation.event_id)
                ),
            )
            .order_by(
                V2RithmicBrokerEvent.received_at,
                V2RithmicBrokerEvent.local_ingest_sequence,
            )
        ).all()
        for candidate in candidates:
            _, candidate_data = _flatten_payload(
                _object_mapping(candidate.event_facts)
            )
            effect = (
                _text(candidate_data.get("execution_effect")) or "NONE"
            ).upper()
            if effect not in {"BUST_FILL", "CORRECT_FILL"}:
                continue
            referenced_sequences = {
                value
                for value in (
                    candidate.original_sequence_number,
                    candidate.correlation_sequence_number,
                )
                if value
            }
            if referenced_sequences:
                if execution.sequence_number not in referenced_sequences:
                    continue
            elif not (
                candidate.fill_id == execution.fill_id
                and candidate.basket_id == execution.basket_id
            ):
                continue
            candidate_data.update(
                {
                    "broker_account_id": candidate.broker_account_id,
                    "basket_id": candidate.basket_id,
                    "fill_id": candidate.fill_id,
                    "exchange_order_id": candidate.exchange_order_id,
                    "ticker_plant_exchange_order_id": (
                        candidate.ticker_plant_exchange_order_id
                    ),
                    "sequence_number": candidate.sequence_number,
                    "original_sequence_number": candidate.original_sequence_number,
                    "correlation_sequence_number": (
                        candidate.correlation_sequence_number
                    ),
                }
            )
            try:
                source = CaptureSource(candidate.source_kind)
            except ValueError:
                source = CaptureSource.UNKNOWN
            deferred_event = CaptureEvent(
                event_id=candidate.event_id,
                account_id=candidate.broker_account_id,
                plant=CapturePlant(candidate.plant),
                source=source,
                generation_id=candidate.generation_id,
                payload={},
                received_at=candidate.received_at,
            )
            self._project_execution(
                session,
                deferred_event,
                candidate,
                candidate_data,
                effect,
                resolve_deferred=False,
            )

    def _project_bracket(
        self,
        session: Session,
        event: CaptureEvent,
        raw_event: V2RithmicBrokerEvent,
        data: Mapping[str, object],
    ) -> None:
        parent = _opaque_identifier(
            _value(data, "parent_basket_id", "basket_id"), "parent basket ID"
        )
        if not parent:
            return
        if event.account_id is None:
            raise ValueError("bracket observation requires an account")
        _, _, broker_account_id = _broker_account(data, event.account_id)

        def tiers(prefix: str) -> list[object]:
            explicit = _json_list(data.get(f"{prefix}_tiers"))
            if explicit:
                return explicit
            ticks = _integer(data.get(f"{prefix}_ticks"))
            total = _integer(
                _value(data, f"{prefix}_total_quantity", f"{prefix}_quantity")
            )
            released = _integer(
                _value(
                    data,
                    f"{prefix}_released_quantity",
                    f"{prefix}_quantity_released",
                )
            )
            if ticks is None and total is None and released is None:
                return []
            # The 0.90 read schemas expose one target and one stop tier per
            # record. Keep the tier shape extensible without inventing extra
            # levels that the broker did not report.
            return [
                {
                    "tier": 1,
                    "ticks": ticks,
                    "total_quantity": total,
                    "released_quantity": released,
                }
            ]

        session.add(
            V2RithmicBracketObservation(
                bracket_observation_id=uuid4().hex,
                event_id=raw_event.event_id,
                account_id=self._internal_account_id(event.account_id),
                broker_account_id=broker_account_id,
                parent_basket_id=parent,
                linked_basket_ids=_json_list(data.get("linked_basket_ids")),
                bracket_type=_text(data.get("bracket_type"), maximum=64),
                operation_type=_text(data.get("operation_type"), maximum=64),
                status=_text(data.get("status"), maximum=64),
                target_total_quantity=_integer(
                    _value(data, "target_total_quantity", "target_quantity")
                ),
                target_released_quantity=_integer(
                    _value(data, "target_released_quantity", "target_quantity_released")
                ),
                stop_total_quantity=_integer(
                    _value(data, "stop_total_quantity", "stop_quantity")
                ),
                stop_released_quantity=_integer(
                    _value(data, "stop_released_quantity", "stop_quantity_released")
                ),
                target_tiers=tiers("target"),
                stop_tiers=tiers("stop"),
                trailing_facts=redact_sensitive(
                    {
                        key: _json_safe(data[key])
                        for key in (
                            "trailing_field_id",
                            "trailing_stop_trigger_ticks",
                            "trailing_stop_ticks",
                        )
                        if key in data
                    }
                ),
                source_kind=event.source.value,
                observed_at=_timestamp_evidence(data, "source"),
                received_at=event.received_at,
                bracket_facts=raw_event.event_facts,
            )
        )

    def _project_reference(
        self,
        session: Session,
        event: CaptureEvent,
        raw_event: V2RithmicBrokerEvent,
        kind: str,
        data: Mapping[str, object],
    ) -> None:
        broker_account_id = None
        internal_account_id = None
        if event.account_id is not None:
            _, _, broker_account_id = _broker_account(data, event.account_id)
            internal_account_id = self._internal_account_id(event.account_id)
        session.add(
            V2RithmicReferenceObservation(
                reference_observation_id=uuid4().hex,
                event_id=raw_event.event_id,
                generation_id=event.generation_id,
                account_id=internal_account_id,
                broker_account_id=broker_account_id,
                observation_kind=kind,
                symbol=_opaque_identifier(data.get("symbol"), "reference symbol", maximum=128),
                exchange=_opaque_identifier(
                    data.get("exchange"), "reference exchange", maximum=64
                ),
                exchange_symbol=_opaque_identifier(
                    data.get("exchange_symbol"), "exchange symbol", maximum=128
                ),
                symbol_name=_text(data.get("symbol_name"), maximum=256),
                trading_symbol=_opaque_identifier(
                    data.get("trading_symbol"), "trading symbol", maximum=128
                ),
                trading_exchange=_opaque_identifier(
                    data.get("trading_exchange"), "trading exchange", maximum=64
                ),
                product_code=_opaque_identifier(
                    data.get("product_code"), "product code", maximum=64
                ),
                instrument_type=_text(data.get("instrument_type"), maximum=64),
                underlying_symbol=_opaque_identifier(
                    data.get("underlying_symbol"), "underlying symbol", maximum=128
                ),
                expiration_date=_date(data.get("expiration_date")),
                currency=_text(data.get("currency"), maximum=16),
                tick_size_type=_text(data.get("tick_size_type"), maximum=64),
                price_display_format=_text(data.get("price_display_format"), maximum=64),
                is_tradable=_optional_boolean(data.get("is_tradable")),
                minimum_quoted_price_change=_decimal(
                    data.get("minimum_quoted_price_change")
                ),
                minimum_feed_price_change=_decimal(data.get("minimum_feed_price_change")),
                single_point_value=_decimal(data.get("single_point_value")),
                quote_to_feed_price_factor=_decimal(data.get("quote_to_feed_price_factor")),
                feed_to_quote_price_factor=_decimal(data.get("feed_to_quote_price_factor")),
                tick_table_first_price=_decimal(data.get("first_price")),
                tick_table_last_price=_decimal(data.get("last_price")),
                tick_table_first_price_operator=_text(
                    data.get("first_price_operator"), maximum=32
                ),
                tick_table_last_price_operator=_text(
                    data.get("last_price_operator"), maximum=32
                ),
                presence_bits=_integer(data.get("presence_bits")),
                source_kind=event.source.value,
                source_at=_timestamp_evidence(data, "source"),
                received_at=event.received_at,
                reference_facts=raw_event.event_facts,
            )
        )

    def _project_pnl(
        self,
        session: Session,
        event: CaptureEvent,
        raw_event: V2RithmicBrokerEvent,
        kind: str,
        data: Mapping[str, object],
    ) -> None:
        _, _, broker_account_id = _broker_account(data, event.account_id)
        scope = "INSTRUMENT" if "INSTRUMENT" in kind or data.get("symbol") else "ACCOUNT"
        session.add(
            V2RithmicPnLObservation(
                pnl_observation_id=uuid4().hex,
                event_id=raw_event.event_id,
                account_id=self._internal_account_id(event.account_id),
                broker_account_id=broker_account_id,
                scope=scope,
                symbol=_text(data.get("symbol"), maximum=128),
                exchange=_text(data.get("exchange"), maximum=64),
                currency=_text(data.get("currency"), maximum=16),
                trade_date=None,
                net_quantity=_integer(data.get("net_quantity")),
                # Official buy/sell quantities are not synonymous with current
                # long/short position.  Populate only explicitly supplied facts.
                long_quantity=_integer(data.get("long_quantity")),
                short_quantity=_integer(data.get("short_quantity")),
                open_quantity=_integer(
                    _value(data, "open_quantity", "open_position_quantity")
                ),
                closed_quantity=_integer(
                    _value(data, "closed_quantity", "closed_position_quantity")
                ),
                working_buy_quantity=_integer(data.get("working_buy_quantity")),
                working_sell_quantity=_integer(data.get("working_sell_quantity")),
                average_open_fill_price=_decimal(data.get("average_open_fill_price")),
                open_position_pnl=_decimal(data.get("open_position_pnl")),
                closed_position_pnl=_decimal(data.get("closed_position_pnl")),
                day_open_pnl=_decimal(data.get("day_open_pnl")),
                day_closed_pnl=_decimal(data.get("day_closed_pnl")),
                day_total_pnl=_decimal(_value(data, "day_total_pnl", "day_pnl")),
                day_open_pnl_offset=_decimal(data.get("day_open_pnl_offset")),
                day_closed_pnl_offset=_decimal(data.get("day_closed_pnl_offset")),
                account_balance=_decimal(data.get("account_balance")),
                cash_on_hand=_decimal(data.get("cash_on_hand")),
                margin_balance=_decimal(data.get("margin_balance")),
                available_buying_power=_decimal(data.get("available_buying_power")),
                used_buying_power=_decimal(data.get("used_buying_power")),
                reserved_buying_power=_decimal(data.get("reserved_buying_power")),
                excess_buy_margin=_decimal(data.get("excess_buy_margin")),
                excess_sell_margin=_decimal(data.get("excess_sell_margin")),
                commission=_decimal(data.get("commission")),
                source_kind=event.source.value,
                is_snapshot=event.source.value == "SNAPSHOT",
                freshness=(
                    "UNKNOWN"
                    if event.source is CaptureSource.UNKNOWN
                    else "HISTORICAL"
                    if event.source in {CaptureSource.HISTORY, CaptureSource.REPLAY}
                    else "CURRENT"
                ),
                source_at=_timestamp_evidence(data, "source"),
                received_at=event.received_at,
                pnl_facts=raw_event.event_facts,
            )
        )

    def _project_rms(
        self,
        session: Session,
        event: CaptureEvent,
        raw_event: V2RithmicBrokerEvent,
        kind: str,
        data: Mapping[str, object],
    ) -> None:
        _, _, broker_account_id = _broker_account(data, event.account_id)
        scope = "PRODUCT" if "PRODUCT" in kind or data.get("product_code") else "ACCOUNT"
        session.add(
            V2RithmicRmsObservation(
                rms_observation_id=uuid4().hex,
                event_id=raw_event.event_id,
                account_id=self._internal_account_id(event.account_id),
                broker_account_id=broker_account_id,
                scope=scope,
                product_code=_text(data.get("product_code"), maximum=64),
                currency=_text(data.get("currency"), maximum=16),
                status=_text(data.get("status"), maximum=64),
                algorithm=_text(data.get("algorithm"), maximum=128),
                loss_limit=_decimal(data.get("loss_limit")),
                minimum_account_balance=_decimal(data.get("minimum_account_balance")),
                minimum_margin_balance=_decimal(data.get("minimum_margin_balance")),
                account_balance=_decimal(data.get("account_balance")),
                current_auto_liquidate_threshold=_decimal(
                    _value(
                        data,
                        "current_auto_liquidate_threshold",
                        "auto_liquidate_threshold",
                    )
                ),
                peak_account_balance=_decimal(data.get("peak_account_balance")),
                peak_account_balance_at=_datetime(
                    _value(data, "peak_account_balance_at", "peak_account_balance_ssboe")
                ),
                auto_liquidate=_optional_boolean(data.get("auto_liquidate")),
                auto_liquidate_criteria=_text(
                    data.get("auto_liquidate_criteria"), maximum=128
                ),
                disable_on_auto_liquidate=_optional_boolean(
                    data.get("disable_on_auto_liquidate")
                ),
                max_order_quantity=_integer(
                    _value(data, "max_order_quantity", "maximum_order_quantity")
                ),
                buy_limit=_integer(data.get("buy_limit")),
                sell_limit=_integer(data.get("sell_limit")),
                buy_margin_rate=_decimal(data.get("buy_margin_rate")),
                sell_margin_rate=_decimal(data.get("sell_margin_rate")),
                commission_rate=_decimal(
                    _value(data, "commission_rate", "commission_fill_rate", "default_commission")
                ),
                source_kind=event.source.value,
                source_at=_timestamp_evidence(data, "source"),
                received_at=event.received_at,
                rms_facts=raw_event.event_facts,
            )
        )


async def create_journal(config: CaptureConfig) -> SQLAlchemyCaptureJournal:
    """Capture-service factory using the existing application ``DATABASE_URL``."""

    journal = SQLAlchemyCaptureJournal(
        config,
        system_name=os.environ.get("RITHMIC_SYSTEM_NAME", "").strip(),
    )
    await journal.initialize()
    return journal


__all__ = ["SQLAlchemyCaptureJournal", "create_journal"]
