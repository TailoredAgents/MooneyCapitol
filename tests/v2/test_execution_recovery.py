from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import MetaData, Table, create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.db.models import Base
from app.v2.db_models import (
    V2CopyControlAudit,
    V2CopyDeliveryEvent,
    V2CopyFollowerControl,
    V2CopyLatency,
    V2CopyLifecycleCommand,
    V2CopyLifecycleEvent,
    V2CopyLifecycleIntent,
    V2CopyOutbox,
    V2CopyProfile,
    V2CopyReconciliationFinding,
    V2CopyReconciliationRun,
    V2ExecutionLease,
)
from app.v2.domain.broker_observation import ObservedOrderState
from app.v2.domain.models import Side
from app.v2.execution.copy_control import FollowerControlState
from app.v2.execution.copy_control_store import SqlAlchemyCopyControlStore
from app.v2.execution.lifecycle import (
    BracketOperation,
    CopyDirective,
    FollowerLifecyclePlan,
    LifecycleReceipt,
    LifecycleStatus,
    MasterEventKind,
    MasterLifecycleEvent,
    ReceiptKind,
)
from app.v2.execution.lifecycle_service import DurableLifecycleCopier
from app.v2.execution.lifecycle_store import (
    ExecutionFenceLost,
    SqlAlchemyLifecycleStore,
)
from app.v2.execution.lease import SqlAlchemyFencedLease
from app.v2.execution.reconciliation import BrokerAuthoritativeReconciler
from app.v2.execution.recovery import (
    BrokerOrderState,
    BrokerPositionState,
    FollowerBrokerSnapshot,
    LatencyStage,
    OutboxStatus,
    ReconciliationStatus,
    summarize_latency,
)


NOW = datetime(2026, 10, 15, 16, 0, tzinfo=timezone.utc)
TABLES = (
    V2CopyProfile.__table__,
    V2CopyFollowerControl.__table__,
    V2CopyControlAudit.__table__,
    V2CopyLifecycleIntent.__table__,
    V2CopyLifecycleCommand.__table__,
    V2CopyLifecycleEvent.__table__,
    V2ExecutionLease.__table__,
    V2CopyOutbox.__table__,
    V2CopyDeliveryEvent.__table__,
    V2CopyReconciliationRun.__table__,
    V2CopyReconciliationFinding.__table__,
    V2CopyLatency.__table__,
)


class _Clock:
    def __init__(self, value: datetime) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value

    def advance(self, **values: int) -> None:
        self.value += timedelta(**values)


def _session() -> Session:
    engine = create_engine(
        "sqlite+pysqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine, tables=list(TABLES))
    session = Session(engine)
    session.add(
        V2CopyProfile(
            profile_id="mooney-five-account",
            name="Mooney five-account copier",
            master_connection_id="master",
            global_paused=False,
            revision=1,
            created_at=NOW,
            updated_at=NOW,
        )
    )
    for number in range(1, 5):
        session.add(
            V2CopyFollowerControl(
                profile_id="mooney-five-account",
                connection_id=f"follower-{number}",
                follower_number=number,
                state=FollowerControlState.ENABLED.value,
                revision=1,
                updated_at=NOW,
            )
        )
    session.commit()
    return session


def _plan(event_id: str = "master-event-1") -> FollowerLifecyclePlan:
    event = MasterLifecycleEvent(
        source_event_id=event_id,
        master_order_id="master-order-1",
        revision=1,
        kind=MasterEventKind.POSITION_TARGET,
        occurred_at=NOW,
        received_at=NOW + timedelta(milliseconds=1),
    )
    return FollowerLifecyclePlan(
        profile_id="mooney-five-account",
        follower_connection_id="follower-1",
        event=event,
        directive=CopyDirective.SYNC_POSITION,
        contract_id="CME:MNQZ6",
        symbol="MNQZ6",
        exchange="CME",
        current_signed_quantity=0,
        evaluated_target_signed_quantity=1,
        achievable_target_signed_quantity=1,
        delta_signed_quantity=1,
        allowed=True,
        reason="cumulative_plan_ready",
        stop_ticks=8,
        bracket_operation=BracketOperation.AFOCCA,
        desired_quantity=Decimal("1"),
        desired_risk_dollars=Decimal("25"),
        achievable_risk_dollars=Decimal("25"),
        desired_risk_fraction=Decimal("0.01"),
        achievable_risk_fraction=Decimal("0.01"),
        desired_exposure_fraction=Decimal("0.01"),
        achievable_exposure_fraction=Decimal("0.01"),
        risk_parity_error=Decimal("0"),
        exposure_parity_error=Decimal("0"),
        parity_error=Decimal("0"),
        target_margin_required=Decimal("100"),
    )


def _snapshot(
    snapshot_id: str,
    *,
    captured_at: datetime | None = None,
    orders: tuple[BrokerOrderState, ...] = (),
    positions: tuple[BrokerPositionState, ...] = (),
) -> FollowerBrokerSnapshot:
    return FollowerBrokerSnapshot(
        snapshot_id=snapshot_id,
        follower_connection_id="follower-1",
        connection_generation="generation-2",
        captured_at=captured_at or NOW + timedelta(seconds=20),
        positions_complete=True,
        orders_complete=True,
        executions_complete=True,
        protection_complete=True,
        orders=orders,
        positions=positions,
    )


class _Adapter:
    connection_id = "follower-1"

    def __init__(self) -> None:
        self.commands = []

    async def dispatch(self, command):
        self.commands.append(command)
        return SimpleNamespace(transmitted=True, broker_order_ids=("broker-1",))


def test_fenced_takeover_marks_abandoned_claim_uncertain_and_never_retries() -> None:
    session = _session()
    try:
        clock = _Clock(NOW + timedelta(milliseconds=2))
        lease = SqlAlchemyFencedLease(session, clock=clock)
        store = SqlAlchemyLifecycleStore(session, clock=clock)
        first = lease.acquire("copy-execution", "owner-1", timedelta(seconds=10))
        assert first is not None
        intent = store.persist_intent(_plan())
        claimed = store.claim_dispatch(
            intent.intent_id,
            token=first,
            claim_ttl=timedelta(seconds=10),
        )
        assert claimed is not None

        clock.advance(seconds=11)
        second = lease.acquire("copy-execution", "owner-2", timedelta(seconds=10))
        assert second is not None and second.fence == first.fence + 1
        with pytest.raises(ExecutionFenceLost):
            store.mark_dispatched(
                claimed.command.command_id,
                token=first,
                broker_order_ids=("possibly-sent",),
            )
        assert store.recover_abandoned_claims(token=second) == (
            claimed.command.command_id,
        )
        recovered = store.get_intent(intent.intent_id)
        assert recovered.status == LifecycleStatus.DISPATCHING
        assert recovered.commands[0].outbox_status == OutboxStatus.UNCERTAIN

        adapter = _Adapter()
        copier = DurableLifecycleCopier(
            store,
            {"follower-1": adapter},
            lease_token=second,
        )
        outcome = asyncio.run(copier.resume_intent(intent.intent_id))
        assert outcome.transmitted is False
        assert adapter.commands == []
    finally:
        session.close()


def test_broker_snapshot_adopts_explained_order_after_crash_without_duplicate() -> None:
    session = _session()
    try:
        clock = _Clock(NOW + timedelta(milliseconds=2))
        lease = SqlAlchemyFencedLease(session, clock=clock)
        store = SqlAlchemyLifecycleStore(session, clock=clock)
        first = lease.acquire("copy-execution", "owner-1", timedelta(seconds=10))
        assert first is not None
        intent = store.persist_intent(_plan())
        claimed = store.claim_dispatch(
            intent.intent_id,
            token=first,
            claim_ttl=timedelta(seconds=10),
        )
        assert claimed is not None
        clock.advance(seconds=11)
        second = lease.acquire("copy-execution", "owner-2", timedelta(seconds=30))
        assert second is not None

        order = BrokerOrderState(
            broker_order_id="broker-bracket-1",
            normalized_state=ObservedOrderState.WORKING,
            symbol="MNQZ6",
            exchange="CME",
            side=Side.BUY,
            quantity=1,
            local_command_id=claimed.command.command_id,
        )
        reconciler = BrokerAuthoritativeReconciler(
            store,
            SqlAlchemyCopyControlStore(session),
        )
        outcome = reconciler.reconcile(
            token=second,
            snapshot=_snapshot(
                "snapshot-adopted",
                captured_at=clock.value,
                orders=(order,),
            ),
        )
        assert outcome.result.status == ReconciliationStatus.ADOPTED
        updated = store.get_intent(intent.intent_id)
        assert updated.status == LifecycleStatus.ACKNOWLEDGED
        assert updated.commands[0].outbox_status == OutboxStatus.CONFIRMED
        control = SqlAlchemyCopyControlStore(session).get_profile()
        assert control.followers[0].state == FollowerControlState.ENABLED
    finally:
        session.close()


def test_unexplained_ambiguous_write_locks_only_that_follower() -> None:
    session = _session()
    try:
        clock = _Clock(NOW + timedelta(milliseconds=2))
        lease = SqlAlchemyFencedLease(session, clock=clock)
        store = SqlAlchemyLifecycleStore(session, clock=clock)
        first = lease.acquire("copy-execution", "owner-1", timedelta(seconds=10))
        assert first is not None
        intent = store.persist_intent(_plan())
        claimed = store.claim_dispatch(
            intent.intent_id,
            token=first,
            claim_ttl=timedelta(seconds=10),
        )
        assert claimed is not None
        clock.advance(seconds=11)
        second = lease.acquire("copy-execution", "owner-2", timedelta(seconds=30))
        assert second is not None
        outcome = BrokerAuthoritativeReconciler(
            store,
            SqlAlchemyCopyControlStore(session),
        ).reconcile(
            token=second,
            snapshot=_snapshot("snapshot-missing", captured_at=clock.value),
        )

        assert outcome.result.status == ReconciliationStatus.LOCKED
        assert {item.finding_kind for item in outcome.result.findings} == {
            "ambiguous_command_missing"
        }
        profile = SqlAlchemyCopyControlStore(session).get_profile()
        assert profile.followers[0].state == FollowerControlState.LOCKED
        assert all(
            item.state == FollowerControlState.ENABLED
            for item in profile.followers[1:]
        )
        assert store.get_intent(intent.intent_id).status == LifecycleStatus.DIVERGED
        run = session.execute(select(V2CopyReconciliationRun)).scalar_one()
        assert run.status == ReconciliationStatus.LOCKED.value
        assert run.finding_count == 1
    finally:
        session.close()


def test_missing_protection_on_filled_position_is_a_locking_divergence() -> None:
    session = _session()
    try:
        clock = _Clock(NOW + timedelta(milliseconds=2))
        lease = SqlAlchemyFencedLease(session, clock=clock)
        token = lease.acquire("copy-execution", "owner", timedelta(seconds=60))
        assert token is not None
        store = SqlAlchemyLifecycleStore(session, clock=clock)
        intent = store.persist_intent(_plan())
        command = store.claim_dispatch(
            intent.intent_id,
            token=token,
            claim_ttl=timedelta(seconds=30),
        )
        assert command is not None
        store.mark_dispatched(
            command.command.command_id,
            token=token,
            broker_order_ids=("broker-filled",),
        )
        store.record_receipt(
            LifecycleReceipt(
                receipt_id="filled-before-restart",
                command_id=command.command.command_id,
                kind=ReceiptKind.FILL,
                occurred_at=NOW + timedelta(seconds=1),
                broker_order_id="broker-filled",
                cumulative_filled_quantity=1,
                fill_price=Decimal("20000"),
            )
        )
        position = BrokerPositionState(
            contract_id="CME:MNQZ6",
            symbol="MNQZ6",
            exchange="CME",
            signed_quantity=1,
        )
        outcome = BrokerAuthoritativeReconciler(
            store,
            SqlAlchemyCopyControlStore(session),
        ).reconcile(
            token=token,
            snapshot=_snapshot(
                "snapshot-no-stop",
                captured_at=clock.value,
                positions=(position,),
            ),
        )
        assert outcome.result.status == ReconciliationStatus.LOCKED
        assert "protective_stop_shortfall" in {
            item.finding_kind for item in outcome.result.findings
        }
    finally:
        session.close()


def test_lifecycle_latency_records_event_decision_submit_ack_and_fill() -> None:
    session = _session()
    try:
        clock = _Clock(NOW + timedelta(milliseconds=10))
        lease = SqlAlchemyFencedLease(session, clock=clock)
        token = lease.acquire("copy-execution", "owner", timedelta(seconds=60))
        assert token is not None
        store = SqlAlchemyLifecycleStore(session, clock=clock)
        intent = store.persist_intent(_plan("latency-event"))
        clock.advance(milliseconds=10)
        command = store.claim_dispatch(
            intent.intent_id,
            token=token,
            claim_ttl=timedelta(seconds=30),
        )
        assert command is not None
        clock.advance(milliseconds=5)
        store.mark_dispatched(
            command.command.command_id,
            token=token,
            broker_order_ids=("broker-latency",),
        )
        store.record_receipt(
            LifecycleReceipt(
                receipt_id="latency-ack",
                command_id=command.command.command_id,
                kind=ReceiptKind.ACKNOWLEDGED,
                occurred_at=NOW + timedelta(milliseconds=40),
                broker_order_id="broker-latency",
            )
        )
        store.record_receipt(
            LifecycleReceipt(
                receipt_id="latency-fill",
                command_id=command.command.command_id,
                kind=ReceiptKind.FILL,
                occurred_at=NOW + timedelta(milliseconds=60),
                broker_order_id="broker-latency",
                cumulative_filled_quantity=1,
                fill_price=Decimal("20000"),
            )
        )

        assert store.latency_summary(
            LatencyStage.EVENT_TO_DECISION
        ).p50_ms == Decimal("10.000")
        assert store.latency_summary(
            LatencyStage.EVENT_RECEIVED_TO_DECISION
        ).p50_ms == Decimal("9.000")
        assert store.latency_summary(
            LatencyStage.SUBMISSION_TO_RESPONSE
        ).p50_ms == Decimal("5.000")
        assert store.latency_summary(
            LatencyStage.SUBMISSION_TO_ACKNOWLEDGEMENT
        ).p50_ms == Decimal("20.000")
        assert store.latency_summary(
            LatencyStage.SUBMISSION_TO_FIRST_FILL
        ).p50_ms == Decimal("40.000")
    finally:
        session.close()


def test_latency_percentiles_use_deterministic_nearest_rank() -> None:
    summary = summarize_latency(
        LatencyStage.EVENT_TO_SUBMISSION,
        (Decimal(index) for index in range(1, 101)),
    )
    assert (summary.p50_ms, summary.p95_ms, summary.p99_ms) == (
        Decimal("50"),
        Decimal("95"),
        Decimal("99"),
    )


def test_execution_recovery_migration_matches_models_and_protects_evidence(
    monkeypatch,
) -> None:
    path = Path("migrations/versions/0014_v2_execution_recovery.py")
    spec = importlib.util.spec_from_file_location("migration_0014", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    declared: dict[str, Table] = {}

    def capture(name, *items, **kwargs):
        declared[name] = Table(name, MetaData(), *items, **kwargs)

    monkeypatch.setattr(module, "_create", capture)
    monkeypatch.setattr(module, "_index", lambda name, table, columns: None)
    monkeypatch.setattr(module, "_backfill_existing_commands", lambda: None)
    monkeypatch.setattr(module, "_install_append_only_triggers", lambda: None)
    module.upgrade()

    assert module.down_revision == "0013_v2_copy_lifecycle"
    for model in (
        V2CopyOutbox,
        V2CopyDeliveryEvent,
        V2CopyReconciliationRun,
        V2CopyReconciliationFinding,
        V2CopyLatency,
    ):
        migration_table = declared[model.__tablename__]
        assert set(migration_table.c.keys()) == set(model.__table__.c.keys())
        for column_name, orm_column in model.__table__.c.items():
            migration_column = migration_table.c[column_name]
            assert type(migration_column.type) is type(orm_column.type)
            assert migration_column.nullable == orm_column.nullable
            assert migration_column.primary_key == orm_column.primary_key

    source = path.read_text(encoding="utf-8")
    assert "BEFORE UPDATE OR DELETE ON {table}" in source
    assert '"v2_copy_delivery_events"' in source
    assert '"v2_copy_reconciliation_findings"' in source
    assert "migration_requires_reconciliation" in source
