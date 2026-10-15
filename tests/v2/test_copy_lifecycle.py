from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import MetaData, Table, create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.db.models import Base
from app.v2.db_models import (
    V2CopyDeliveryEvent,
    V2CopyLatency,
    V2CopyLifecycleCommand,
    V2CopyLifecycleEvent,
    V2CopyLifecycleIntent,
    V2CopyOutbox,
    V2CopyProfile,
    V2ExecutionLease,
)
from app.v2.domain.models import Side
from app.v2.execution.lifecycle import (
    BracketOperation,
    CopyAction,
    CopyDirective,
    FollowerLifecyclePlan,
    LifecycleReceipt,
    LifecycleStatus,
    MasterEventKind,
    MasterLifecycleEvent,
    OrderType,
    ReceiptKind,
    compile_commands,
)
from app.v2.execution.lifecycle_service import DurableLifecycleCopier
from app.v2.execution.lifecycle_bridge import lifecycle_plan_from_cumulative
from app.v2.execution.lifecycle_store import (
    DuplicateMasterEventConflict,
    SqlAlchemyLifecycleStore,
    StaleMasterEvent,
)
from app.v2.execution.lease import SqlAlchemyFencedLease
from app.v2.execution.recovery import OutboxStatus


NOW = datetime(2026, 10, 15, 14, 0, tzinfo=timezone.utc)
TABLES = (
    V2CopyProfile.__table__,
    V2CopyLifecycleIntent.__table__,
    V2CopyLifecycleCommand.__table__,
    V2CopyLifecycleEvent.__table__,
    V2CopyOutbox.__table__,
    V2CopyDeliveryEvent.__table__,
    V2CopyLatency.__table__,
    V2ExecutionLease.__table__,
)


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
    session.commit()
    return session


def _lease(session: Session):
    token = SqlAlchemyFencedLease(session).acquire(
        "v2-copy-execution",
        "test-owner",
        timedelta(seconds=30),
    )
    assert token is not None
    return token


def _event(
    event_id: str,
    *,
    revision: int = 1,
    kind: MasterEventKind = MasterEventKind.POSITION_TARGET,
) -> MasterLifecycleEvent:
    return MasterLifecycleEvent(
        source_event_id=event_id,
        master_order_id="master-order-1",
        revision=revision,
        kind=kind,
        occurred_at=NOW,
        received_at=NOW + timedelta(milliseconds=1),
    )


def _plan(
    follower: str = "follower-1",
    *,
    event: MasterLifecycleEvent | None = None,
    current: int = 0,
    target: int = 1,
    allowed: bool = True,
    reason: str = "cumulative_plan_ready",
    directive: CopyDirective = CopyDirective.SYNC_POSITION,
    **values: object,
) -> FollowerLifecyclePlan:
    if (
        allowed
        and directive == CopyDirective.SYNC_POSITION
        and target != 0
        and (current == 0 or (current > 0) != (target > 0) or abs(target) > abs(current))
    ):
        values.setdefault("stop_ticks", 8)
        values.setdefault("bracket_operation", BracketOperation.AFOCCA)
    return FollowerLifecyclePlan(
        profile_id="mooney-five-account",
        follower_connection_id=follower,
        event=event or _event("master-event-1"),
        directive=directive,
        contract_id="CME:MNQZ6",
        symbol="MNQZ6",
        exchange="CME",
        current_signed_quantity=current,
        evaluated_target_signed_quantity=target,
        achievable_target_signed_quantity=target,
        delta_signed_quantity=target - current,
        allowed=allowed,
        reason=reason,
        desired_quantity=Decimal(abs(target)),
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
        **values,
    )


def test_position_lifecycle_compiles_add_reduce_exit_and_ordered_reversal() -> None:
    opening = compile_commands(_plan(current=0, target=2))
    assert [(item.action, item.side, item.quantity) for item in opening] == [
        (CopyAction.BRACKET, Side.BUY, 2)
    ]

    reducing = compile_commands(_plan(current=3, target=1))
    assert [(item.action, item.side, item.quantity) for item in reducing] == [
        (CopyAction.SUBMIT, Side.SELL, 2)
    ]

    exiting = compile_commands(_plan(current=-2, target=0))
    assert [(item.action, item.side, item.quantity) for item in exiting] == [
        (CopyAction.EXIT_POSITION, Side.BUY, 2)
    ]

    reversing = compile_commands(_plan(current=2, target=-1))
    assert [item.action for item in reversing] == [
        CopyAction.EXIT_POSITION,
        CopyAction.BRACKET,
    ]
    assert reversing[0].quantity == 2
    assert reversing[1].quantity == 1
    assert reversing[1].depends_on_command_id == reversing[0].command_id


def test_cumulative_sizing_facts_flow_unchanged_into_lifecycle_plan() -> None:
    sizing = SimpleNamespace(
        evaluated_contract_id="CME:MNQZ6",
        current_contract_id="CME:MNQZ6",
        current_signed_quantity=1,
        evaluated_target_signed_quantity=3,
        achievable_target_signed_quantity=3,
        delta_signed_quantity=2,
        allowed=True,
        reason="cumulative_plan_ready",
        desired_target_quantity=Decimal("2.8"),
        desired_risk_dollars=Decimal("70"),
        achievable_risk_dollars=Decimal("75"),
        desired_risk_fraction=Decimal("0.028"),
        achievable_risk_fraction=Decimal("0.03"),
        desired_point_exposure_fraction=Decimal("0.031"),
        achievable_point_exposure_fraction=Decimal("0.033"),
        risk_parity_error=Decimal("0.071"),
        exposure_parity_error=Decimal("0.064"),
        parity_error=Decimal("0.071"),
        over_target=True,
        target_margin_required=Decimal("300"),
    )
    plan = lifecycle_plan_from_cumulative(
        profile_id="mooney-five-account",
        follower_connection_id="follower-1",
        event=_event("sized-master-event"),
        sizing=sizing,
        contract_id="CME:MNQZ6",
        symbol="MNQZ6",
        exchange="CME",
        stop_ticks=8,
        bracket_operation=BracketOperation.AFOCCA,
    )
    assert plan.desired_quantity == Decimal("2.8")
    assert plan.achievable_target_signed_quantity == 3
    assert plan.delta_signed_quantity == 2
    assert plan.desired_risk_dollars == Decimal("70")
    assert plan.parity_error == Decimal("0.071")
    assert plan.over_target is True
    with pytest.raises(ValueError, match="match"):
        lifecycle_plan_from_cumulative(
            profile_id="mooney-five-account",
            follower_connection_id="follower-1",
            event=_event("wrong-contract-event"),
            sizing=sizing,
            contract_id="CME:MNQH7",
            symbol="MNQH7",
            exchange="CME",
            stop_ticks=8,
            bracket_operation=BracketOperation.AFOCCA,
        )


def test_bracket_oco_replace_cancel_and_stop_limit_contracts() -> None:
    bracket = compile_commands(
        _plan(
            directive=CopyDirective.SUBMIT_BRACKET,
            stop_ticks=8,
            target_ticks=16,
            bracket_operation=BracketOperation.AFOCCA,
        )
    )[0]
    assert bracket.action == CopyAction.BRACKET

    oco = compile_commands(
        _plan(
            current=1,
            target=1,
            directive=CopyDirective.SUBMIT_OCO,
            requested_side=Side.SELL,
            requested_quantity=1,
            limit_price=Decimal("20100"),
            trigger_price=Decimal("19900"),
        )
    )[0]
    assert (oco.action, oco.side, oco.quantity) == (CopyAction.OCO, Side.SELL, 1)

    replace = compile_commands(
        _plan(
            current=1,
            target=1,
            directive=CopyDirective.REPLACE_WORKING,
            existing_broker_order_id="basket-1",
            requested_side=Side.SELL,
            requested_quantity=1,
            order_type=OrderType.STOP_LIMIT,
            limit_price=Decimal("19995"),
            trigger_price=Decimal("20000"),
        )
    )[0]
    assert replace.action == CopyAction.MODIFY

    cancel = compile_commands(
        _plan(
            current=1,
            target=1,
            directive=CopyDirective.CANCEL_WORKING,
            existing_broker_order_id="basket-1",
        )
    )[0]
    assert cancel.action == CopyAction.CANCEL


def test_position_increase_cannot_be_compiled_without_attached_stop_policy() -> None:
    with pytest.raises(ValueError, match="protective stop"):
        _plan(stop_ticks=None, bracket_operation=None)
    with pytest.raises(ValueError, match="protective stop"):
        _plan(
            directive=CopyDirective.SUBMIT_BRACKET,
            stop_ticks=None,
            target_ticks=16,
            bracket_operation=BracketOperation.AFOCCA,
        )


def test_store_persists_before_dispatch_and_deduplicates_source_event() -> None:
    session = _session()
    try:
        store = SqlAlchemyLifecycleStore(session)
        first = store.persist_intent(_plan())
        duplicate = store.persist_intent(_plan())
        delayed_duplicate = store.persist_intent(
            _plan(
                event=replace(
                    _event("master-event-1"),
                    received_at=NOW + timedelta(seconds=30),
                )
            )
        )

        assert first.created is True
        assert duplicate.created is False
        assert delayed_duplicate.created is False
        assert duplicate.intent_id == first.intent_id
        assert duplicate.next_dispatchable is not None
        assert len(session.execute(select(V2CopyLifecycleIntent)).scalars().all()) == 1
        assert len(session.execute(select(V2CopyLifecycleCommand)).scalars().all()) == 1
        event_rows = session.execute(select(V2CopyLifecycleEvent)).scalars().all()
        assert [row.event_kind for row in event_rows] == ["INTENT_CREATED"]

        changed = _plan(reason="different_reason")
        with pytest.raises(DuplicateMasterEventConflict):
            store.persist_intent(changed)
    finally:
        session.close()


def test_store_rejects_stale_revision_and_duplicate_revision_identity() -> None:
    session = _session()
    try:
        store = SqlAlchemyLifecycleStore(session)
        store.persist_intent(_plan(event=_event("master-event-2", revision=2)))
        with pytest.raises(StaleMasterEvent):
            store.persist_intent(_plan(event=_event("master-event-1", revision=1)))
        with pytest.raises(DuplicateMasterEventConflict):
            store.persist_intent(_plan(event=_event("master-event-other", revision=2)))
    finally:
        session.close()


def test_partial_fill_fill_and_duplicate_receipt_are_monotonic() -> None:
    session = _session()
    try:
        store = SqlAlchemyLifecycleStore(session)
        intent = store.persist_intent(_plan(target=2))
        command_id = intent.commands[0].command.command_id
        token = _lease(session)
        claimed = store.claim_dispatch(
            intent.intent_id,
            token=token,
            claim_ttl=timedelta(seconds=15),
        )
        assert claimed is not None
        store.mark_dispatched(
            command_id,
            token=token,
            broker_order_ids=("basket-1",),
        )
        partial = LifecycleReceipt(
            receipt_id="receipt-partial-1",
            command_id=command_id,
            kind=ReceiptKind.PARTIAL_FILL,
            occurred_at=NOW + timedelta(seconds=1),
            broker_order_id="basket-1",
            cumulative_filled_quantity=1,
            fill_price=Decimal("20001.25"),
        )
        updated = store.record_receipt(partial)
        duplicate = store.record_receipt(partial)
        assert updated.status == LifecycleStatus.PARTIALLY_FILLED
        assert duplicate.revision == updated.revision
        with pytest.raises(DuplicateMasterEventConflict, match="different"):
            store.record_receipt(
                replace(partial, cumulative_filled_quantity=2)
            )

        with pytest.raises(ValueError, match="move backward"):
            store.record_receipt(
                LifecycleReceipt(
                    receipt_id="receipt-stale-fill",
                    command_id=command_id,
                    kind=ReceiptKind.ACKNOWLEDGED,
                    occurred_at=NOW + timedelta(seconds=2),
                    cumulative_filled_quantity=0,
                )
            )

        filled = store.record_receipt(
            LifecycleReceipt(
                receipt_id="receipt-fill-1",
                command_id=command_id,
                kind=ReceiptKind.FILL,
                occurred_at=NOW + timedelta(seconds=3),
                cumulative_filled_quantity=2,
                fill_price=Decimal("20002"),
            )
        )
        assert filled.status == LifecycleStatus.FILLED
        with pytest.raises(ValueError, match="terminal"):
            store.record_receipt(
                LifecycleReceipt(
                    receipt_id="receipt-late-ack",
                    command_id=command_id,
                    kind=ReceiptKind.ACKNOWLEDGED,
                    occurred_at=NOW + timedelta(seconds=4),
                    cumulative_filled_quantity=2,
                )
            )
    finally:
        session.close()


def test_reversal_open_is_not_dispatchable_until_close_fill() -> None:
    session = _session()
    try:
        store = SqlAlchemyLifecycleStore(session)
        intent = store.persist_intent(_plan(current=2, target=-1))
        close, opening = intent.commands
        assert intent.next_dispatchable == close
        token = _lease(session)
        claimed = store.claim_dispatch(
            intent.intent_id,
            token=token,
            claim_ttl=timedelta(seconds=15),
        )
        assert claimed is not None
        store.mark_dispatched(
            close.command.command_id,
            token=token,
            broker_order_ids=("exit-position",),
        )
        assert store.get_intent(intent.intent_id).next_dispatchable is None
        updated = store.record_receipt(
            LifecycleReceipt(
                receipt_id="close-fill",
                command_id=close.command.command_id,
                kind=ReceiptKind.FILL,
                occurred_at=NOW + timedelta(seconds=1),
                cumulative_filled_quantity=2,
            )
        )
        assert updated.next_dispatchable is not None
        assert updated.next_dispatchable.command.command_id == opening.command.command_id
    finally:
        session.close()


class _FakeAdapter:
    def __init__(self, connection_id: str, *, fail: bool = False) -> None:
        self.connection_id = connection_id
        self.fail = fail
        self.commands = []

    async def dispatch(self, command):
        self.commands.append(command)
        await asyncio.sleep(0)
        if self.fail:
            raise ConnectionError("isolated follower failure")
        return SimpleNamespace(
            transmitted=True,
            broker_order_ids=(f"broker-{command.command_id}",),
        )


def test_parallel_dispatch_isolates_one_follower_failure() -> None:
    session = _session()
    try:
        store = SqlAlchemyLifecycleStore(session)
        first = _FakeAdapter("follower-1")
        second = _FakeAdapter("follower-2", fail=True)
        copier = DurableLifecycleCopier(
            store,
            {"follower-1": first, "follower-2": second},
            lease_token=_lease(session),
        )
        event = _event("shared-master-event")
        outcomes = asyncio.run(
            copier.persist_and_dispatch(
                (
                    _plan("follower-1", event=event),
                    _plan("follower-2", event=event),
                )
            )
        )
        assert outcomes[0].succeeded and outcomes[0].transmitted
        assert not outcomes[1].succeeded and not outcomes[1].transmitted
        assert store.get_intent(outcomes[0].intent_id).status == (
            LifecycleStatus.DISPATCHED
        )
        failed = store.get_intent(outcomes[1].intent_id)
        assert failed.status == LifecycleStatus.DISPATCHING
        assert failed.commands[0].outbox_status == OutboxStatus.UNCERTAIN
        assert len(first.commands) == len(second.commands) == 1
    finally:
        session.close()


def test_dispatcher_rejects_adapter_account_slot_mismatch() -> None:
    session = _session()
    try:
        with pytest.raises(ValueError, match="does not match"):
            DurableLifecycleCopier(
                SqlAlchemyLifecycleStore(session),
                {"follower-1": _FakeAdapter("follower-2")},
            )
    finally:
        session.close()


def test_blocked_plan_is_durable_and_never_dispatchable() -> None:
    session = _session()
    try:
        store = SqlAlchemyLifecycleStore(session)
        blocked = store.persist_intent(
            _plan(allowed=False, reason="minimum_contract_exceeds_target")
        )
        assert blocked.status == LifecycleStatus.BLOCKED
        assert blocked.commands == ()
        assert blocked.next_dispatchable is None
    finally:
        session.close()


def test_copy_lifecycle_migration_is_single_head_and_event_history_is_append_only() -> None:
    scripts = ScriptDirectory.from_config(Config("alembic.ini"))
    assert scripts.get_current_head() == "0014_v2_execution_recovery"
    source = Path("migrations/versions/0013_v2_copy_lifecycle.py").read_text(
        encoding="utf-8"
    )
    assert 'down_revision = "0012_v2_copy_control"' in source
    assert "BEFORE UPDATE OR DELETE ON v2_copy_lifecycle_events" in source
    assert "v2_reject_copy_lifecycle_event_mutation" in source


def test_copy_lifecycle_migration_columns_match_orm(monkeypatch) -> None:
    path = Path("migrations/versions/0013_v2_copy_lifecycle.py")
    spec = importlib.util.spec_from_file_location("migration_0013", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    declared: dict[str, Table] = {}

    def capture(name, *items, **kwargs):
        declared[name] = Table(name, MetaData(), *items, **kwargs)

    monkeypatch.setattr(module, "_create", capture)
    monkeypatch.setattr(module, "_index", lambda name, table, columns: None)
    monkeypatch.setattr(module, "_install_append_only_trigger", lambda: None)
    module.upgrade()

    for table_name in (
        "v2_copy_lifecycle_intents",
        "v2_copy_lifecycle_commands",
        "v2_copy_lifecycle_events",
    ):
        migration_table = declared[table_name]
        orm_table = Base.metadata.tables[table_name]
        assert set(migration_table.c.keys()) == set(orm_table.c.keys())
        for column_name in orm_table.c.keys():
            migration_column = migration_table.c[column_name]
            orm_column = orm_table.c[column_name]
            assert type(migration_column.type) is type(orm_column.type)
            assert migration_column.nullable == orm_column.nullable
            assert migration_column.primary_key == orm_column.primary_key
