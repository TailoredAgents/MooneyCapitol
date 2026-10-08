from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import Base
from app.v2.capture.config import CaptureConfig
from app.v2.capture.contracts import (
    CaptureEvent,
    CapturePlant,
    CaptureSource,
    ReconciliationCheckpoint,
)
from app.v2.capture.persistence import SQLAlchemyCaptureJournal
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


@compiles(JSONB, "sqlite")
def _compile_jsonb_for_sqlite(_type, _compiler, **_kwargs):
    return "JSON"


TABLES = (
    V2BrokerConnection.__table__,
    V2BrokerAccount.__table__,
    V2RithmicConnectionGeneration.__table__,
    V2RithmicReplayBatch.__table__,
    V2RithmicBrokerEvent.__table__,
    V2RithmicAccountObservation.__table__,
    V2RithmicOrderObservation.__table__,
    V2RithmicExecutionObservation.__table__,
    V2RithmicBracketObservation.__table__,
    V2RithmicReferenceObservation.__table__,
    V2RithmicPnLObservation.__table__,
    V2RithmicRmsObservation.__table__,
    V2RithmicReconciliationCheckpoint.__table__,
)


def _journal(account_id: str | tuple[str, ...] = "allowed-account"):
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine, tables=list(TABLES))
    sessions = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    account_ids = (account_id,) if isinstance(account_id, str) else account_id
    config = CaptureConfig(
        connectivity_enabled=True,
        environment="TEST",
        account_allowlist=frozenset(account_ids),
    )
    journal = SQLAlchemyCaptureJournal(
        config,
        session_factory=sessions,
        system_name="RITHMIC-TEST",
    )
    asyncio.run(journal.initialize())
    return journal, sessions


def _event(
    event_id: str,
    observation_type: str,
    normalized: dict,
    *,
    plant=CapturePlant.ORDER,
    account_id: str = "allowed-account",
):
    return CaptureEvent(
        event_id=event_id,
        account_id=account_id,
        plant=plant,
        source=CaptureSource.LIVE,
        generation_id=f"{plant.value.lower()}-generation-1",
        received_at=datetime(2026, 10, 7, 14, 30, tzinfo=timezone.utc),
        payload={
            "schema_version": "rithmic-read-v1",
            "template_id": 352,
            "template_name": "ExchangeOrderNotification",
            "observation_type": observation_type,
            "broker_identity": {
                "fcm_id": "synthetic-fcm",
                "ib_id": "synthetic-ib",
                "account_id": account_id,
            },
            "native_identity": {
                "basket_id": "synthetic-basket",
                "fill_id": "synthetic-fill",
                "exchange_order_id": "synthetic-exchange-order",
                "sequence_number": "000000000000000123",
            },
            "dedupe_key": event_id,
            "normalized": normalized,
            "raw_frame_sha256": "a" * 64,
            "raw_frame_b64": "c3ludGhldGljLWZyYW1l",
        },
    )


def test_durable_journal_deduplicates_redacts_and_projects_order_execution():
    journal, sessions = _journal()
    event = _event(
        "event-order-fill-1",
        "ORDER_EXECUTION",
        {
            "normalized_state": "PARTIALLY_FILLED",
            "execution_effect": "APPLY_FILL",
            "quantity": 2**40,
            "fill_size": 2**35,
            "cumulative_fill_size": 2**35,
            "unfilled_size": (2**40) - (2**35),
            "fill_price": "20123.2500",
            "average_fill_price": "20123.2500",
            "status": "OPEN",
            "raw_notify_type": "FILL",
            "origin": {
                "user_id": "synthetic-user",
                "application": "R|Trader",
                "user_tag": "synthetic-tag",
            },
        },
    )

    assert asyncio.run(journal.append(event)) is True
    assert asyncio.run(journal.append(event)) is False
    bust = _event(
        "event-order-bust-1",
        "ORDER_EXECUTION",
        {
            "normalized_state": "UNKNOWN",
            "execution_effect": "BUST_FILL",
            "status": "COMPLETE",
            "report_type": "BUST",
            "sequence_number": "000000000000000124",
            "original_sequence_number": "000000000000000123",
        },
    )
    assert asyncio.run(journal.append(bust)) is True
    correction = _event(
        "event-order-correction-1",
        "ORDER_EXECUTION",
        {
            "normalized_state": "UNKNOWN",
            "execution_effect": "CORRECT_FILL",
            "status": "COMPLETE",
            "report_type": "TRADE CORRECT",
            "fill_size": (2**35) + 7,
            "fill_price": "20123.5000",
            "sequence_number": "000000000000000125",
            "original_sequence_number": "000000000000000123",
            "correlation_sequence_number": "000000000000000123",
        },
    )
    assert asyncio.run(journal.append(correction)) is True
    assert journal.depth == 3

    with sessions() as session:
        raw = session.get(V2RithmicBrokerEvent, "event-order-fill-1")
        assert raw is not None
        assert raw.payload_fingerprint == "a" * 64
        assert raw.event_facts["raw_frame_b64"] == "[REDACTED]"
        assert raw.event_facts["broker_identity"]["fcm_id"] == "[REDACTED]"
        assert raw.event_facts["raw_frame_sha256"] == "a" * 64

        order = session.scalar(select(V2RithmicOrderObservation))
        assert order is not None
        assert order.quantity == 2**40
        assert order.total_fill_size == 2**35
        assert order.normalized_state == "PARTIALLY_FILLED"
        assert order.mooney_owned is False
        assert order.user_tag == "synthetic-tag"

        execution = session.scalar(
            select(V2RithmicExecutionObservation).where(
                V2RithmicExecutionObservation.execution_kind == "APPLY_FILL"
            )
        )
        assert execution is not None
        assert execution.quantity == 2**35
        assert execution.effective_quantity_delta == 2**35
        assert execution.execution_kind == "APPLY_FILL"

        bust_execution = session.scalar(
            select(V2RithmicExecutionObservation).where(
                V2RithmicExecutionObservation.execution_kind == "BUST_FILL"
            )
        )
        assert bust_execution is not None
        assert bust_execution.quantity is None
        assert bust_execution.price is None
        assert bust_execution.corrects_execution_id == execution.execution_observation_id
        assert bust_execution.effective_quantity_delta == -(2**35)

        correction_execution = session.scalar(
            select(V2RithmicExecutionObservation).where(
                V2RithmicExecutionObservation.execution_kind == "CORRECT_FILL"
            )
        )
        assert correction_execution is not None
        assert (
            correction_execution.corrects_execution_id
            == execution.execution_observation_id
        )
        assert correction_execution.quantity == (2**35) + 7
        assert correction_execution.effective_quantity_delta == 7


def test_execution_correction_that_arrives_first_is_projected_after_original():
    journal, sessions = _journal()
    correction = _event(
        "event-order-correction-first",
        "ORDER_EXECUTION",
        {
            "normalized_state": "UNKNOWN",
            "execution_effect": "CORRECT_FILL",
            "report_type": "TRADE CORRECT",
            "fill_size": (2**35) + 7,
            "fill_price": "20123.5000",
            "sequence_number": "000000000000000125",
            "original_sequence_number": "000000000000000123",
            "correlation_sequence_number": "000000000000000123",
        },
    )
    original = _event(
        "event-order-fill-after-correction",
        "ORDER_EXECUTION",
        {
            "normalized_state": "PARTIALLY_FILLED",
            "execution_effect": "APPLY_FILL",
            "report_type": "FILL",
            "fill_size": 2**35,
            "fill_price": "20123.2500",
            "sequence_number": "000000000000000123",
        },
    )

    assert asyncio.run(journal.append(correction)) is True
    with sessions() as session:
        assert session.scalar(select(func.count(V2RithmicBrokerEvent.event_id))) == 1
        assert (
            session.scalar(
                select(func.count(V2RithmicExecutionObservation.execution_observation_id))
            )
            == 0
        )

    assert asyncio.run(journal.append(original)) is True
    with sessions() as session:
        executions = session.scalars(
            select(V2RithmicExecutionObservation).order_by(
                V2RithmicExecutionObservation.execution_kind
            )
        ).all()
        assert {item.execution_kind for item in executions} == {
            "APPLY_FILL",
            "CORRECT_FILL",
        }
        applied = next(item for item in executions if item.execution_kind == "APPLY_FILL")
        corrected = next(
            item for item in executions if item.execution_kind == "CORRECT_FILL"
        )
        assert corrected.corrects_execution_id == applied.execution_observation_id
        assert corrected.quantity == (2**35) + 7
        assert corrected.effective_quantity_delta == 7


def test_journal_persists_account_pnl_rms_and_bracket_projections():
    journal, sessions = _journal()
    events = (
        _event(
            "event-account-1",
            "ACCOUNT",
            {
                "account_name": "Synthetic Account",
                "currency": "USD",
                    "access_type": "READ_ONLY",
                    "status": "ACTIVE",
                    "user_id": "synthetic-user",
                    "user_type": "USER_TYPE_TRADER",
                    "user_status": "enabled",
                    "country_code": "US",
                    "state_code": "IL",
            },
        ),
        _event(
            "event-position-1",
            "POSITION",
            {
                "symbol": "MNQZ6",
                "exchange": "CME",
                "net_quantity": 2**35,
                "working_buy_quantity": 4,
                "working_sell_quantity": 5,
                "average_open_fill_price": "20100.25",
                "open_position_pnl": "25.50",
                "day_pnl": "30.25",
            },
            plant=CapturePlant.PNL,
        ),
        _event(
            "event-rms-1",
            "PRODUCT_RMS",
            {
                "product_code": "MNQ",
                "maximum_order_quantity": 2**40,
                "buy_limit": 2**39,
                "sell_limit": 2**39,
                "buy_margin_rate": "100.25",
                "sell_margin_rate": "100.25",
                "auto_liquidate": "ENABLED",
                "disable_on_auto_liquidate": "DISABLED",
                "peak_account_balance_ssboe": 1_791_382_800,
            },
        ),
        _event(
            "event-bracket-1",
            "BRACKET",
            {
                "parent_basket_id": "synthetic-parent",
                "linked_basket_ids": ["synthetic-stop", "synthetic-target"],
                "bracket_type": "TARGET_AND_STOP",
                "target_quantity": 2**34,
                "target_quantity_released": 2**33,
                "target_ticks": 40,
                "stop_quantity": 2**34,
                "stop_quantity_released": 2**33,
                "stop_ticks": 20,
                "trailing_stop_trigger_ticks": 8,
            },
        ),
    )

    for event in events:
        assert asyncio.run(journal.append(event)) is True

    with sessions() as session:
        account = session.scalar(select(V2RithmicAccountObservation))
        assert account is not None
        assert account.allowlisted is True
        assert account.access_type == "READ_ONLY"
        assert account.user_id == "synthetic-user"
        assert account.user_status == "enabled"
        assert account.country_code == "US"
        assert account.state_code == "IL"

        pnl = session.scalar(select(V2RithmicPnLObservation))
        assert pnl is not None
        assert pnl.scope == "INSTRUMENT"
        assert pnl.net_quantity == 2**35
        assert pnl.long_quantity is None
        assert pnl.short_quantity is None
        assert str(pnl.day_total_pnl) == "30.2500000000"

        rms = session.scalar(select(V2RithmicRmsObservation))
        assert rms is not None
        assert rms.scope == "PRODUCT"
        assert rms.max_order_quantity == 2**40
        assert rms.auto_liquidate is True
        assert rms.disable_on_auto_liquidate is False
        assert rms.peak_account_balance_at is not None

        bracket = session.scalar(select(V2RithmicBracketObservation))
        assert bracket is not None
        assert bracket.target_total_quantity == 2**34
        assert bracket.stop_released_quantity == 2**33
        assert bracket.linked_basket_ids == ["synthetic-stop", "synthetic-target"]
        assert bracket.target_tiers == [
            {
                "tier": 1,
                "ticks": 40,
                "total_quantity": 2**34,
                "released_quantity": 2**33,
            }
        ]
        assert bracket.stop_tiers == [
            {
                "tier": 1,
                "ticks": 20,
                "total_quantity": 2**34,
                "released_quantity": 2**33,
            }
        ]

        connection = session.get(V2BrokerConnection, "v2-rithmic-test")
        broker_account = session.scalar(select(V2BrokerAccount))
        assert connection is not None and connection.enabled is False
        assert broker_account is not None and broker_account.enabled is False
        assert broker_account.account_id != "allowed-account"
        assert broker_account.broker_account_ref == "allowed-account"


def test_reconciliation_checkpoints_gate_generation_ready_for_all_allowlisted_accounts():
    journal, sessions = _journal()
    generations = {
        CapturePlant.ORDER: "order-generation-1",
        CapturePlant.PNL: "pnl-generation-1",
    }
    asyncio.run(journal.begin_reconciliation("allowed-account", generations))
    asyncio.run(
        journal.save_checkpoint(
            ReconciliationCheckpoint(
                account_id="allowed-account",
                generations=generations,
                checkpoint="synthetic-clean-checkpoint",
                buffered_events_applied=3,
                completed_at=datetime(2026, 10, 7, 14, 35, tzinfo=timezone.utc),
            )
        )
    )

    with sessions() as session:
        generations_by_id = {
            item.generation_id: item
            for item in session.scalars(select(V2RithmicConnectionGeneration))
        }
        assert set(generations_by_id) == {"order-generation-1", "pnl-generation-1"}
        assert all(item.ready for item in generations_by_id.values())
        assert all(item.state == "READY" for item in generations_by_id.values())
        assert session.scalar(select(func.count()).select_from(V2RithmicReconciliationCheckpoint)) == 4
        completed = session.scalars(
            select(V2RithmicReconciliationCheckpoint).where(
                V2RithmicReconciliationCheckpoint.status == "COMPLETE"
            )
        ).all()
        assert len(completed) == 2
        assert all(item.live_subscription_active for item in completed)
        assert all(item.ready for item in completed)
        assert all(item.discrepancy_count == 0 for item in completed)
        assert all(item.checkpoint_facts["buffered_events_applied"] == 3 for item in completed)
        assert all("observer_checkpoint" not in item.checkpoint_facts for item in completed)
        assert all(len(item.checkpoint_facts["observer_checkpoint_sha256"]) == 64 for item in completed)


def test_nonzero_reconciliation_discrepancy_cannot_be_checkpointed_ready():
    journal, _sessions = _journal()
    generations = {CapturePlant.ORDER: "order-generation-discrepant"}
    asyncio.run(journal.begin_reconciliation("allowed-account", generations))

    with pytest.raises(ValueError, match="cannot be READY"):
        asyncio.run(
            journal.save_checkpoint(
                ReconciliationCheckpoint(
                    account_id="allowed-account",
                    generations=generations,
                    checkpoint="must-not-be-ready",
                    buffered_events_applied=0,
                    discrepancy_count=1,
                )
            )
        )


def test_new_generation_supersedes_prior_ready_generation_in_same_plant():
    journal, sessions = _journal()
    first = {CapturePlant.ORDER: "order-generation-first"}
    asyncio.run(journal.begin_reconciliation("allowed-account", first))
    asyncio.run(
        journal.save_checkpoint(
            ReconciliationCheckpoint(
                account_id="allowed-account",
                generations=first,
                checkpoint="first-clean",
                buffered_events_applied=0,
            )
        )
    )
    successor = _event(
        "event-new-order-generation",
        "ORDER",
        {"basket_id": "basket-new", "normalized_state": "WORKING"},
    )
    successor = CaptureEvent(
        event_id=successor.event_id,
        account_id=successor.account_id,
        plant=successor.plant,
        source=successor.source,
        generation_id="order-generation-second",
        payload=successor.payload,
        received_at=successor.received_at,
    )
    assert asyncio.run(journal.append(successor)) is True

    with sessions() as session:
        prior = session.get(
            V2RithmicConnectionGeneration, "order-generation-first"
        )
        current = session.get(
            V2RithmicConnectionGeneration, "order-generation-second"
        )
        assert prior is not None and current is not None
        assert prior.state == "DISCONNECTED"
        assert prior.ready is False
        assert prior.disconnected_at == successor.received_at.replace(tzinfo=None)
        assert prior.disconnect_reason == "superseded_by_new_generation"
        assert current.disconnected_at is None


def test_local_disconnect_boundary_closes_ready_generation_durably():
    journal, sessions = _journal()
    generations = {CapturePlant.ORDER: "order-generation-shutdown"}
    asyncio.run(journal.begin_reconciliation("allowed-account", generations))
    asyncio.run(
        journal.save_checkpoint(
            ReconciliationCheckpoint(
                account_id="allowed-account",
                generations=generations,
                checkpoint="shutdown-clean",
                buffered_events_applied=0,
            )
        )
    )
    disconnected_at = datetime(2026, 10, 7, 15, 0, tzinfo=timezone.utc)
    event = CaptureEvent(
        event_id="event-local-disconnect",
        account_id="allowed-account",
        plant=CapturePlant.ORDER,
        source=CaptureSource.SYSTEM,
        generation_id="order-generation-shutdown",
        received_at=disconnected_at,
        payload={
            "template_id": 0,
            "template_name": "LOCAL_DISCONNECTED",
            "observation_type": "CONTROL",
            "control_kind": "DISCONNECTED",
            "normalized": {
                "control_kind": "DISCONNECTED",
                "disconnect_reason": "service_shutdown",
            },
            "dedupe_key": "local-disconnect",
            "raw_frame_sha256": "b" * 64,
        },
    )
    assert asyncio.run(journal.append(event)) is True

    with sessions() as session:
        generation = session.get(
            V2RithmicConnectionGeneration, "order-generation-shutdown"
        )
        assert generation is not None
        assert generation.state == "DISCONNECTED"
        assert generation.ready is False
        assert generation.disconnected_at == disconnected_at.replace(tzinfo=None)
        assert generation.disconnect_reason == "service_shutdown"


def test_failed_reconciliation_is_append_only_and_generation_stays_unready():
    journal, sessions = _journal()
    generations = {
        CapturePlant.ORDER: "order-generation-failed",
        CapturePlant.PNL: "pnl-generation-failed",
    }
    asyncio.run(journal.begin_reconciliation("allowed-account", generations))
    asyncio.run(
        journal.fail_reconciliation(
            "allowed-account",
            generations,
            "recovery_phase_timeout",
        )
    )

    with sessions() as session:
        checkpoints = session.scalars(
            select(V2RithmicReconciliationCheckpoint).order_by(
                V2RithmicReconciliationCheckpoint.recorded_at,
                V2RithmicReconciliationCheckpoint.status,
            )
        ).all()
        assert len(checkpoints) == 4
        assert {row.status for row in checkpoints} == {"STARTED", "FAILED"}
        failed = [row for row in checkpoints if row.status == "FAILED"]
        assert len(failed) == 2
        assert all(row.failure_reason == "recovery_phase_timeout" for row in failed)
        assert all(not row.ready for row in failed)
        generations_rows = session.scalars(
            select(V2RithmicConnectionGeneration)
        ).all()
        assert all(row.state == "FAILED" for row in generations_rows)
        assert all(not row.ready for row in generations_rows)


def test_checkpoint_high_water_mark_is_scoped_to_its_account():
    journal, sessions = _journal(("account-a", "account-b"))
    generations = {CapturePlant.ORDER: "order-generation-1"}
    asyncio.run(journal.begin_reconciliation("account-a", generations))
    assert asyncio.run(
        journal.append(
            _event(
                "event-account-a",
                "ORDER",
                {"basket_id": "basket-a", "normalized_state": "WORKING"},
                account_id="account-a",
            )
        )
    )
    assert asyncio.run(
        journal.append(
            _event(
                "event-account-b",
                "ORDER",
                {"basket_id": "basket-b", "normalized_state": "WORKING"},
                account_id="account-b",
            )
        )
    )
    asyncio.run(
        journal.save_checkpoint(
            ReconciliationCheckpoint(
                account_id="account-a",
                generations=generations,
                checkpoint="account-a-clean",
                buffered_events_applied=0,
            )
        )
    )

    with sessions() as session:
        checkpoint = session.scalar(
            select(V2RithmicReconciliationCheckpoint).where(
                V2RithmicReconciliationCheckpoint.status == "COMPLETE"
            )
        )
        event_a = session.get(V2RithmicBrokerEvent, "event-account-a")
        event_b = session.get(V2RithmicBrokerEvent, "event-account-b")
        assert checkpoint is not None and event_a is not None and event_b is not None
        assert event_b.local_ingest_sequence > event_a.local_ingest_sequence
        assert checkpoint.highest_ingest_sequence == event_a.local_ingest_sequence


def test_login_heartbeat_and_forced_logout_update_only_generation_health_facts():
    journal, sessions = _journal()
    login = CaptureEvent(
        event_id="event-login-1",
        account_id="allowed-account",
        plant=CapturePlant.ORDER,
        source=CaptureSource.SYSTEM,
        generation_id="order-generation-health",
        received_at=datetime(2026, 10, 7, 14, 20, tzinfo=timezone.utc),
        payload={
            "template_id": 11,
            "template_name": "ResponseLogin",
            "observation_type": "UNKNOWN",
            "dedupe_key": "event-login-1",
            "normalized": {"success": True, "heartbeat_interval_seconds": "30"},
        },
    )
    forced_logout = CaptureEvent(
        event_id="event-forced-logout-1",
        account_id="allowed-account",
        plant=CapturePlant.ORDER,
        source=CaptureSource.SYSTEM,
        generation_id="order-generation-health",
        received_at=datetime(2026, 10, 7, 14, 21, tzinfo=timezone.utc),
        payload={
            "template_id": 77,
            "template_name": "ForcedLogout",
            "observation_type": "UNKNOWN",
            "dedupe_key": "event-forced-logout-1",
        },
    )

    assert asyncio.run(journal.append(login)) is True
    assert asyncio.run(journal.append(forced_logout)) is True

    with sessions() as session:
        generation = session.get(V2RithmicConnectionGeneration, "order-generation-health")
        assert generation is not None
        assert generation.authenticated_at.replace(tzinfo=timezone.utc) == login.received_at
        assert generation.heartbeat_interval_ms == 30_000
        assert generation.last_message_at.replace(tzinfo=timezone.utc) == forced_logout.received_at
        assert generation.disconnected_at.replace(tzinfo=timezone.utc) == forced_logout.received_at
        assert generation.forced_logout is True
        assert generation.ready is False
        assert generation.state == "FORCED_LOGOUT"


def test_opaque_broker_account_id_is_not_used_as_internal_pk_or_truncated():
    broker_account_id = "opaque-account-" + ("x" * 170)
    journal, sessions = _journal(broker_account_id)
    event = _event(
        "event-long-account-1",
        "ORDER",
        {
            "normalized_state": "WORKING",
            "quantity": 1,
            "price": "20100.00",
        },
        account_id=broker_account_id,
    )

    assert asyncio.run(journal.append(event)) is True

    with sessions() as session:
        account = session.scalar(select(V2BrokerAccount))
        raw = session.get(V2RithmicBrokerEvent, event.event_id)
        order = session.scalar(select(V2RithmicOrderObservation))
        assert account is not None and raw is not None and order is not None
        assert account.broker_account_ref == broker_account_id
        assert account.account_id != broker_account_id
        assert len(account.account_id) <= 96
        assert raw.account_id == account.account_id
        assert raw.broker_account_id == broker_account_id
        assert order.account_id == account.account_id
        assert order.broker_account_id == broker_account_id


def test_clean_reconciliation_closes_unfinished_replay_batch_without_inventing_terminal_reply():
    journal, sessions = _journal()
    generations = {CapturePlant.ORDER: "order-generation-replay"}
    asyncio.run(journal.begin_reconciliation("allowed-account", generations))
    replay_event = CaptureEvent(
        event_id="event-replay-1",
        account_id="allowed-account",
        plant=CapturePlant.ORDER,
        source=CaptureSource.REPLAY,
        generation_id="order-generation-replay",
        received_at=datetime(2026, 10, 7, 14, 33, tzinfo=timezone.utc),
        payload={
            "template_id": 352,
            "template_name": "ExchangeOrderNotification",
            "observation_type": "ORDER",
            "broker_identity": {
                "fcm_id": "synthetic-fcm",
                "ib_id": "synthetic-ib",
                "account_id": "allowed-account",
            },
            "native_identity": {"basket_id": "synthetic-replay-basket"},
            "replay_batch_id": "replay-batch-1",
            "batch_kind": "ACCOUNT_RECOVERY",
            "dedupe_key": "event-replay-1",
            "normalized": {"normalized_state": "WORKING", "quantity": 1},
        },
    )
    assert asyncio.run(journal.append(replay_event)) is True
    asyncio.run(
        journal.save_checkpoint(
            ReconciliationCheckpoint(
                account_id="allowed-account",
                generations=generations,
                checkpoint="clean-recovery",
                buffered_events_applied=0,
                completed_at=datetime(2026, 10, 7, 14, 34, tzinfo=timezone.utc),
            )
        )
    )

    with sessions() as session:
        replay = session.get(V2RithmicReplayBatch, "replay-batch-1")
        assert replay is not None
        assert replay.status == "COMPLETE"
        assert replay.completed_at is not None
        assert replay.terminal_response_received is False
        assert replay.boundary_facts["closed_by_reconciliation_checkpoint"] is True


def test_replay_batch_uses_deterministic_order_anchor_and_full_generation_map():
    journal, sessions = _journal()
    generations = {
        CapturePlant.ORDER: "order-generation-batch",
        CapturePlant.PNL: "pnl-generation-batch",
    }
    asyncio.run(journal.begin_reconciliation("allowed-account", generations))
    generation_facts = {plant.value: value for plant, value in generations.items()}
    pnl_first = CaptureEvent(
        event_id="event-pnl-first",
        account_id="allowed-account",
        plant=CapturePlant.PNL,
        source=CaptureSource.SNAPSHOT,
        generation_id=generations[CapturePlant.PNL],
        payload={
            "template_id": 403,
            "template_name": "ResponsePnLPositionSnapshot",
            "observation_type": "ACCOUNT_PNL",
            "broker_identity": {
                "fcm_id": "synthetic-fcm",
                "ib_id": "synthetic-ib",
                "account_id": "allowed-account",
            },
            "replay_batch_id": "deterministic-batch",
            "generations": generation_facts,
            "dedupe_key": "event-pnl-first",
            "normalized": {"cash_on_hand": "1000.00"},
        },
    )
    terminal = CaptureEvent(
        event_id="event-batch-terminal",
        account_id="allowed-account",
        plant=CapturePlant.ORDER,
        source=CaptureSource.REPLAY,
        generation_id=generations[CapturePlant.ORDER],
        payload={
            "template_id": 0,
            "template_name": "LOCAL_RECOVERY_CHECKPOINT",
            "observation_type": "RECOVERY_CHECKPOINT",
            "broker_identity": {
                "fcm_id": "synthetic-fcm",
                "ib_id": "synthetic-ib",
                "account_id": "allowed-account",
            },
            "replay_batch_id": "deterministic-batch",
            "generations": generation_facts,
            "batch_kind": "ACCOUNT_RECOVERY",
            "batch_status": "COMPLETE",
            "terminal_response_received": True,
            "clean": True,
            "dedupe_key": "event-batch-terminal",
            "normalized": {},
        },
    )

    assert asyncio.run(journal.append(pnl_first)) is True
    assert asyncio.run(journal.append(terminal)) is True

    with sessions() as session:
        replay = session.get(V2RithmicReplayBatch, "deterministic-batch")
        assert replay is not None
        assert replay.generation_id == generations[CapturePlant.ORDER]
        assert replay.generation_map == generation_facts
        assert replay.batch_kind == "ACCOUNT_RECOVERY"
        assert replay.source_kind == "REPLAY"
        assert replay.status == "COMPLETE"
        assert replay.boundary_facts["generations"] == generation_facts


def test_unknown_pnl_source_does_not_claim_current_freshness():
    journal, sessions = _journal()
    base = _event(
        "event-pnl-unknown-source",
        "POSITION",
        {"symbol": "MNQZ6", "exchange": "CME", "net_quantity": 1},
        plant=CapturePlant.PNL,
    )
    event = CaptureEvent(
        event_id=base.event_id,
        account_id=base.account_id,
        plant=base.plant,
        source=CaptureSource.UNKNOWN,
        generation_id=base.generation_id,
        received_at=base.received_at,
        payload=base.payload,
    )

    assert asyncio.run(journal.append(event)) is True
    with sessions() as session:
        pnl = session.scalar(
            select(V2RithmicPnLObservation).where(
                V2RithmicPnLObservation.event_id == event.event_id
            )
        )
        assert pnl is not None
        assert pnl.freshness == "UNKNOWN"


def test_reference_and_tick_table_observations_are_append_only_projections():
    journal, sessions = _journal()
    common = {
        "schema_version": "rithmic-read-v1",
        "broker_identity": {"fcm_id": None, "ib_id": None, "account_id": None},
        "native_identity": {},
        "raw_frame_b64": "",
    }
    reference = CaptureEvent(
        event_id="event-reference-1",
        account_id=None,
        plant=CapturePlant.TICKER,
        source=CaptureSource.SNAPSHOT,
        generation_id="ticker-generation-1",
        received_at=datetime(2026, 10, 7, 14, 25, tzinfo=timezone.utc),
        payload={
            **common,
            "template_id": 15,
            "template_name": "ResponseReferenceData",
            "observation_type": "REFERENCE",
            "dedupe_key": "event-reference-1",
            "raw_frame_sha256": "b" * 64,
            "normalized": {
                "symbol": "MNQZ6",
                "exchange": "CME",
                "exchange_symbol": "MNQZ6",
                "trading_symbol": "MNQZ6",
                "trading_exchange": "CME",
                "product_code": "MNQ",
                "instrument_type": "FUTURE",
                "underlying_symbol": "MNQ",
                "expiration_date": "20261218",
                "currency": "USD",
                "tick_size_type": "SIMPLE",
                "price_display_format": "DECIMAL",
                "is_tradable": "1",
                "minimum_quoted_price_change": "0.25",
                "minimum_feed_price_change": "0.25",
                "single_point_value": "2",
                "quote_to_feed_price_factor": "1",
                "feed_to_quote_price_factor": "1",
                "presence_bits": 2**35,
                "unknown_extension": "retained-in-facts",
            },
        },
    )
    tick_table = CaptureEvent(
        event_id="event-reference-tick-1",
        account_id=None,
        plant=CapturePlant.TICKER,
        source=CaptureSource.SNAPSHOT,
        generation_id="ticker-generation-1",
        received_at=datetime(2026, 10, 7, 14, 26, tzinfo=timezone.utc),
        payload={
            **common,
            "template_id": 108,
            "template_name": "ResponseTickSizeTable",
            "observation_type": "REFERENCE_TICK_SIZE",
            "dedupe_key": "event-reference-tick-1",
            "raw_frame_sha256": "c" * 64,
            "normalized": {
                "tick_size_type": "VARIABLE",
                "minimum_feed_price_change": "0.25",
                "first_price": "0",
                "last_price": "1000000",
                "first_price_operator": "GREATER_THAN_OR_EQUAL",
                "last_price_operator": "LESS_THAN",
                "presence_bits": 2**36,
            },
        },
    )

    assert asyncio.run(journal.append(reference)) is True
    assert asyncio.run(journal.append(tick_table)) is True

    with sessions() as session:
        rows = session.scalars(
            select(V2RithmicReferenceObservation).order_by(
                V2RithmicReferenceObservation.received_at
            )
        ).all()
        assert len(rows) == 2
        instrument, tick = rows
        assert instrument.account_id is None
        assert instrument.broker_account_id is None
        assert instrument.observation_kind == "REFERENCE"
        assert instrument.symbol == "MNQZ6"
        assert instrument.expiration_date.isoformat() == "2026-12-18"
        assert str(instrument.minimum_feed_price_change) == "0.2500000000"
        assert str(instrument.single_point_value) == "2.0000000000"
        assert instrument.is_tradable is True
        assert instrument.presence_bits == 2**35
        assert instrument.reference_facts["normalized"]["unknown_extension"] == (
            "retained-in-facts"
        )
        assert tick.observation_kind == "REFERENCE_TICK_SIZE"
        assert str(tick.tick_table_first_price) == "0E-10"
        assert str(tick.tick_table_last_price) == "1000000.0000000000"
        assert tick.tick_table_last_price_operator == "LESS_THAN"
        assert tick.presence_bits == 2**36


def test_oversize_native_identifier_is_rejected_never_silently_truncated():
    journal, sessions = _journal()
    event = _event(
        "event-oversize-native-id",
        "ORDER",
        {"basket_id": "x" * 257, "normalized_state": "WORKING"},
    )
    payload = dict(event.payload)
    payload["native_identity"] = {"basket_id": "x" * 257}
    event = CaptureEvent(
        event_id=event.event_id,
        account_id=event.account_id,
        plant=event.plant,
        source=event.source,
        generation_id=event.generation_id,
        payload=payload,
        received_at=event.received_at,
    )

    with pytest.raises(ValueError, match="basket ID exceeds"):
        asyncio.run(journal.append(event))

    assert journal.depth == 0
    with sessions() as session:
        assert session.scalar(select(func.count()).select_from(V2RithmicBrokerEvent)) == 0
