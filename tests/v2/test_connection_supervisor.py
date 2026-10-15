from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from app.v2.domain.models import NormalizedOrderEvent, OrderStatus, Side
from app.v2.execution.account_config import IndependentAccountTopology
from app.v2.execution.connection_supervisor import (
    ConnectionDataUnavailable,
    ConnectionPhase,
    ConnectionSignal,
    ConnectionSignalKind,
    IndependentAccountConnectionSupervisor,
    MasterEventEnvelope,
    NormalizedFollowerAccountState,
    SessionKey,
    SimulatedConnectionFixture,
    SimulatedConnectionOutcome,
    SimulatedDiscoveryOutcome,
    SupervisedPlant,
)
from app.v2.execution.config import ExecutionConfigSnapshot
from app.v2.execution.service import FuturesExecutionService


NOW = datetime(2026, 10, 14, 14, 0, tzinfo=timezone.utc)


def _supervisor() -> IndependentAccountConnectionSupervisor:
    return IndependentAccountConnectionSupervisor(
        IndependentAccountTopology.from_mapping({}),
        stale_after=timedelta(seconds=15),
    )


def _master_event() -> NormalizedOrderEvent:
    return NormalizedOrderEvent(
        event_id="event-1",
        broker="simulated",
        account_id="private-master-account",
        broker_order_id="order-1",
        contract_id=f"CME:NQ:{date(2027, 3, 19).isoformat()}",
        revision=1,
        status=OrderStatus.WORKING,
        side=Side.BUY,
        quantity=1,
        event_at=NOW,
        received_at=NOW,
    )


def test_supervisor_defines_ten_independent_stopped_sessions() -> None:
    supervisor = _supervisor()
    health = supervisor.health()

    assert supervisor.session_count == 10
    assert health.running is False
    assert health.global_new_copy_blocked is True
    assert health.global_blockers == (
        "connection_supervisor_not_running",
        "master_connection_not_ready",
    )
    assert len(health.accounts) == 5
    assert all(account.discovery_phase.value == "STOPPED" for account in health.accounts)
    assert all(len(account.sessions) == 2 for account in health.accounts)
    assert all(
        session.phase == ConnectionPhase.STOPPED
        for account in health.accounts
        for session in account.sessions
    )

    with pytest.raises(ValueError, match="unknown connection"):
        _supervisor().start_simulation(
            SimulatedConnectionFixture(
                {
                    SessionKey("unknown", SupervisedPlant.ORDER): (
                        SimulatedConnectionOutcome.READY
                    )
                }
            ),
            now=NOW,
        )


def test_all_ready_simulation_is_operational_and_fully_ready() -> None:
    supervisor = _supervisor()
    health = supervisor.start_simulation(now=NOW)

    assert health.running is True
    assert health.mode == "SIMULATED"
    assert health.master_ready is True
    assert health.global_new_copy_blocked is False
    assert health.operational is True
    assert health.fully_ready is True
    assert health.ready_follower_count == 4
    assert health.eligible_follower_connection_ids == (
        "follower-1",
        "follower-2",
        "follower-3",
        "follower-4",
    )
    assert all(account.ready for account in health.accounts)
    assert all(account.discovery_ready for account in health.accounts)
    assert all(
        account.discovery_generation_id == f"{account.connection_id}:discovery:1"
        for account in health.accounts
    )
    assert all(
        session.generation_id
        == f"{account.connection_id}:{session.plant.value.lower()}:1"
        for account in health.accounts
        for session in account.sessions
    )


def test_failed_follower_is_isolated_from_healthy_followers() -> None:
    failed_key = SessionKey("follower-2", SupervisedPlant.PNL)
    fixture = SimulatedConnectionFixture(
        {failed_key: SimulatedConnectionOutcome.AUTH_REJECTED}
    )
    supervisor = _supervisor()
    health = supervisor.start_simulation(fixture, now=NOW)

    assert health.master_ready is True
    assert health.global_new_copy_blocked is False
    assert health.operational is True
    assert health.fully_ready is False
    assert health.ready_follower_count == 3
    assert health.eligible_follower_connection_ids == (
        "follower-1",
        "follower-3",
        "follower-4",
    )
    failed = next(
        account for account in health.accounts if account.connection_id == "follower-2"
    )
    assert failed.ready is False
    assert failed.can_accept_new_copy is False
    assert "pnl_simulated_auth_rejected" in failed.blocker_codes


def test_failed_account_discovery_blocks_only_that_follower() -> None:
    fixture = SimulatedConnectionFixture(
        discovery_outcomes={
            "follower-2": SimulatedDiscoveryOutcome.FAILED,
        }
    )
    supervisor = _supervisor()
    health = supervisor.start_simulation(fixture, now=NOW)

    assert health.master_ready is True
    assert health.global_new_copy_blocked is False
    assert health.ready_follower_count == 3
    failed = next(
        account for account in health.accounts if account.connection_id == "follower-2"
    )
    assert failed.discovery_ready is False
    assert failed.discovery_phase.value == "FAILED"
    assert "discovery_simulated_discovery_failed" in failed.blocker_codes
    assert all(session.phase == ConnectionPhase.STOPPED for session in failed.sessions)


def test_stale_master_blocks_all_new_copying_without_staling_followers() -> None:
    supervisor = _supervisor()
    supervisor.start_simulation(now=NOW)
    follower_heartbeat_at = NOW + timedelta(seconds=10)
    for follower_number in range(1, 5):
        for plant in SupervisedPlant:
            key = SessionKey(f"follower-{follower_number}", plant)
            supervisor.apply(
                ConnectionSignal(
                    key,
                    f"{key.connection_id}:{plant.value.lower()}:1",
                    ConnectionSignalKind.HEARTBEAT,
                    follower_heartbeat_at,
                )
            )

    stale = supervisor.mark_stale(now=NOW + timedelta(seconds=16))
    health = supervisor.health()

    assert stale == (
        SessionKey("master", SupervisedPlant.ORDER),
        SessionKey("master", SupervisedPlant.PNL),
    )
    assert health.master_ready is False
    assert health.global_new_copy_blocked is True
    assert health.ready_follower_count == 0
    assert health.eligible_follower_connection_ids == ()
    assert all(
        account.ready
        for account in health.accounts
        if account.connection_id.startswith("follower-")
    )
    with pytest.raises(ConnectionDataUnavailable, match="required master state"):
        supervisor.accept_master_event(
            MasterEventEnvelope(
                "master",
                "master:order:1",
                NOW + timedelta(seconds=16),
                _master_event(),
            ),
            now=NOW + timedelta(seconds=16),
        )


def test_old_generation_signals_cannot_corrupt_reconnected_session() -> None:
    supervisor = _supervisor()
    supervisor.start_simulation(now=NOW)
    key = SessionKey("follower-3", SupervisedPlant.ORDER)
    old_generation = "follower-3:order:1"
    new_generation = supervisor.reconnect_simulated(
        key,
        now=NOW + timedelta(seconds=1),
    )

    applied = supervisor.apply(
        ConnectionSignal(
            key,
            old_generation,
            ConnectionSignalKind.DISCONNECTED,
            NOW + timedelta(seconds=2),
            "late_disconnect",
        )
    )
    account = next(
        item
        for item in supervisor.health().accounts
        if item.connection_id == "follower-3"
    )
    order = next(
        item for item in account.sessions if item.plant == SupervisedPlant.ORDER
    )

    assert new_generation == "follower-3:order:2"
    assert applied is False
    assert order.generation_id == new_generation
    assert order.ready is True
    assert order.reconnect_attempt == 1


def test_broker_neutral_master_and_follower_data_require_fresh_ready_generation() -> None:
    supervisor = _supervisor()
    supervisor.start_simulation(now=NOW)
    master_event = _master_event()
    envelope = MasterEventEnvelope(
        "master",
        "master:order:1",
        NOW + timedelta(seconds=1),
        master_event,
    )
    follower_state = NormalizedFollowerAccountState(
        connection_id="follower-1",
        generation_id="follower-1:pnl:1",
        observed_at=NOW + timedelta(seconds=1),
        equity=Decimal("500"),
        available_margin=Decimal("450"),
        current_open_risk=Decimal("10"),
        daily_realized_pnl=Decimal("5"),
        daily_total_pnl=Decimal("7"),
        account_kill_switch=False,
    )

    assert supervisor.accept_master_event(
        envelope,
        now=NOW + timedelta(seconds=1),
    ) is master_event
    assert supervisor.accept_follower_state(
        follower_state,
        now=NOW + timedelta(seconds=1),
    ) is follower_state

    with pytest.raises(ConnectionDataUnavailable, match="generation is stale"):
        supervisor.accept_follower_state(
            NormalizedFollowerAccountState(
                connection_id="follower-1",
                generation_id="follower-1:pnl:0",
                observed_at=NOW + timedelta(seconds=2),
            ),
            now=NOW + timedelta(seconds=2),
        )
    with pytest.raises(ConnectionDataUnavailable, match="stale connection data"):
        supervisor.accept_master_event(
            MasterEventEnvelope(
                "master",
                "master:order:1",
                NOW,
                master_event,
            ),
            now=NOW + timedelta(seconds=16),
        )


def test_supervisor_health_and_repr_never_contain_connection_secrets() -> None:
    values = {
        "V2_COPIER_CONNECTIVITY_ENABLED": "1",
        "V2_COPIER_ENVIRONMENT": "TEST",
    }
    for number, slot in enumerate(
        ("MASTER", "FOLLOWER_1", "FOLLOWER_2", "FOLLOWER_3", "FOLLOWER_4"),
        start=1,
    ):
        values[f"V2_RITHMIC_{slot}_ACCOUNT_ID"] = f"private-account-{number}"
        values[f"V2_RITHMIC_{slot}_USERNAME"] = f"private-user-{number}"
        values[f"V2_RITHMIC_{slot}_PASSWORD"] = f"private-password-{number}"
        values[f"V2_COPIER_{slot}_DISCOVERY_URI"] = (
            f"wss://private-{number}.example.invalid"
        )
        values[f"V2_COPIER_{slot}_SYSTEM_NAME"] = f"private-system-{number}"

    supervisor = IndependentAccountConnectionSupervisor(
        IndependentAccountTopology.from_mapping(values)
    )
    rendered = repr(supervisor) + repr(supervisor.start_simulation(now=NOW))

    for sensitive_prefix in (
        "private-account",
        "private-user",
        "private-password",
        "wss://private",
        "private-system",
    ):
        assert sensitive_prefix not in rendered


def test_execution_service_owns_simulated_supervisor_lifecycle() -> None:
    supervisor = _supervisor()
    fixture = SimulatedConnectionFixture(
        {
            SessionKey("follower-4", SupervisedPlant.ORDER): (
                SimulatedConnectionOutcome.DISCONNECTED
            )
        }
    )
    service = FuturesExecutionService(
        ExecutionConfigSnapshot("phase-2-simulation"),
        connection_supervisor=supervisor,
        simulation_fixture=fixture,
    )

    started = asyncio.run(service.start())
    assert started.connection_supervision_mode == "SIMULATED"
    assert started.supervised_account_count == 5
    assert started.ready_supervised_account_count == 4
    assert started.master_connection_ready is True
    assert started.ready_follower_connection_count == 3

    stopped = asyncio.run(service.stop())
    assert stopped.connection_supervision_mode == "OFF"
    assert stopped.ready_supervised_account_count == 0
