from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone

from app.v2.capture.config import CaptureConfig
from app.v2.capture.contracts import (
    CaptureEvent,
    CapturePlant,
    CaptureSource,
    ObservedAccount,
    PlantHealth,
    RecoveryResult,
)
from app.v2.capture.journal import InMemoryCaptureJournal
from app.v2.capture.service import RithmicCaptureService


class DurableJournal(InMemoryCaptureJournal):
    durable = True


class FakeObserver:
    def __init__(self, account_ids=("allowed-a", "allowed-b", "not-allowed")) -> None:
        self.account_ids = account_ids
        self.calls: list[tuple[str, str, str | None]] = []
        self.sink = None
        self.started = False
        self.abort_calls = 0
        self.health_by_plant = {
            CapturePlant.ORDER: PlantHealth(
                plant=CapturePlant.ORDER,
                connected=True,
                authenticated=True,
                generation_id="order-generation-1",
            ),
            CapturePlant.PNL: PlantHealth(
                plant=CapturePlant.PNL,
                connected=True,
                authenticated=True,
                generation_id="pnl-generation-1",
            ),
        }

    async def start(self, event_sink) -> None:
        self.sink = event_sink
        self.started = True
        self.calls.append(("start", "", None))

    async def stop(self) -> None:
        self.started = False
        self.calls.append(("stop", "", None))

    async def discover_accounts(self):
        self.calls.append(("discover", "", None))
        return tuple(ObservedAccount(account_id=value) for value in self.account_ids)

    async def prepare_reconciliation(self, account_id, generations) -> None:
        self.calls.append(("prepare", account_id, None))

    async def subscribe_account(self, account_id, plant) -> None:
        self.calls.append(("subscribe", account_id, plant.value))

    async def reconcile_account(self, account_id, generations):
        self.calls.append(("reconcile", account_id, None))
        assert self.sink is not None
        await self.sink(
            CaptureEvent(
                event_id=f"live-{account_id}",
                account_id=account_id,
                plant=CapturePlant.ORDER,
                source=CaptureSource.LIVE,
                generation_id=generations[CapturePlant.ORDER],
                payload={"sensitive": "not-for-health"},
            )
        )
        return RecoveryResult(clean=True, checkpoint=f"checkpoint-{account_id}")

    async def apply_buffered(self, account_id, events) -> None:
        self.calls.append(("apply", account_id, str(len(events))))

    async def finalize_reconciliation(self, account_id, generations):
        self.calls.append(("finalize", account_id, None))
        return RecoveryResult(clean=True, checkpoint=f"checkpoint-{account_id}")

    async def abort_recovery(self, generations) -> None:
        self.abort_calls += 1
        self.calls.append(("abort", "", None))

    def plant_health(self):
        return dict(self.health_by_plant)


def enabled_config(
    *,
    allowlist=frozenset({"allowed-a", "allowed-b"}),
    poll=0.01,
    plants=frozenset({CapturePlant.ORDER, CapturePlant.PNL}),
    reconcile_timeout=1,
    max_buffered_events=50_000,
):
    return CaptureConfig(
        connectivity_enabled=True,
        environment="TEST",
        account_allowlist=allowlist,
        enabled_plants=plants,
        observer_factory="app.v2.rithmic_protocol.factory:create_observer",
        journal_factory="app.v2.capture.persistence:create_journal",
        bindings_configured=True,
        poll_seconds=poll,
        reconcile_timeout_seconds=reconcile_timeout,
        max_buffered_events=max_buffered_events,
    )


def test_default_configuration_is_live_fail_closed_and_never_starts_observer():
    async def scenario():
        observer = FakeObserver()
        service = RithmicCaptureService(
            CaptureConfig.from_mapping({}), observer=observer, journal=DurableJournal()
        )
        health = await service.start()
        assert health.live
        assert not health.ready
        assert not health.connectivity_enabled
        assert health.submission_enabled is False
        assert "external_connectivity_disabled" in health.blockers
        assert not observer.started
        await service.stop()

    asyncio.run(scenario())


def test_connectivity_requires_test_allowlist_both_plants_secrets_bindings_and_factories():
    values = {
        "RITHMIC_CAPTURE_CONNECTIVITY_ENABLED": "1",
        "RITHMIC_ENVIRONMENT": "LIVE",
        "RITHMIC_ENABLED_PLANTS": "ORDER",
    }
    config = CaptureConfig.from_mapping(values)

    assert not config.connectivity_permitted
    assert "environment_must_be_test" in config.preflight_blockers
    assert "account_allowlist_required" in config.preflight_blockers
    assert "order_and_pnl_plants_required" in config.preflight_blockers
    assert "required_connection_settings_missing" in config.preflight_blockers
    assert "external_protocol_bindings_required" in config.preflight_blockers
    assert "observer_factory_required" in config.preflight_blockers
    assert "durable_journal_factory_required" in config.preflight_blockers


def test_archive_bindings_require_checksum_and_invalid_limits_fail_closed():
    values = {
        "RITHMIC_CAPTURE_CONNECTIVITY_ENABLED": "1",
        "RITHMIC_ENVIRONMENT": "TEST",
        "RITHMIC_ENABLED_PLANTS": "ORDER,PNL",
        "RITHMIC_ACCOUNT_ALLOWLIST": "account-a",
        "RITHMIC_DISCOVERY_URI": "wss://example.invalid",
        "RITHMIC_SYSTEM_NAME": "test-system",
        "RITHMIC_USERNAME": "not-a-real-user",
        "RITHMIC_PASSWORD": "not-a-real-password",
        "RITHMIC_GENERATED_BINDINGS_ARCHIVE_B64_FILE": "C:/external/bindings.b64",
        "RITHMIC_CAPTURE_OBSERVER_FACTORY": "app.v2.brokers.rithmic_protocol.adapter:create_observer",
        "RITHMIC_CAPTURE_JOURNAL_FACTORY": "app.v2.capture.persistence:create_journal",
        "RITHMIC_CAPTURE_POLL_SECONDS": "not-a-number",
    }
    config = CaptureConfig.from_mapping(values)

    assert "bindings_archive_checksum_required" in config.preflight_blockers
    assert "invalid_capture_runtime_setting" in config.preflight_blockers


def test_all_subscriptions_precede_snapshot_replay_and_buffer_is_applied_before_ready():
    async def scenario():
        observer = FakeObserver()
        journal = DurableJournal()
        service = RithmicCaptureService(enabled_config(), observer=observer, journal=journal)
        await service.start()
        assert await service.wait_until_ready(timeout=1)

        ordered = [call[0] for call in observer.calls]
        last_subscribe = max(index for index, value in enumerate(ordered) if value == "subscribe")
        first_reconcile = min(index for index, value in enumerate(ordered) if value == "reconcile")
        assert last_subscribe < first_reconcile
        assert ordered.count("subscribe") == 4
        assert ordered.count("reconcile") == 2
        assert ordered.count("apply") == 2
        assert journal.depth == 2
        assert len(journal.checkpoints) == 2

        health = service.health()
        serialized = str(health.as_dict())
        assert health.ready
        assert health.reconciled_account_count == 2
        assert "allowed-a" not in serialized
        assert "allowed-b" not in serialized
        assert "not-allowed" not in serialized
        assert "not-for-health" not in serialized
        await service.stop()

    asyncio.run(scenario())


def test_non_allowlisted_events_are_rejected_before_journaling():
    async def scenario():
        observer = FakeObserver(account_ids=("allowed-a",))
        journal = DurableJournal()
        service = RithmicCaptureService(
            enabled_config(allowlist=frozenset({"allowed-a"})),
            observer=observer,
            journal=journal,
        )
        accepted = await service.ingest(
            CaptureEvent(
                event_id="unauthorized",
                account_id="not-allowed",
                plant=CapturePlant.ORDER,
                source=CaptureSource.LIVE,
                generation_id="generation",
                payload={},
            )
        )
        assert accepted is False
        assert journal.depth == 0
        assert "event_for_non_allowlisted_account_rejected" in service.health().blockers

    asyncio.run(scenario())


def test_only_ticker_reference_events_may_be_accountless():
    async def scenario():
        journal = DurableJournal()
        service = RithmicCaptureService(
            enabled_config(
                allowlist=frozenset({"allowed-a"}),
                plants=frozenset(
                    {CapturePlant.ORDER, CapturePlant.PNL, CapturePlant.TICKER}
                ),
            ),
            observer=FakeObserver(account_ids=("allowed-a",)),
            journal=journal,
        )
        reference = CaptureEvent(
            event_id="reference-system-event",
            account_id=None,
            plant=CapturePlant.TICKER,
            source=CaptureSource.SNAPSHOT,
            generation_id="ticker-generation",
            payload={"observation_type": "REFERENCE", "symbol": "MNQZ6"},
        )
        assert await service.ingest(reference) is True

        unscoped_order = CaptureEvent(
            event_id="unscoped-order",
            account_id=None,
            plant=CapturePlant.ORDER,
            source=CaptureSource.LIVE,
            generation_id="order-generation",
            payload={"observation_type": "ORDER"},
        )
        assert await service.ingest(unscoped_order) is False
        assert journal.depth == 1
        assert "unscoped_non_reference_event_rejected" in service.health().blockers

    asyncio.run(scenario())


def test_order_and_pnl_health_are_independent_and_both_gate_readiness():
    async def scenario():
        observer = FakeObserver(account_ids=("allowed-a",))
        observer.health_by_plant[CapturePlant.PNL] = PlantHealth(
            plant=CapturePlant.PNL,
            connected=False,
            authenticated=False,
            reconnecting=True,
        )
        service = RithmicCaptureService(
            enabled_config(allowlist=frozenset({"allowed-a"})),
            observer=observer,
            journal=DurableJournal(),
        )
        await service.start()
        await asyncio.sleep(0.05)
        health = service.health()
        assert not health.ready
        assert not any(call[0] == "discover" for call in observer.calls)
        by_plant = {plant.plant: plant for plant in health.plants}
        assert by_plant["ORDER"].connected
        assert not by_plant["PNL"].connected
        await service.stop()

    asyncio.run(scenario())


def test_new_plant_generation_forces_fresh_subscribe_then_reconciliation():
    async def scenario():
        observer = FakeObserver(account_ids=("allowed-a",))
        service = RithmicCaptureService(
            enabled_config(allowlist=frozenset({"allowed-a"})),
            observer=observer,
            journal=DurableJournal(),
        )
        await service.start()
        assert await service.wait_until_ready(timeout=1)
        observer.health_by_plant[CapturePlant.ORDER] = PlantHealth(
            plant=CapturePlant.ORDER,
            connected=True,
            authenticated=True,
            generation_id="order-generation-2",
            last_message_at=datetime.now(timezone.utc),
        )
        for _ in range(50):
            if sum(call[0] == "reconcile" for call in observer.calls) == 2:
                break
            await asyncio.sleep(0.01)
        assert sum(call[0] == "reconcile" for call in observer.calls) == 2
        assert service.health().ready
        await service.stop()

    asyncio.run(scenario())


def test_in_memory_journal_cannot_make_connected_service_ready():
    async def scenario():
        service = RithmicCaptureService(
            enabled_config(allowlist=frozenset({"allowed-a"})),
            observer=FakeObserver(account_ids=("allowed-a",)),
            journal=InMemoryCaptureJournal(),
        )
        health = await service.start()
        assert health.lifecycle == "blocked"
        assert "durable_journal_required" in health.blockers

    asyncio.run(scenario())


def test_recovery_timeout_aborts_generation_and_never_checkpoints():
    class HangingObserver(FakeObserver):
        async def reconcile_account(self, account_id, generations):
            self.calls.append(("reconcile", account_id, None))
            await asyncio.Event().wait()

        async def abort_recovery(self, generations) -> None:
            await super().abort_recovery(generations)
            for plant in tuple(self.health_by_plant):
                self.health_by_plant[plant] = PlantHealth(
                    plant=plant,
                    connected=False,
                    authenticated=False,
                    reconnecting=True,
                )

    async def scenario():
        observer = HangingObserver(account_ids=("allowed-a",))
        journal = DurableJournal()
        service = RithmicCaptureService(
            enabled_config(
                allowlist=frozenset({"allowed-a"}),
                reconcile_timeout=0.03,
            ),
            observer=observer,
            journal=journal,
        )
        await service.start()
        for _ in range(100):
            if observer.abort_calls:
                break
            await asyncio.sleep(0.01)
        assert observer.abort_calls == 1
        assert not service.health().ready
        assert "reconciliation_timed_out" in service.health().blockers
        assert journal.checkpoints == ()
        assert len(journal.failed_reconciliations) == 1
        failed_account, failed_generations, failure_reason = (
            journal.failed_reconciliations[0]
        )
        assert failed_account == "allowed-a"
        assert failed_generations == {
            CapturePlant.ORDER: "order-generation-1",
            CapturePlant.PNL: "pnl-generation-1",
        }
        assert failure_reason == "recovery_phase_timeout"
        await service.stop()

    asyncio.run(scenario())


def test_recovery_timeout_is_per_account_not_one_shared_deadline():
    class SlowObserver(FakeObserver):
        async def reconcile_account(self, account_id, generations):
            self.calls.append(("reconcile", account_id, None))
            await asyncio.sleep(0.035)
            return RecoveryResult(clean=True, checkpoint=f"checkpoint-{account_id}")

    async def scenario():
        observer = SlowObserver(account_ids=("allowed-a", "allowed-b"))
        service = RithmicCaptureService(
            enabled_config(reconcile_timeout=0.06),
            observer=observer,
            journal=DurableJournal(),
        )
        await service.start()
        assert await service.wait_until_ready(timeout=1)
        assert observer.abort_calls == 0
        assert sum(call[0] == "reconcile" for call in observer.calls) == 2
        await service.stop()

    asyncio.run(scenario())


def test_checkpoint_persistence_failure_aborts_clean_fold_and_never_readies():
    class FailingCheckpointJournal(DurableJournal):
        async def save_checkpoint(self, checkpoint) -> None:
            del checkpoint
            raise RuntimeError("synthetic checkpoint outage")

    async def scenario():
        observer = FakeObserver(account_ids=("allowed-a",))
        journal = FailingCheckpointJournal()
        service = RithmicCaptureService(
            enabled_config(allowlist=frozenset({"allowed-a"})),
            observer=observer,
            journal=journal,
        )
        await service.start()
        for _ in range(100):
            if observer.abort_calls:
                break
            await asyncio.sleep(0.01)

        assert observer.abort_calls == 1
        assert not service.health().ready
        assert journal.checkpoints == ()
        assert len(journal.failed_reconciliations) == 1
        await service.stop()

    asyncio.run(scenario())


def test_failed_start_retries_without_requiring_process_restart():
    class FlakyStartObserver(FakeObserver):
        def __init__(self):
            super().__init__(account_ids=("allowed-a",))
            self.start_attempts = 0

        async def start(self, event_sink) -> None:
            self.start_attempts += 1
            if self.start_attempts == 1:
                raise ConnectionError("synthetic discovery outage")
            await super().start(event_sink)

    async def scenario():
        observer = FlakyStartObserver()
        service = RithmicCaptureService(
            enabled_config(allowlist=frozenset({"allowed-a"})),
            observer=observer,
            journal=DurableJournal(),
        )
        health = await service.start()
        assert health.lifecycle == "starting"
        assert "observer_start_failed" in health.blockers
        assert await service.wait_until_ready(timeout=2)
        assert observer.start_attempts == 2
        assert "observer_start_failed" not in service.health().blockers
        await service.stop()

    asyncio.run(scenario())


def test_buffer_overflow_persists_observations_but_aborts_readiness():
    class OverflowObserver(FakeObserver):
        async def reconcile_account(self, account_id, generations):
            assert self.sink is not None
            for suffix in ("one", "two"):
                await self.sink(
                    CaptureEvent(
                        event_id=f"live-{suffix}",
                        account_id=account_id,
                        plant=CapturePlant.ORDER,
                        source=CaptureSource.LIVE,
                        generation_id=generations[CapturePlant.ORDER],
                        payload={"observation_type": "ORDER"},
                    )
                )
            return RecoveryResult(clean=True, checkpoint="must-not-commit")

    async def scenario():
        observer = OverflowObserver(account_ids=("allowed-a",))
        journal = DurableJournal()
        service = RithmicCaptureService(
            enabled_config(
                allowlist=frozenset({"allowed-a"}),
                max_buffered_events=1,
            ),
            observer=observer,
            journal=journal,
        )
        await service.start()
        for _ in range(100):
            if observer.abort_calls:
                break
            await asyncio.sleep(0.01)
        assert observer.abort_calls == 1
        assert [event.event_id for event in journal.events] == ["live-one", "live-two"]
        assert journal.checkpoints == ()
        assert "live_event_buffer_overflow" in service.health().blockers
        assert not service.health().ready
        await service.stop()

    asyncio.run(scenario())


def test_drain_timeout_requeues_unprocessed_tail_before_abort_persistence():
    class SlowFirstAppendJournal(DurableJournal):
        def __init__(self):
            super().__init__()
            self.delayed = False

        async def append(self, event):
            if not self.delayed:
                self.delayed = True
                await asyncio.sleep(0.2)
            return await super().append(event)

    class TwoEventObserver(FakeObserver):
        async def reconcile_account(self, account_id, generations):
            assert self.sink is not None
            for suffix in ("one", "two"):
                await self.sink(
                    CaptureEvent(
                        event_id=f"tail-{suffix}",
                        account_id=account_id,
                        plant=CapturePlant.ORDER,
                        source=CaptureSource.LIVE,
                        generation_id=generations[CapturePlant.ORDER],
                        payload={"observation_type": "ORDER"},
                    )
                )
            return RecoveryResult(clean=True, checkpoint="must-not-commit")

    async def scenario():
        observer = TwoEventObserver(account_ids=("allowed-a",))
        journal = SlowFirstAppendJournal()
        service = RithmicCaptureService(
            enabled_config(
                allowlist=frozenset({"allowed-a"}),
                reconcile_timeout=0.04,
            ),
            observer=observer,
            journal=journal,
        )
        await service.start()
        for _ in range(100):
            if observer.abort_calls:
                break
            await asyncio.sleep(0.01)
        assert observer.abort_calls == 1
        for _ in range(100):
            if journal.depth == 2:
                break
            await asyncio.sleep(0.01)
        assert [event.event_id for event in journal.events] == ["tail-one", "tail-two"]
        assert journal.checkpoints == ()
        assert not service.health().ready
        await service.stop()

    asyncio.run(scenario())


def test_failed_buffer_flush_timeout_retains_tail_for_next_generation():
    class SlowFirstAppendJournal(DurableJournal):
        def __init__(self):
            super().__init__()
            self.delayed = False

        async def append(self, event):
            if not self.delayed:
                self.delayed = True
                await asyncio.sleep(0.2)
            return await super().append(event)

    class FailOnceObserver(FakeObserver):
        def __init__(self):
            super().__init__(account_ids=("allowed-a",))
            self.reconcile_attempts = 0

        async def reconcile_account(self, account_id, generations):
            self.reconcile_attempts += 1
            if self.reconcile_attempts == 1:
                assert self.sink is not None
                for suffix in ("one", "two"):
                    await self.sink(
                        CaptureEvent(
                            event_id=f"retained-{suffix}",
                            account_id=account_id,
                            plant=CapturePlant.ORDER,
                            source=CaptureSource.LIVE,
                            generation_id=generations[CapturePlant.ORDER],
                            payload={"observation_type": "ORDER"},
                        )
                    )
                raise RuntimeError("synthetic recovery failure")
            return RecoveryResult(clean=True, checkpoint="new-generation-clean")

    async def scenario():
        observer = FailOnceObserver()
        journal = SlowFirstAppendJournal()
        service = RithmicCaptureService(
            enabled_config(
                allowlist=frozenset({"allowed-a"}),
                reconcile_timeout=0.04,
            ),
            observer=observer,
            journal=journal,
        )
        await service.start()
        for _ in range(100):
            if "failed_recovery_buffer_persistence_failed" in service.health().blockers:
                break
            await asyncio.sleep(0.01)
        assert service.health().buffered_event_count == 2

        observer.health_by_plant = {
            plant: PlantHealth(
                plant=plant,
                connected=True,
                authenticated=True,
                generation_id=f"{plant.value.lower()}-generation-2",
            )
            for plant in (CapturePlant.ORDER, CapturePlant.PNL)
        }
        assert await service.wait_until_ready(timeout=1)
        assert [event.event_id for event in journal.events] == [
            "retained-one",
            "retained-two",
        ]
        assert service.health().buffered_event_count == 0
        assert "failed_recovery_buffer_persistence_failed" not in service.health().blockers
        await service.stop()

    asyncio.run(scenario())
