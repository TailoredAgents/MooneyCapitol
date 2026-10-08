from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from app.v2.brokers.rithmic_protocol.adapter import RithmicReadOnlyObserver
from app.v2.brokers.rithmic_protocol.constants import Plant, Template
from app.v2.brokers.rithmic_protocol.dispatch import DispatchEvent, DispatchKind
from app.v2.brokers.rithmic_protocol.errors import ProtocolRejected, UnauthorizedAccount
from app.v2.brokers.rithmic_protocol.normalization import (
    BrokerAccountKey,
    normalize_account,
)
from app.v2.brokers.rithmic_protocol.session import ReceivedMessage
from app.v2.brokers.rithmic_protocol.state import PlantSessionState, SessionState
from app.v2.capture.contracts import CaptureEvent, CapturePlant, CaptureSource


def _received(template: Template, message: dict) -> ReceivedMessage:
    return ReceivedMessage(
        plant=Plant.ORDER,
        generation_id="order-generation-1",
        template_id=int(template),
        fingerprint="a" * 64,
        dispatch=DispatchEvent(
            DispatchKind.MESSAGE,
            int(template),
            message=message,
        ),
        raw_frame=b"synthetic-frame",
    )


def _observer(account: BrokerAccountKey, sink):
    observer = object.__new__(RithmicReadOnlyObserver)
    observer.runtime = SimpleNamespace(account_ids=frozenset({account.account_id}))
    observer._event_sink = sink
    observer._accounts = {account.account_id: account}
    observer._account_rows = {}
    observer._ambiguous_accounts = set()
    observer._account_metadata_observed = set()
    observer._account_metadata_denied = set()
    observer._account_metadata_clock = {}
    observer._login_info = SimpleNamespace(user="test-user")
    observer._active_replay_batch = {}
    observer._reconcile_record_counts = {}
    observer._pending_reconciliations = {}
    observer._waiters = {}
    observer._subscriptions = {
        (account.account_id, Plant.ORDER, "order-generation-1"),
        (account.account_id, Plant.PNL, "pnl-generation-1"),
    }
    observer._accounts_discovered = asyncio.Event()
    observer._accounts_discovered.set()
    observer._sessions = {
        Plant.ORDER: SimpleNamespace(
            state=SimpleNamespace(generation_id="order-generation-1")
        ),
        Plant.PNL: SimpleNamespace(
            state=SimpleNamespace(generation_id="pnl-generation-1")
        ),
    }
    return observer


def test_inbound_event_requires_exact_discovered_fcm_ib_account_identity():
    async def scenario():
        events = []

        async def sink(event):
            events.append(event)
            return True

        expected = BrokerAccountKey("expected-fcm", "expected-ib", "allowed-account")
        observer = _observer(expected, sink)
        received = _received(
            Template.EXCHANGE_ORDER_NOTIFICATION,
            {
                "template_id": int(Template.EXCHANGE_ORDER_NOTIFICATION),
                "fcm_id": "wrong-fcm",
                "ib_id": "expected-ib",
                "account_id": "allowed-account",
                "basket_id": "basket-1",
                "notify_type": 1,
            },
        )

        with pytest.raises(UnauthorizedAccount, match="did not match discovery"):
            await observer._normalize_and_emit(received)
        assert events == []

    asyncio.run(scenario())


def test_forced_logout_audit_keeps_codes_and_hashes_but_not_free_text():
    async def scenario():
        events = []

        async def sink(event):
            events.append(event)
            return True

        account = BrokerAccountKey("fcm", "ib", "allowed-account")
        observer = _observer(account, sink)
        message = {
            "template_id": int(Template.FORCED_LOGOUT),
            "rp_code": ["7"],
            "reason_code": "SESSION_LIMIT",
            "reason": "sensitive broker diagnostic text",
        }
        received = ReceivedMessage(
            plant=Plant.ORDER,
            generation_id="order-generation-1",
            template_id=int(Template.FORCED_LOGOUT),
            fingerprint="c" * 64,
            dispatch=DispatchEvent(
                DispatchKind.FORCED_LOGOUT,
                int(Template.FORCED_LOGOUT),
                message=message,
                response_codes=("7",),
            ),
            raw_frame=b"synthetic-forced-logout",
        )

        await observer._emit_control(received)

        normalized = events[0].payload["normalized"]
        assert normalized["response_codes"] == ("7",)
        assert normalized["reason_code"] == "SESSION_LIMIT"
        assert len(normalized["text_sha256"][0]) == 64
        assert "sensitive broker diagnostic text" not in str(events[0].payload)

    asyncio.run(scenario())


def test_authorization_revocation_is_journaled_then_invalidates_both_plants():
    async def scenario():
        events = []
        aborts = []

        async def sink(event):
            events.append(event)
            return True

        async def abort(generations):
            aborts.append(dict(generations))

        account = BrokerAccountKey("fcm", "ib", "allowed-account")
        observer = _observer(account, sink)
        observer.abort_recovery = abort
        received = _received(
            Template.USER_ACCOUNT_UPDATE,
            {
                "template_id": int(Template.USER_ACCOUNT_UPDATE),
                "update_type": "remove_account_from_user",
                "fcm_id": "fcm",
                "ib_id": "ib",
                "account_id": "allowed-account",
                "account_status": "disabled",
            },
        )

        await observer._capture_user_account_update(received)

        assert len(events) == 1
        assert events[0].payload["normalized"]["authorization_revoked"] is True
        assert "allowed-account" not in observer._accounts
        assert not observer._accounts_discovered.is_set()
        assert aborts == [
            {
                CapturePlant.ORDER: "order-generation-1",
                CapturePlant.PNL: "pnl-generation-1",
            }
        ]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "authorization_field",
    [
        {"access_type": "unexpected-access"},
        {"account_status": "future-unknown-status"},
        {"status": "future-unknown-user-status"},
    ],
)
def test_unknown_live_authorization_values_fail_closed(authorization_field):
    async def scenario():
        events = []
        aborts = []

        async def sink(event):
            events.append(event)
            return True

        async def abort(generations):
            aborts.append(dict(generations))

        account = BrokerAccountKey("fcm", "ib", "allowed-account")
        observer = _observer(account, sink)
        observer.abort_recovery = abort
        await observer._capture_user_account_update(
            _received(
                Template.USER_ACCOUNT_UPDATE,
                {
                    "template_id": int(Template.USER_ACCOUNT_UPDATE),
                    "update_type": "modify_account",
                    "fcm_id": "fcm",
                    "ib_id": "ib",
                    "account_id": "allowed-account",
                    "user": "test-user",
                    **authorization_field,
                },
            )
        )

        assert len(events) == 1
        assert events[0].payload["normalized"]["authorization_revoked"] is True
        assert "allowed-account" not in observer._accounts
        assert len(aborts) == 1

    asyncio.run(scenario())


def test_other_users_removal_cannot_revoke_current_user_live_access():
    async def scenario():
        events = []

        async def sink(event):
            events.append(event)
            return True

        account = BrokerAccountKey("fcm", "ib", "allowed-account")
        observer = _observer(account, sink)
        await observer._capture_user_account_update(
            _received(
                Template.USER_ACCOUNT_UPDATE,
                {
                    "template_id": int(Template.USER_ACCOUNT_UPDATE),
                    "update_type": "remove_account_from_user",
                    "fcm_id": "fcm",
                    "ib_id": "ib",
                    "account_id": "allowed-account",
                    "user": "another-user",
                },
            )
        )

        assert observer._accounts == {"allowed-account": account}
        assert events == []

    asyncio.run(scenario())


def test_allowlisted_live_account_update_requires_full_broker_identity():
    async def scenario():
        account = BrokerAccountKey("fcm", "ib", "allowed-account")
        observer = _observer(account, lambda _event: True)
        with pytest.raises(UnauthorizedAccount, match="omitted full"):
            await observer._capture_user_account_update(
                _received(
                    Template.USER_ACCOUNT_UPDATE,
                    {
                        "template_id": int(Template.USER_ACCOUNT_UPDATE),
                        "update_type": "modify_account",
                        "account_id": "allowed-account",
                        "user": "test-user",
                        "access_type": "0",
                        "account_status": "active",
                    },
                )
            )

    asyncio.run(scenario())


def test_account_and_user_update_projects_live_risk_and_cash_facts():
    async def scenario():
        events = []

        async def sink(event):
            events.append(event)
            return True

        account = BrokerAccountKey("fcm", "ib", "allowed-account")
        observer = _observer(account, sink)
        received = _received(
            Template.ACCOUNT_AND_USER_UPDATE,
            {
                "template_id": int(Template.ACCOUNT_AND_USER_UPDATE),
                "update_type": "modify_account",
                "fcm_id": "fcm",
                "ib_id": "ib",
                "account_id": "allowed-account",
                "product_code": "NQ",
                "cash_on_hand": "12500.25",
                "loss_limit": "1500.50",
                "min_margin_balance": "2500",
                "max_limit_quantity": "20",
                "buy_margin_rate": "100.25",
                "sell_margin_rate": "101.25",
                "commission_fill_rate": "2.40",
                "ssboe": 1_791_382_800,
                "usecs": 5,
            },
        )

        await observer._capture_user_account_update(received)

        by_kind = {event.payload["observation_type"]: event for event in events}
        assert set(by_kind) == {"ACCOUNT", "PRODUCT_RMS", "ACCOUNT_PNL"}
        rms = by_kind["PRODUCT_RMS"].payload["normalized"]
        assert rms["loss_limit"] == "1500.50"
        assert rms["max_order_quantity"] == "20"
        assert rms["commission_rate"] == "2.40"
        pnl = by_kind["ACCOUNT_PNL"].payload["normalized"]
        assert pnl["cash_on_hand"] == "12500.25"
        assert pnl["timestamps"]["ssboe"] == 1_791_382_800

    asyncio.run(scenario())


def test_multi_account_rms_rows_are_scoped_to_the_requesting_recovery_batch():
    async def scenario():
        events = []

        async def sink(event):
            events.append(event)
            return True

        account_a = BrokerAccountKey("fcm", "ib", "account-a")
        account_b = BrokerAccountKey("fcm", "ib", "account-b")
        observer = _observer(account_a, sink)
        observer.runtime = SimpleNamespace(
            account_ids=frozenset({"account-a", "account-b"})
        )
        observer._accounts["account-b"] = account_b
        observer._active_replay_batch = {
            "account-a": "batch-a",
            "account-b": "batch-b",
        }
        row_for_b = {
            "template_id": int(Template.ACCOUNT_RMS_RESPONSE),
            "user_msg": ["rms-a"],
            "fcm_id": "fcm",
            "ib_id": "ib",
            "account_id": "account-b",
            "loss_limit": "500",
        }
        observer._waiters = {
            "rms-a": SimpleNamespace(
                account=account_a,
                plant=Plant.ORDER,
                generation_id="order-generation-1",
            )
        }

        await observer._normalize_and_emit(
            _received(Template.ACCOUNT_RMS_RESPONSE, row_for_b)
        )
        assert events == []

        row_for_b["user_msg"] = ["rms-b"]
        observer._waiters = {
            "rms-b": SimpleNamespace(
                account=account_b,
                plant=Plant.ORDER,
                generation_id="order-generation-1",
            )
        }
        await observer._normalize_and_emit(
            _received(Template.ACCOUNT_RMS_RESPONSE, row_for_b)
        )
        assert len(events) == 1
        assert events[0].account_id == "account-b"
        assert events[0].payload["replay_batch_id"] == "batch-b"

    asyncio.run(scenario())


def test_cancelled_recovery_emits_failed_boundary_before_reraising():
    class AnyMessageFactory:
        def __getattr__(self, _name):
            return lambda *args, **kwargs: object()

    class Transport:
        def __init__(self):
            self.closed = []

        async def close(self, *, reason):
            self.closed.append(reason)

    async def scenario():
        events = []

        async def sink(event):
            events.append(event)
            return True

        async def hang(_session, _message, _response_template, **_kwargs):
            await asyncio.Event().wait()

        account = BrokerAccountKey("fcm", "ib", "allowed-account")
        observer = _observer(account, sink)
        observer.message_factory = AnyMessageFactory()
        observer._login_info = SimpleNamespace(user_type_code=3)
        observer.runtime = SimpleNamespace(
            account_ids=frozenset({"allowed-account"}),
            fill_history_start_index=20261007,
            fill_history_finish_index=20261007,
            recovery_trade_date=20261007,
        )
        observer._send_and_wait = hang
        observer._fail_waiters = lambda _plant, _generation=None: None
        sessions = {}
        generations = {}
        for plant, capture_plant in (
            (Plant.ORDER, CapturePlant.ORDER),
            (Plant.PNL, CapturePlant.PNL),
        ):
            state = PlantSessionState(plant)
            state.begin_connect()
            state.transport_connected()
            state.login_succeeded(5)
            transport = Transport()
            sessions[plant] = SimpleNamespace(state=state, transport=transport)
            generations[capture_plant] = str(state.generation_id)
        observer._sessions = sessions
        observer._subscriptions = {
            ("allowed-account", plant, str(sessions[plant].state.generation_id))
            for plant in (Plant.ORDER, Plant.PNL)
        }

        task = asyncio.create_task(
            observer.reconcile_account("allowed-account", generations)
        )
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert len(events) == 2
        assert events[0].payload["observation_type"] == "RECOVERY_START"
        assert events[1].payload["observation_type"] == "RECOVERY_CHECKPOINT"
        assert events[1].payload["batch_status"] == "FAILED"
        assert events[1].payload["terminal_response_received"] is False
        assert observer._active_replay_batch == {}
        assert all(
            session.state.state is SessionState.DISCONNECTED
            for session in sessions.values()
        )
        assert all(session.transport.closed for session in sessions.values())

    asyncio.run(scenario())


def test_failed_old_recovery_does_not_close_reconnected_generation():
    class Transport:
        def __init__(self):
            self.closed = []

        async def close(self, *, reason):
            self.closed.append(reason)

    async def scenario():
        events = []

        async def sink(event):
            events.append(event)
            return True

        account = BrokerAccountKey("fcm", "ib", "allowed-account")
        observer = _observer(account, sink)
        observer._reconcile_record_counts = {"allowed-account": 2}
        old_generations = {}
        sessions = {}
        for plant, capture_plant in (
            (Plant.ORDER, CapturePlant.ORDER),
            (Plant.PNL, CapturePlant.PNL),
        ):
            state = PlantSessionState(plant)
            state.begin_connect()
            old_generations[capture_plant] = str(state.generation_id)
            state.connection_lost("synthetic reconnect")
            state.begin_connect()
            state.transport_connected()
            state.login_succeeded(5)
            sessions[plant] = SimpleNamespace(state=state, transport=Transport())
        observer._sessions = sessions

        await observer._fail_account_reconciliation(
            "allowed-account",
            account,
            "old-batch",
            old_generations,
            (Plant.ORDER, Plant.PNL),
        )

        assert all(
            session.state.state is SessionState.CONNECTED
            for session in sessions.values()
        )
        assert all(not session.transport.closed for session in sessions.values())
        assert len(events) == 1
        assert events[0].generation_id == old_generations[CapturePlant.ORDER]
        assert events[0].payload["generations"] == {
            plant.value: generation
            for plant, generation in old_generations.items()
        }

    asyncio.run(scenario())


def test_invalid_fill_history_range_never_activates_a_replay_batch():
    class AnyMessageFactory:
        def __getattr__(self, _name):
            return lambda *args, **kwargs: object()

    async def scenario():
        events = []

        async def sink(event):
            events.append(event)
            return True

        account = BrokerAccountKey("fcm", "ib", "allowed-account")
        observer = _observer(account, sink)
        observer.message_factory = AnyMessageFactory()
        observer._login_info = SimpleNamespace(user_type_code=3)
        observer.runtime = SimpleNamespace(
            account_ids=frozenset({"allowed-account"}),
            fill_history_start_index=20261008,
            fill_history_finish_index=20261007,
            recovery_trade_date=20261007,
        )
        sessions = {}
        generations = {}
        for plant, capture_plant in (
            (Plant.ORDER, CapturePlant.ORDER),
            (Plant.PNL, CapturePlant.PNL),
        ):
            state = PlantSessionState(plant)
            state.begin_connect()
            state.transport_connected()
            state.login_succeeded(5)
            sessions[plant] = SimpleNamespace(state=state)
            generations[capture_plant] = str(state.generation_id)
        observer._sessions = sessions
        observer._subscriptions = {
            ("allowed-account", plant, str(sessions[plant].state.generation_id))
            for plant in (Plant.ORDER, Plant.PNL)
        }

        with pytest.raises(Exception, match="trade-date range"):
            await observer.reconcile_account("allowed-account", generations)

        assert observer._active_replay_batch == {}
        assert observer._reconcile_record_counts == {}
        assert events == []
        assert all(
            session.state.state is SessionState.CONNECTED
            for session in sessions.values()
        )

    asyncio.run(scenario())


def test_discover_accounts_has_no_single_request_deadline():
    async def scenario():
        account = BrokerAccountKey("fcm", "ib", "allowed-account")
        observer = _observer(account, lambda _event: True)
        observer._accounts_discovered.clear()
        observer.runtime = SimpleNamespace(
            account_ids=frozenset({"allowed-account"}),
            request_timeout_seconds=0.001,
        )
        task = asyncio.create_task(observer.discover_accounts())
        await asyncio.sleep(0.01)
        assert not task.done()
        observer._accounts_discovered.set()
        discovered = await task
        assert [item.account_id for item in discovered] == ["allowed-account"]

    asyncio.run(scenario())


def test_tracked_request_timeout_resets_plant_before_releasing_waiter():
    class Transport:
        def __init__(self):
            self.close_reasons = []

        async def close(self, *, reason):
            self.close_reasons.append(reason)

    class Session:
        plant = Plant.TICKER

        def __init__(self):
            self.state = SimpleNamespace(generation_id="ticker-generation-1")
            self.transport = Transport()

        async def send(self, _message):
            return None

    async def scenario():
        account = BrokerAccountKey("fcm", "ib", "allowed-account")
        observer = _observer(account, lambda _event: True)
        observer.runtime = SimpleNamespace(request_timeout_seconds=0.001)
        session = Session()

        with pytest.raises(TimeoutError):
            await observer._send_and_wait(
                session,
                {"user_msg": ["timed-out-query"]},
                Template.SEARCH_SYMBOLS_RESPONSE,
            )

        assert observer._waiters == {}
        assert session.transport.close_reasons == ["tracked request timed out"]

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("code_field", "codes"),
    [
        ("rq_handler_rp_code", ["7", "row rejected"]),
        ("rp_code", ["0", "malformed extra code"]),
    ],
)
def test_multipart_error_or_malformed_terminal_fails_without_emitting(
    code_field, codes
):
    async def scenario():
        events = []

        async def sink(event):
            events.append(event)
            return True

        account = BrokerAccountKey("fcm", "ib", "allowed-account")
        observer = _observer(account, sink)
        correlation = "rms-correlation"
        future = asyncio.get_running_loop().create_future()
        observer._waiters = {
            correlation: SimpleNamespace(
                account=account,
                plant=Plant.ORDER,
                generation_id="order-generation-1",
                response_template=int(Template.ACCOUNT_RMS_RESPONSE),
                future=future,
                error=None,
            )
        }
        observer._response_rows = {correlation: []}
        message = {
            "template_id": int(Template.ACCOUNT_RMS_RESPONSE),
            "user_msg": [correlation],
            code_field: codes,
            "fcm_id": "fcm",
            "ib_id": "ib",
            "account_id": "allowed-account",
            "loss_limit": "500",
        }

        await observer._handle_message(
            _received(Template.ACCOUNT_RMS_RESPONSE, message)
        )

        if code_field == "rq_handler_rp_code":
            assert not future.done()
            assert isinstance(observer._waiters[correlation].error, ProtocolRejected)
            await observer._handle_message(
                _received(
                    Template.ACCOUNT_RMS_RESPONSE,
                    {
                        "template_id": int(Template.ACCOUNT_RMS_RESPONSE),
                        "user_msg": [correlation],
                        "rp_code": ["0"],
                    },
                )
            )
        assert isinstance(future.exception(), ProtocolRejected)
        assert observer._response_rows[correlation] == []
        assert events == []

    asyncio.run(scenario())


def test_semantic_dedupe_preserves_changes_and_is_generation_bounded():
    @dataclass(frozen=True)
    class SyntheticOrder:
        status: str
        source_kind: str

        @property
        def stable_identity(self):
            return ("order", "basket-1", "OPEN")

    async def scenario():
        events = []

        async def sink(event):
            events.append(event)
            return True

        account = BrokerAccountKey("fcm", "ib", "allowed-account")
        observer = _observer(account, sink)
        base = _received(
            Template.EXCHANGE_ORDER_NOTIFICATION,
            {"template_id": int(Template.EXCHANGE_ORDER_NOTIFICATION)},
        )

        await observer._emit(
            base,
            SyntheticOrder("OPEN", "SNAPSHOT"),
            "ORDER",
            account,
            CaptureSource.SNAPSHOT,
        )
        await observer._emit(
            base,
            SyntheticOrder("MODIFIED", "LIVE"),
            "ORDER",
            account,
            CaptureSource.LIVE,
        )
        assert events[0].event_id != events[1].event_id

        await observer._emit(
            base,
            SyntheticOrder("OPEN", "LIVE"),
            "ORDER",
            account,
            CaptureSource.LIVE,
        )
        assert events[0].event_id == events[2].event_id

        next_generation = ReceivedMessage(
            plant=base.plant,
            generation_id="order-generation-2",
            template_id=base.template_id,
            fingerprint=base.fingerprint,
            dispatch=base.dispatch,
            raw_frame=base.raw_frame,
        )
        await observer._emit(
            next_generation,
            SyntheticOrder("OPEN", "LIVE"),
            "ORDER",
            account,
            CaptureSource.LIVE,
        )
        assert events[0].event_id != events[3].event_id

        modified_fact = _received(
            Template.EXCHANGE_ORDER_NOTIFICATION,
            {
                "template_id": int(Template.EXCHANGE_ORDER_NOTIFICATION),
                "modified_id": "official-change-1",
            },
        )
        await observer._emit(
            modified_fact,
            SyntheticOrder("OPEN", "LIVE"),
            "ORDER",
            account,
            CaptureSource.LIVE,
        )
        assert events[0].event_id != events[4].event_id

        snapshot_delivery_only = _received(
            Template.EXCHANGE_ORDER_NOTIFICATION,
            {
                "template_id": int(Template.EXCHANGE_ORDER_NOTIFICATION),
                "is_snapshot": True,
            },
        )
        await observer._emit(
            snapshot_delivery_only,
            SyntheticOrder("OPEN", "LIVE"),
            "ORDER",
            account,
            CaptureSource.LIVE,
        )
        live_delivery_only = _received(
            Template.EXCHANGE_ORDER_NOTIFICATION,
            {
                "template_id": int(Template.EXCHANGE_ORDER_NOTIFICATION),
                "is_snapshot": False,
            },
        )
        await observer._emit(
            live_delivery_only,
            SyntheticOrder("OPEN", "LIVE"),
            "ORDER",
            account,
            CaptureSource.LIVE,
        )
        assert events[5].event_id == events[6].event_id

    asyncio.run(scenario())


def test_empty_terminal_responses_from_distinct_templates_are_not_deduplicated():
    async def scenario():
        events = []

        async def sink(event):
            events.append(event)
            return True

        account = BrokerAccountKey("fcm", "ib", "allowed-account")
        observer = _observer(account, sink)
        for template in (
            Template.SHOW_BRACKETS_RESPONSE,
            Template.SHOW_BRACKET_STOPS_RESPONSE,
        ):
            received = _received(
                template,
                {"template_id": int(template), "rp_code": ["0"]},
            )
            await observer._emit(
                received,
                None,
                "UNKNOWN",
                account,
                CaptureSource.SNAPSHOT,
            )

        assert len(events) == 2
        assert events[0].event_id != events[1].event_id

    asyncio.run(scenario())


def test_buffered_live_state_is_folded_after_snapshot_before_final_checkpoint():
    async def scenario():
        account = BrokerAccountKey("fcm", "ib", "allowed-account")
        observer = _observer(account, lambda _event: True)
        generations = {
            CapturePlant.ORDER: "order-generation-1",
            CapturePlant.PNL: "pnl-generation-1",
        }
        sessions = {}
        for plant in (Plant.ORDER, Plant.PNL):
            state = PlantSessionState(plant)
            state.begin_connect()
            state.generation_id = generations[CapturePlant[plant.name]]
            state.transport_connected()
            state.login_succeeded(5)
            state.begin_reconciliation()
            sessions[plant] = SimpleNamespace(state=state)
        observer._sessions = sessions
        await observer.prepare_reconciliation("allowed-account", generations)
        pending = observer._pending_reconciliations["allowed-account"]
        snapshot = CaptureEvent(
            event_id="snapshot-order",
            account_id="allowed-account",
            plant=CapturePlant.ORDER,
            source=CaptureSource.SNAPSHOT,
            generation_id="order-generation-1",
            payload={
                "observation_type": "ORDER",
                "native_identity": {"basket_id": "basket-1"},
                "normalized": {"basket_id": "basket-1", "status": "OPEN"},
                "decoded_facts": {"status": "OPEN"},
                "dedupe_key": "snapshot-order",
            },
        )
        observer._fold_reconciliation_event(pending, snapshot)
        pending.snapshot_complete = True
        before = observer._reconciliation_digest(pending)
        live = CaptureEvent(
            event_id="live-order",
            account_id="allowed-account",
            plant=CapturePlant.ORDER,
            source=CaptureSource.LIVE,
            generation_id="order-generation-1",
            payload={
                "observation_type": "ORDER",
                "native_identity": {"basket_id": "basket-1"},
                "normalized": {"basket_id": "basket-1", "status": "MODIFIED"},
                "decoded_facts": {"status": "MODIFIED", "modified_id": "m-1"},
                "dedupe_key": "live-order",
            },
        )
        await observer.apply_buffered("allowed-account", (live,))
        result = await observer.finalize_reconciliation(
            "allowed-account", generations
        )

        assert result.clean and result.discrepancy_count == 0
        assert result.checkpoint != before
        assert result.records_seen == 2
        assert all(
            session.state.state is SessionState.READY
            for session in sessions.values()
        )
        assert observer._pending_reconciliations == {}

    asyncio.run(scenario())


def test_empty_multipart_terminal_ack_is_not_a_state_discrepancy():
    account = BrokerAccountKey("fcm", "ib", "allowed-account")
    observer = _observer(account, lambda _event: True)
    pending = SimpleNamespace(
        account=account,
        generations={CapturePlant.ORDER: "order-generation-1"},
        state={},
        seen_event_ids=set(),
        records_seen=0,
        discrepancy_count=0,
    )
    terminal = CaptureEvent(
        event_id="empty-fill-history-terminal",
        account_id="allowed-account",
        plant=CapturePlant.ORDER,
        source=CaptureSource.HISTORY,
        generation_id="order-generation-1",
        payload={
            "observation_type": "ORDER",
            "normalized": {},
            "native_identity": {},
            "decoded_facts": {"rp_code": ["0"], "user_msg": ["fills-1"]},
            "dedupe_key": "empty-fill-history-terminal",
        },
    )

    observer._fold_reconciliation_event(pending, terminal)

    assert pending.discrepancy_count == 0
    assert len(pending.state) == 1


def test_abort_drops_old_pending_recovery_without_erasing_new_generation():
    async def scenario():
        account = BrokerAccountKey("fcm", "ib", "allowed-account")
        observer = _observer(account, lambda _event: True)
        old = {
            CapturePlant.ORDER: "order-generation-1",
            CapturePlant.PNL: "pnl-generation-1",
        }
        await observer.prepare_reconciliation("allowed-account", old)
        observer._sessions[Plant.ORDER].state.generation_id = "order-generation-2"
        observer._sessions[Plant.PNL].state.generation_id = "pnl-generation-2"

        await observer.abort_recovery(old)

        assert observer._pending_reconciliations == {}
        new = {
            CapturePlant.ORDER: "order-generation-2",
            CapturePlant.PNL: "pnl-generation-2",
        }
        await observer.prepare_reconciliation("allowed-account", new)
        assert (
            observer._pending_reconciliations["allowed-account"].generations
            == new
        )

    asyncio.run(scenario())


def test_account_metadata_playback_gates_on_exact_access_and_status():
    async def scenario():
        events = []

        async def sink(event):
            events.append(event)
            return True

        account = BrokerAccountKey("fcm", "ib", "allowed-account")
        observer = _observer(account, sink)
        observer._account_rows[account.account_id] = normalize_account(
            {
                "fcm_id": "fcm",
                "ib_id": "ib",
                "account_id": "allowed-account",
                "account_name": "Before playback",
                "account_currency": "USD",
            }
        )
        await observer._capture_account_metadata_playback(
            _received(
                Template.PLAYBACK_ACCOUNT_USERS_RESPONSE,
                {
                    "template_id": int(Template.PLAYBACK_ACCOUNT_USERS_RESPONSE),
                    "update_type": "assign_account_to_user",
                    "fcm_id": "fcm",
                    "ib_id": "ib",
                    "account_id": "allowed-account",
                    "account_name": "Test account",
                    "account_access_type": "0",
                    "account_status": "active",
                    "user": "test-user",
                    "status": "enabled",
                    "ssboe": 1_791_382_800,
                    "usecs": 7,
                },
            )
        )

        assert observer._account_metadata_observed == {"allowed-account"}
        assert observer._account_metadata_denied == set()
        row = observer._account_rows["allowed-account"]
        assert row.account_name == "Test account"
        assert row.access_type == "0"
        assert row.account_status == "active"
        assert row.user_status == "enabled"
        assert len(events) == 1
        assert events[0].source is CaptureSource.HISTORY
        assert events[0].payload["normalized"]["access_type"] == "0"

        with pytest.raises(UnauthorizedAccount, match="metadata identity"):
            await observer._capture_account_metadata_playback(
                _received(
                    Template.PLAYBACK_ACCOUNT_USERS_RESPONSE,
                    {
                        "fcm_id": "wrong-fcm",
                        "ib_id": "ib",
                        "account_id": "allowed-account",
                        "account_access_type": "view",
                        "account_status": "enabled",
                    },
                )
            )

    asyncio.run(scenario())


def test_account_metadata_removal_cannot_be_cleared_by_another_user_row():
    async def scenario():
        account = BrokerAccountKey("fcm", "ib", "allowed-account")
        observer = _observer(account, lambda _event: True)
        observer._account_rows[account.account_id] = normalize_account(
            {
                "fcm_id": "fcm",
                "ib_id": "ib",
                "account_id": "allowed-account",
            }
        )

        async def playback(ssboe, **fields):
            await observer._capture_account_metadata_playback(
                _received(
                    Template.PLAYBACK_ACCOUNT_USERS_RESPONSE,
                    {
                        "template_id": int(
                            Template.PLAYBACK_ACCOUNT_USERS_RESPONSE
                        ),
                        "fcm_id": "fcm",
                        "ib_id": "ib",
                        "account_id": "allowed-account",
                        "ssboe": ssboe,
                        **fields,
                    },
                )
            )

        await playback(
            100,
            update_type="assign_account_to_user",
            user="test-user",
            account_access_type="0",
            account_status="active",
            status="enabled",
        )
        assert observer._account_metadata_observed == {"allowed-account"}

        await playback(
            101,
            update_type="remove_account_from_user",
            user="test-user",
        )
        assert observer._account_metadata_observed == set()
        assert observer._account_metadata_denied == {"allowed-account"}

        await playback(
            102,
            update_type="assign_account_to_user",
            user="unrelated-user",
            account_access_type="1",
            account_status="active",
            status="enabled",
        )
        assert observer._account_metadata_observed == set()
        assert observer._account_metadata_denied == {"allowed-account"}

    asyncio.run(scenario())


def test_disabled_account_metadata_fails_authorization_gate():
    async def scenario():
        account = BrokerAccountKey("fcm", "ib", "allowed-account")
        observer = _observer(account, lambda _event: True)
        observer._account_rows[account.account_id] = normalize_account(
            {
                "fcm_id": "fcm",
                "ib_id": "ib",
                "account_id": "allowed-account",
            }
        )
        await observer._capture_account_metadata_playback(
            _received(
                Template.PLAYBACK_ACCOUNT_USERS_RESPONSE,
                {
                    "template_id": int(Template.PLAYBACK_ACCOUNT_USERS_RESPONSE),
                    "update_type": "modify_account",
                    "fcm_id": "fcm",
                    "ib_id": "ib",
                    "account_id": "allowed-account",
                    "account_access_type": "0",
                    "account_status": "inactive",
                    "user": "test-user",
                    "status": "enabled",
                    "ssboe": 200,
                },
            )
        )
        assert observer._account_metadata_observed == set()
        assert observer._account_metadata_denied == {"allowed-account"}

    asyncio.run(scenario())


def test_malformed_allowlisted_live_message_is_preserved_as_unknown():
    async def scenario():
        events = []

        async def sink(event):
            events.append(event)
            return True

        account = BrokerAccountKey("fcm", "ib", "allowed-account")
        observer = _observer(account, sink)
        message = {
            "template_id": int(Template.EXCHANGE_ORDER_NOTIFICATION),
            "fcm_id": "fcm",
            "ib_id": "ib",
            "account_id": "allowed-account",
            "basket_id": "basket-malformed",
            "notify_type": 1,
            "fill_size_64": "not-an-integer",
        }
        await observer._normalize_and_emit(
            _received(Template.EXCHANGE_ORDER_NOTIFICATION, message)
        )

        assert len(events) == 1
        event = events[0]
        assert event.account_id == "allowed-account"
        assert event.payload["observation_type"] == "UNKNOWN"
        assert event.payload["normalized"] == {}
        assert event.payload["decoded_facts"]["basket_id"] == "basket-malformed"
        assert event.payload["raw_frame_b64"]

    asyncio.run(scenario())
