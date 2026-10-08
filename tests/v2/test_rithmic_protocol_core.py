from __future__ import annotations

import asyncio
import ssl
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from app.v2.brokers.rithmic_protocol.adapter import (
    RithmicReadOnlyObserver,
    _cme_recovery_trade_date,
)
from app.v2.calendar import EASTERN
from app.v2.brokers.rithmic_protocol.constants import (
    BROKER_MUTATION_TEMPLATE_IDS,
    OUTBOUND_READ_ONLY_TEMPLATE_IDS,
    PROTOCOL_TEMPLATE_VERSION,
    Plant,
    Template,
)
from app.v2.brokers.rithmic_protocol.dispatch import DispatchKind, MessageDispatcher
from app.v2.brokers.rithmic_protocol.errors import (
    InsecureEndpoint,
    InvalidFrame,
    InvalidStateTransition,
    MutationTemplateRejected,
    OutboundTemplateRejected,
)
from app.v2.brokers.rithmic_protocol.framing import extract_template_id
from app.v2.brokers.rithmic_protocol.multipart import MultipartResponseTracker
from app.v2.brokers.rithmic_protocol.recovery import RecoveryCoordinator, RecoveryPhase
from app.v2.brokers.rithmic_protocol.session import PlantSession
from app.v2.brokers.rithmic_protocol.state import (
    PlantSessionState,
    ReconnectPolicy,
    SessionState,
    all_required_plants_ready,
)
from app.v2.brokers.rithmic_protocol.transport import (
    ReadOnlyOutboundPolicy,
    WssTransport,
    create_client_ssl_context,
    validate_wss_endpoint,
)


def _varint(value: int) -> bytes:
    result = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            result.append(byte | 0x80)
        else:
            result.append(byte)
            return bytes(result)


def _frame(template_id: int, *, prefix: bytes = b"") -> bytes:
    key = (154467 << 3) | 0
    return prefix + _varint(key) + _varint(template_id)


class _Socket:
    def __init__(self) -> None:
        self.sent: list[bytes] = []
        self.closed = False

    async def send(self, message: bytes) -> None:
        self.sent.append(message)

    async def recv(self):
        return _frame(19)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        del code, reason
        self.closed = True


def test_v2_binary_frame_has_no_length_prefix_and_finds_large_template_field():
    # An unrelated length-delimited field before template_id proves extraction
    # does not assume field order or a legacy four-byte message length.
    prefix = _varint((1 << 3) | 2) + _varint(3) + b"abc"
    assert extract_template_id(_frame(351, prefix=prefix)) == 351
    with pytest.raises(InvalidFrame):
        extract_template_id(b"")
    with pytest.raises(InvalidFrame):
        extract_template_id("not-binary")  # type: ignore[arg-type]


def test_duplicate_template_id_cannot_smuggle_a_mutation_past_policy():
    read_request = _frame(int(Template.SHOW_ORDERS_REQUEST))
    mutation_request = _frame(int(Template.NEW_ORDER_REQUEST))

    with pytest.raises(InvalidFrame, match="duplicate template_id"):
        extract_template_id(read_request + mutation_request)
    with pytest.raises(InvalidFrame, match="duplicate template_id"):
        extract_template_id(read_request + read_request)


def test_outbound_policy_default_denies_unknowns_and_every_broker_mutation():
    policy = ReadOnlyOutboundPolicy()
    assert not ({int(item) for item in BROKER_MUTATION_TEMPLATE_IDS} & policy.allowed)
    for template_id in BROKER_MUTATION_TEMPLATE_IDS:
        with pytest.raises(MutationTemplateRejected):
            policy.assert_allowed(template_id)
    with pytest.raises(OutboundTemplateRejected):
        policy.assert_allowed(999999)
    for template_id in OUTBOUND_READ_ONLY_TEMPLATE_IDS:
        policy.assert_allowed(template_id)


def test_observer_surface_has_no_broker_mutation_methods():
    for method in (
        "submit",
        "modify",
        "cancel",
        "cancel_all",
        "flatten",
        "exit_position",
        "submit_bracket",
        "submit_oco",
        "link_orders",
        "modify_stop",
        "modify_target",
    ):
        assert not hasattr(RithmicReadOnlyObserver, method)


def test_wss_and_tls_verification_are_mandatory():
    endpoint = validate_wss_endpoint("wss://example.test/rithmic")
    assert endpoint.hostname == "example.test"
    for uri in (
        "ws://example.test/rithmic",
        "https://example.test/rithmic",
        "wss://user:secret@example.test/rithmic",
        "wss://example.test/rithmic?token=secret",
    ):
        with pytest.raises(InsecureEndpoint):
            validate_wss_endpoint(uri)
    insecure = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    insecure.check_hostname = False
    insecure.verify_mode = ssl.CERT_NONE
    with pytest.raises(InsecureEndpoint):
        WssTransport("wss://example.test", ssl_context=insecure)
    secure = create_client_ssl_context()
    assert secure.check_hostname and secure.verify_mode == ssl.CERT_REQUIRED


def test_transport_rechecks_template_immediately_before_socket_send():
    socket = _Socket()

    async def connect_impl(*args, **kwargs):
        del args, kwargs
        return socket

    async def run() -> None:
        transport = WssTransport("wss://example.test", connect_impl=connect_impl)
        await transport.connect()
        assert await transport.send_frame(_frame(18)) == 18
        with pytest.raises(MutationTemplateRejected):
            await transport.send_frame(_frame(312))
        await transport.close()

    asyncio.run(run())
    assert socket.sent == [_frame(18)]
    assert socket.closed


def test_order_and_pnl_health_are_independent_and_require_reconciliation():
    order = PlantSessionState(Plant.ORDER)
    pnl = PlantSessionState(Plant.PNL)
    for state in (order, pnl):
        state.begin_connect()
        state.transport_connected()
        state.login_succeeded(5)
    order.begin_reconciliation()
    order.reconciliation_succeeded()
    assert order.ready and not pnl.ready
    assert not all_required_plants_ready((order, pnl))
    pnl.begin_reconciliation()
    pnl.reconciliation_succeeded()
    assert all_required_plants_ready((order, pnl))
    assert order.generation_id != pnl.generation_id

    # A second account reconciles on the same authenticated plant sessions.
    # READY must support re-entry without manufacturing a new generation.
    generations = (order.generation_id, pnl.generation_id)
    for state in (order, pnl):
        state.begin_reconciliation()
        state.reconciliation_succeeded()
    assert all_required_plants_ready((order, pnl))
    assert (order.generation_id, pnl.generation_id) == generations


def test_heartbeat_due_and_unanswered_heartbeat_becomes_stale():
    now = datetime(2026, 10, 7, tzinfo=timezone.utc)
    state = PlantSessionState(Plant.ORDER)
    state.begin_connect(now=now)
    state.transport_connected(now=now)
    state.login_succeeded(5, now=now)
    assert not state.heartbeat_due(now=now + timedelta(seconds=4))
    assert state.heartbeat_due(now=now + timedelta(seconds=5))
    sent = now + timedelta(seconds=5)
    state.record_outbound(heartbeat=True, now=sent)
    assert state.heartbeat_response_pending()
    assert not state.heartbeat_due(now=sent + timedelta(seconds=5))
    assert not state.heartbeat_response_overdue(now=sent + timedelta(seconds=9))
    assert state.heartbeat_response_overdue(now=sent + timedelta(seconds=10))
    state.record_message(now=sent + timedelta(seconds=11))
    assert not state.heartbeat_response_pending()
    assert not state.heartbeat_response_overdue(now=sent + timedelta(seconds=20))
    assert state.heartbeat_due(now=sent + timedelta(seconds=20))


def test_dispatch_login_heartbeat_reject_and_forced_logout_are_deterministic():
    dispatcher = MessageDispatcher()
    state = PlantSessionState(Plant.ORDER)
    state.begin_connect()
    state.transport_connected()
    event = dispatcher.dispatch(
        11,
        {
            "template_id": 11,
            "rp_code": ["0"],
            "heartbeat_interval": "5.5",
            "template_version": PROTOCOL_TEMPLATE_VERSION,
        },
        state,
    )
    assert event.kind is DispatchKind.LOGIN_SUCCEEDED
    assert state.state is SessionState.CONNECTED
    event = dispatcher.dispatch(19, {"template_id": 19, "rp_code": ["0"]}, state)
    assert event.kind is DispatchKind.HEARTBEAT
    assert state.last_heartbeat_received_at is not None
    event = dispatcher.dispatch(75, {"template_id": 75, "rp_code": ["error"]}, state)
    assert event.kind is DispatchKind.REJECT
    assert state.state is SessionState.DEGRADED

    forced = PlantSessionState(Plant.PNL)
    forced.begin_connect()
    forced.transport_connected()
    forced.login_succeeded(5)
    event = dispatcher.dispatch(77, {"template_id": 77, "rp_code": ["7"]}, forced)
    assert event.kind is DispatchKind.FORCED_LOGOUT
    assert event.response_codes == ("7",)
    assert forced.state is SessionState.FORCED_LOGOUT


@pytest.mark.parametrize("template_version", [None, "0.49", "5.54", "5.55 "])
def test_login_rejects_omitted_or_incompatible_protocol_version(template_version):
    state = PlantSessionState(Plant.ORDER)
    state.begin_connect()
    state.transport_connected()
    message = {
        "template_id": int(Template.LOGIN_RESPONSE),
        "rp_code": ["0"],
        "heartbeat_interval": "5",
    }
    if template_version is not None:
        message["template_version"] = template_version

    event = MessageDispatcher().dispatch(Template.LOGIN_RESPONSE, message, state)

    assert event.kind is DispatchKind.LOGIN_FAILED
    assert state.state is SessionState.DEGRADED


def test_multipart_tracker_distinguishes_rows_from_terminal_response():
    tracker = MultipartResponseTracker()
    tracker.begin("accounts-1", {303})
    row = tracker.consume(
        303,
        {"user_msg": ["accounts-1"], "rq_handler_rp_code": ["0"], "account_id": "x"},
    )
    assert row and not row.terminal and row.row_count == 1 and row.success is None
    terminal = tracker.consume(303, {"user_msg": ["accounts-1"], "rp_code": ["0"]})
    assert terminal and terminal.terminal and terminal.success and terminal.row_count == 1
    assert tracker.pending_count == 0


def test_multipart_correlation_prefers_echoed_user_message_over_server_key():
    tracker = MultipartResponseTracker()
    tracker.begin("fills-client-1", {3513})
    terminal = tracker.consume(
        3513,
        {
            "request_key": "server-flow-key",
            "user_msg": ["fills-client-1"],
            "rp_code": ["0"],
        },
    )

    assert terminal is not None
    assert terminal.correlation_id == "fills-client-1"
    assert terminal.terminal and terminal.success


@pytest.mark.parametrize(
    ("row_codes", "terminal_codes"),
    [
        (["7"], ["0"]),
        (["0", "extra"], ["0"]),
        (["0"], ["0", "extra"]),
    ],
)
def test_multipart_tracker_requires_exact_success_codes(row_codes, terminal_codes):
    tracker = MultipartResponseTracker()
    tracker.begin("strict-codes", {303})
    tracker.consume(
        303,
        {"user_msg": ["strict-codes"], "rq_handler_rp_code": row_codes},
    )

    terminal = tracker.consume(
        303,
        {"user_msg": ["strict-codes"], "rp_code": terminal_codes},
    )

    assert terminal is not None
    assert terminal.terminal
    assert terminal.success is False


def test_subscribe_before_snapshot_and_replay_live_overlap_deduplication():
    recovery = RecoveryCoordinator({"orders", "pnl"})
    recovery.begin("generation-1")
    recovery.mark_subscribed("orders")
    with pytest.raises(InvalidStateTransition):
        recovery.begin_snapshot("orders")
    recovery.mark_subscribed("pnl")
    assert recovery.phase is RecoveryPhase.BUFFERING
    recovery.record_live("orders", {"status": "open"}, ("basket", "1"))
    recovery.begin_snapshot("orders")
    recovery.record_snapshot("orders", {"status": "open"}, ("basket", "1"))
    recovery.complete_snapshot("orders")
    recovery.begin_snapshot("pnl")
    recovery.record_snapshot("pnl", {"net": 1}, ("position", "NQ"))
    recovery.record_live("pnl", {"net": 2}, ("position", "NQ", "update-2"))
    recovery.complete_snapshot("pnl")
    applied = recovery.reconcile(now=datetime(2026, 10, 7, tzinfo=timezone.utc))
    assert [item.dedupe_key for item in applied] == [
        ("basket", "1"),
        ("position", "NQ"),
        ("position", "NQ", "update-2"),
    ]
    assert recovery.ready and recovery.last_checkpoint is not None
    assert recovery.last_checkpoint.buffered_event_count == 2


def test_reconnect_backoff_is_bounded_and_deterministic_when_sampled():
    policy = ReconnectPolicy(initial_seconds=1, maximum_seconds=8, jitter_ratio=0.2)
    assert policy.delay(1, random_value=0.5) == 1
    assert policy.delay(4, random_value=0.5) == 8
    assert policy.delay(100, random_value=1) == 8


def test_recovery_history_uses_cme_trade_date_not_utc_calendar_date():
    sunday_evening = datetime(2026, 10, 4, 18, 1, tzinfo=EASTERN)
    monday_maintenance = datetime(2026, 10, 5, 17, 30, tzinfo=EASTERN)
    saturday = datetime(2026, 10, 10, 12, tzinfo=EASTERN)

    assert _cme_recovery_trade_date(sunday_evening) == 20261005
    assert _cme_recovery_trade_date(monday_maintenance) == 20261005
    assert _cme_recovery_trade_date(saturday) == 20261009
    assert _cme_recovery_trade_date(saturday, 20261231) == 20261231


class _SerializedMessage:
    template_id = int(Template.LOGIN_REQUEST)

    def SerializeToString(self) -> bytes:
        return _frame(self.template_id)


class _SilentTransport:
    def __init__(self, stop_event: asyncio.Event) -> None:
        self.stop_event = stop_event
        self.connect_count = 0
        self.close_reasons: list[str] = []

    async def connect(self) -> None:
        self.connect_count += 1
        if self.connect_count == 2:
            self.stop_event.set()

    async def send_frame(self, frame: bytes) -> int:
        return extract_template_id(frame)

    async def receive_frame(self) -> bytes:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def close(self, *, reason: str = "client shutdown") -> None:
        self.close_reasons.append(reason)


def test_silent_authentication_has_a_deadline_and_reconnects():
    async def run() -> tuple[_SilentTransport, PlantSession, list[tuple[Plant, str, str]]]:
        stop_event = asyncio.Event()
        disconnects: list[tuple[Plant, str, str]] = []
        transport = _SilentTransport(stop_event)
        session = PlantSession(
            Plant.ORDER,
            transport,  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            login_timeout_seconds=0.01,
            reconnect_policy=ReconnectPolicy(
                initial_seconds=0.001,
                maximum_seconds=0.001,
                jitter_ratio=0,
            ),
        )
        await asyncio.wait_for(
            session.run_forever(
                login_factory=_SerializedMessage,
                heartbeat_factory=_SerializedMessage,
                on_event=lambda _event: None,
                stop_event=stop_event,
                on_disconnect=lambda plant, generation, reason: disconnects.append(
                    (plant, generation, reason)
                ),
            ),
            timeout=1,
        )
        return transport, session, disconnects

    transport, session, disconnects = asyncio.run(run())
    assert transport.connect_count == 2
    assert "connection recovery" in transport.close_reasons
    assert session.state.state is SessionState.STOPPED
    assert [item[2] for item in disconnects] == [
        "connection_interrupted",
        "service_shutdown",
    ]
    assert all(item[0] is Plant.ORDER and item[1] for item in disconnects)
