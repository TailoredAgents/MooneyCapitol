from __future__ import annotations

import asyncio
import base64
import hashlib
import inspect
import os
from dataclasses import dataclass, field, fields, is_dataclass, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

from app.v2.capture.contracts import (
    CaptureEvent,
    CapturePlant,
    CaptureSource,
    EventSink,
    ObservedAccount,
    PlantHealth,
    RecoveryResult,
)
from app.v2.calendar import CmeEquityIndexCalendar, EASTERN

from .bindings import (
    ExternalBindingRegistry,
    ExternalBindingsConfig,
    PreparedBindings,
    prepare_bindings,
)
from .constants import PROTOCOL_TEMPLATE_VERSION, Plant, Template, template_name
from .dispatch import DispatchKind
from .errors import NormalizationError, ProtocolRejected, UnauthorizedAccount
from .factory import AccountAllowlist, LoginCredentials, ReadOnlyMessageFactory
from .fields import FieldView, is_exact_success_response, optional_text
from .framing import encode_message, extract_template_id
from .multipart import response_correlation
from .normalization import (
    AccountObservation,
    BrokerAccountKey,
    ExecutionEffect,
    SourceKind,
    normalize_account,
    normalize_account_pnl,
    normalize_account_rms,
    normalize_bracket,
    normalize_instrument_pnl,
    normalize_login_info,
    normalize_order,
    normalize_product_rms,
    normalize_reference,
    normalize_rms_update,
    normalize_tick_size,
)
from .recovery import stable_fingerprint
from .session import PlantSession, ReceivedMessage
from .state import SessionState
from .transport import ReadOnlyOutboundPolicy, WssTransport, create_client_ssl_context


_SCHEMA_VERSION = "rithmic-read-v1"
_ALLOWED_ACCOUNT_ACCESS_TYPES = frozenset({"0", "1", "read_only", "read_write"})
_ALLOWED_ACCOUNT_STATUSES = frozenset(
    {"active", "enabled", "admin_only", "done_for_day"}
)
_ALLOWED_USER_STATUSES = frozenset({"active", "enabled"})
_DEDUPE_TRANSPORT_FIELDS = frozenset(
    {
        "template_id",
        "request_key",
        "user_msg",
        "rq_handler_rp_code",
        "rp_code",
        "is_snapshot",
        "source_kind",
    }
)


def _authorization_token(value: str | None) -> str:
    return (value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _validated_trade_date(value: str, *, setting: str) -> int:
    try:
        parsed = datetime.strptime(value, "%Y%m%d").date()
    except ValueError as exc:
        raise ValueError(f"{setting} must be a valid YYYYMMDD trade date") from exc
    return int(parsed.strftime("%Y%m%d"))


def _cme_recovery_trade_date(now: datetime, override: int | None = None) -> int:
    """Resolve the active or most recently closed CME equity-index trade date."""

    if override is not None:
        return override
    calendar = CmeEquityIndexCalendar()
    active = calendar.trade_date_at(now)
    if active is not None:
        return int(active.strftime("%Y%m%d"))

    local_now = now.astimezone(EASTERN)
    for offset in range(8):
        candidate = local_now.date() - timedelta(days=offset)
        schedule = calendar.schedule_for(candidate)
        if (
            not schedule.holiday
            and schedule.closes_at is not None
            and schedule.closes_at <= local_now
        ):
            return int(candidate.strftime("%Y%m%d"))
    raise RuntimeError("unable to resolve a recent CME recovery trade date")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {item.name: _json_safe(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    return str(value)


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ObserverRuntimeConfig:
    discovery_uri: str
    order_uri: str | None
    pnl_uri: str | None
    ticker_uri: str | None
    gateway_name: str | None
    order_gateway_name: str | None
    pnl_gateway_name: str | None
    ticker_gateway_name: str | None
    system_name: str
    username: str = field(repr=False)
    password: str = field(repr=False)
    account_ids: frozenset[str]
    template_version: str = PROTOCOL_TEMPLATE_VERSION
    app_name: str = "MooneyCapitol"
    app_version: str = "2-read-only"
    ca_file: str | None = None
    login_timeout_seconds: float = 30.0
    request_timeout_seconds: float = 30.0
    reconcile_timeout_seconds: float = 120.0
    recovery_trade_date: int | None = None
    fill_history_start_index: int | None = None
    fill_history_finish_index: int | None = None
    account_metadata_start_ssboe: int = 0

    @classmethod
    def from_capture_config(cls, capture_config: Any) -> "ObserverRuntimeConfig":
        discovery = os.environ.get("RITHMIC_DISCOVERY_URI", "").strip()
        username = os.environ.get("RITHMIC_USERNAME", "")
        password = os.environ.get("RITHMIC_PASSWORD", "")
        system = os.environ.get("RITHMIC_SYSTEM_NAME", "").strip()
        if not discovery or not username or not password or not system:
            raise ValueError("required Rithmic TEST connection settings are missing")
        if str(getattr(capture_config, "environment", "")).upper() != "TEST":
            raise ValueError("the read-only observer may connect only to Rithmic TEST")
        if not bool(getattr(capture_config, "connectivity_enabled", False)):
            raise ValueError("Rithmic capture connectivity is disabled")
        account_ids = frozenset(str(item) for item in capture_config.account_allowlist)
        if not account_ids:
            raise ValueError("an explicit Rithmic account allowlist is required")

        start = os.getenv("RITHMIC_FILL_HISTORY_START_INDEX", "").strip()
        finish = os.getenv("RITHMIC_FILL_HISTORY_FINISH_INDEX", "").strip()
        recovery_trade_date = os.getenv("RITHMIC_RECOVERY_TRADE_DATE", "").strip()
        account_metadata_start = os.getenv(
            "RITHMIC_ACCOUNT_METADATA_START_SSBOE", "0"
        ).strip()
        try:
            account_metadata_start_ssboe = int(account_metadata_start)
        except ValueError as exc:
            raise ValueError(
                "RITHMIC_ACCOUNT_METADATA_START_SSBOE must be an integer"
            ) from exc
        if not 0 <= account_metadata_start_ssboe <= 2_147_483_647:
            raise ValueError(
                "RITHMIC_ACCOUNT_METADATA_START_SSBOE must fit an unsigned epoch value in proto int32"
            )
        request_timeout = float(os.getenv("RITHMIC_REQUEST_TIMEOUT_SECONDS", "30"))
        login_timeout = float(os.getenv("RITHMIC_LOGIN_TIMEOUT_SECONDS", "30"))
        template_version = os.getenv(
            "RITHMIC_TEMPLATE_VERSION", PROTOCOL_TEMPLATE_VERSION
        ).strip()
        if template_version != PROTOCOL_TEMPLATE_VERSION:
            raise ValueError(
                f"RITHMIC_TEMPLATE_VERSION must be {PROTOCOL_TEMPLATE_VERSION} for this runtime"
            )
        reconcile_timeout = float(
            getattr(capture_config, "reconcile_timeout_seconds", 120.0)
        )
        return cls(
            discovery_uri=discovery,
            order_uri=os.getenv("RITHMIC_ORDER_URI", "").strip() or None,
            pnl_uri=os.getenv("RITHMIC_PNL_URI", "").strip() or None,
            ticker_uri=os.getenv("RITHMIC_TICKER_URI", "").strip() or None,
            gateway_name=os.getenv("RITHMIC_GATEWAY_NAME", "").strip() or None,
            order_gateway_name=os.getenv("RITHMIC_ORDER_GATEWAY_NAME", "").strip() or None,
            pnl_gateway_name=os.getenv("RITHMIC_PNL_GATEWAY_NAME", "").strip() or None,
            ticker_gateway_name=os.getenv("RITHMIC_TICKER_GATEWAY_NAME", "").strip() or None,
            system_name=system,
            username=username,
            password=password,
            account_ids=account_ids,
            template_version=template_version,
            app_name=(
                os.getenv("RITHMIC_APPLICATION_NAME")
                or os.getenv("RITHMIC_APP_NAME")
                or "MooneyCapitol"
            ).strip()
            or "MooneyCapitol",
            app_version=(
                os.getenv("RITHMIC_APPLICATION_VERSION")
                or os.getenv("RITHMIC_APP_VERSION")
                or "2-read-only"
            ).strip()
            or "2-read-only",
            ca_file=os.getenv("RITHMIC_CA_FILE", "").strip() or None,
            login_timeout_seconds=max(login_timeout, 1.0),
            request_timeout_seconds=max(request_timeout, 1.0),
            reconcile_timeout_seconds=max(reconcile_timeout, 1.0),
            recovery_trade_date=(
                _validated_trade_date(
                    recovery_trade_date,
                    setting="RITHMIC_RECOVERY_TRADE_DATE",
                )
                if recovery_trade_date
                else None
            ),
            fill_history_start_index=(
                _validated_trade_date(
                    start,
                    setting="RITHMIC_FILL_HISTORY_START_INDEX",
                )
                if start
                else None
            ),
            fill_history_finish_index=(
                _validated_trade_date(
                    finish,
                    setting="RITHMIC_FILL_HISTORY_FINISH_INDEX",
                )
                if finish
                else None
            ),
            account_metadata_start_ssboe=account_metadata_start_ssboe,
        )


@dataclass
class _Waiter:
    plant: Plant
    generation_id: str
    response_template: int
    future: asyncio.Future[ReceivedMessage]
    account: BrokerAccountKey | None = None
    error: ProtocolRejected | None = None


@dataclass
class _PendingReconciliation:
    account: BrokerAccountKey
    batch_id: str
    generations: dict[CapturePlant, str]
    state: dict[str, Any] = field(default_factory=dict, repr=False)
    seen_event_ids: set[str] = field(default_factory=set, repr=False)
    records_seen: int = 0
    buffered_events_applied: int = 0
    discrepancy_count: int = 0
    snapshot_complete: bool = False


class RithmicReadOnlyObserver:
    """Direct R|Protocol observer implementing the capture service hooks.

    This class deliberately has no submit, modify, cancel, bracket-send,
    flatten or position-exit method. All outbound bytes also pass the
    default-deny transport allowlist.
    """

    def __init__(
        self,
        runtime: ObserverRuntimeConfig,
        capture_config: Any,
        prepared_bindings: PreparedBindings,
    ) -> None:
        self.runtime = runtime
        self.capture_config = capture_config
        self._prepared_bindings = prepared_bindings
        self.registry = ExternalBindingRegistry(prepared_bindings.path)
        self.policy = ReadOnlyOutboundPolicy()
        self.message_factory = ReadOnlyMessageFactory(
            self.registry,
            policy=self.policy,
            template_version=runtime.template_version,
        )
        self.credentials = LoginCredentials(runtime.username, runtime.password)
        self._ssl_context = create_client_ssl_context(cafile=runtime.ca_file)
        self._event_sink: EventSink | None = None
        self._stop_event = asyncio.Event()
        self._tasks: dict[Plant, asyncio.Task[None]] = {}
        self._account_discovery_task: asyncio.Task[None] | None = None
        self._sessions: dict[Plant, PlantSession] = {}
        self._waiters: dict[str, _Waiter] = {}
        self._response_rows: dict[str, list[Any]] = {}
        self._accounts: dict[str, BrokerAccountKey] = {}
        self._account_rows: dict[str, AccountObservation] = {}
        self._login_info: Any | None = None
        self._ambiguous_accounts: set[str] = set()
        self._account_metadata_observed: set[str] = set()
        self._account_metadata_denied: set[str] = set()
        self._account_metadata_clock: dict[str, tuple[int, int]] = {}
        self._accounts_discovered = asyncio.Event()
        self._subscriptions: set[tuple[str, Plant, str]] = set()
        self._active_replay_batch: dict[str, str] = {}
        self._reconcile_record_counts: dict[str, int] = {}
        self._pending_reconciliations: dict[str, _PendingReconciliation] = {}
        self._started = False

    def _build_sessions(self, gateways: Mapping[str, str]) -> None:
        selected = lambda name: self._select_gateway(gateways, name)
        endpoints: dict[Plant, str] = {
            Plant.ORDER: self.runtime.order_uri
            or selected(self.runtime.order_gateway_name or self.runtime.gateway_name),
            Plant.PNL: self.runtime.pnl_uri
            or selected(self.runtime.pnl_gateway_name or self.runtime.gateway_name),
        }
        enabled_values = {
            getattr(item, "value", str(item)).upper()
            for item in getattr(self.capture_config, "enabled_plants", ())
        }
        if "TICKER" in enabled_values:
            endpoints[Plant.TICKER] = self.runtime.ticker_uri or selected(
                self.runtime.ticker_gateway_name or self.runtime.gateway_name
            )
        for plant, endpoint in endpoints.items():
            transport = WssTransport(
                endpoint,
                policy=self.policy,
                ssl_context=self._ssl_context,
            )
            self._sessions[plant] = PlantSession(
                plant,
                transport,
                self.registry,
                required=plant in {Plant.ORDER, Plant.PNL},
                login_timeout_seconds=self.runtime.login_timeout_seconds,
            )

    async def start(self, event_sink: EventSink) -> None:
        if self._started:
            return
        self._event_sink = event_sink
        self._stop_event.clear()
        gateways = await self._verify_system_discovery()
        self._build_sessions(gateways)
        self._started = True
        for plant, session in self._sessions.items():
            task = asyncio.create_task(
                session.run_forever(
                    login_factory=lambda plant=plant: self.message_factory.login(
                        self.credentials,
                        plant=plant,
                        system_name=self.runtime.system_name,
                        app_name=self.runtime.app_name,
                        app_version=self.runtime.app_version,
                    ),
                    heartbeat_factory=self.message_factory.heartbeat,
                    on_event=self._handle_message,
                    stop_event=self._stop_event,
                    on_disconnect=self._emit_disconnect,
                ),
                name=f"rithmic-{plant.name.lower()}-read-only",
            )
            self._tasks[plant] = task

    async def stop(self) -> None:
        self._stop_event.set()
        discovery_task, self._account_discovery_task = (
            self._account_discovery_task,
            None,
        )
        if discovery_task is not None:
            discovery_task.cancel()
            await asyncio.gather(discovery_task, return_exceptions=True)
        tasks = tuple(self._tasks.values())
        self._tasks.clear()
        if tasks:
            done, pending = await asyncio.wait(tasks, timeout=10)
            del done
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
        disconnect_error: Exception | None = None
        for session in self._sessions.values():
            try:
                await self._emit_disconnect(
                    session.plant,
                    str(session.state.generation_id or ""),
                    "service_shutdown",
                )
            except Exception as exc:
                disconnect_error = disconnect_error or exc
            if session.transport.connected:
                await session.transport.close(reason="capture service shutdown")
            session.state.stop()
        for waiter in self._waiters.values():
            if not waiter.future.done():
                waiter.future.cancel()
        self._waiters.clear()
        self._pending_reconciliations.clear()
        self._active_replay_batch.clear()
        self._reconcile_record_counts.clear()
        self._prepared_bindings.close()
        self._started = False
        if disconnect_error is not None:
            raise disconnect_error

    async def _discovery_request(self, request: Any, expected_template: Template) -> Any:
        transport = WssTransport(
            self.runtime.discovery_uri,
            policy=self.policy,
            ssl_context=self._ssl_context,
        )
        try:
            await transport.connect()
            await transport.send_frame(encode_message(request).payload)
            frame = await asyncio.wait_for(
                transport.receive_frame(), timeout=self.runtime.request_timeout_seconds
            )
            if extract_template_id(frame) != expected_template:
                raise ProtocolRejected("unexpected response during Rithmic discovery")
            return self.registry.decode(frame)
        finally:
            await transport.close(reason="discovery request complete")

    async def _verify_system_discovery(self) -> Mapping[str, str]:
        system_response = await self._discovery_request(
            self.message_factory.system_info(), Template.SYSTEM_INFO_RESPONSE
        )
        system_view = FieldView(system_response)
        codes = system_view.strings("rp_code")
        systems = system_view.strings("system_name")
        if not is_exact_success_response(codes) or self.runtime.system_name not in systems:
            raise ProtocolRejected("configured Rithmic system was not discovered")

        gateway_response = await self._discovery_request(
            self.message_factory.gateway_info(self.runtime.system_name),
            Template.GATEWAY_INFO_RESPONSE,
        )
        gateway_view = FieldView(gateway_response)
        gateway_codes = gateway_view.strings("rp_code")
        names = gateway_view.strings("gateway_name")
        uris = gateway_view.strings("gateway_uri")
        if (
            not is_exact_success_response(gateway_codes)
            or not names
            or len(names) != len(uris)
        ):
            raise ProtocolRejected("Rithmic gateway discovery returned no usable gateways")
        gateways = dict(zip(names, uris, strict=True))
        if len(gateways) != len(names):
            raise ProtocolRejected("Rithmic gateway discovery returned duplicate names")
        return gateways

    @staticmethod
    def _select_gateway(gateways: Mapping[str, str], requested_name: str | None) -> str:
        if requested_name:
            try:
                return gateways[requested_name]
            except KeyError as exc:
                raise ProtocolRejected("configured Rithmic gateway name was not discovered") from exc
        if len(gateways) != 1:
            raise ProtocolRejected(
                "multiple Rithmic gateways were discovered; configure an explicit gateway name"
            )
        return next(iter(gateways.values()))

    async def discover_accounts(self) -> Sequence[ObservedAccount]:
        # The capture coordinator owns the overall discovery/reconciliation
        # budget. Each protocol request is bounded independently below; adding
        # another single-request deadline here would reject a valid four-step
        # discovery sequence.
        await self._accounts_discovered.wait()
        if self._ambiguous_accounts.intersection(self.runtime.account_ids):
            raise UnauthorizedAccount("an allowlisted account ID was ambiguous across FCM/IB identities")
        return tuple(ObservedAccount(account_id=item) for item in sorted(self._accounts))

    async def prepare_reconciliation(
        self,
        account_id: str,
        generations: Mapping[CapturePlant, str],
    ) -> None:
        """Open account recovery provenance before any live subscription.

        The capture service prepares every account before its subscription
        pass. Consequently, live events received while another account is
        snapshotting still carry this account's recovery batch and generation
        map when the durable buffer is later drained.
        """

        account = self._require_account(account_id)
        self._assert_current_generations(generations)
        current = self._pending_reconciliations.get(account_id)
        if current is not None:
            if current.generations != dict(generations):
                raise ConnectionError(
                    "account recovery was already prepared for another generation"
                )
            return
        batch_id = str(uuid4())
        self._pending_reconciliations[account_id] = _PendingReconciliation(
            account=account,
            batch_id=batch_id,
            generations=dict(generations),
        )
        self._active_replay_batch[account_id] = batch_id
        self._reconcile_record_counts[account_id] = 0
        try:
            await self._emit_recovery_start(
                account_id,
                account,
                batch_id,
                generations,
            )
        except BaseException:
            self._pending_reconciliations.pop(account_id, None)
            self._active_replay_batch.pop(account_id, None)
            self._reconcile_record_counts.pop(account_id, None)
            raise

    def _start_account_discovery(self, generation_id: str) -> None:
        if self._account_discovery_task is not None:
            self._account_discovery_task.cancel()
        self._account_discovery_task = asyncio.create_task(
            self._discover_order_accounts(generation_id),
            name="rithmic-order-account-discovery",
        )

    async def _discover_order_accounts(self, generation_id: str) -> None:
        session = self._sessions[Plant.ORDER]
        try:
            response = await self._send_and_wait(
                session,
                self.message_factory.login_info(),
                Template.LOGIN_INFO_RESPONSE,
            )
            self._assert_current_generations(
                {CapturePlant.ORDER: generation_id}
            )
            login_info = normalize_login_info(response.dispatch.message)
            self._login_info = login_info
            if not login_info.fcm_id or not login_info.ib_id:
                raise ProtocolRejected("Rithmic login info omitted FCM/IB identity")
            if not login_info.user:
                raise ProtocolRejected("Rithmic login info omitted user identity")
            if login_info.user_type_code not in {1, 2, 3}:
                raise ProtocolRejected("Rithmic login info returned unsupported user type")
            await self._send_and_wait(
                session,
                self.message_factory.user_info(
                    fcm_id=login_info.fcm_id,
                    ib_id=login_info.ib_id,
                    user=login_info.user,
                ),
                Template.USER_INFO_RESPONSE,
            )
            self._assert_current_generations(
                {CapturePlant.ORDER: generation_id}
            )
            login_info = self._login_info or login_info
            await self._send_and_wait(
                session,
                self.message_factory.account_list(
                    fcm_id=login_info.fcm_id,
                    ib_id=login_info.ib_id,
                    user_type=login_info.user_type_code,
                ),
                Template.ACCOUNT_LIST_RESPONSE,
            )
            self._assert_current_generations(
                {CapturePlant.ORDER: generation_id}
            )
            await self._send_and_wait(
                session,
                self.message_factory.playback_account_users(
                    start_index=self.runtime.account_metadata_start_ssboe,
                    playback="accounts",
                ),
                Template.PLAYBACK_ACCOUNT_USERS_RESPONSE,
            )
            self._assert_current_generations(
                {CapturePlant.ORDER: generation_id}
            )
            required_metadata = {
                account_id
                for account_id in self.runtime.account_ids
                if account_id in self._accounts
                and account_id not in self._ambiguous_accounts
            }
            if required_metadata & self._account_metadata_denied:
                raise ProtocolRejected(
                    "Rithmic account metadata denied configured account access"
                )
            if required_metadata - self._account_metadata_observed:
                raise ProtocolRejected(
                    "Rithmic account metadata playback omitted access or status"
                )
            self._accounts_discovered.set()
        except asyncio.CancelledError:
            raise
        except (ProtocolRejected, UnauthorizedAccount, ValueError):
            if str(session.state.generation_id or "") == generation_id:
                session.state.rejected("account discovery rejected")
                await session.transport.close(reason="account discovery rejected")
        except Exception:
            if str(session.state.generation_id or "") == generation_id:
                session.state.connection_lost("account discovery interrupted")
                await session.transport.close(reason="account discovery recovery")

    async def subscribe_account(self, account_id: str, plant: CapturePlant) -> None:
        key = self._require_account(account_id)
        protocol_plant = Plant[plant.value]
        session = self._sessions.get(protocol_plant)
        if session is None or not session.state.authenticated:
            raise ConnectionError(f"{plant.value} Plant is not authenticated")
        generation = str(session.state.generation_id or "")
        marker = (account_id, protocol_plant, generation)
        if marker in self._subscriptions:
            return
        if protocol_plant is Plant.ORDER:
            await self._send_and_wait(
                session,
                self.message_factory.subscribe_orders(key),
                Template.ORDER_UPDATES_RESPONSE,
            )
            await self._send_and_wait(
                session,
                self.message_factory.subscribe_brackets(key),
                Template.BRACKET_UPDATES_RESPONSE,
            )
            await self._send_and_wait(
                session,
                self.message_factory.subscribe_rms(key),
                Template.RMS_UPDATES_RESPONSE,
            )
        elif protocol_plant is Plant.PNL:
            await self._send_and_wait(
                session,
                self.message_factory.subscribe_pnl(key),
                Template.PNL_UPDATES_RESPONSE,
            )
        else:
            # Reference data is request/response and has no account stream.
            return
        self._subscriptions.add(marker)

    async def reconcile_account(
        self,
        account_id: str,
        generations: Mapping[CapturePlant, str],
    ) -> RecoveryResult:
        key = self._require_account(account_id)
        self._assert_current_generations(generations)
        required = (Plant.ORDER, Plant.PNL)
        for plant in required:
            session = self._sessions[plant]
            marker = (account_id, plant, str(session.state.generation_id or ""))
            if marker not in self._subscriptions:
                return RecoveryResult(False, "", blocker="subscribe_before_snapshot_required")
        if self._login_info is None or self._login_info.user_type_code not in {1, 2, 3}:
            raise ProtocolRejected("authenticated Rithmic user type is unavailable")
        user_type_code = self._login_info.user_type_code
        trade_date = _cme_recovery_trade_date(
            _utc_now(), self.runtime.recovery_trade_date
        )
        start_index = self.runtime.fill_history_start_index or trade_date
        finish_index = self.runtime.fill_history_finish_index or trade_date
        if start_index > finish_index:
            raise ProtocolRejected("fill-history trade-date range is invalid")
        order = self._sessions[Plant.ORDER]
        pnl = self._sessions[Plant.PNL]
        request_specs = (
            (order, self.message_factory.show_orders(key), Template.SHOW_ORDERS_RESPONSE, None),
            (
                order,
                self.message_factory.replay_executions(key),
                Template.REPLAY_EXECUTIONS_RESPONSE,
                None,
            ),
            (
                order,
                self.message_factory.order_history_summary(key, str(trade_date)),
                Template.ORDER_HISTORY_SUMMARY_RESPONSE,
                None,
            ),
            (
                order,
                self.message_factory.fill_history(
                    key,
                    start_index=start_index,
                    finish_index=finish_index,
                    index_format="trade_date",
                ),
                Template.FILL_HISTORY_RESPONSE,
                None,
            ),
            (
                order,
                self.message_factory.show_brackets(key),
                Template.SHOW_BRACKETS_RESPONSE,
                None,
            ),
            (
                order,
                self.message_factory.show_bracket_stops(key),
                Template.SHOW_BRACKET_STOPS_RESPONSE,
                None,
            ),
            (
                order,
                self.message_factory.account_rms(
                    key,
                    user_type=user_type_code,
                ),
                Template.ACCOUNT_RMS_RESPONSE,
                key,
            ),
            (
                order,
                self.message_factory.product_rms(key),
                Template.PRODUCT_RMS_RESPONSE,
                None,
            ),
            (
                pnl,
                self.message_factory.pnl_snapshot(key),
                Template.PNL_SNAPSHOT_RESPONSE,
                None,
            ),
        )
        await self.prepare_reconciliation(account_id, generations)
        pending = self._pending_reconciliations[account_id]
        batch_id = pending.batch_id
        try:
            for plant in required:
                session = self._sessions[plant]
                if session.state.state in {SessionState.CONNECTED, SessionState.READY}:
                    session.state.begin_reconciliation()
            # The capture coordinator owns the per-account recovery deadline.
            # Individual request waits remain bounded here; a second identical
            # outer deadline would always win the race and cancel this method
            # before it could journal a FAILED replay boundary.
            await asyncio.gather(
                *(
                    self._send_and_wait(
                        session,
                        message,
                        response_template,
                        account=request_account,
                    )
                    for session, message, response_template, request_account in request_specs
                )
            )
            self._assert_current_generations(generations)
            pending.snapshot_complete = True
            checkpoint = self._reconciliation_digest(pending)
            return RecoveryResult(
                True,
                checkpoint,
                records_seen=pending.records_seen,
                discrepancy_count=pending.discrepancy_count,
            )
        except asyncio.CancelledError:
            await asyncio.shield(
                self._fail_account_reconciliation(
                    account_id,
                    key,
                    batch_id,
                    generations,
                    required,
                )
            )
            raise
        except Exception:
            await self._fail_account_reconciliation(
                account_id,
                key,
                batch_id,
                generations,
                required,
            )
            return RecoveryResult(
                False,
                "",
                discrepancy_count=pending.discrepancy_count,
                blocker="rithmic_reconciliation_failed",
            )

    async def _fail_account_reconciliation(
        self,
        account_id: str,
        key: BrokerAccountKey,
        batch_id: str,
        generations: Mapping[CapturePlant, str],
        required: Sequence[Plant],
    ) -> None:
        closures = []
        for plant in required:
            session = self._sessions[plant]
            expected_generation = generations.get(self._capture_plant(plant), "")
            if str(session.state.generation_id or "") != expected_generation:
                continue
            self._fail_waiters(plant, expected_generation)
            if session.state.state is not SessionState.DISCONNECTED:
                session.state.connection_lost("reconciliation failed")
            closures.append(
                session.transport.close(reason="reconciliation recovery")
            )
        # Closing both sockets unblocks receive loops and establishes new,
        # independent generations. A failed terminal boundary is journaled even
        # when the service deadline cancelled the recovery coroutine.
        if closures:
            await asyncio.gather(*closures, return_exceptions=True)
        try:
            await self._emit_recovery_completion(
                account_id,
                key,
                batch_id,
                generations,
                clean=False,
                checkpoint="",
            )
        finally:
            self._pending_reconciliations.pop(account_id, None)
            self._active_replay_batch.pop(account_id, None)
            self._reconcile_record_counts.pop(account_id, None)

    async def apply_buffered(
        self,
        account_id: str,
        events: Sequence[CaptureEvent],
    ) -> None:
        self._require_account(account_id)
        try:
            pending = self._pending_reconciliations[account_id]
        except KeyError as exc:
            raise RuntimeError("buffered events arrived without an active recovery") from exc
        self._assert_current_generations(pending.generations)
        if not pending.snapshot_complete:
            raise RuntimeError("buffered events cannot precede snapshot completion")
        for event in events:
            self._fold_reconciliation_event(pending, event)
            pending.buffered_events_applied += 1

    async def finalize_reconciliation(
        self,
        account_id: str,
        generations: Mapping[CapturePlant, str],
    ) -> RecoveryResult:
        key = self._require_account(account_id)
        try:
            pending = self._pending_reconciliations[account_id]
        except KeyError as exc:
            raise RuntimeError("account recovery was not prepared") from exc
        if pending.generations != dict(generations):
            pending.discrepancy_count += 1
        try:
            self._assert_current_generations(generations)
            if not pending.snapshot_complete:
                pending.discrepancy_count += 1
            if pending.discrepancy_count:
                await self._fail_account_reconciliation(
                    account_id,
                    key,
                    pending.batch_id,
                    pending.generations,
                    (Plant.ORDER, Plant.PNL),
                )
                return RecoveryResult(
                    False,
                    "",
                    records_seen=pending.records_seen,
                    discrepancy_count=pending.discrepancy_count,
                    blocker="reconciliation_state_discrepancy",
                )
            checkpoint = self._reconciliation_digest(pending)
            # The service persists the checkpoint after this call.  No clean
            # terminal event is emitted here: the durable checkpoint is the
            # only operation allowed to close the replay batch as COMPLETE.
            sessions = [self._sessions[plant] for plant in (Plant.ORDER, Plant.PNL)]
            if any(
                session.state.state is not SessionState.RECONCILING
                for session in sessions
            ):
                raise ConnectionError(
                    "Rithmic plant left reconciliation before the state fold completed"
                )
            for session in sessions:
                session.state.reconciliation_succeeded()
            return RecoveryResult(
                True,
                checkpoint,
                records_seen=pending.records_seen,
                discrepancy_count=0,
            )
        finally:
            self._pending_reconciliations.pop(account_id, None)
            self._active_replay_batch.pop(account_id, None)
            self._reconcile_record_counts.pop(account_id, None)

    async def abort_recovery(
        self,
        generations: Mapping[CapturePlant, str],
    ) -> None:
        """Invalidate only the affected generations and let sessions reconnect."""

        affected_accounts = {
            account_id
            for account_id, pending in self._pending_reconciliations.items()
            if any(
                pending.generations.get(plant) == generation
                for plant, generation in generations.items()
            )
        }
        closures = []
        for capture_plant, expected_generation in generations.items():
            plant = Plant[capture_plant.value]
            session = self._sessions.get(plant)
            if session is None:
                continue
            if str(session.state.generation_id or "") != expected_generation:
                continue
            self._fail_waiters(plant, expected_generation)
            if session.state.state is not SessionState.DISCONNECTED:
                session.state.connection_lost("capture recovery aborted")
            closures.append(
                session.transport.close(reason="capture recovery aborted")
            )
        # Recovery batches are account-scoped but depend on both required plant
        # generations. Drop only work tied to the attempted generation so a
        # stale abort cannot erase freshly prepared recovery.
        for account_id in affected_accounts:
            self._pending_reconciliations.pop(account_id, None)
            self._active_replay_batch.pop(account_id, None)
            self._reconcile_record_counts.pop(account_id, None)
        if closures:
            await asyncio.gather(*closures, return_exceptions=True)

    def plant_health(self) -> Mapping[CapturePlant, PlantHealth]:
        result: dict[CapturePlant, PlantHealth] = {}
        mapping = {Plant.ORDER: CapturePlant.ORDER, Plant.PNL: CapturePlant.PNL, Plant.TICKER: CapturePlant.TICKER}
        for plant, session in self._sessions.items():
            health = session.state.snapshot()
            blocker = None
            if session.state.state in {
                SessionState.DEGRADED,
                SessionState.FORCED_LOGOUT,
                SessionState.DISCONNECTED,
            }:
                blocker = session.state.state.value.lower()
            result[mapping[plant]] = PlantHealth(
                plant=mapping[plant],
                connected=health.connected,
                authenticated=health.authenticated,
                generation_id=health.generation_id,
                last_message_at=health.last_message_at,
                reconnecting=session.state.state in {SessionState.CONNECTING, SessionState.DISCONNECTED},
                blocker=blocker,
            )
        return result

    async def reference_data(self, symbol: str, exchange: str) -> Any:
        session = self._sessions.get(Plant.TICKER)
        if session is None or not session.state.authenticated:
            raise ConnectionError("optional Ticker Plant is not authenticated")
        response = await self._send_and_wait(
            session,
            self.message_factory.reference_data(symbol, exchange),
            Template.REFERENCE_DATA_RESPONSE,
        )
        return normalize_reference(response.dispatch.message)

    async def search_symbols(
        self,
        search_text: str,
        *,
        exchange: str | None = None,
        product_code: str | None = None,
    ) -> tuple[Mapping[str, Any], ...]:
        session = self._require_ticker_session()
        correlation = self.message_factory.correlation_id("symbol-search")
        self._response_rows[correlation] = []
        try:
            await self._send_and_wait(
                session,
                self.message_factory.search_symbols(
                    search_text,
                    exchange=exchange,
                    product_code=product_code,
                    correlation_id=correlation,
                ),
                Template.SEARCH_SYMBOLS_RESPONSE,
            )
            rows = tuple(self._response_rows.get(correlation, ()))
        finally:
            self._response_rows.pop(correlation, None)
        return tuple(self._message_fields(message) for message in rows)

    async def tick_size_table(self, tick_size_type: str) -> tuple[Any, ...]:
        session = self._require_ticker_session()
        correlation = self.message_factory.correlation_id("tick-table")
        self._response_rows[correlation] = []
        try:
            await self._send_and_wait(
                session,
                self.message_factory.tick_size_table(
                    tick_size_type, correlation_id=correlation
                ),
                Template.TICK_SIZE_TABLE_RESPONSE,
            )
            rows = tuple(self._response_rows.get(correlation, ()))
        finally:
            self._response_rows.pop(correlation, None)
        return tuple(normalize_tick_size(message) for message in rows)

    def _require_ticker_session(self) -> PlantSession:
        session = self._sessions.get(Plant.TICKER)
        if session is None or not session.state.authenticated:
            raise ConnectionError("optional Ticker Plant is not authenticated")
        return session

    async def _send_and_wait(
        self,
        session: PlantSession,
        message: Any,
        response_template: int | Template,
        *,
        account: BrokerAccountKey | None = None,
    ) -> ReceivedMessage:
        view = FieldView(message)
        user_messages = view.strings("user_msg")
        if not user_messages:
            raise ValueError("tracked request requires user_msg correlation")
        correlation = user_messages[-1]
        future: asyncio.Future[ReceivedMessage] = asyncio.get_running_loop().create_future()
        request_account = account or self._account_from_view(view)
        self._waiters[correlation] = _Waiter(
            session.plant,
            str(session.state.generation_id or ""),
            int(response_template),
            future,
            request_account,
        )
        try:
            await session.send(message)
            try:
                return await asyncio.wait_for(
                    future, timeout=self.runtime.request_timeout_seconds
                )
            except TimeoutError:
                # A timed-out multipart stream no longer has a trustworthy
                # terminal boundary. Reset the plant so late frames cannot be
                # mistaken for unsolicited live observations or a later query.
                await session.transport.close(reason="tracked request timed out")
                raise
        finally:
            self._waiters.pop(correlation, None)

    async def _handle_message(self, received: ReceivedMessage) -> None:
        dispatch = received.dispatch
        if dispatch.kind is DispatchKind.LOGIN_SUCCEEDED:
            self._drop_generation_state(received.plant, received.generation_id)
            await self._emit_control(received)
            if received.plant is Plant.ORDER:
                # Request/response waits run outside the receive-loop callback,
                # so terminal rp_code values and timeouts are enforced without
                # deadlocking dispatch.
                self._start_account_discovery(received.generation_id)
            return
        if dispatch.kind is DispatchKind.HEARTBEAT:
            await self._emit_control(received)
            return
        if dispatch.kind in {DispatchKind.REJECT, DispatchKind.FORCED_LOGOUT, DispatchKind.LOGIN_FAILED}:
            self._fail_waiters(received.plant, received.generation_id)
            await self._emit_control(received)
            return

        message = dispatch.message
        correlation = response_correlation(message)
        message_view = FieldView(message)
        row_codes = message_view.strings("rq_handler_rp_code")
        terminal_codes = message_view.strings("rp_code")
        invalid_row = bool(row_codes) and not is_exact_success_response(row_codes)
        invalid_terminal = bool(terminal_codes) and not is_exact_success_response(
            terminal_codes
        )
        waiter = self._waiters.get(correlation) if correlation else None
        matching_waiter = bool(
            waiter
            and waiter.plant is received.plant
            and waiter.generation_id == received.generation_id
            and waiter.response_template == received.template_id
        )

        if matching_waiter and invalid_row and waiter.error is None:
            # A rejected intermediate row taints the whole request, but the
            # multipart stream must remain registered until its terminal frame
            # is drained. Otherwise later rows can escape request ownership.
            waiter.error = ProtocolRejected("read-only request was rejected")
        if matching_waiter and invalid_terminal and waiter.error is None:
            waiter.error = ProtocolRejected("read-only request was rejected")

        if matching_waiter and row_codes:
            if waiter.error is None and correlation in self._response_rows:
                self._response_rows[correlation].append(message)
        if matching_waiter and terminal_codes and not waiter.future.done():
            if waiter.error is not None:
                waiter.future.set_exception(waiter.error)
            else:
                waiter.future.set_result(received)

        if invalid_row or invalid_terminal or (matching_waiter and waiter.error is not None):
            # Error rows and every later row in a tainted stream are drained,
            # but never normalized, journaled, or folded into recovered state.
            return

        if received.template_id == Template.LOGIN_INFO_RESPONSE:
            login_info = normalize_login_info(message)
            self._login_info = login_info
            return

        if received.template_id == Template.USER_INFO_RESPONSE:
            user_info = normalize_login_info(message)
            if self._login_info is None:
                self._login_info = user_info
            else:
                current = self._login_info
                self._login_info = replace(
                    current,
                    fcm_id=user_info.fcm_id or current.fcm_id,
                    ib_id=user_info.ib_id or current.ib_id,
                    user=user_info.user or current.user,
                    user_type=user_info.user_type or current.user_type,
                    user_type_code=user_info.user_type_code
                    if user_info.user_type_code is not None
                    else current.user_type_code,
                    status=user_info.status or current.status,
                    order_copy_status=user_info.order_copy_status
                    or current.order_copy_status,
                    ticker_session_max=user_info.ticker_session_max
                    if user_info.ticker_session_max is not None
                    else current.ticker_session_max,
                    order_session_max=user_info.order_session_max
                    if user_info.order_session_max is not None
                    else current.order_session_max,
                    country_code=user_info.country_code or current.country_code,
                    state_code=user_info.state_code or current.state_code,
                    sensitive_metadata={
                        **current.sensitive_metadata,
                        **user_info.sensitive_metadata,
                    },
                )
            return

        if received.template_id == Template.ACCOUNT_LIST_RESPONSE:
            try:
                account = normalize_account(message)
            except ValueError:
                account = None
            if account is not None:
                if self._login_info is not None:
                    if (
                        account.key.fcm_id != self._login_info.fcm_id
                        or account.key.ib_id != self._login_info.ib_id
                    ):
                        self._ambiguous_accounts.add(account.key.account_id)
                        raise UnauthorizedAccount(
                            "account discovery returned an unexpected FCM/IB identity"
                        )
                    account = replace(
                        account,
                        user_id=self._login_info.user,
                        user_type=self._login_info.user_type,
                        user_status=self._login_info.status,
                        order_copy_status=self._login_info.order_copy_status,
                        ticker_session_max=self._login_info.ticker_session_max,
                        order_session_max=self._login_info.order_session_max,
                        country_code=self._login_info.country_code,
                        state_code=self._login_info.state_code,
                        sensitive_metadata=self._login_info.sensitive_metadata,
                    )
                existing = self._accounts.get(account.key.account_id)
                if existing is not None and existing != account.key:
                    self._ambiguous_accounts.add(account.key.account_id)
                else:
                    self._accounts[account.key.account_id] = account.key
                    self._account_rows[account.key.account_id] = account
                if account.key.account_id in self.runtime.account_ids:
                    await self._emit(received, account, "ACCOUNT", account.key, CaptureSource.SYSTEM)
            terminal_codes = FieldView(message).strings("rp_code")
            if is_exact_success_response(terminal_codes):
                allowed_keys = [
                    key
                    for account_id, key in self._accounts.items()
                    if account_id in self.runtime.account_ids and account_id not in self._ambiguous_accounts
                ]
                if allowed_keys:
                    self.message_factory.account_allowlist = AccountAllowlist(allowed_keys)
            return

        if received.template_id == Template.PLAYBACK_ACCOUNT_USERS_RESPONSE:
            await self._capture_account_metadata_playback(received)
            return

        if received.template_id in {
            Template.USER_ACCOUNT_UPDATE,
            Template.USER_INFO_UPDATE,
            Template.ACCOUNT_AND_USER_UPDATE,
        }:
            await self._capture_user_account_update(received)
            return

        await self._normalize_and_emit(received)

    async def _normalize_and_emit(self, received: ReceivedMessage) -> None:
        message = received.dispatch.message
        template_id = received.template_id
        observation: Any | None = None
        observation_type = "UNKNOWN"
        account: BrokerAccountKey | None = None
        source = CaptureSource.LIVE
        correlation = response_correlation(message)
        correlated_waiter = self._waiters.get(correlation) if correlation else None
        correlated_account = (
            correlated_waiter.account
            if correlated_waiter is not None
            and correlated_waiter.plant is received.plant
            and correlated_waiter.generation_id == received.generation_id
            else None
        )
        try:
            if template_id in {
                Template.RITHMIC_ORDER_NOTIFICATION,
                Template.EXCHANGE_ORDER_NOTIFICATION,
                Template.FILL_HISTORY_RESPONSE,
            }:
                observation = normalize_order(message, template_id=template_id)
                account = observation.account
                observation_type = (
                    "ORDER_EXECUTION"
                    if observation.execution_effect is not ExecutionEffect.NONE
                    else "ORDER"
                )
                source = self._capture_source(observation.source_kind)
            elif template_id == Template.INSTRUMENT_PNL_UPDATE:
                observation = normalize_instrument_pnl(message)
                account = observation.account
                observation_type = "POSITION"
                source = self._capture_source(observation.source_kind)
            elif template_id == Template.ACCOUNT_PNL_UPDATE:
                observation = normalize_account_pnl(message)
                account = observation.account
                observation_type = "ACCOUNT_PNL"
                source = self._capture_source(observation.source_kind)
            elif template_id == Template.ACCOUNT_RMS_RESPONSE:
                observation = normalize_account_rms(message)
                account = observation.account
                if correlated_account is not None and account != correlated_account:
                    # Account RMS requests are FCM/IB/user scoped and may return
                    # rows for multiple accounts. Each sequential account
                    # recovery owns only its exact row; other rows are captured
                    # when that account's own request/batch runs.
                    return
                observation_type = "ACCOUNT_RMS"
                source = CaptureSource.SNAPSHOT
            elif template_id == Template.PRODUCT_RMS_RESPONSE:
                observation = normalize_product_rms(message)
                account = observation.account
                observation_type = "PRODUCT_RMS"
                source = CaptureSource.SNAPSHOT
            elif template_id == Template.ACCOUNT_RMS_UPDATE:
                observation = normalize_rms_update(message)
                account = observation.account
                observation_type = "ACCOUNT_RMS"
            elif template_id in {
                Template.BRACKET_UPDATE,
                Template.SHOW_BRACKETS_RESPONSE,
                Template.SHOW_BRACKET_STOPS_RESPONSE,
            }:
                observation = normalize_bracket(message)
                account = observation.account
                observation_type = "BRACKET"
                source = (
                    CaptureSource.LIVE
                    if template_id == Template.BRACKET_UPDATE
                    else CaptureSource.SNAPSHOT
                )
            elif template_id == Template.REFERENCE_DATA_RESPONSE:
                observation = normalize_reference(message)
                observation_type = "REFERENCE"
                source = CaptureSource.SNAPSHOT
            elif template_id == Template.TICK_SIZE_TABLE_RESPONSE:
                observation = normalize_tick_size(message)
                observation_type = "REFERENCE_TICK_SIZE"
                source = CaptureSource.SNAPSHOT
            elif template_id == Template.SEARCH_SYMBOLS_RESPONSE:
                observation = self._message_fields(message)
                observation_type = "REFERENCE_SEARCH"
                source = CaptureSource.SNAPSHOT
            elif template_id in {
                Template.SHOW_ORDER_HISTORY_RESPONSE,
                Template.ORDER_HISTORY_SUMMARY_RESPONSE,
                Template.ORDER_HISTORY_DETAIL_RESPONSE,
            }:
                observation = self._message_fields(message)
                observation_type = "ORDER_HISTORY"
                account = correlated_account
                source = CaptureSource.HISTORY
        except (NormalizationError, TypeError, ValueError):
            # Preserve the raw fingerprint/frame and unknown classification;
            # never invent values from a malformed or newly extended message.
            observation = None
            observation_type = "UNKNOWN"
            account = self._account_from_view(FieldView(message))

        account = account or correlated_account
        if observation_type in {
            "REFERENCE",
            "REFERENCE_SEARCH",
            "REFERENCE_TICK_SIZE",
        }:
            meaningful = _json_safe(observation)
            if isinstance(meaningful, Mapping):
                ignored = {
                    "template_id",
                    "user_msg",
                    "rq_handler_rp_code",
                    "rp_code",
                }
                if not any(value is not None and value != [] for key, value in meaningful.items() if key not in ignored):
                    return
            await self._emit(
                received,
                observation,
                observation_type,
                None,
                source,
            )
            return
        if account is None or account.account_id not in self.runtime.account_ids:
            return
        expected = self._accounts.get(account.account_id)
        if expected is None or expected != account:
            raise UnauthorizedAccount(
                "inbound Rithmic account identity did not match discovery"
            )
        await self._emit(received, observation, observation_type, account, source)

    async def _capture_account_metadata_playback(
        self, received: ReceivedMessage
    ) -> None:
        """Journal authoritative account metadata without broadening access.

        ResponsePlaybackAccUsers is a mixed account/user history stream.  This
        observer requests only account records and applies rows solely to an
        exact FCM/IB/account identity already discovered for an allowlisted
        account.  Every row is journaled; only the newest timestamped row is
        used for the current account projection.
        """

        view = FieldView(received.dispatch.message)
        account = self._account_from_view(view)
        if account is None or account.account_id not in self.runtime.account_ids:
            return
        expected = self._accounts.get(account.account_id)
        if expected is None or expected != account:
            raise UnauthorizedAccount(
                "account metadata identity did not match account discovery"
            )
        row = self._account_rows.get(account.account_id)
        if row is None:
            raise ProtocolRejected(
                "account metadata arrived before the matching account-list row"
            )

        try:
            clock = (
                int(view.get("ssboe")) if view.get("ssboe") is not None else -1,
                int(view.get("usecs")) if view.get("usecs") is not None else -1,
            )
        except (TypeError, ValueError) as exc:
            raise ProtocolRejected(
                "Rithmic account metadata contained an invalid source timestamp"
            ) from exc

        sensitive_names = (
            "first_name",
            "last_name",
            "email_address",
            "address_street_1",
            "address_street_2",
            "address_city",
            "address_state",
            "address_country",
            "address_zip",
            "phone_residence",
            "phone_work",
            "phone_mobile",
        )
        sensitive = {
            name: value
            for name in sensitive_names
            if (value := optional_text(view.get(name))) is not None
        }

        current_user = optional_text(getattr(self._login_info, "user", None))
        record_user = optional_text(view.get("user"))
        applies_to_current_user = record_user is None or record_user == current_user
        access_type = (
            optional_text(view.first("account_access_type", "access_type"))
            if applies_to_current_user
            else None
        )
        update_type = (optional_text(view.get("update_type")) or "").strip().lower()

        if clock >= self._account_metadata_clock.get(account.account_id, (-1, -1)):
            def count(name: str, fallback: int | None) -> int | None:
                if not applies_to_current_user:
                    return fallback
                value = view.get(name)
                return int(value) if value is not None else fallback

            updated = replace(
                row,
                account_name=optional_text(view.get("account_name")) or row.account_name,
                account_status=(
                    optional_text(view.get("account_status"))
                    if applies_to_current_user
                    else None
                )
                or row.account_status,
                access_type=access_type or row.access_type,
                user_id=(record_user if applies_to_current_user else None) or row.user_id,
                user_type=(
                    optional_text(view.get("type")) if applies_to_current_user else None
                )
                or row.user_type,
                user_status=(
                    optional_text(view.get("status"))
                    if applies_to_current_user
                    else None
                )
                or row.user_status,
                order_copy_status=(
                    optional_text(view.get("order_copy_status"))
                    if applies_to_current_user
                    else None
                )
                or row.order_copy_status,
                ticker_session_max=count(
                    "tp_max_session_count", row.ticker_session_max
                ),
                order_session_max=count(
                    "op_max_session_count", row.order_session_max
                ),
                country_code=(
                    optional_text(view.get("country_code"))
                    if applies_to_current_user
                    else None
                )
                or row.country_code,
                state_code=(
                    optional_text(view.get("state_code"))
                    if applies_to_current_user
                    else None
                )
                or row.state_code,
                sensitive_metadata=(
                    {**row.sensitive_metadata, **sensitive}
                    if applies_to_current_user
                    else row.sensitive_metadata
                ),
            )
            self._account_rows[account.account_id] = updated
            self._account_metadata_clock[account.account_id] = clock
            if applies_to_current_user:
                access_state = _authorization_token(updated.access_type)
                account_state = _authorization_token(updated.account_status)
                user_state = _authorization_token(updated.user_status)
                explicit_removal = update_type in {
                    "remove_account_from_user",
                    "remove_account",
                    "delete_account",
                    "disable_account",
                }
                explicitly_disabled_user = (
                    bool(user_state) and user_state not in _ALLOWED_USER_STATUSES
                )
                authorized = (
                    not explicit_removal
                    and access_state in _ALLOWED_ACCOUNT_ACCESS_TYPES
                    and account_state in _ALLOWED_ACCOUNT_STATUSES
                    and not explicitly_disabled_user
                )
                if authorized:
                    self._account_metadata_denied.discard(account.account_id)
                    self._account_metadata_observed.add(account.account_id)
                else:
                    self._account_metadata_denied.add(account.account_id)
                    self._account_metadata_observed.discard(account.account_id)
            row = updated

        await self._emit(
            received,
            row,
            "ACCOUNT",
            account,
            CaptureSource.HISTORY,
        )

    async def _emit_recovery_completion(
        self,
        account_id: str,
        account: BrokerAccountKey,
        batch_id: str,
        generations: Mapping[CapturePlant, str],
        *,
        clean: bool,
        checkpoint: str,
    ) -> None:
        if self._event_sink is None:
            return
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "template_id": 0,
            "template_name": "LOCAL_RECOVERY_CHECKPOINT",
            "observation_type": "RECOVERY_CHECKPOINT",
            "source": CaptureSource.REPLAY.value,
            "broker_identity": {
                "fcm_id": account.fcm_id,
                "ib_id": account.ib_id,
                "account_id": account.account_id,
            },
            "native_identity": {},
            "request_key": None,
            "user_msg": [],
            "replay_batch_id": batch_id,
            "batch_kind": "ACCOUNT_RECOVERY",
            # A clean batch observed every required terminal response.  On
            # failure the exact boundary may be a reject, timeout, disconnect,
            # or partial multipart response, so do not invent a terminal ACK.
            "terminal_response_received": clean,
            "batch_status": "COMPLETE" if clean else "FAILED",
            "failure_reason": None if clean else "reconciliation_failed",
            "clean": clean,
            "checkpoint": checkpoint,
            "records_seen": self._reconcile_record_counts.get(account_id, 0),
            "generations": {plant.value: value for plant, value in generations.items()},
            "dedupe_key": f"recovery:{batch_id}:terminal",
            "normalized": {},
            "raw_frame_sha256": hashlib.sha256(
                f"recovery:{batch_id}:{clean}".encode("ascii")
            ).hexdigest(),
            "raw_frame_b64": "",
        }
        event = CaptureEvent(
            event_id=hashlib.sha256(
                f"{account.fcm_id}\0{account.ib_id}\0{account.account_id}\0{batch_id}\0terminal".encode()
            ).hexdigest(),
            account_id=account_id,
            plant=CapturePlant.ORDER,
            source=CaptureSource.REPLAY,
            generation_id=generations.get(CapturePlant.ORDER, ""),
            payload=payload,
        )
        result = self._event_sink(event)
        if inspect.isawaitable(result):
            await result

    async def _emit_recovery_start(
        self,
        account_id: str,
        account: BrokerAccountKey,
        batch_id: str,
        generations: Mapping[CapturePlant, str],
    ) -> None:
        if self._event_sink is None:
            return
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "template_id": 0,
            "template_name": "LOCAL_RECOVERY_START",
            "observation_type": "RECOVERY_START",
            "source": CaptureSource.REPLAY.value,
            "broker_identity": {
                "fcm_id": account.fcm_id,
                "ib_id": account.ib_id,
                "account_id": account.account_id,
            },
            "native_identity": {},
            "request_key": None,
            "user_msg": [],
            "replay_batch_id": batch_id,
            "batch_kind": "ACCOUNT_RECOVERY",
            "terminal_response_received": False,
            "generations": {
                plant.value: generation
                for plant, generation in generations.items()
            },
            "dedupe_key": f"recovery:{batch_id}:start",
            "normalized": {},
            "raw_frame_sha256": hashlib.sha256(
                f"recovery:{batch_id}:start".encode("ascii")
            ).hexdigest(),
            "raw_frame_b64": "",
        }
        event = CaptureEvent(
            event_id=hashlib.sha256(
                f"{account.fcm_id}\0{account.ib_id}\0{account.account_id}\0{batch_id}\0start".encode()
            ).hexdigest(),
            account_id=account_id,
            plant=CapturePlant.ORDER,
            source=CaptureSource.REPLAY,
            generation_id=generations.get(CapturePlant.ORDER, ""),
            payload=payload,
        )
        result = self._event_sink(event)
        if inspect.isawaitable(result):
            await result

    async def _emit_disconnect(
        self,
        plant: Plant,
        generation_id: str,
        reason: str,
    ) -> None:
        if self._event_sink is None or not generation_id:
            return
        reason_code = (
            reason
            if reason
            in {
                "connection_interrupted",
                "service_shutdown",
                "terminal_protocol_response",
                "terminal_protocol_state",
            }
            else "connection_interrupted"
        )
        capture_plant = self._capture_plant(plant)
        for account_id in sorted(self.runtime.account_ids):
            account = self._accounts.get(account_id)
            dedupe_key = stable_fingerprint(
                ("DISCONNECTED", account_id, capture_plant.value, generation_id)
            )
            payload = {
                "schema_version": _SCHEMA_VERSION,
                "template_id": 0,
                "template_name": "LOCAL_DISCONNECTED",
                "observation_type": "CONTROL",
                "control_kind": "DISCONNECTED",
                "source": CaptureSource.SYSTEM.value,
                "broker_identity": {
                    "fcm_id": account.fcm_id if account else None,
                    "ib_id": account.ib_id if account else None,
                    "account_id": account_id,
                },
                "native_identity": {},
                "request_key": None,
                "user_msg": [],
                "replay_batch_id": None,
                "dedupe_key": dedupe_key,
                "normalized": {
                    "control_kind": "DISCONNECTED",
                    "disconnect_reason": reason_code,
                },
                "raw_frame_sha256": hashlib.sha256(
                    f"disconnect:{capture_plant.value}:{generation_id}".encode("ascii")
                ).hexdigest(),
                "raw_frame_b64": "",
            }
            event = CaptureEvent(
                event_id=hashlib.sha256(
                    f"{account_id}\0{capture_plant.value}\0{generation_id}\0disconnect".encode()
                ).hexdigest(),
                account_id=account_id,
                plant=capture_plant,
                source=CaptureSource.SYSTEM,
                generation_id=generation_id,
                payload=payload,
            )
            result = self._event_sink(event)
            if inspect.isawaitable(result):
                await result

    async def _emit_control(self, received: ReceivedMessage) -> None:
        if self._event_sink is None:
            return
        control_kind = received.dispatch.kind.value
        normalized_control = _json_safe(received.dispatch.control)
        control_mapping = (
            normalized_control if isinstance(normalized_control, Mapping) else {}
        )
        message_view = FieldView(received.dispatch.message)

        def safe_token(value: Any) -> str | None:
            token = optional_text(value)
            if (
                token
                and len(token) <= 64
                and all(character.isalnum() or character in "-_.:" for character in token)
            ):
                return token
            return None

        raw_codes = tuple(received.dispatch.response_codes) or message_view.strings(
            "rp_code"
        )
        response_codes = tuple(
            token for value in raw_codes if (token := safe_token(value)) is not None
        )
        response_code_hashes = tuple(
            hashlib.sha256(str(value).encode("utf-8")).hexdigest()
            for value in raw_codes
        )
        reason_value = message_view.first(
            "reason_code", "logout_reason_code", "reject_reason_code"
        )
        reason_code = safe_token(reason_value)
        reason_hash = (
            hashlib.sha256(str(reason_value).encode("utf-8")).hexdigest()
            if reason_value is not None
            else None
        )
        text_hashes = tuple(
            hashlib.sha256(str(value).encode("utf-8")).hexdigest()
            for name in ("reason", "text", "report_text")
            if (value := message_view.get(name)) not in (None, "")
        )
        for account_id in sorted(self.runtime.account_ids):
            if account_id in self._ambiguous_accounts:
                continue
            account = self._accounts.get(account_id)
            fcm_id = account.fcm_id if account else control_mapping.get("fcm_id")
            ib_id = account.ib_id if account else control_mapping.get("ib_id")
            dedupe_key = stable_fingerprint(
                (
                    account_id,
                    received.plant.name,
                    received.generation_id,
                    received.template_id,
                    received.fingerprint,
                )
            )
            payload = {
                "schema_version": _SCHEMA_VERSION,
                "template_id": received.template_id,
                "template_name": template_name(received.template_id),
                "observation_type": "CONTROL",
                "control_kind": control_kind,
                "forced_logout": received.dispatch.kind is DispatchKind.FORCED_LOGOUT,
                "source": CaptureSource.SYSTEM.value,
                "broker_identity": {
                    "fcm_id": fcm_id,
                    "ib_id": ib_id,
                    "account_id": account_id,
                },
                "native_identity": {},
                "request_key": None,
                "user_msg": [],
                "replay_batch_id": None,
                "dedupe_key": dedupe_key,
                "normalized": {
                    "control_kind": control_kind,
                    **control_mapping,
                    "response_codes": response_codes,
                    "response_code_sha256": response_code_hashes,
                    "reason_code": reason_code,
                    "reason_code_sha256": reason_hash,
                    "text_sha256": text_hashes,
                },
                "raw_frame_sha256": received.fingerprint,
                # Control error text can contain sensitive broker context; only
                # its digest is retained in this audit event.
                "raw_frame_b64": "",
            }
            event = CaptureEvent(
                event_id=hashlib.sha256(
                    f"{fcm_id}\0{ib_id}\0{account_id}\0{dedupe_key}".encode()
                ).hexdigest(),
                account_id=account_id,
                plant=self._capture_plant(received.plant),
                source=CaptureSource.SYSTEM,
                generation_id=received.generation_id,
                payload=payload,
            )
            result = self._event_sink(event)
            if inspect.isawaitable(result):
                await result

    async def _capture_user_account_update(self, received: ReceivedMessage) -> None:
        view = FieldView(received.dispatch.message)
        update_facts = self._message_fields(received.dispatch.message)

        def present(name: str) -> Any | None:
            value = view.get(name)
            return value if value is not None else None

        def single_fact(*names: str) -> Any | None:
            values = [present(name) for name in names]
            values = [value for value in values if value is not None]
            if not values:
                return None
            distinct = {str(value) for value in values}
            return values[0] if len(distinct) == 1 else None

        timestamps = {
            key: value
            for key in ("ssboe", "usecs")
            if (value := present(key)) is not None
        }
        rms_update = {
            key: value
            for key, value in {
                "product_code": present("product_code"),
                "status": present("account_status"),
                "loss_limit": present("loss_limit"),
                "minimum_account_balance": present("min_account_balance"),
                "minimum_margin_balance": present("min_margin_balance"),
                "max_order_quantity": present("max_limit_quantity"),
                "buy_limit": present("buy_limit"),
                "sell_limit": present("sell_limit"),
                "buy_margin_rate": present("buy_margin_rate"),
                "sell_margin_rate": present("sell_margin_rate"),
                "commission_rate": present("commission_fill_rate"),
                "auto_liquidate": single_fact(
                    "multiple_liq_auto_liquidate",
                    "ib_account_auto_liq",
                    "user_account_auto_liq",
                ),
                "auto_liquidate_criteria": single_fact(
                    "multiple_liq_auto_liq_criteria",
                    "ib_account_auto_liq_criteria",
                    "user_account_auto_liq_criteria",
                ),
                "current_auto_liquidate_threshold": single_fact(
                    "mulitple_liq_auto_liq_threshold",
                    "ib_account_auto_liq_threshold",
                    "user_account_auto_liq_threshold",
                ),
            }.items()
            if value is not None
        }
        pnl_update = {
            key: value
            for key, value in {
                "cash_on_hand": present("cash_on_hand"),
            }.items()
            if value is not None
        }
        has_rms_update = bool(rms_update)
        has_pnl_update = bool(pnl_update)
        if timestamps and has_rms_update:
            rms_update["timestamps"] = timestamps
        if timestamps and has_pnl_update:
            pnl_update["timestamps"] = timestamps
        if has_rms_update:
            rms_update["official_update"] = update_facts
        if has_pnl_update:
            pnl_update["official_update"] = update_facts
        update_type = (optional_text(view.get("update_type")) or "").strip().lower()
        record_user = optional_text(view.get("user"))
        current_user = optional_text(getattr(self._login_info, "user", None))
        applies_to_current_user = record_user is None or record_user == current_user
        if received.template_id == Template.USER_INFO_UPDATE:
            fcm = optional_text(view.get("fcm_id"))
            ib = optional_text(view.get("ib_id"))
            if not fcm or not ib or not record_user:
                raise UnauthorizedAccount(
                    "user update omitted the full logged-in broker identity"
                )
            if not applies_to_current_user:
                return
        if (
            received.template_id == Template.USER_ACCOUNT_UPDATE
            and not applies_to_current_user
            and update_type
            in {"assign_account_to_user", "remove_account_from_user"}
        ):
            return

        raw_account_status = optional_text(view.get("account_status"))
        raw_user_status = optional_text(view.get("status"))
        raw_access_type = optional_text(
            view.first("access_type", "account_access_type")
        )
        authorization_revoked = (
            (
                applies_to_current_user
                and update_type
                in {
                    "remove_account_from_user",
                    "remove_account",
                    "delete_account",
                    "disable_account",
                }
            )
            or (
                raw_account_status is not None
                and _authorization_token(raw_account_status)
                not in _ALLOWED_ACCOUNT_STATUSES
            )
            or (
                applies_to_current_user
                and raw_user_status is not None
                and _authorization_token(raw_user_status)
                not in _ALLOWED_USER_STATUSES
            )
            or (
                applies_to_current_user
                and raw_access_type is not None
                and _authorization_token(raw_access_type)
                not in _ALLOWED_ACCOUNT_ACCESS_TYPES
            )
        )
        direct = self._account_from_view(view)
        named_account_id = optional_text(view.get("account_id"))
        if (
            named_account_id in self.runtime.account_ids
            and direct is None
        ):
            raise UnauthorizedAccount(
                "allowlisted account update omitted full FCM/IB/account identity"
            )
        targets: list[BrokerAccountKey] = []
        if direct and direct.account_id in self.runtime.account_ids:
            expected = self._accounts.get(direct.account_id)
            if expected is None or expected != direct:
                raise UnauthorizedAccount(
                    "account update identity did not match discovery"
                )
            targets.append(expected)
        elif received.template_id == Template.USER_INFO_UPDATE:
            fcm = optional_text(view.get("fcm_id"))
            ib = optional_text(view.get("ib_id"))
            targets.extend(
                key
                for key in self._accounts.values()
                if key.account_id in self.runtime.account_ids
                and key.fcm_id == fcm
                and key.ib_id == ib
            )
        try:
            for account in targets:
                normalized: Mapping[str, Any] = {
                    **update_facts,
                    "authorization_revoked": authorization_revoked,
                }
                row = self._account_rows.get(account.account_id)
                if row is not None:
                    updated = replace(
                        row,
                        account_status=raw_account_status or row.account_status,
                        access_type=(
                            raw_access_type if applies_to_current_user else None
                        )
                        or row.access_type,
                        user_id=(record_user if applies_to_current_user else None)
                        or row.user_id,
                        user_type=(
                            optional_text(view.get("type"))
                            if applies_to_current_user
                            else None
                        )
                        or row.user_type,
                        user_status=(
                            raw_user_status if applies_to_current_user else None
                        )
                        or row.user_status,
                        order_copy_status=(
                            optional_text(view.get("order_copy_status"))
                            if applies_to_current_user
                            else None
                        )
                        or row.order_copy_status,
                        ticker_session_max=(
                            int(view.get("tp_max_session_count"))
                            if view.get("tp_max_session_count") is not None
                            else row.ticker_session_max
                        ),
                        order_session_max=(
                            int(view.get("op_max_session_count"))
                            if view.get("op_max_session_count") is not None
                            else row.order_session_max
                        ),
                    )
                    self._account_rows[account.account_id] = updated
                    normalized = {
                        "account": _json_safe(updated),
                        "update": update_facts,
                        "authorization_revoked": authorization_revoked,
                    }
                await self._emit(
                    received,
                    normalized,
                    "ACCOUNT",
                    account,
                    CaptureSource.LIVE,
                )
                if received.template_id == Template.ACCOUNT_AND_USER_UPDATE:
                    if rms_update:
                        await self._emit(
                            received,
                            rms_update,
                            "PRODUCT_RMS"
                            if rms_update.get("product_code") is not None
                            else "ACCOUNT_RMS",
                            account,
                            CaptureSource.LIVE,
                        )
                    if pnl_update:
                        await self._emit(
                            received,
                            pnl_update,
                            "ACCOUNT_PNL",
                            account,
                            CaptureSource.LIVE,
                        )
        finally:
            if authorization_revoked and targets:
                revoked_ids = {account.account_id for account in targets}
                for account_id in revoked_ids:
                    self._accounts.pop(account_id, None)
                    self._account_rows.pop(account_id, None)
                self._subscriptions = {
                    marker for marker in self._subscriptions if marker[0] not in revoked_ids
                }
                self._accounts_discovered.clear()
                generations = {
                    capture_plant: str(self._sessions[plant].state.generation_id or "")
                    for plant, capture_plant in (
                        (Plant.ORDER, CapturePlant.ORDER),
                        (Plant.PNL, CapturePlant.PNL),
                    )
                    if plant in self._sessions
                    and self._sessions[plant].state.generation_id
                }
                await self.abort_recovery(generations)

    async def _emit(
        self,
        received: ReceivedMessage,
        observation: Any,
        observation_type: str,
        account: BrokerAccountKey | None,
        source: CaptureSource,
    ) -> None:
        if self._event_sink is None:
            return
        stable_identity = getattr(observation, "stable_identity", None)
        normalized_payload = _json_safe(observation)
        decoded_facts = self._message_fields(received.dispatch.message)
        decoded_identity = {
            key: value
            for key, value in decoded_facts.items()
            if key not in _DEDUPE_TRANSPORT_FIELDS
        }
        if FieldView(received.dispatch.message).strings("rp_code"):
            # Empty multipart boundaries from different required requests are
            # distinct audit facts even when every domain field is absent.
            decoded_identity["terminal_template_id"] = received.template_id
        identity_payload = normalized_payload
        if isinstance(identity_payload, Mapping):
            identity_payload = {
                key: value
                for key, value in identity_payload.items()
                if key
                not in {
                    "source_kind",
                    "user_msg",
                    "rq_handler_rp_code",
                    "rp_code",
                }
            }
        identity = (
            (stable_identity, identity_payload, decoded_identity)
            if stable_identity
            else (
                (identity_payload, decoded_identity)
                if identity_payload or decoded_identity
                else received.fingerprint
            )
        )
        account_identity = (
            (account.fcm_id, account.ib_id, account.account_id)
            if account
            else ("SYSTEM", received.plant.name)
        )
        dedupe_key = stable_fingerprint(
            (
                received.generation_id,
                account_identity,
                observation_type,
                identity,
            )
        )
        native = {
            "basket_id": getattr(observation, "basket_id", None),
            "fill_id": getattr(observation, "fill_id", None),
            "exchange_order_id": getattr(observation, "exchange_order_id", None),
            "sequence_number": getattr(observation, "sequence_number", None),
        }
        message_view = FieldView(received.dispatch.message)
        batch_id = (
            self._active_replay_batch.get(account.account_id) if account else None
        )
        pending = (
            self._pending_reconciliations.get(account.account_id)
            if account is not None
            else None
        )
        replay_generations = (
            {plant.value: value for plant, value in pending.generations.items()}
            if pending is not None
            else {
                self._capture_plant(plant).value: str(session.state.generation_id or "")
                for plant, session in self._sessions.items()
                if plant in {Plant.ORDER, Plant.PNL} and session.state.generation_id
            }
            if batch_id
            else None
        )
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "template_id": received.template_id,
            "template_name": template_name(received.template_id),
            "observation_type": observation_type,
            "source": source.value,
            "broker_identity": {
                "fcm_id": account.fcm_id if account else None,
                "ib_id": account.ib_id if account else None,
                "account_id": account.account_id if account else None,
            },
            "native_identity": _json_safe(native),
            "request_key": optional_text(message_view.get("request_key")),
            "user_msg": list(message_view.strings("user_msg")),
            "replay_batch_id": batch_id,
            "generations": replay_generations,
            "dedupe_key": dedupe_key,
            "normalized": normalized_payload if observation is not None else {},
            # Persistence recursively redacts this decoded map before storage.
            # Keeping the non-sensitive official fields closes the audit gap
            # where a newer schema field could otherwise be lost merely because
            # the curated normalizer had not learned it yet.
            "decoded_facts": decoded_facts,
            "raw_frame_sha256": received.fingerprint,
            "raw_frame_b64": base64.b64encode(received.raw_frame).decode("ascii"),
        }
        event_id = hashlib.sha256(
            (
                f"{account.fcm_id if account else ''}\0"
                f"{account.ib_id if account else ''}\0"
                f"{account.account_id if account else 'SYSTEM'}\0{dedupe_key}"
            ).encode()
        ).hexdigest()
        event = CaptureEvent(
            event_id=event_id,
            account_id=account.account_id if account else None,
            plant=self._capture_plant(received.plant),
            source=source,
            generation_id=received.generation_id,
            payload=payload,
        )
        result = self._event_sink(event)
        if inspect.isawaitable(result):
            await result
        if (
            batch_id
            and pending is not None
            and source not in {CaptureSource.LIVE, CaptureSource.UNKNOWN}
        ):
            self._fold_reconciliation_event(pending, event)
            self._reconcile_record_counts[account.account_id] = (
                pending.records_seen
            )

    @staticmethod
    def _fold_reconciliation_event(
        pending: _PendingReconciliation,
        event: CaptureEvent,
    ) -> None:
        """Fold one persisted observation into an account's deterministic view."""

        if event.event_id in pending.seen_event_ids:
            return
        pending.seen_event_ids.add(event.event_id)
        pending.records_seen += 1
        if event.account_id != pending.account.account_id:
            pending.discrepancy_count += 1
            return
        expected_generation = pending.generations.get(event.plant)
        if not expected_generation or expected_generation != event.generation_id:
            pending.discrepancy_count += 1
            return
        if not isinstance(event.payload, Mapping):
            pending.discrepancy_count += 1
            return

        observation_type = str(
            event.payload.get("observation_type") or "UNKNOWN"
        ).upper()
        normalized_value = event.payload.get("normalized")
        normalized = (
            dict(normalized_value) if isinstance(normalized_value, Mapping) else {}
        )
        native_value = event.payload.get("native_identity")
        native = dict(native_value) if isinstance(native_value, Mapping) else {}
        decoded_value = event.payload.get("decoded_facts")
        decoded = (
            {
                str(name): value
                for name, value in decoded_value.items()
                if str(name) not in _DEDUPE_TRANSPORT_FIELDS
            }
            if isinstance(decoded_value, Mapping)
            else {}
        )

        def first(*names: str) -> Any:
            for name in names:
                value = native.get(name)
                if value not in (None, "", [], {}):
                    return value
                value = normalized.get(name)
                if value not in (None, "", [], {}):
                    return value
            return None

        entity: tuple[Any, ...]
        missing_identity = False
        if observation_type in {"ORDER", "ORDER_EXECUTION"}:
            basket_id = first("basket_id")
            missing_identity = basket_id is None
            if observation_type == "ORDER_EXECUTION":
                fill_id = first("fill_id")
                missing_identity = missing_identity or fill_id is None
                entity = (
                    basket_id,
                    fill_id,
                    first("sequence_number"),
                    normalized.get("execution_effect"),
                )
            else:
                entity = (basket_id,)
        elif observation_type == "POSITION":
            symbol = first("symbol")
            product = first("product_code")
            missing_identity = symbol is None and product is None
            entity = (first("exchange"), symbol, product)
        elif observation_type == "BRACKET":
            parent = first("parent_basket_id", "basket_id")
            missing_identity = parent is None
            entity = (parent, first("bracket_type"))
        elif observation_type == "PRODUCT_RMS":
            product = first("product_code")
            missing_identity = product is None
            entity = (product,)
        elif observation_type in {"ACCOUNT", "ACCOUNT_PNL", "ACCOUNT_RMS"}:
            entity = (pending.account.account_id,)
        else:
            # History and unknown-but-decodable rows remain part of the digest,
            # but cannot safely replace a known current-state entity.
            entity = (event.payload.get("dedupe_key") or event.event_id,)

        terminal_ack = (
            isinstance(decoded_value, Mapping)
            and decoded_value.get("rp_code") not in (None, "", [], {})
        )
        if missing_identity and terminal_ack:
            # Multipart terminal ACKs intentionally carry no order/fill/bracket
            # identity. Their success is already enforced by the request waiter;
            # retain receipt evidence without treating an empty result set as a
            # state discrepancy.
            terminal_key = stable_fingerprint(
                ("TERMINAL_ACK", event.payload.get("dedupe_key") or event.event_id)
            )
            pending.state[terminal_key] = {
                "plant": event.plant.value,
                "observation_type": "TERMINAL_ACK",
            }
            return
        if missing_identity:
            pending.discrepancy_count += 1
            return
        value = {
            "plant": event.plant.value,
            "observation_type": observation_type,
            "normalized": {
                name: item
                for name, item in normalized.items()
                if name not in {"source_kind", "user_msg", "rq_handler_rp_code", "rp_code"}
            },
            "decoded_facts": decoded,
        }
        state_key = stable_fingerprint((event.plant.value, observation_type, entity))
        pending.state[state_key] = value
        if observation_type == "ORDER_EXECUTION":
            # An execution notification is also the newest broker order view.
            order_key = stable_fingerprint(
                (event.plant.value, "ORDER", (first("basket_id"),))
            )
            pending.state[order_key] = {**value, "observation_type": "ORDER"}

    @staticmethod
    def _reconciliation_digest(pending: _PendingReconciliation) -> str:
        return stable_fingerprint(
            {
                "account": (
                    pending.account.fcm_id,
                    pending.account.ib_id,
                    pending.account.account_id,
                ),
                "batch_id": pending.batch_id,
                "generations": {
                    plant.value: generation
                    for plant, generation in pending.generations.items()
                },
                "snapshot_complete": pending.snapshot_complete,
                "records_seen": pending.records_seen,
                "buffered_events_applied": pending.buffered_events_applied,
                "discrepancy_count": pending.discrepancy_count,
                "state": sorted(pending.state.items()),
            }
        )

    def _require_account(self, account_id: str) -> BrokerAccountKey:
        if account_id not in self.runtime.account_ids:
            raise UnauthorizedAccount("account is not explicitly allowlisted")
        if account_id in self._ambiguous_accounts:
            raise UnauthorizedAccount("account identifier is ambiguous")
        try:
            return self._accounts[account_id]
        except KeyError as exc:
            raise UnauthorizedAccount("allowlisted account was not discovered") from exc

    @staticmethod
    def _account_from_view(view: FieldView) -> BrokerAccountKey | None:
        fcm = optional_text(view.get("fcm_id"))
        ib = optional_text(view.get("ib_id"))
        account = optional_text(view.get("account_id"))
        if fcm and ib and account:
            return BrokerAccountKey(fcm, ib, account)
        return None

    @staticmethod
    def _message_fields(message: Any) -> Mapping[str, Any]:
        if isinstance(message, Mapping):
            return {str(key): _json_safe(value) for key, value in message.items()}
        list_fields = getattr(message, "ListFields", None)
        if not callable(list_fields):
            return {}
        result: dict[str, Any] = {}
        for descriptor, value in list_fields():
            # Full decoded messages are persisted only inside the hidden event
            # payload and are never emitted to logs or health endpoints.
            result[descriptor.name] = _json_safe(value)
        return result

    def _assert_current_generations(self, generations: Mapping[CapturePlant, str]) -> None:
        for capture_plant, expected in generations.items():
            plant = Plant[capture_plant.value]
            actual = str(self._sessions[plant].state.generation_id or "")
            if not expected or expected != actual:
                raise ConnectionError("Rithmic plant generation changed during recovery")

    def _drop_generation_state(self, plant: Plant, generation: str) -> None:
        self._subscriptions = {
            item for item in self._subscriptions if item[1] is not plant or item[2] == generation
        }
        capture_plant = self._capture_plant(plant)
        stale_accounts = (
            {
                account_id
                for account_id, pending in self._pending_reconciliations.items()
                if pending.generations.get(capture_plant) != generation
            }
            if capture_plant in {CapturePlant.ORDER, CapturePlant.PNL}
            else set()
        )
        for account_id in stale_accounts:
            self._pending_reconciliations.pop(account_id, None)
            self._active_replay_batch.pop(account_id, None)
            self._reconcile_record_counts.pop(account_id, None)
        if plant is Plant.ORDER:
            self._accounts.clear()
            self._account_rows.clear()
            self._ambiguous_accounts.clear()
            self._account_metadata_observed.clear()
            self._account_metadata_denied.clear()
            self._account_metadata_clock.clear()
            self._accounts_discovered.clear()
            self._login_info = None

    def _fail_waiters(self, plant: Plant, generation_id: str | None = None) -> None:
        for waiter in self._waiters.values():
            if (
                waiter.plant is plant
                and (generation_id is None or waiter.generation_id == generation_id)
                and not waiter.future.done()
            ):
                waiter.future.set_exception(ProtocolRejected("Rithmic session became unavailable"))

    @staticmethod
    def _capture_source(source: SourceKind) -> CaptureSource:
        return {
            SourceKind.SNAPSHOT: CaptureSource.SNAPSHOT,
            SourceKind.HISTORY: CaptureSource.HISTORY,
            SourceKind.LIVE: CaptureSource.LIVE,
            SourceKind.UNKNOWN: getattr(CaptureSource, "UNKNOWN", CaptureSource.SYSTEM),
        }[source]

    @staticmethod
    def _capture_plant(plant: Plant) -> CapturePlant:
        return {Plant.ORDER: CapturePlant.ORDER, Plant.PNL: CapturePlant.PNL, Plant.TICKER: CapturePlant.TICKER}[plant]


def create_observer(capture_config: Any) -> RithmicReadOnlyObserver:
    """Production factory loaded by ``app.v2.capture.main``.

    Binding preparation performs no network activity. The observer's ``start``
    method is the sole external-connectivity boundary.
    """

    runtime = ObserverRuntimeConfig.from_capture_config(capture_config)
    binding_config = ExternalBindingsConfig.from_environment(
        forbid_workspace_root=_workspace_root()
    )
    prepared = prepare_bindings(binding_config)
    try:
        return RithmicReadOnlyObserver(runtime, capture_config, prepared)
    except Exception:
        prepared.close()
        raise
