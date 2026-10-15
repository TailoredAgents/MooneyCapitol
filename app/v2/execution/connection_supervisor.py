from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Mapping

from app.v2.domain.models import MasterOrderEvent
from app.v2.execution.account_config import (
    CopierAccountRole,
    IndependentAccountTopology,
)


_SAFE_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _require_aware(value: datetime, label: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")


class SupervisedPlant(str, Enum):
    ORDER = "ORDER"
    PNL = "PNL"


class ConnectionPhase(str, Enum):
    STOPPED = "STOPPED"
    CONNECTING = "CONNECTING"
    AUTHENTICATING = "AUTHENTICATING"
    RECONCILING = "RECONCILING"
    READY = "READY"
    DISCONNECTED = "DISCONNECTED"
    STALE = "STALE"
    FAILED = "FAILED"


class DiscoveryPhase(str, Enum):
    STOPPED = "STOPPED"
    DISCOVERING = "DISCOVERING"
    READY = "READY"
    FAILED = "FAILED"


class ConnectionSignalKind(str, Enum):
    TRANSPORT_CONNECTED = "TRANSPORT_CONNECTED"
    AUTHENTICATED = "AUTHENTICATED"
    RECONCILED = "RECONCILED"
    MESSAGE = "MESSAGE"
    HEARTBEAT = "HEARTBEAT"
    DISCONNECTED = "DISCONNECTED"
    AUTH_REJECTED = "AUTH_REJECTED"


class SimulatedConnectionOutcome(str, Enum):
    READY = "READY"
    CONNECTING = "CONNECTING"
    AUTHENTICATING = "AUTHENTICATING"
    RECONCILING = "RECONCILING"
    DISCONNECTED = "DISCONNECTED"
    AUTH_REJECTED = "AUTH_REJECTED"


class SimulatedDiscoveryOutcome(str, Enum):
    READY = "READY"
    DISCOVERING = "DISCOVERING"
    FAILED = "FAILED"


class ConnectionSupervisorError(RuntimeError):
    pass


class InvalidConnectionTransition(ConnectionSupervisorError):
    pass


class ConnectionDataUnavailable(ConnectionSupervisorError):
    pass


@dataclass(frozen=True)
class SessionKey:
    connection_id: str
    plant: SupervisedPlant


@dataclass(frozen=True)
class ConnectionSignal:
    key: SessionKey
    generation_id: str
    kind: ConnectionSignalKind
    occurred_at: datetime
    blocker_code: str | None = None

    def __post_init__(self) -> None:
        _require_aware(self.occurred_at, "signal time")
        if self.blocker_code and not _SAFE_CODE.fullmatch(self.blocker_code):
            raise ValueError("blocker_code must be a safe low-cardinality code")


@dataclass(frozen=True)
class SessionHealth:
    connection_id: str
    plant: SupervisedPlant
    phase: ConnectionPhase
    generation_id: str | None
    authenticated: bool
    reconciled: bool
    ready: bool
    reconnect_attempt: int
    last_activity_at: datetime | None
    last_heartbeat_at: datetime | None
    blocker_code: str | None


@dataclass(frozen=True)
class AccountConnectionHealth:
    connection_id: str
    display_name: str
    role: CopierAccountRole
    follower_number: int | None
    discovery_phase: DiscoveryPhase
    discovery_generation_id: str | None
    discovery_ready: bool
    last_discovered_at: datetime | None
    ready: bool
    can_accept_new_copy: bool
    sessions: tuple[SessionHealth, ...]
    blocker_codes: tuple[str, ...]


@dataclass(frozen=True)
class ConnectionSupervisorHealth:
    running: bool
    mode: str
    master_ready: bool
    global_new_copy_blocked: bool
    operational: bool
    fully_ready: bool
    ready_follower_count: int
    eligible_follower_connection_ids: tuple[str, ...]
    global_blockers: tuple[str, ...]
    degraded_connection_ids: tuple[str, ...]
    accounts: tuple[AccountConnectionHealth, ...]


@dataclass(frozen=True)
class MasterEventEnvelope:
    connection_id: str
    generation_id: str
    observed_at: datetime
    event: MasterOrderEvent = field(repr=False)

    def __post_init__(self) -> None:
        _require_aware(self.observed_at, "master event time")


@dataclass(frozen=True)
class NormalizedFollowerAccountState:
    connection_id: str
    generation_id: str
    observed_at: datetime
    equity: Decimal | None = None
    available_margin: Decimal | None = None
    current_open_risk: Decimal | None = None
    daily_realized_pnl: Decimal | None = None
    daily_total_pnl: Decimal | None = None
    account_kill_switch: bool = True

    def __post_init__(self) -> None:
        _require_aware(self.observed_at, "follower account-state time")
        for label, value in (
            ("equity", self.equity),
            ("available_margin", self.available_margin),
            ("current_open_risk", self.current_open_risk),
        ):
            if value is not None and (not value.is_finite() or value < 0):
                raise ValueError(f"{label} must be finite and non-negative")
        for label, value in (
            ("daily_realized_pnl", self.daily_realized_pnl),
            ("daily_total_pnl", self.daily_total_pnl),
        ):
            if value is not None and not value.is_finite():
                raise ValueError(f"{label} must be finite")


@dataclass(frozen=True)
class SimulatedConnectionFixture:
    outcomes: Mapping[SessionKey, SimulatedConnectionOutcome] = field(
        default_factory=dict,
        repr=False,
    )
    discovery_outcomes: Mapping[str, SimulatedDiscoveryOutcome] = field(
        default_factory=dict,
        repr=False,
    )

    def __post_init__(self) -> None:
        if any(
            not isinstance(key, SessionKey)
            or not isinstance(outcome, SimulatedConnectionOutcome)
            for key, outcome in self.outcomes.items()
        ):
            raise ValueError("simulated session outcomes must use typed keys and values")
        if any(
            not isinstance(connection_id, str)
            or not isinstance(outcome, SimulatedDiscoveryOutcome)
            for connection_id, outcome in self.discovery_outcomes.items()
        ):
            raise ValueError("simulated discovery outcomes must use typed values")

    def outcome_for(self, key: SessionKey) -> SimulatedConnectionOutcome:
        return self.outcomes.get(key, SimulatedConnectionOutcome.READY)

    def discovery_outcome_for(self, connection_id: str) -> SimulatedDiscoveryOutcome:
        return self.discovery_outcomes.get(
            connection_id,
            SimulatedDiscoveryOutcome.READY,
        )


@dataclass(frozen=True)
class _AccountMetadata:
    connection_id: str
    display_name: str
    role: CopierAccountRole
    follower_number: int | None


class _AccountDiscovery:
    def __init__(self, connection_id: str) -> None:
        self.connection_id = connection_id
        self.phase = DiscoveryPhase.STOPPED
        self.generation = 0
        self.generation_id: str | None = None
        self.last_discovered_at: datetime | None = None
        self.blocker_code: str | None = None

    @property
    def ready(self) -> bool:
        return self.phase == DiscoveryPhase.READY

    def begin(self, now: datetime) -> str:
        _require_aware(now, "discovery start time")
        self.generation += 1
        self.generation_id = f"{self.connection_id}:discovery:{self.generation}"
        self.phase = DiscoveryPhase.DISCOVERING
        self.last_discovered_at = None
        self.blocker_code = None
        return self.generation_id

    def complete(
        self,
        generation_id: str,
        *,
        succeeded: bool,
        now: datetime,
        blocker_code: str | None = None,
    ) -> bool:
        _require_aware(now, "discovery completion time")
        if generation_id != self.generation_id:
            return False
        if self.phase != DiscoveryPhase.DISCOVERING:
            raise InvalidConnectionTransition(
                "discovery completed from an invalid phase"
            )
        if blocker_code and not _SAFE_CODE.fullmatch(blocker_code):
            raise ValueError("blocker_code must be a safe low-cardinality code")
        if succeeded:
            self.phase = DiscoveryPhase.READY
            self.last_discovered_at = now
            self.blocker_code = None
        else:
            self.phase = DiscoveryPhase.FAILED
            self.blocker_code = blocker_code or "discovery_failed"
        return True

    def stop(self) -> None:
        self.phase = DiscoveryPhase.STOPPED
        self.blocker_code = "supervisor_stopped"


class _PlantSession:
    def __init__(self, key: SessionKey) -> None:
        self.key = key
        self.phase = ConnectionPhase.STOPPED
        self.generation = 0
        self.generation_id: str | None = None
        self.authenticated = False
        self.reconciled = False
        self.reconnect_attempt = 0
        self.last_activity_at: datetime | None = None
        self.last_heartbeat_at: datetime | None = None
        self.blocker_code: str | None = None

    @property
    def ready(self) -> bool:
        return (
            self.phase == ConnectionPhase.READY
            and self.authenticated
            and self.reconciled
        )

    def begin_connection(self, now: datetime) -> str:
        _require_aware(now, "connection start time")
        if self.generation:
            self.reconnect_attempt += 1
        self.generation += 1
        self.generation_id = (
            f"{self.key.connection_id}:{self.key.plant.value.lower()}:{self.generation}"
        )
        self.phase = ConnectionPhase.CONNECTING
        self.authenticated = False
        self.reconciled = False
        self.last_activity_at = now
        self.last_heartbeat_at = None
        self.blocker_code = None
        return self.generation_id

    def _record_activity(self, occurred_at: datetime) -> None:
        if self.last_activity_at is None or occurred_at > self.last_activity_at:
            self.last_activity_at = occurred_at

    def apply(self, signal: ConnectionSignal) -> bool:
        if signal.generation_id != self.generation_id:
            return False
        if signal.key != self.key:
            raise ValueError("signal session key does not match target session")

        if signal.kind == ConnectionSignalKind.DISCONNECTED:
            self.phase = ConnectionPhase.DISCONNECTED
            self.authenticated = False
            self.reconciled = False
            self.blocker_code = signal.blocker_code or "connection_lost"
            return True
        if signal.kind == ConnectionSignalKind.AUTH_REJECTED:
            if self.phase != ConnectionPhase.AUTHENTICATING:
                raise InvalidConnectionTransition(
                    "authentication can only be rejected while authenticating"
                )
            self.phase = ConnectionPhase.FAILED
            self.authenticated = False
            self.reconciled = False
            self.blocker_code = signal.blocker_code or "authentication_rejected"
            return True
        if signal.kind == ConnectionSignalKind.TRANSPORT_CONNECTED:
            if self.phase != ConnectionPhase.CONNECTING:
                raise InvalidConnectionTransition(
                    "transport connection completed from an invalid phase"
                )
            self.phase = ConnectionPhase.AUTHENTICATING
            self._record_activity(signal.occurred_at)
            return True
        if signal.kind == ConnectionSignalKind.AUTHENTICATED:
            if self.phase != ConnectionPhase.AUTHENTICATING:
                raise InvalidConnectionTransition(
                    "authentication completed from an invalid phase"
                )
            self.phase = ConnectionPhase.RECONCILING
            self.authenticated = True
            self._record_activity(signal.occurred_at)
            return True
        if signal.kind == ConnectionSignalKind.RECONCILED:
            if self.phase != ConnectionPhase.RECONCILING:
                raise InvalidConnectionTransition(
                    "reconciliation completed from an invalid phase"
                )
            self.phase = ConnectionPhase.READY
            self.reconciled = True
            self._record_activity(signal.occurred_at)
            self.blocker_code = None
            return True
        if signal.kind in {
            ConnectionSignalKind.MESSAGE,
            ConnectionSignalKind.HEARTBEAT,
        }:
            if not self.authenticated:
                raise InvalidConnectionTransition(
                    "authenticated connection required for inbound activity"
                )
            self._record_activity(signal.occurred_at)
            if signal.kind == ConnectionSignalKind.HEARTBEAT:
                if (
                    self.last_heartbeat_at is None
                    or signal.occurred_at > self.last_heartbeat_at
                ):
                    self.last_heartbeat_at = signal.occurred_at
            return True
        raise ValueError("unsupported connection signal")

    def mark_stale(self, now: datetime, stale_after: timedelta) -> bool:
        if not self.ready or self.last_activity_at is None:
            return False
        if now - self.last_activity_at <= stale_after:
            return False
        self.phase = ConnectionPhase.STALE
        self.reconciled = False
        self.blocker_code = "connection_state_stale"
        return True

    def stop(self) -> None:
        self.phase = ConnectionPhase.STOPPED
        self.authenticated = False
        self.reconciled = False
        self.blocker_code = "supervisor_stopped"

    def snapshot(self) -> SessionHealth:
        blocker = self.blocker_code
        if blocker is None and not self.ready:
            blocker = {
                ConnectionPhase.STOPPED: "not_started",
                ConnectionPhase.CONNECTING: "connecting",
                ConnectionPhase.AUTHENTICATING: "authenticating",
                ConnectionPhase.RECONCILING: "reconciling",
                ConnectionPhase.DISCONNECTED: "connection_lost",
                ConnectionPhase.STALE: "connection_state_stale",
                ConnectionPhase.FAILED: "connection_failed",
            }.get(self.phase)
        return SessionHealth(
            connection_id=self.key.connection_id,
            plant=self.key.plant,
            phase=self.phase,
            generation_id=self.generation_id,
            authenticated=self.authenticated,
            reconciled=self.reconciled,
            ready=self.ready,
            reconnect_attempt=self.reconnect_attempt,
            last_activity_at=self.last_activity_at,
            last_heartbeat_at=self.last_heartbeat_at,
            blocker_code=blocker,
        )


class IndependentAccountConnectionSupervisor:
    """Generation-safe health supervision for five independent account links.

    This component owns no credentials and has no order submission methods.
    Transport drivers feed low-cardinality lifecycle signals into the same state
    machine used by the deterministic simulation fixture.
    """

    def __init__(
        self,
        topology: IndependentAccountTopology,
        *,
        stale_after: timedelta = timedelta(seconds=15),
    ) -> None:
        if topology.simulation_blockers:
            raise ValueError("account topology is not structurally valid")
        if stale_after <= timedelta(0):
            raise ValueError("stale_after must be positive")
        self.stale_after = stale_after
        self._accounts = tuple(
            _AccountMetadata(
                connection_id=account.connection_id,
                display_name=account.display_name,
                role=account.role,
                follower_number=account.follower_number,
            )
            for account in topology.accounts
        )
        self._sessions = {
            SessionKey(account.connection_id, plant): _PlantSession(
                SessionKey(account.connection_id, plant)
            )
            for account in self._accounts
            for plant in SupervisedPlant
        }
        self._discoveries = {
            account.connection_id: _AccountDiscovery(account.connection_id)
            for account in self._accounts
        }
        self._running = False
        self._mode = "OFF"

    @property
    def session_count(self) -> int:
        return len(self._sessions)

    def start(self, *, now: datetime | None = None) -> Mapping[str, str]:
        if self._running:
            raise ConnectionSupervisorError("connection supervisor is already running")
        timestamp = now or _utc_now()
        _require_aware(timestamp, "supervisor start time")
        self._running = True
        self._mode = "EVENT_DRIVEN"
        return {
            connection_id: discovery.begin(timestamp)
            for connection_id, discovery in self._discoveries.items()
        }

    def start_simulation(
        self,
        fixture: SimulatedConnectionFixture | None = None,
        *,
        now: datetime | None = None,
    ) -> ConnectionSupervisorHealth:
        selected_fixture = fixture or SimulatedConnectionFixture()
        timestamp = now or _utc_now()
        unknown_sessions = set(selected_fixture.outcomes) - set(self._sessions)
        unknown_discoveries = set(selected_fixture.discovery_outcomes) - set(
            self._discoveries
        )
        if unknown_sessions or unknown_discoveries:
            raise ValueError("simulation fixture references an unknown connection")
        discovery_generations = self.start(now=timestamp)
        self._mode = "SIMULATED"
        for connection_id, generation_id in discovery_generations.items():
            discovery_outcome = selected_fixture.discovery_outcome_for(connection_id)
            if discovery_outcome == SimulatedDiscoveryOutcome.DISCOVERING:
                continue
            session_generations = self.complete_discovery(
                connection_id,
                generation_id,
                succeeded=discovery_outcome == SimulatedDiscoveryOutcome.READY,
                now=timestamp,
                blocker_code=(
                    "simulated_discovery_failed"
                    if discovery_outcome == SimulatedDiscoveryOutcome.FAILED
                    else None
                ),
            )
            for key, session_generation_id in session_generations.items():
                self._apply_simulated_outcome(
                    key,
                    session_generation_id,
                    selected_fixture.outcome_for(key),
                    timestamp,
                )
        return self.health()

    def complete_discovery(
        self,
        connection_id: str,
        generation_id: str,
        *,
        succeeded: bool,
        now: datetime | None = None,
        blocker_code: str | None = None,
    ) -> Mapping[SessionKey, str]:
        self._require_running()
        timestamp = now or _utc_now()
        discovery = self._discovery(connection_id)
        applied = discovery.complete(
            generation_id,
            succeeded=succeeded,
            now=timestamp,
            blocker_code=blocker_code,
        )
        if not applied or not succeeded:
            return {}
        return {
            key: session.begin_connection(timestamp)
            for key, session in self._sessions.items()
            if key.connection_id == connection_id
        }

    def reconnect_simulated(
        self,
        key: SessionKey,
        *,
        outcome: SimulatedConnectionOutcome = SimulatedConnectionOutcome.READY,
        now: datetime | None = None,
    ) -> str:
        self._require_running()
        timestamp = now or _utc_now()
        if not self._discovery(key.connection_id).ready:
            raise ConnectionSupervisorError(
                "account discovery must be ready before session reconnect"
            )
        session = self._session(key)
        generation_id = session.begin_connection(timestamp)
        self._apply_simulated_outcome(
            key,
            generation_id,
            outcome,
            timestamp,
        )
        return generation_id

    def apply(self, signal: ConnectionSignal) -> bool:
        self._require_running()
        return self._session(signal.key).apply(signal)

    def mark_stale(self, *, now: datetime | None = None) -> tuple[SessionKey, ...]:
        self._require_running()
        timestamp = now or _utc_now()
        _require_aware(timestamp, "staleness check time")
        return tuple(
            key
            for key, session in self._sessions.items()
            if session.mark_stale(timestamp, self.stale_after)
        )

    def accept_master_event(
        self,
        envelope: MasterEventEnvelope,
        *,
        now: datetime | None = None,
    ) -> MasterOrderEvent:
        master = next(
            account
            for account in self._accounts
            if account.role == CopierAccountRole.MASTER
        )
        if envelope.connection_id != master.connection_id:
            raise ConnectionDataUnavailable("master event came from a non-master link")
        if not all(
            self._session(SessionKey(master.connection_id, plant)).ready
            for plant in SupervisedPlant
        ):
            raise ConnectionDataUnavailable("required master state is not ready")
        key = SessionKey(master.connection_id, SupervisedPlant.ORDER)
        self._accept_fresh_data(
            key,
            envelope.generation_id,
            envelope.observed_at,
            now=now,
        )
        return envelope.event

    def accept_follower_state(
        self,
        state: NormalizedFollowerAccountState,
        *,
        now: datetime | None = None,
    ) -> NormalizedFollowerAccountState:
        account = self._account(state.connection_id)
        if account.role != CopierAccountRole.FOLLOWER:
            raise ConnectionDataUnavailable(
                "follower account state came from the master link"
            )
        key = SessionKey(state.connection_id, SupervisedPlant.PNL)
        self._accept_fresh_data(
            key,
            state.generation_id,
            state.observed_at,
            now=now,
        )
        return state

    def stop(self) -> ConnectionSupervisorHealth:
        for discovery in self._discoveries.values():
            discovery.stop()
        for session in self._sessions.values():
            session.stop()
        self._running = False
        self._mode = "OFF"
        return self.health()

    def health(self) -> ConnectionSupervisorHealth:
        session_health = {
            key: session.snapshot() for key, session in self._sessions.items()
        }
        raw_accounts: list[tuple[_AccountMetadata, tuple[SessionHealth, ...], bool]] = []
        for account in self._accounts:
            discovery = self._discoveries[account.connection_id]
            sessions = tuple(
                session_health[SessionKey(account.connection_id, plant)]
                for plant in SupervisedPlant
            )
            raw_accounts.append(
                (
                    account,
                    sessions,
                    discovery.ready and all(item.ready for item in sessions),
                )
            )

        master_ready = next(
            ready
            for account, _sessions, ready in raw_accounts
            if account.role == CopierAccountRole.MASTER
        )
        eligible_followers = tuple(
            account.connection_id
            for account, _sessions, ready in raw_accounts
            if account.role == CopierAccountRole.FOLLOWER
            and self._running
            and master_ready
            and ready
        )
        accounts = tuple(
            AccountConnectionHealth(
                connection_id=account.connection_id,
                display_name=account.display_name,
                role=account.role,
                follower_number=account.follower_number,
                discovery_phase=self._discoveries[account.connection_id].phase,
                discovery_generation_id=(
                    self._discoveries[account.connection_id].generation_id
                ),
                discovery_ready=self._discoveries[account.connection_id].ready,
                last_discovered_at=(
                    self._discoveries[account.connection_id].last_discovered_at
                ),
                ready=ready,
                can_accept_new_copy=bool(
                    self._running
                    and master_ready
                    and (
                        ready
                        if account.role == CopierAccountRole.FOLLOWER
                        else master_ready
                    )
                ),
                sessions=sessions,
                blocker_codes=(
                    (
                        f"discovery_{self._discoveries[account.connection_id].blocker_code}",
                    )
                    if self._discoveries[account.connection_id].blocker_code
                    else ()
                )
                + tuple(
                    f"{session.plant.value.lower()}_{session.blocker_code}"
                    for session in sessions
                    if session.blocker_code
                ),
            )
            for account, sessions, ready in raw_accounts
        )
        global_blockers: list[str] = []
        if not self._running:
            global_blockers.append("connection_supervisor_not_running")
        if not master_ready:
            global_blockers.append("master_connection_not_ready")
        elif not eligible_followers:
            global_blockers.append("no_follower_connections_ready")
        degraded = tuple(
            account.connection_id
            for account, _sessions, ready in raw_accounts
            if not ready
        )
        return ConnectionSupervisorHealth(
            running=self._running,
            mode=self._mode,
            master_ready=master_ready,
            global_new_copy_blocked=bool(global_blockers),
            operational=not global_blockers and bool(eligible_followers),
            fully_ready=master_ready and len(eligible_followers) == 4,
            ready_follower_count=len(eligible_followers),
            eligible_follower_connection_ids=eligible_followers,
            global_blockers=tuple(global_blockers),
            degraded_connection_ids=degraded,
            accounts=accounts,
        )

    def _apply_simulated_outcome(
        self,
        key: SessionKey,
        generation_id: str,
        outcome: SimulatedConnectionOutcome,
        timestamp: datetime,
    ) -> None:
        if outcome == SimulatedConnectionOutcome.CONNECTING:
            return
        self.apply(
            ConnectionSignal(
                key,
                generation_id,
                ConnectionSignalKind.TRANSPORT_CONNECTED,
                timestamp,
            )
        )
        if outcome == SimulatedConnectionOutcome.AUTHENTICATING:
            return
        if outcome == SimulatedConnectionOutcome.DISCONNECTED:
            self.apply(
                ConnectionSignal(
                    key,
                    generation_id,
                    ConnectionSignalKind.DISCONNECTED,
                    timestamp,
                    "simulated_disconnect",
                )
            )
            return
        if outcome == SimulatedConnectionOutcome.AUTH_REJECTED:
            self.apply(
                ConnectionSignal(
                    key,
                    generation_id,
                    ConnectionSignalKind.AUTH_REJECTED,
                    timestamp,
                    "simulated_auth_rejected",
                )
            )
            return
        self.apply(
            ConnectionSignal(
                key,
                generation_id,
                ConnectionSignalKind.AUTHENTICATED,
                timestamp,
            )
        )
        if outcome == SimulatedConnectionOutcome.RECONCILING:
            return
        self.apply(
            ConnectionSignal(
                key,
                generation_id,
                ConnectionSignalKind.RECONCILED,
                timestamp,
            )
        )
        self.apply(
            ConnectionSignal(
                key,
                generation_id,
                ConnectionSignalKind.HEARTBEAT,
                timestamp,
            )
        )

    def _accept_fresh_data(
        self,
        key: SessionKey,
        generation_id: str,
        observed_at: datetime,
        *,
        now: datetime | None,
    ) -> None:
        self._require_running()
        current = now or _utc_now()
        _require_aware(current, "data acceptance time")
        if observed_at > current:
            raise ConnectionDataUnavailable("future-dated connection data rejected")
        if current - observed_at > self.stale_after:
            raise ConnectionDataUnavailable("stale connection data rejected")
        session = self._session(key)
        if not session.ready:
            raise ConnectionDataUnavailable("connection is not ready")
        if session.generation_id != generation_id:
            raise ConnectionDataUnavailable("connection generation is stale")
        session.apply(
            ConnectionSignal(
                key,
                generation_id,
                ConnectionSignalKind.MESSAGE,
                observed_at,
            )
        )

    def _session(self, key: SessionKey) -> _PlantSession:
        try:
            return self._sessions[key]
        except KeyError as exc:
            raise KeyError("unknown supervised account session") from exc

    def _discovery(self, connection_id: str) -> _AccountDiscovery:
        try:
            return self._discoveries[connection_id]
        except KeyError as exc:
            raise KeyError("unknown supervised account discovery") from exc

    def _account(self, connection_id: str) -> _AccountMetadata:
        try:
            return next(
                account
                for account in self._accounts
                if account.connection_id == connection_id
            )
        except StopIteration as exc:
            raise KeyError("unknown supervised account connection") from exc

    def _require_running(self) -> None:
        if not self._running:
            raise ConnectionSupervisorError("connection supervisor is not running")
