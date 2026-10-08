from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from uuid import UUID, uuid4

from .constants import Plant
from .errors import InvalidStateTransition


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SessionState(str, Enum):
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    AUTHENTICATING = "AUTHENTICATING"
    CONNECTED = "CONNECTED"
    RECONCILING = "RECONCILING"
    READY = "READY"
    DEGRADED = "DEGRADED"
    FORCED_LOGOUT = "FORCED_LOGOUT"
    STOPPED = "STOPPED"


@dataclass(frozen=True)
class PlantHealthSnapshot:
    plant: str
    required: bool
    state: str
    generation_id: str | None
    connected: bool
    authenticated: bool
    reconciled: bool
    ready: bool
    last_message_at: datetime | None
    last_outbound_at: datetime | None
    last_heartbeat_sent_at: datetime | None
    last_heartbeat_received_at: datetime | None
    reconnect_attempt: int
    has_error: bool


class PlantSessionState:
    """Independent connection/readiness state for exactly one infrastructure plant."""

    def __init__(self, plant: Plant, *, required: bool = True) -> None:
        self.plant = plant
        self.required = required
        self.state = SessionState.DISCONNECTED
        self.generation_id: UUID | None = None
        self.connected_at: datetime | None = None
        self.last_message_at: datetime | None = None
        self.last_outbound_at: datetime | None = None
        self.last_heartbeat_sent_at: datetime | None = None
        self.last_heartbeat_received_at: datetime | None = None
        self.heartbeat_interval_seconds: float | None = None
        self.reconnect_attempt = 0
        self.last_error: str | None = None
        self.reconciled = False

    @property
    def connected(self) -> bool:
        return self.state in {
            SessionState.AUTHENTICATING,
            SessionState.CONNECTED,
            SessionState.RECONCILING,
            SessionState.READY,
        }

    @property
    def authenticated(self) -> bool:
        return self.state in {
            SessionState.CONNECTED,
            SessionState.RECONCILING,
            SessionState.READY,
        }

    @property
    def ready(self) -> bool:
        return self.state is SessionState.READY and self.reconciled

    def begin_connect(self, *, now: datetime | None = None) -> UUID:
        if self.state is SessionState.STOPPED:
            raise InvalidStateTransition("stopped session cannot reconnect")
        self.state = SessionState.CONNECTING
        self.generation_id = uuid4()
        self.connected_at = None
        self.last_message_at = None
        self.last_outbound_at = None
        self.last_heartbeat_sent_at = None
        self.last_heartbeat_received_at = None
        self.heartbeat_interval_seconds = None
        self.reconciled = False
        self.last_error = None
        return self.generation_id

    def transport_connected(self, *, now: datetime | None = None) -> None:
        if self.state is not SessionState.CONNECTING:
            raise InvalidStateTransition("transport connection completed from an invalid state")
        self.state = SessionState.AUTHENTICATING
        self.connected_at = now or utc_now()

    def login_succeeded(self, heartbeat_interval_seconds: float, *, now: datetime | None = None) -> None:
        if self.state is not SessionState.AUTHENTICATING:
            raise InvalidStateTransition("login response received outside authentication")
        if heartbeat_interval_seconds <= 0:
            raise ValueError("heartbeat interval must be positive")
        self.heartbeat_interval_seconds = float(heartbeat_interval_seconds)
        self.state = SessionState.CONNECTED
        self.record_message(now=now)
        self.reconnect_attempt = 0

    def begin_reconciliation(self) -> None:
        if self.state not in {SessionState.CONNECTED, SessionState.RECONCILING, SessionState.READY}:
            raise InvalidStateTransition("reconciliation requires an authenticated connection")
        self.state = SessionState.RECONCILING
        self.reconciled = False

    def reconciliation_succeeded(self) -> None:
        if self.state is not SessionState.RECONCILING:
            raise InvalidStateTransition("session was not reconciling")
        self.reconciled = True
        self.state = SessionState.READY

    def record_message(self, *, now: datetime | None = None) -> None:
        self.last_message_at = now or utc_now()

    def record_outbound(self, *, heartbeat: bool = False, now: datetime | None = None) -> None:
        timestamp = now or utc_now()
        self.last_outbound_at = timestamp
        if heartbeat:
            self.last_heartbeat_sent_at = timestamp

    def record_heartbeat(self, *, now: datetime | None = None) -> None:
        timestamp = now or utc_now()
        self.last_heartbeat_received_at = timestamp
        self.last_message_at = timestamp

    def heartbeat_due(self, *, now: datetime | None = None) -> bool:
        if not self.authenticated or self.heartbeat_interval_seconds is None:
            return False
        # Never replace the timestamp of an unanswered heartbeat.  Keeping the
        # first outstanding send time stable is what lets the timeout deadline
        # expire and force a reconnect.
        if self.heartbeat_response_pending():
            return False
        current = now or utc_now()
        activity = max(
            (item for item in (self.last_message_at, self.last_outbound_at, self.connected_at) if item),
            default=current,
        )
        return current >= activity + timedelta(seconds=self.heartbeat_interval_seconds)

    def heartbeat_response_pending(self) -> bool:
        if self.last_heartbeat_sent_at is None:
            return False
        latest_inbound = max(
            (item for item in (self.last_message_at, self.last_heartbeat_received_at) if item),
            default=None,
        )
        return latest_inbound is None or latest_inbound < self.last_heartbeat_sent_at

    def heartbeat_response_overdue(
        self,
        *,
        timeout_multiplier: float = 2.0,
        now: datetime | None = None,
    ) -> bool:
        if timeout_multiplier < 1:
            raise ValueError("heartbeat timeout multiplier must be at least one")
        if self.last_heartbeat_sent_at is None or self.heartbeat_interval_seconds is None:
            return False
        if not self.heartbeat_response_pending():
            return False
        deadline = self.last_heartbeat_sent_at + timedelta(
            seconds=self.heartbeat_interval_seconds * timeout_multiplier
        )
        return (now or utc_now()) >= deadline

    def connection_lost(self, reason: str) -> None:
        if self.state is SessionState.STOPPED:
            return
        self.state = SessionState.DISCONNECTED
        self.reconciled = False
        self.last_error = reason
        self.reconnect_attempt += 1

    def rejected(self, reason: str) -> None:
        self.state = SessionState.DEGRADED
        self.reconciled = False
        self.last_error = reason

    def forced_logout(self) -> None:
        self.state = SessionState.FORCED_LOGOUT
        self.reconciled = False
        self.last_error = "forced logout received"

    def stop(self) -> None:
        self.state = SessionState.STOPPED
        self.reconciled = False

    def snapshot(self) -> PlantHealthSnapshot:
        return PlantHealthSnapshot(
            plant=self.plant.name,
            required=self.required,
            state=self.state.value,
            generation_id=str(self.generation_id) if self.generation_id else None,
            connected=self.connected,
            authenticated=self.authenticated,
            reconciled=self.reconciled,
            ready=self.ready,
            last_message_at=self.last_message_at,
            last_outbound_at=self.last_outbound_at,
            last_heartbeat_sent_at=self.last_heartbeat_sent_at,
            last_heartbeat_received_at=self.last_heartbeat_received_at,
            reconnect_attempt=self.reconnect_attempt,
            has_error=self.last_error is not None,
        )


@dataclass(frozen=True)
class ReconnectPolicy:
    initial_seconds: float = 1.0
    maximum_seconds: float = 30.0
    multiplier: float = 2.0
    jitter_ratio: float = 0.2

    def __post_init__(self) -> None:
        if self.initial_seconds <= 0 or self.maximum_seconds < self.initial_seconds:
            raise ValueError("invalid reconnect bounds")
        if self.multiplier < 1 or not 0 <= self.jitter_ratio <= 1:
            raise ValueError("invalid reconnect policy")

    def delay(self, attempt: int, *, random_value: float | None = None) -> float:
        if attempt < 1:
            raise ValueError("reconnect attempt starts at one")
        base = min(self.maximum_seconds, self.initial_seconds * self.multiplier ** (attempt - 1))
        sample = random.random() if random_value is None else random_value
        if not 0 <= sample <= 1:
            raise ValueError("random_value must be between zero and one")
        jitter = base * self.jitter_ratio * ((sample * 2) - 1)
        return max(0.0, min(self.maximum_seconds, base + jitter))


def all_required_plants_ready(states: list[PlantSessionState] | tuple[PlantSessionState, ...]) -> bool:
    required = [state for state in states if state.required]
    return bool(required) and all(state.ready for state in required)
