from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


DEFAULT_COPY_PROFILE_ID = "mooney-five-account"
MASTER_CONNECTION_ID = "master"
FOLLOWER_CONNECTION_IDS = (
    "follower-1",
    "follower-2",
    "follower-3",
    "follower-4",
)
CONTROL_REASON_CODES = frozenset(
    {
        "operator_requested",
        "simulation",
        "scheduled_pause",
        "connection_issue",
        "risk_review",
        "reconciliation_issue",
        "emergency",
        "recovery",
        "system_confirmed_flat",
    }
)


class FollowerControlState(str, Enum):
    DISABLED = "DISABLED"
    ENABLED = "ENABLED"
    PAUSED = "PAUSED"
    FLATTEN_REQUESTED = "FLATTEN_REQUESTED"
    LOCKED = "LOCKED"

    @property
    def permits_new_entries(self) -> bool:
        return self == FollowerControlState.ENABLED

    @property
    def requires_lifecycle_management(self) -> bool:
        return self in {
            FollowerControlState.ENABLED,
            FollowerControlState.PAUSED,
            FollowerControlState.FLATTEN_REQUESTED,
        }


class RestartDirective(str, Enum):
    HOLD_DISABLED = "HOLD_DISABLED"
    RECONCILE_BEFORE_NEW_ENTRIES = "RECONCILE_BEFORE_NEW_ENTRIES"
    RESUME_MANAGEMENT_WITHOUT_NEW_ENTRIES = (
        "RESUME_MANAGEMENT_WITHOUT_NEW_ENTRIES"
    )
    RECONCILE_AND_RESUME_FLATTEN = "RECONCILE_AND_RESUME_FLATTEN"
    HOLD_LOCKED = "HOLD_LOCKED"


class FollowerControlAction(str, Enum):
    ENABLE = "ENABLE"
    PAUSE = "PAUSE"
    REQUEST_FLATTEN_AND_DISABLE = "REQUEST_FLATTEN_AND_DISABLE"
    LOCK = "LOCK"
    UNLOCK_TO_PAUSED = "UNLOCK_TO_PAUSED"
    CONFIRM_FLAT_AND_DISABLE = "CONFIRM_FLAT_AND_DISABLE"


class InvalidControlTransition(ValueError):
    pass


def next_follower_state(
    current: FollowerControlState,
    action: FollowerControlAction,
) -> FollowerControlState:
    allowed = {
        (FollowerControlState.DISABLED, FollowerControlAction.ENABLE): (
            FollowerControlState.ENABLED
        ),
        (FollowerControlState.DISABLED, FollowerControlAction.LOCK): (
            FollowerControlState.LOCKED
        ),
        (FollowerControlState.ENABLED, FollowerControlAction.PAUSE): (
            FollowerControlState.PAUSED
        ),
        (
            FollowerControlState.ENABLED,
            FollowerControlAction.REQUEST_FLATTEN_AND_DISABLE,
        ): FollowerControlState.FLATTEN_REQUESTED,
        (FollowerControlState.ENABLED, FollowerControlAction.LOCK): (
            FollowerControlState.LOCKED
        ),
        (FollowerControlState.PAUSED, FollowerControlAction.ENABLE): (
            FollowerControlState.ENABLED
        ),
        (
            FollowerControlState.PAUSED,
            FollowerControlAction.REQUEST_FLATTEN_AND_DISABLE,
        ): FollowerControlState.FLATTEN_REQUESTED,
        (FollowerControlState.PAUSED, FollowerControlAction.LOCK): (
            FollowerControlState.LOCKED
        ),
        (
            FollowerControlState.FLATTEN_REQUESTED,
            FollowerControlAction.CONFIRM_FLAT_AND_DISABLE,
        ): FollowerControlState.DISABLED,
        (
            FollowerControlState.FLATTEN_REQUESTED,
            FollowerControlAction.LOCK,
        ): FollowerControlState.LOCKED,
        (
            FollowerControlState.LOCKED,
            FollowerControlAction.UNLOCK_TO_PAUSED,
        ): FollowerControlState.PAUSED,
        (
            FollowerControlState.LOCKED,
            FollowerControlAction.REQUEST_FLATTEN_AND_DISABLE,
        ): FollowerControlState.FLATTEN_REQUESTED,
    }
    try:
        return allowed[(current, action)]
    except KeyError as exc:
        raise InvalidControlTransition(
            f"{action.value} is not allowed from {current.value}"
        ) from exc


@dataclass(frozen=True)
class FollowerControlSnapshot:
    connection_id: str
    follower_number: int
    state: FollowerControlState
    revision: int
    reason_code: str | None
    flatten_requested_at: datetime | None
    locked_at: datetime | None
    updated_at: datetime

    @property
    def restart_directive(self) -> RestartDirective:
        return {
            FollowerControlState.DISABLED: RestartDirective.HOLD_DISABLED,
            FollowerControlState.ENABLED: (
                RestartDirective.RECONCILE_BEFORE_NEW_ENTRIES
            ),
            FollowerControlState.PAUSED: (
                RestartDirective.RESUME_MANAGEMENT_WITHOUT_NEW_ENTRIES
            ),
            FollowerControlState.FLATTEN_REQUESTED: (
                RestartDirective.RECONCILE_AND_RESUME_FLATTEN
            ),
            FollowerControlState.LOCKED: RestartDirective.HOLD_LOCKED,
        }[self.state]


@dataclass(frozen=True)
class CopyProfileSnapshot:
    profile_id: str
    name: str
    master_connection_id: str
    global_paused: bool
    revision: int
    followers: tuple[FollowerControlSnapshot, ...]
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        if self.master_connection_id != MASTER_CONNECTION_ID:
            raise ValueError("copy profile must use the fixed master connection")
        if tuple(item.connection_id for item in self.followers) != (
            FOLLOWER_CONNECTION_IDS
        ):
            raise ValueError("copy profile must contain the four fixed followers")
        if tuple(item.follower_number for item in self.followers) != (1, 2, 3, 4):
            raise ValueError("copy profile follower numbers must be one through four")

    @property
    def entry_enabled_connection_ids(self) -> tuple[str, ...]:
        if self.global_paused:
            return ()
        return tuple(
            follower.connection_id
            for follower in self.followers
            if follower.state.permits_new_entries
        )


@dataclass(frozen=True)
class CopyControlAuditSnapshot:
    audit_id: str
    profile_id: str
    connection_id: str | None
    action: str
    actor: str
    auth_method: str
    expected_revision: int
    previous_revision: int
    resulting_revision: int
    previous_state: str
    resulting_state: str
    reason_code: str | None
    request_id: str
    occurred_at: datetime
