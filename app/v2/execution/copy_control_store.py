from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Protocol
from uuid import uuid4

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.v2.db_models import (
    V2CopyControlAudit,
    V2CopyFollowerControl,
    V2CopyProfile,
)
from app.v2.execution.copy_control import (
    CONTROL_REASON_CODES,
    DEFAULT_COPY_PROFILE_ID,
    FOLLOWER_CONNECTION_IDS,
    CopyControlAuditSnapshot,
    CopyProfileSnapshot,
    FollowerControlAction,
    FollowerControlSnapshot,
    FollowerControlState,
    next_follower_state,
)


_REQUEST_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,95}$")
_AUTH_METHODS = {"api_token", "basic", "dashboard_session", "local_unconfigured", "system"}


class CopyControlStoreError(RuntimeError):
    pass


class CopyProfileNotFound(CopyControlStoreError):
    pass


class FollowerControlNotFound(CopyControlStoreError):
    pass


class ControlRevisionConflict(CopyControlStoreError):
    def __init__(self, current_revision: int) -> None:
        super().__init__("copy control revision conflict")
        self.current_revision = current_revision


class DuplicateControlRequest(CopyControlStoreError):
    pass


class CopyControlTopologyCorrupt(CopyControlStoreError):
    pass


class CopyControlStore(Protocol):
    def get_profile(
        self,
        profile_id: str = DEFAULT_COPY_PROFILE_ID,
    ) -> CopyProfileSnapshot: ...

    def transition_follower(
        self,
        *,
        profile_id: str,
        connection_id: str,
        action: FollowerControlAction,
        expected_revision: int,
        actor: str,
        auth_method: str,
        request_id: str,
        reason_code: str | None,
    ) -> CopyProfileSnapshot: ...

    def set_global_pause(
        self,
        *,
        profile_id: str,
        paused: bool,
        expected_revision: int,
        actor: str,
        auth_method: str,
        request_id: str,
        reason_code: str | None,
    ) -> CopyProfileSnapshot: ...

    def list_audit(
        self,
        *,
        profile_id: str,
        limit: int,
    ) -> tuple[CopyControlAuditSnapshot, ...]: ...


class SqlAlchemyCopyControlStore:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get_profile(
        self,
        profile_id: str = DEFAULT_COPY_PROFILE_ID,
    ) -> CopyProfileSnapshot:
        profile = self.session.get(V2CopyProfile, profile_id)
        if profile is None:
            raise CopyProfileNotFound("copy profile does not exist")
        followers = self.session.execute(
            select(V2CopyFollowerControl)
            .where(V2CopyFollowerControl.profile_id == profile_id)
            .order_by(V2CopyFollowerControl.follower_number)
        ).scalars().all()
        try:
            return _profile_snapshot(profile, followers)
        except (ValueError, KeyError) as exc:
            raise CopyControlTopologyCorrupt(
                "copy profile membership is incomplete or invalid"
            ) from exc

    def transition_follower(
        self,
        *,
        profile_id: str,
        connection_id: str,
        action: FollowerControlAction,
        expected_revision: int,
        actor: str,
        auth_method: str,
        request_id: str,
        reason_code: str | None,
    ) -> CopyProfileSnapshot:
        if action == FollowerControlAction.CONFIRM_FLAT_AND_DISABLE:
            raise ValueError("flatten completion is reserved for the execution system")
        return self._transition_follower(
            profile_id=profile_id,
            connection_id=connection_id,
            action=action,
            expected_revision=expected_revision,
            actor=actor,
            auth_method=auth_method,
            request_id=request_id,
            reason_code=reason_code,
        )

    def confirm_flattened(
        self,
        *,
        profile_id: str,
        connection_id: str,
        expected_revision: int,
        request_id: str,
        reason_code: str | None = "system_confirmed_flat",
    ) -> CopyProfileSnapshot:
        return self._transition_follower(
            profile_id=profile_id,
            connection_id=connection_id,
            action=FollowerControlAction.CONFIRM_FLAT_AND_DISABLE,
            expected_revision=expected_revision,
            actor="execution-system",
            auth_method="system",
            request_id=request_id,
            reason_code=reason_code,
        )

    def _transition_follower(
        self,
        *,
        profile_id: str,
        connection_id: str,
        action: FollowerControlAction,
        expected_revision: int,
        actor: str,
        auth_method: str,
        request_id: str,
        reason_code: str | None,
    ) -> CopyProfileSnapshot:
        _validate_command(
            expected_revision,
            actor,
            auth_method,
            request_id,
        )
        selected_reason_code = _reason_code(reason_code)
        if action in {
            FollowerControlAction.REQUEST_FLATTEN_AND_DISABLE,
            FollowerControlAction.LOCK,
            FollowerControlAction.UNLOCK_TO_PAUSED,
        } and not selected_reason_code:
            raise ValueError(f"{action.value} requires a reason_code")
        if connection_id not in FOLLOWER_CONNECTION_IDS:
            raise FollowerControlNotFound("unknown follower connection")
        profile = self._locked_profile(profile_id)
        self._reject_duplicate_request(request_id)
        follower = self.session.execute(
            select(V2CopyFollowerControl)
            .where(
                V2CopyFollowerControl.profile_id == profile_id,
                V2CopyFollowerControl.connection_id == connection_id,
            )
            .with_for_update()
        ).scalar_one_or_none()
        if follower is None:
            raise FollowerControlNotFound("follower control does not exist")
        if follower.revision != expected_revision:
            raise ControlRevisionConflict(follower.revision)

        now = datetime.now(timezone.utc)
        previous_state = FollowerControlState(follower.state)
        resulting_state = next_follower_state(previous_state, action)
        previous_revision = follower.revision
        follower.state = resulting_state.value
        follower.revision += 1
        follower.reason_code = selected_reason_code
        follower.updated_at = now
        if resulting_state == FollowerControlState.FLATTEN_REQUESTED:
            follower.flatten_requested_at = now
        elif resulting_state == FollowerControlState.DISABLED:
            follower.flatten_requested_at = None
        if resulting_state == FollowerControlState.LOCKED:
            follower.locked_at = now
        elif previous_state == FollowerControlState.LOCKED:
            follower.locked_at = None
        profile.revision += 1
        profile.updated_at = now
        self.session.add(
            _audit_row(
                profile_id=profile_id,
                connection_id=connection_id,
                action=action.value,
                actor=actor,
                auth_method=auth_method,
                expected_revision=expected_revision,
                previous_revision=previous_revision,
                resulting_revision=follower.revision,
                previous_state=previous_state.value,
                resulting_state=resulting_state.value,
                reason_code=selected_reason_code,
                request_id=request_id,
                occurred_at=now,
            )
        )
        self.session.flush()
        return self.get_profile(profile_id)

    def set_global_pause(
        self,
        *,
        profile_id: str,
        paused: bool,
        expected_revision: int,
        actor: str,
        auth_method: str,
        request_id: str,
        reason_code: str | None,
    ) -> CopyProfileSnapshot:
        _validate_command(expected_revision, actor, auth_method, request_id)
        selected_reason_code = _reason_code(reason_code)
        if not selected_reason_code:
            raise ValueError("global pause changes require a reason_code")
        profile = self._locked_profile(profile_id)
        self._reject_duplicate_request(request_id)
        if profile.revision != expected_revision:
            raise ControlRevisionConflict(profile.revision)

        now = datetime.now(timezone.utc)
        previous_revision = profile.revision
        previous_state = "PAUSED" if profile.global_paused else "ACTIVE"
        resulting_state = "PAUSED" if paused else "ACTIVE"
        profile.global_paused = paused
        profile.revision += 1
        profile.updated_at = now
        self.session.add(
            _audit_row(
                profile_id=profile_id,
                connection_id=None,
                action="SET_GLOBAL_PAUSE" if paused else "CLEAR_GLOBAL_PAUSE",
                actor=actor,
                auth_method=auth_method,
                expected_revision=expected_revision,
                previous_revision=previous_revision,
                resulting_revision=profile.revision,
                previous_state=previous_state,
                resulting_state=resulting_state,
                reason_code=selected_reason_code,
                request_id=request_id,
                occurred_at=now,
            )
        )
        self.session.flush()
        return self.get_profile(profile_id)

    def list_audit(
        self,
        *,
        profile_id: str = DEFAULT_COPY_PROFILE_ID,
        limit: int = 100,
    ) -> tuple[CopyControlAuditSnapshot, ...]:
        if not 1 <= limit <= 500:
            raise ValueError("audit limit must be between one and 500")
        if self.session.get(V2CopyProfile, profile_id) is None:
            raise CopyProfileNotFound("copy profile does not exist")
        rows = self.session.execute(
            select(V2CopyControlAudit)
            .where(V2CopyControlAudit.profile_id == profile_id)
            .order_by(desc(V2CopyControlAudit.occurred_at))
            .limit(limit)
        ).scalars().all()
        return tuple(_audit_snapshot(row) for row in rows)

    def _locked_profile(self, profile_id: str) -> V2CopyProfile:
        profile = self.session.execute(
            select(V2CopyProfile)
            .where(V2CopyProfile.profile_id == profile_id)
            .with_for_update()
        ).scalar_one_or_none()
        if profile is None:
            raise CopyProfileNotFound("copy profile does not exist")
        return profile

    def _reject_duplicate_request(self, request_id: str) -> None:
        existing = self.session.execute(
            select(V2CopyControlAudit.audit_id).where(
                V2CopyControlAudit.request_id == request_id
            )
        ).scalar_one_or_none()
        if existing is not None:
            raise DuplicateControlRequest("control request ID has already been used")


def _validate_command(
    expected_revision: int,
    actor: str,
    auth_method: str,
    request_id: str,
) -> None:
    if isinstance(expected_revision, bool) or expected_revision < 1:
        raise ValueError("expected_revision must be positive")
    if (
        not actor
        or len(actor) > 128
        or any(ord(character) < 32 for character in actor)
    ):
        raise ValueError("actor is invalid")
    if auth_method not in _AUTH_METHODS:
        raise ValueError("auth_method is invalid")
    if not _REQUEST_ID.fullmatch(request_id):
        raise ValueError("request_id is invalid")


def _reason_code(value: str | None) -> str | None:
    cleaned = (value or "").strip()
    if cleaned and cleaned not in CONTROL_REASON_CODES:
        raise ValueError("reason_code is not recognized")
    return cleaned or None


def _audit_row(**values: object) -> V2CopyControlAudit:
    return V2CopyControlAudit(audit_id=str(uuid4()), **values)


def _profile_snapshot(
    profile: V2CopyProfile,
    followers: list[V2CopyFollowerControl],
) -> CopyProfileSnapshot:
    return CopyProfileSnapshot(
        profile_id=profile.profile_id,
        name=profile.name,
        master_connection_id=profile.master_connection_id,
        global_paused=profile.global_paused,
        revision=profile.revision,
        followers=tuple(
            FollowerControlSnapshot(
                connection_id=row.connection_id,
                follower_number=row.follower_number,
                state=FollowerControlState(row.state),
                revision=row.revision,
                reason_code=row.reason_code,
                flatten_requested_at=row.flatten_requested_at,
                locked_at=row.locked_at,
                updated_at=row.updated_at,
            )
            for row in followers
        ),
        created_at=profile.created_at,
        updated_at=profile.updated_at,
    )


def _audit_snapshot(row: V2CopyControlAudit) -> CopyControlAuditSnapshot:
    return CopyControlAuditSnapshot(
        audit_id=row.audit_id,
        profile_id=row.profile_id,
        connection_id=row.connection_id,
        action=row.action,
        actor=row.actor,
        auth_method=row.auth_method,
        expected_revision=row.expected_revision,
        previous_revision=row.previous_revision,
        resulting_revision=row.resulting_revision,
        previous_state=row.previous_state,
        resulting_state=row.resulting_state,
        reason_code=row.reason_code,
        request_id=row.request_id,
        occurred_at=row.occurred_at,
    )
