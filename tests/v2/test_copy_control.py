from __future__ import annotations

from datetime import datetime, timezone
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
from sqlalchemy import MetaData, Table, create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.db.models import Base
from app.v2.db_models import (
    V2CopyControlAudit,
    V2CopyFollowerControl,
    V2CopyProfile,
)
from app.v2.execution.copy_control import (
    CONTROL_REASON_CODES,
    DEFAULT_COPY_PROFILE_ID,
    FollowerControlAction,
    FollowerControlState,
    InvalidControlTransition,
    RestartDirective,
    next_follower_state,
)
from app.v2.execution.copy_control_store import (
    ControlRevisionConflict,
    DuplicateControlRequest,
    SqlAlchemyCopyControlStore,
)


NOW = datetime(2026, 10, 14, 14, 0, tzinfo=timezone.utc)
TABLES = (
    V2CopyProfile.__table__,
    V2CopyFollowerControl.__table__,
    V2CopyControlAudit.__table__,
)


def _session() -> Session:
    engine = create_engine(
        "sqlite+pysqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine, tables=list(TABLES))
    session = Session(engine)
    session.add(
        V2CopyProfile(
            profile_id=DEFAULT_COPY_PROFILE_ID,
            name="Mooney five-account copier",
            master_connection_id="master",
            global_paused=True,
            revision=1,
            created_at=NOW,
            updated_at=NOW,
        )
    )
    for number in range(1, 5):
        session.add(
            V2CopyFollowerControl(
                profile_id=DEFAULT_COPY_PROFILE_ID,
                connection_id=f"follower-{number}",
                follower_number=number,
                state=FollowerControlState.DISABLED.value,
                revision=1,
                updated_at=NOW,
            )
        )
    session.commit()
    return session


def _load_migration():
    path = Path("migrations/versions/0012_v2_copy_control.py")
    spec = spec_from_file_location("migration_0012", path)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_follower_control_transition_contract_is_fail_closed() -> None:
    assert next_follower_state(
        FollowerControlState.DISABLED,
        FollowerControlAction.ENABLE,
    ) == FollowerControlState.ENABLED
    assert next_follower_state(
        FollowerControlState.ENABLED,
        FollowerControlAction.PAUSE,
    ) == FollowerControlState.PAUSED
    assert next_follower_state(
        FollowerControlState.PAUSED,
        FollowerControlAction.REQUEST_FLATTEN_AND_DISABLE,
    ) == FollowerControlState.FLATTEN_REQUESTED
    assert next_follower_state(
        FollowerControlState.FLATTEN_REQUESTED,
        FollowerControlAction.CONFIRM_FLAT_AND_DISABLE,
    ) == FollowerControlState.DISABLED
    with pytest.raises(InvalidControlTransition):
        next_follower_state(
            FollowerControlState.FLATTEN_REQUESTED,
            FollowerControlAction.ENABLE,
        )


def test_store_updates_one_follower_with_revision_and_immutable_audit_fact() -> None:
    session = _session()
    try:
        store = SqlAlchemyCopyControlStore(session)
        initial = store.get_profile()
        assert initial.global_paused is True
        assert [item.state for item in initial.followers] == [
            FollowerControlState.DISABLED
        ] * 4
        assert initial.entry_enabled_connection_ids == ()
        assert initial.followers[0].restart_directive == RestartDirective.HOLD_DISABLED

        updated = store.transition_follower(
            profile_id=DEFAULT_COPY_PROFILE_ID,
            connection_id="follower-1",
            action=FollowerControlAction.ENABLE,
            expected_revision=1,
            actor="test-operator",
            auth_method="api_token",
            request_id="request-enable-1",
            reason_code="simulation",
        )

        assert updated.revision == 2
        assert updated.followers[0].state == FollowerControlState.ENABLED
        assert updated.followers[0].revision == 2
        assert updated.entry_enabled_connection_ids == ()
        assert updated.followers[0].restart_directive == (
            RestartDirective.RECONCILE_BEFORE_NEW_ENTRIES
        )
        assert all(item.revision == 1 for item in updated.followers[1:])
        audit = store.list_audit(profile_id=DEFAULT_COPY_PROFILE_ID, limit=10)
        assert len(audit) == 1
        assert audit[0].previous_state == "DISABLED"
        assert audit[0].resulting_state == "ENABLED"
        assert audit[0].previous_revision == 1
        assert audit[0].resulting_revision == 2
        assert audit[0].request_id == "request-enable-1"
    finally:
        session.close()


def test_store_rejects_stale_revision_duplicate_request_and_invalid_transition() -> None:
    session = _session()
    try:
        store = SqlAlchemyCopyControlStore(session)
        store.transition_follower(
            profile_id=DEFAULT_COPY_PROFILE_ID,
            connection_id="follower-1",
            action=FollowerControlAction.ENABLE,
            expected_revision=1,
            actor="test-operator",
            auth_method="api_token",
            request_id="request-1",
            reason_code=None,
        )
        with pytest.raises(ControlRevisionConflict) as conflict:
            store.transition_follower(
                profile_id=DEFAULT_COPY_PROFILE_ID,
                connection_id="follower-1",
                action=FollowerControlAction.PAUSE,
                expected_revision=1,
                actor="test-operator",
                auth_method="api_token",
                request_id="request-2",
                reason_code=None,
            )
        assert conflict.value.current_revision == 2
        with pytest.raises(DuplicateControlRequest):
            store.transition_follower(
                profile_id=DEFAULT_COPY_PROFILE_ID,
                connection_id="follower-2",
                action=FollowerControlAction.ENABLE,
                expected_revision=1,
                actor="test-operator",
                auth_method="api_token",
                request_id="request-1",
                reason_code=None,
            )
        with pytest.raises(InvalidControlTransition):
            store.transition_follower(
                profile_id=DEFAULT_COPY_PROFILE_ID,
                connection_id="follower-2",
                action=FollowerControlAction.PAUSE,
                expected_revision=1,
                actor="test-operator",
                auth_method="api_token",
                request_id="request-3",
                reason_code=None,
            )
    finally:
        session.close()


def test_flatten_request_cannot_claim_disabled_until_system_confirms_flat() -> None:
    session = _session()
    try:
        store = SqlAlchemyCopyControlStore(session)
        enabled = store.transition_follower(
            profile_id=DEFAULT_COPY_PROFILE_ID,
            connection_id="follower-3",
            action=FollowerControlAction.ENABLE,
            expected_revision=1,
            actor="test-operator",
            auth_method="api_token",
            request_id="enable-3",
            reason_code=None,
        )
        assert enabled.followers[2].state == FollowerControlState.ENABLED
        with pytest.raises(ValueError, match="requires a reason_code"):
            store.transition_follower(
                profile_id=DEFAULT_COPY_PROFILE_ID,
                connection_id="follower-3",
                action=FollowerControlAction.REQUEST_FLATTEN_AND_DISABLE,
                expected_revision=2,
                actor="test-operator",
                auth_method="api_token",
                request_id="flatten-no-reason",
                reason_code=None,
            )
        requested = store.transition_follower(
            profile_id=DEFAULT_COPY_PROFILE_ID,
            connection_id="follower-3",
            action=FollowerControlAction.REQUEST_FLATTEN_AND_DISABLE,
            expected_revision=2,
            actor="test-operator",
            auth_method="api_token",
            request_id="flatten-3",
            reason_code="emergency",
        )
        assert requested.followers[2].state == FollowerControlState.FLATTEN_REQUESTED
        assert requested.followers[2].flatten_requested_at is not None
        assert requested.followers[2].restart_directive == (
            RestartDirective.RECONCILE_AND_RESUME_FLATTEN
        )

        disabled = store.confirm_flattened(
            profile_id=DEFAULT_COPY_PROFILE_ID,
            connection_id="follower-3",
            expected_revision=3,
            request_id="flat-confirmed-3",
        )
        assert disabled.followers[2].state == FollowerControlState.DISABLED
        assert disabled.followers[2].flatten_requested_at is None
    finally:
        session.close()


def test_global_pause_uses_profile_revision_and_is_audited() -> None:
    session = _session()
    try:
        store = SqlAlchemyCopyControlStore(session)
        active = store.set_global_pause(
            profile_id=DEFAULT_COPY_PROFILE_ID,
            paused=False,
            expected_revision=1,
            actor="test-operator",
            auth_method="dashboard_session",
            request_id="global-resume-1",
            reason_code="simulation",
        )
        assert active.global_paused is False
        assert active.revision == 2
        audit = store.list_audit(profile_id=DEFAULT_COPY_PROFILE_ID, limit=10)[0]
        assert audit.action == "CLEAR_GLOBAL_PAUSE"
        assert audit.previous_state == "PAUSED"
        assert audit.resulting_state == "ACTIVE"
    finally:
        session.close()


def test_database_rejects_uncontrolled_reason_codes() -> None:
    session = _session()
    try:
        follower = session.get(
            V2CopyFollowerControl,
            (DEFAULT_COPY_PROFILE_ID, "follower-1"),
        )
        assert follower is not None
        follower.reason_code = "free form operator note"
        with pytest.raises(IntegrityError):
            session.flush()
    finally:
        session.rollback()
        session.close()


def test_copy_control_migration_matches_models_and_protects_audit(monkeypatch) -> None:
    module = _load_migration()
    declared: dict[str, Table] = {}

    def capture(name, *items, **kwargs):
        declared[name] = Table(name, MetaData(), *items, **kwargs)

    monkeypatch.setattr(module, "_create", capture)
    monkeypatch.setattr(module, "_index", lambda name, table, columns: None)
    monkeypatch.setattr(module, "_seed_default_profile", lambda: None)
    monkeypatch.setattr(module, "_install_audit_trigger", lambda: None)
    module.upgrade()

    assert module.down_revision == "0011_rithmic_read_capture"
    for model in (V2CopyProfile, V2CopyFollowerControl, V2CopyControlAudit):
        migration_table = declared[model.__tablename__]
        assert set(migration_table.c.keys()) == set(model.__table__.c.keys())
        for column_name, orm_column in model.__table__.c.items():
            migration_column = migration_table.c[column_name]
            assert type(migration_column.type) is type(orm_column.type)
            assert migration_column.nullable == orm_column.nullable
            assert migration_column.primary_key == orm_column.primary_key
        assert {
            item.name for item in migration_table.constraints if item.name
        } == {
            item.name for item in model.__table__.constraints if item.name
        }

    follower_constraints = " ".join(
        str(item.sqltext)
        for item in V2CopyFollowerControl.__table__.constraints
        if hasattr(item, "sqltext")
    )
    assert all(reason_code in follower_constraints for reason_code in CONTROL_REASON_CODES)

    source = Path("migrations/versions/0012_v2_copy_control.py").read_text(
        encoding="utf-8"
    )
    assert "BEFORE UPDATE OR DELETE ON v2_copy_control_audit" in source
    assert "mooney-five-account" in source
    assert "'DISABLED', 1" in source
