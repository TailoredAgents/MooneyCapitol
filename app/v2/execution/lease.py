from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Callable, Protocol

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.v2.db_models import V2ExecutionLease


@dataclass(frozen=True)
class LeaseToken:
    name: str
    owner_id: str
    fence: int
    expires_at: datetime


class FencedLease(Protocol):
    durable: bool

    def acquire(
        self, name: str, owner_id: str, ttl: timedelta
    ) -> LeaseToken | None: ...

    def renew(self, token: LeaseToken, ttl: timedelta) -> LeaseToken | None: ...

    def is_current(self, token: LeaseToken) -> bool: ...

    def release(self, token: LeaseToken) -> bool: ...


class InMemoryFencedLease:
    """Test/local abstraction; production must replace this with shared durable storage."""

    durable = False

    def __init__(self) -> None:
        self._lock = Lock()
        self._token: LeaseToken | None = None
        self._fence = 0

    def acquire(self, name: str, owner_id: str, ttl: timedelta) -> LeaseToken | None:
        _validate_lease_request(name, owner_id, ttl)
        now = datetime.now(timezone.utc)
        with self._lock:
            if self._token and self._token.expires_at > now:
                if self._token.name != name or self._token.owner_id != owner_id:
                    return None
                self._token = LeaseToken(name, owner_id, self._token.fence, now + ttl)
                return self._token
            self._fence += 1
            self._token = LeaseToken(name, owner_id, self._fence, now + ttl)
            return self._token

    def renew(self, token: LeaseToken, ttl: timedelta) -> LeaseToken | None:
        _validate_lease_request(token.name, token.owner_id, ttl)
        now = datetime.now(timezone.utc)
        with self._lock:
            if self._token != token or token.expires_at <= now:
                return None
            self._token = LeaseToken(
                token.name,
                token.owner_id,
                token.fence,
                now + ttl,
            )
            return self._token

    def is_current(self, token: LeaseToken) -> bool:
        now = datetime.now(timezone.utc)
        with self._lock:
            return self._token == token and token.expires_at > now

    def release(self, token: LeaseToken) -> bool:
        with self._lock:
            if self._token != token:
                return False
            self._token = None
            return True


class SqlAlchemyFencedLease:
    """Shared durable lease whose monotonically increasing fence prevents stale writes."""

    durable = True

    def __init__(
        self,
        session: Session,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.session = session
        self._clock = clock

    def acquire(
        self,
        name: str,
        owner_id: str,
        ttl: timedelta,
    ) -> LeaseToken | None:
        _validate_lease_request(name, owner_id, ttl)
        now = self._now()
        row = self.session.execute(
            select(V2ExecutionLease)
            .where(V2ExecutionLease.lease_name == name)
            .execution_options(populate_existing=True)
            .with_for_update()
        ).scalar_one_or_none()
        if row is None:
            row = V2ExecutionLease(
                lease_name=name,
                owner_id=owner_id,
                fence=1,
                expires_at=now + ttl,
                updated_at=now,
            )
            self.session.add(row)
        else:
            expires_at = _aware_utc(row.expires_at)
            if expires_at > now:
                if row.owner_id != owner_id:
                    self.session.rollback()
                    return None
            else:
                row.fence += 1
            row.owner_id = owner_id
            row.expires_at = now + ttl
            row.updated_at = now
        selected_fence = int(row.fence)
        selected_expiry = _aware_utc(row.expires_at)
        try:
            self.session.commit()
        except IntegrityError:
            # A concurrent first acquisition may win the primary-key insert.
            self.session.rollback()
            return None
        return LeaseToken(name, owner_id, selected_fence, selected_expiry)

    def renew(self, token: LeaseToken, ttl: timedelta) -> LeaseToken | None:
        _validate_lease_request(token.name, token.owner_id, ttl)
        now = self._now()
        row = self._locked(token.name)
        if row is None or not _matches(row, token) or _aware_utc(row.expires_at) <= now:
            self.session.rollback()
            return None
        row.expires_at = now + ttl
        row.updated_at = now
        selected_expiry = _aware_utc(row.expires_at)
        self.session.commit()
        return LeaseToken(
            token.name,
            token.owner_id,
            token.fence,
            selected_expiry,
        )

    def is_current(self, token: LeaseToken) -> bool:
        now = self._now()
        row = self.session.execute(
            select(V2ExecutionLease)
            .where(V2ExecutionLease.lease_name == token.name)
            .execution_options(populate_existing=True)
        ).scalar_one_or_none()
        current = bool(
            row is not None
            and _matches(row, token)
            and _aware_utc(row.expires_at) > now
        )
        self.session.rollback()
        return current

    def release(self, token: LeaseToken) -> bool:
        now = self._now()
        row = self._locked(token.name)
        if row is None or not _matches(row, token):
            self.session.rollback()
            return False
        # Preserve the row and fence history so a future owner can only advance it.
        row.expires_at = now
        row.updated_at = now
        self.session.commit()
        return True

    def _locked(self, name: str) -> V2ExecutionLease | None:
        return self.session.execute(
            select(V2ExecutionLease)
            .where(V2ExecutionLease.lease_name == name)
            .execution_options(populate_existing=True)
            .with_for_update()
        ).scalar_one_or_none()

    def _now(self) -> datetime:
        value = self._clock() if self._clock is not None else self.session.scalar(
            select(func.now())
        )
        if value is None:  # pragma: no cover - database contract guard
            raise RuntimeError("database did not return its current time")
        return _aware_utc(value)


def _matches(row: V2ExecutionLease, token: LeaseToken) -> bool:
    return row.owner_id == token.owner_id and int(row.fence) == token.fence


def _validate_lease_request(name: str, owner_id: str, ttl: timedelta) -> None:
    if not name.strip() or len(name) > 96:
        raise ValueError("lease name is invalid")
    if not owner_id.strip() or len(owner_id) > 96:
        raise ValueError("lease owner is invalid")
    if ttl <= timedelta(0):
        raise ValueError("lease TTL must be positive")


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
