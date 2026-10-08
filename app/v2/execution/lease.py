from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import Lock


@dataclass(frozen=True)
class LeaseToken:
    name: str
    owner_id: str
    fence: int
    expires_at: datetime


class InMemoryFencedLease:
    """Test/local abstraction; production must replace this with shared durable storage."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._token: LeaseToken | None = None
        self._fence = 0

    def acquire(self, name: str, owner_id: str, ttl: timedelta) -> LeaseToken | None:
        now = datetime.now(timezone.utc)
        with self._lock:
            if self._token and self._token.expires_at > now and self._token.owner_id != owner_id:
                return None
            self._fence += 1
            self._token = LeaseToken(name, owner_id, self._fence, now + ttl)
            return self._token

    def release(self, token: LeaseToken) -> bool:
        with self._lock:
            if self._token != token:
                return False
            self._token = None
            return True
