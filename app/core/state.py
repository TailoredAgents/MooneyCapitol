from __future__ import annotations

from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta


class SetupState(str, Enum):
    ARMED = "armed"
    PRIMED = "primed"
    ACTIVE = "active"
    RESOLVED = "resolved"


@dataclass
class TTL:
    created_at: datetime
    bars_valid: int

    def expires_at(self, bar_seconds: int) -> datetime:
        return self.created_at + timedelta(seconds=self.bars_valid * bar_seconds)

    def is_expired(self, now: datetime, bar_seconds: int) -> bool:
        return now >= self.expires_at(bar_seconds)

