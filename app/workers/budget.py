from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class PingBudget:
    limit: int
    window_seconds: int = 60
    events: deque[datetime] = field(default_factory=deque)

    def allow(self, now: datetime) -> bool:
        while self.events and (now - self.events[0]).total_seconds() > self.window_seconds:
            self.events.popleft()
        if len(self.events) < self.limit:
            self.events.append(now)
            return True
        return False
