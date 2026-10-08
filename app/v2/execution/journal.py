from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping


@dataclass(frozen=True)
class JournalRecord:
    record_id: str
    kind: str
    occurred_at: datetime
    payload: Mapping[str, Any]


class JournalQueue:
    def __init__(self, maxsize: int = 10000) -> None:
        self._queue: asyncio.Queue[JournalRecord] = asyncio.Queue(maxsize=maxsize)

    async def append(self, record: JournalRecord) -> None:
        await self._queue.put(record)

    async def next(self) -> JournalRecord:
        return await self._queue.get()

    @property
    def depth(self) -> int:
        return self._queue.qsize()
