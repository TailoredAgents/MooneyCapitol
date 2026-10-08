from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AccountMessage:
    account_id: str
    kind: str
    payload: Any


class AccountActor:
    """Serializes lifecycle handling for one account; it has no model access."""

    def __init__(self, account_id: str) -> None:
        self.account_id = account_id
        self._inbox: asyncio.Queue[AccountMessage] = asyncio.Queue()
        self._stopping = False

    async def send(self, message: AccountMessage) -> None:
        if message.account_id != self.account_id:
            raise ValueError("account actor received a message for another account")
        await self._inbox.put(message)

    async def stop(self) -> None:
        self._stopping = True

    @property
    def queue_depth(self) -> int:
        return self._inbox.qsize()
