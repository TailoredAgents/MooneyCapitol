from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, Protocol

from app.observability.logging import get_logger
from app.services.depth import set_status


logger = get_logger("depth")


@dataclass
class DepthSnapshot:
    bid_total: float
    ask_total: float
    levels: dict

    @property
    def total(self) -> float:
        return (self.bid_total or 0.0) + (self.ask_total or 0.0)

    @property
    def bid_ratio(self) -> float:
        total = self.total or 1e-6
        return (self.bid_total or 0.0) / total

    @property
    def ask_ratio(self) -> float:
        total = self.total or 1e-6
        return (self.ask_total or 0.0) / total


class DepthProvider(Protocol):
    async def subscribe(self, symbols: Iterable[str]) -> None: ...

    async def unsubscribe(self, symbols: Iterable[str]) -> None: ...

    async def snapshot(self, symbol: str) -> DepthSnapshot | None: ...


def get_depth_provider():
    from app.adapters.ibkr_depth import DemoDepthProvider, IBKRDepthProvider

    mode = os.getenv("DEPTH_MODE", "demo").lower()
    if mode == "demo":
        logger.info("depth.mode.demo")
        status = {"mode": "demo", "status": "ok"}
        set_status(status)
        return DemoDepthProvider()

    provider = IBKRDepthProvider()
    if not provider.credentials_present:
        logger.warning("depth.ibkr.credentials_missing")
        status = {"mode": "ibkr", "status": "disabled", "reason": "no credentials", "fallback": "demo"}
        set_status(status)
        return DemoDepthProvider()

    logger.info("depth.mode.ibkr")
    status = {"mode": "ibkr", "status": "initializing"}
    set_status(status)
    return provider
