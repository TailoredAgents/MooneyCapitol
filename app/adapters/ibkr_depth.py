from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Iterable

from app.adapters.depth_provider import DepthSnapshot, DepthProvider
from app.observability.logging import get_logger


logger = get_logger("depth.ibkr")


@dataclass
class _DemoState:
    phase: float = 0.0


class DemoDepthProvider(DepthProvider):
    """Deterministic L2 snapshots for testing PRIMED flow."""

    def __init__(self):
        self.state: dict[str, _DemoState] = {}

    async def subscribe(self, symbols: Iterable[str]) -> None:
        for sym in symbols:
            self.state.setdefault(sym, _DemoState())

    async def unsubscribe(self, symbols: Iterable[str]) -> None:  # pragma: no cover - noop cleanup
        for sym in symbols:
            self.state.pop(sym, None)

    async def snapshot(self, symbol: str) -> DepthSnapshot | None:
        state = self.state.setdefault(symbol, _DemoState())
        # Use time-based sine wave to alternate lean
        ts = time.time()
        phase = (math.sin(ts / 6 + hash(symbol) % 10) + 1) / 2  # 0..1
        state.phase = phase
        bid_total = 1000 * (1 - phase) + 500
        ask_total = 1000 * phase + 500
        levels = {
            "bid_levels": [bid_total * 0.4, bid_total * 0.3, bid_total * 0.2, bid_total * 0.07, bid_total * 0.03],
            "ask_levels": [ask_total * 0.4, ask_total * 0.3, ask_total * 0.2, ask_total * 0.07, ask_total * 0.03],
        }
        return DepthSnapshot(bid_total=bid_total, ask_total=ask_total, levels=levels)


class IBKRDepthProvider(DepthProvider):
    """IBKR SMART depth adapter (skeleton)."""

    def __init__(self):
        self.host = os.getenv("IBKR_HOST", "127.0.0.1")
        self.port = int(os.getenv("IBKR_PORT", "7497"))
        self.client_id = int(os.getenv("IBKR_CLIENT_ID", "123"))
        self.connected = False

    @property
    def credentials_present(self) -> bool:
        return bool(os.getenv("IBKR_HOST") and os.getenv("IBKR_PORT") and os.getenv("IBKR_CLIENT_ID"))

    async def subscribe(self, symbols: Iterable[str]) -> None:  # pragma: no cover - requires IBKR
        logger.info("ibkr.depth.subscribe", symbols=list(symbols))

    async def unsubscribe(self, symbols: Iterable[str]) -> None:  # pragma: no cover - requires IBKR
        logger.info("ibkr.depth.unsubscribe", symbols=list(symbols))

    async def snapshot(self, symbol: str) -> DepthSnapshot | None:  # pragma: no cover - requires IBKR
        logger.warning("ibkr.depth.snapshot.unimplemented", symbol=symbol)
        return None
