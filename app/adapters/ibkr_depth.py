from __future__ import annotations

import asyncio
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
    """IBKR SMART depth adapter.

    Requires IB Gateway or TWS reachable at IBKR_HOST/IBKR_PORT.
    This provider is intended for market depth (L2) confirmation only.
    """

    def __init__(self, smart_depth: bool = True, num_rows: int = 5):
        self.host = os.getenv("IBKR_HOST", "127.0.0.1")
        self.port = int(os.getenv("IBKR_PORT", "7497"))
        self.client_id = int(os.getenv("IBKR_CLIENT_ID", "123"))
        self._ib = None
        self._connect_lock = asyncio.Lock()
        self._tickers: dict[str, object] = {}
        self._contracts: dict[str, object] = {}
        self._subscribed: set[str] = set()
        self._smart_depth = smart_depth
        self._num_rows = int(num_rows)

    @property
    def credentials_present(self) -> bool:
        return bool(os.getenv("IBKR_HOST") and os.getenv("IBKR_PORT") and os.getenv("IBKR_CLIENT_ID"))

    async def _ensure_connected(self) -> None:  # pragma: no cover - requires IBKR
        if self._ib is not None:
            try:
                if self._ib.isConnected():
                    return
            except Exception:
                pass
        async with self._connect_lock:
            if self._ib is not None:
                try:
                    if self._ib.isConnected():
                        return
                except Exception:
                    pass
            try:
                from ib_insync import IB
            except Exception as exc:
                raise RuntimeError("ib_insync is not installed; cannot use IBKR depth") from exc

            ib = IB()

            def _on_error(req_id, error_code, error_string, contract):
                logger.warning(
                    "ibkr.error",
                    req_id=req_id,
                    code=error_code,
                    err=error_string,
                    contract=getattr(contract, "localSymbol", None) or getattr(contract, "symbol", None),
                )

            try:
                ib.errorEvent += _on_error
            except Exception:
                pass

            logger.info("ibkr.connecting", host=self.host, port=self.port, client_id=self.client_id, readonly=True)
            await ib.connectAsync(self.host, self.port, clientId=self.client_id, readonly=True, timeout=4.0)
            try:
                ib.reqMarketDataType(1)  # live
            except Exception:
                pass
            self._ib = ib
            logger.info("ibkr.connected", serverVersion=getattr(ib.client, "serverVersion", None))

    async def subscribe(self, symbols: Iterable[str]) -> None:  # pragma: no cover - requires IBKR
        await self._ensure_connected()
        assert self._ib is not None
        try:
            from ib_insync import Stock
        except Exception as exc:
            raise RuntimeError("ib_insync missing Stock helper") from exc

        symbols = [s.upper() for s in symbols if s]
        for symbol in symbols:
            if symbol in self._subscribed:
                continue
            contract = Stock(symbol, "SMART", "USD")
            try:
                await self._ib.qualifyContractsAsync(contract)
            except Exception as exc:
                logger.warning("ibkr.qualify_failed", symbol=symbol, err=str(exc))
                continue
            try:
                ticker = self._ib.reqMktDepth(contract, numRows=self._num_rows, isSmartDepth=self._smart_depth)
                self._contracts[symbol] = contract
                self._tickers[symbol] = ticker
                self._subscribed.add(symbol)
                logger.info("ibkr.depth.subscribed", symbol=symbol, smart=self._smart_depth, rows=self._num_rows)
            except Exception as exc:
                logger.warning("ibkr.depth.subscribe_failed", symbol=symbol, err=str(exc))

    async def unsubscribe(self, symbols: Iterable[str]) -> None:  # pragma: no cover - requires IBKR
        if self._ib is None:
            return
        symbols = [s.upper() for s in symbols if s]
        for symbol in symbols:
            ticker = self._tickers.pop(symbol, None)
            contract = self._contracts.pop(symbol, None)
            self._subscribed.discard(symbol)
            if ticker is None:
                continue
            try:
                if contract is not None:
                    self._ib.cancelMktDepth(contract, isSmartDepth=self._smart_depth)
                else:
                    self._ib.cancelMktDepth(ticker.contract, isSmartDepth=self._smart_depth)
                logger.info("ibkr.depth.unsubscribed", symbol=symbol)
            except Exception as exc:
                logger.warning("ibkr.depth.unsubscribe_failed", symbol=symbol, err=str(exc))

    async def snapshot(self, symbol: str) -> DepthSnapshot | None:  # pragma: no cover - requires IBKR
        symbol = (symbol or "").upper()
        if not symbol:
            return None
        if self._ib is None or not getattr(self._ib, "isConnected", lambda: False)():
            await self._ensure_connected()
        if symbol not in self._subscribed:
            return None
        ticker = self._tickers.get(symbol)
        if ticker is None:
            return None

        bids = getattr(ticker, "domBids", None) or []
        asks = getattr(ticker, "domAsks", None) or []
        if not bids and not asks:
            return None

        bid_sizes = [float(getattr(level, "size", 0.0) or 0.0) for level in bids[: self._num_rows]]
        ask_sizes = [float(getattr(level, "size", 0.0) or 0.0) for level in asks[: self._num_rows]]
        bid_prices = [float(getattr(level, "price", 0.0) or 0.0) for level in bids[: self._num_rows]]
        ask_prices = [float(getattr(level, "price", 0.0) or 0.0) for level in asks[: self._num_rows]]

        best_bid = bid_prices[0] if bid_prices else None
        best_ask = ask_prices[0] if ask_prices else None
        spread = (best_ask - best_bid) if (best_bid and best_ask) else None

        bid_total = sum(bid_sizes)
        ask_total = sum(ask_sizes)
        levels = {
            "bid_prices": bid_prices,
            "bid_sizes": bid_sizes,
            "ask_prices": ask_prices,
            "ask_sizes": ask_sizes,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": spread,
        }
        return DepthSnapshot(bid_total=bid_total, ask_total=ask_total, levels=levels)
