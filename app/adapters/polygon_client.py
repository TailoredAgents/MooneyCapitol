from __future__ import annotations

import asyncio
import os
from datetime import datetime, time, timedelta
from typing import Any

import httpx
from zoneinfo import ZoneInfo

from app.data import demo


BASE = "https://api.polygon.io"
ET_ZONE = ZoneInfo("America/New_York")


class PolygonClient:
    def __init__(self, api_key: str | None = None, timeout: float = 10.0):
        self.api_key = api_key or os.getenv("POLYGON_API_KEY", "")
        self.demo_mode = bool(os.getenv("DEMO_MODE")) or not self.api_key
        self._client = httpx.AsyncClient(timeout=timeout) if not self.demo_mode else None

    async def close(self):
        if self._client:
            await self._client.aclose()

    async def _request(self, method: str, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if self.demo_mode:
            raise RuntimeError("PolygonClient is in demo mode; real requests disabled")
        params = params or {}
        params["apiKey"] = self.api_key
        url = f"{BASE}{path}"
        for attempt in range(4):
            try:
                resp = await self._client.request(method, url, params=params)
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status >= 500 and attempt < 3:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise
            except httpx.RequestError:
                if attempt < 3:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise
        raise RuntimeError("Polygon request failed after retries")

    async def get_premarket_ref_price(self, ticker: str, as_of_et: str = "09:25") -> float | None:
        if self.demo_mode:
            return demo.get_demo_premarket_ref_price(ticker)
        date = datetime.now(tz=ET_ZONE).date()
        as_of_time = datetime.strptime(as_of_et, "%H:%M").time()
        start = datetime.combine(date, time(4, 0), tzinfo=ET_ZONE)
        end = datetime.combine(date, as_of_time, tzinfo=ET_ZONE)
        path = f"/v2/aggs/ticker/{ticker.upper()}/range/1/minute/{start.isoformat()}/{end.isoformat()}"
        data = await self._request("GET", path, {"adjusted": "true", "sort": "desc", "limit": 1})
        results = data.get("results") or []
        return float(results[0]["c"]) if results else None

    async def get_prev_close(self, ticker: str) -> float | None:
        if self.demo_mode:
            return demo.get_demo_prev_close(ticker)
        path = f"/v2/aggs/ticker/{ticker.upper()}/prev"
        data = await self._request("GET", path, {"adjusted": "true"})
        results = data.get("results") or []
        return float(results[0]["c"]) if results else None

    async def get_prev_day_volume(self, ticker: str) -> int | None:
        if self.demo_mode:
            return demo.get_demo_prev_volume(ticker)
        path = f"/v2/aggs/ticker/{ticker.upper()}/prev"
        data = await self._request("GET", path, {"adjusted": "true"})
        results = data.get("results") or []
        return int(results[0]["v"]) if results else None

    async def list_top_gappers(
        self,
        limit: int,
        gap_threshold_abs: float,
        price_min: float,
        price_max: float,
        min_volume_prev_day: int,
    ) -> list[dict[str, Any]]:
        if self.demo_mode:
            return demo.get_demo_watchlist(limit)

        resp = await self._request(
            "GET",
            "/v2/snapshot/locale/us/markets/stocks/tickers",
            {"include_otc": "false", "limit": 1000, "sort": "todaysChangePerc", "order": "desc"},
        )
        tickers = resp.get("tickers", [])
        items: list[dict[str, Any]] = []
        for t in tickers:
            t_type = t.get("type")
            if t_type and t_type != "CS":  # common stock
                continue
            last = t.get("preMarket") or t.get("lastTrade") or {}
            price = last.get("p") or last.get("price")
            if price is None:
                continue
            if not (price_min <= price <= price_max):
                continue
            prev_close = t.get("prevDay") or {}
            prev_c = prev_close.get("c") or prev_close.get("close")
            if not prev_c:
                continue
            gap_pct = (price - prev_c) / prev_c
            if abs(gap_pct) < gap_threshold_abs:
                continue
            prev_volume = prev_close.get("v") or prev_close.get("volume")
            if not prev_volume or prev_volume < min_volume_prev_day:
                continue
            premkt = t.get("preMarket") or t.get("day") or {}
            premkt_vol = premkt.get("v") or premkt.get("volume") or 0
            if premkt_vol < 20000:
                continue
            items.append(
                {
                    "ticker": t.get("ticker"),
                    "price": float(price),
                    "gap_pct": float(gap_pct),
                    "direction": "up" if gap_pct >= 0 else "down",
                    "premkt_volume": int(premkt_vol),
                }
            )
            if len(items) >= limit * 3:
                break

        # Fill missing via per-ticker fallback
        if len(items) < limit:
            # TODO: Additional discovery (e.g., gappers endpoint) if needed
            pass

        items.sort(key=lambda x: abs(x["gap_pct"]), reverse=True)
        return items[:limit]

    async def get_aggregates(
        self,
        ticker: str,
        multiplier: int = 1,
        timespan: str = "minute",
        limit: int = 120,
    ) -> list[dict[str, Any]]:
        if self.demo_mode:
            return demo.get_demo_candles(ticker, timespan, limit)

        end = datetime.utcnow()
        delta = timedelta(minutes=limit * multiplier)
        start = end - delta
        path = f"/v2/aggs/ticker/{ticker.upper()}/range/{multiplier}/{timespan}/{start.isoformat()}/{end.isoformat()}"
        data = await self._request("GET", path, {"adjusted": "true", "sort": "asc", "limit": limit})
        return data.get("results", [])
