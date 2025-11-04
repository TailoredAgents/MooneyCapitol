from __future__ import annotations

import os
from typing import Any, Dict, List

import httpx

from app.observability.logging import get_logger


logger = get_logger("tradestation")


class TradeStationClient:
    def __init__(self) -> None:
        self.client_id = os.getenv("TS_CLIENT_ID")
        self.client_secret = os.getenv("TS_CLIENT_SECRET")
        self.refresh_token = os.getenv("TS_REFRESH_TOKEN")
        self.base_url = "https://api.tradestation.com/v2"
        self.demo_mode = os.getenv("TRADES_DEMO", "0").lower() in {"1", "true", "yes"}
        self._client = httpx.AsyncClient(timeout=10)

    async def close(self) -> None:
        await self._client.aclose()

    async def refresh_access_token(self) -> str | None:
        if self.demo_mode:
            return "demo-token"
        if not all([self.client_id, self.client_secret, self.refresh_token]):
            logger.warning("tradestation.credentials_missing")
            return None
        try:
            resp = await self._client.post(
                "https://signin.tradestation.com/oauth/token",
                data={
                    "grant_type": "refresh_token",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "refresh_token": self.refresh_token,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("access_token")
        except httpx.HTTPError as exc:
            logger.error("tradestation.refresh_failed", err=str(exc))
            return None

    async def list_fills(self, since_iso: str | None = None) -> List[Dict[str, Any]]:
        if self.demo_mode:
            return []
        token = await self.refresh_access_token()
        if not token:
            return []
        params = {"since": since_iso} if since_iso else {}
        try:
            resp = await self._client.get(
                f"{self.base_url}/orderexecution/trades",
                headers={"Authorization": f"Bearer {token}"},
                params=params,
            )
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else data.get("trades", [])
        except httpx.HTTPError as exc:
            logger.error("tradestation.list_fills_failed", err=str(exc))
            return []
