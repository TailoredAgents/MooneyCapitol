from __future__ import annotations

import asyncio
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.live_state import get_lanes
from app.services.kv_store import StateStoreError


router = APIRouter()


@router.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            try:
                lanes = get_lanes()
                payload = {
                    "type": "lanes",
                    "t": datetime.utcnow().isoformat() + "Z",
                    "lanes": lanes,
                }
            except StateStoreError as exc:
                payload = {
                    "type": "error",
                    "t": datetime.utcnow().isoformat() + "Z",
                    "error": str(exc),
                }
            await websocket.send_json(payload)
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        return
