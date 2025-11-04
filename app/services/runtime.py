from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
WORKER_TICK_PATH = ARTIFACT_DIR / "worker_tick.json"


def update_worker_tick(meta: Dict[str, Any] | None = None) -> None:
    payload = {
        "ts": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
        "meta": meta or {},
    }
    WORKER_TICK_PATH.write_text(json.dumps(payload))


def get_worker_tick() -> Dict[str, Any] | None:
    if not WORKER_TICK_PATH.exists():
        return None
    try:
        return json.loads(WORKER_TICK_PATH.read_text())
    except json.JSONDecodeError:
        return None
