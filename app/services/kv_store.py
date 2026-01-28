from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from sqlalchemy import select

from app.db.models import KVStore
from app.db.session import get_session
from app.observability.logging import get_logger


logger = get_logger("kv")

_MEM_JSON: dict[str, Any] = {}
_MEM_BYTES: dict[str, bytes] = {}


class StateStoreError(RuntimeError):
    pass


StoreMode = Literal["db", "mem"]


def get_store_mode() -> StoreMode:
    """Returns the configured state store mode.

    - `STATE_STORE=db` forces Postgres-backed storage (recommended on Render).
    - `STATE_STORE=mem` forces process-local memory (local dev/tests only).
    - Default/auto: use db when `DATABASE_URL` is set to a non-placeholder value.
    """
    mode = (os.getenv("STATE_STORE") or "auto").lower().strip()
    if mode in {"mem", "memory"}:
        return "mem"
    if mode in {"db", "postgres", "postgresql"}:
        return "db"

    db_url = os.getenv("DATABASE_URL", "").strip()
    if not db_url:
        return "mem"
    # SQLAlchemy's default in this repo is a placeholder local URL; treat that as "no DB".
    if db_url.startswith("postgresql://user:pass@localhost"):
        return "mem"
    return "db"


def _db_strict() -> bool:
    return (os.getenv("STATE_STORE") or "auto").lower().strip() in {"db", "postgres", "postgresql"}

def assert_store_available() -> None:
    """Fail-fast check for DB-backed store availability.

    In `STATE_STORE=db` mode this should be called at process startup for both
    API and worker, so we don't run with broken cross-service state.
    """
    if get_store_mode() != "db":
        return
    probe_key = "__kv_probe"
    payload = {"ts": datetime.now(tz=timezone.utc).isoformat()}
    set_json(probe_key, payload)
    readback = get_json(probe_key)
    if not isinstance(readback, dict) or readback.get("ts") != payload["ts"]:
        raise StateStoreError("KV store probe failed (readback mismatch)")


def set_json(key: str, value: Any) -> None:
    if get_store_mode() != "db":
        _MEM_JSON[key] = value
        return
    try:
        with get_session() as session:
            row = session.get(KVStore, key)
            if row is None:
                row = KVStore(key=key, value_json=value, value_bytes=None)
                session.add(row)
            else:
                row.value_json = value
            row.updated_at = datetime.now(tz=timezone.utc)
            session.flush()
    except Exception as exc:
        logger.error("kv.db.write_failed", key=key, err=str(exc))
        if _db_strict():
            raise StateStoreError(f"Failed to write key '{key}' to DB") from exc
        _MEM_JSON[key] = value


def get_json(key: str) -> Any | None:
    if get_store_mode() != "db":
        return _MEM_JSON.get(key)
    try:
        with get_session() as session:
            row = session.get(KVStore, key)
            if row is None:
                return None
            return row.value_json
    except Exception as exc:
        logger.error("kv.db.read_failed", key=key, err=str(exc))
        if _db_strict():
            raise StateStoreError(f"Failed to read key '{key}' from DB") from exc
        return _MEM_JSON.get(key)


def set_bytes(key: str, value: bytes) -> None:
    if get_store_mode() != "db":
        _MEM_BYTES[key] = value
        return
    try:
        with get_session() as session:
            row = session.get(KVStore, key)
            if row is None:
                row = KVStore(key=key, value_json=None, value_bytes=value)
                session.add(row)
            else:
                row.value_bytes = value
            row.updated_at = datetime.now(tz=timezone.utc)
            session.flush()
    except Exception as exc:
        logger.error("kv.db.write_failed", key=key, err=str(exc))
        if _db_strict():
            raise StateStoreError(f"Failed to write key '{key}' to DB") from exc
        _MEM_BYTES[key] = value


def get_bytes(key: str) -> bytes | None:
    if get_store_mode() != "db":
        return _MEM_BYTES.get(key)
    try:
        with get_session() as session:
            row = session.get(KVStore, key)
            if row is None:
                return None
            return row.value_bytes
    except Exception as exc:
        logger.error("kv.db.read_failed", key=key, err=str(exc))
        if _db_strict():
            raise StateStoreError(f"Failed to read key '{key}' from DB") from exc
        return _MEM_BYTES.get(key)


def get_updated_at(key: str) -> Optional[datetime]:
    if get_store_mode() != "db":
        return None
    try:
        with get_session() as session:
            row = session.execute(select(KVStore.updated_at).where(KVStore.key == key)).scalar_one_or_none()
            return row
    except Exception as exc:
        logger.error("kv.db.read_failed", key=key, err=str(exc))
        if _db_strict():
            raise StateStoreError(f"Failed to read key '{key}' from DB") from exc
        return None
