from __future__ import annotations

import os

from app.core.config import CopierConfig


_TRUE = {"1", "true", "yes", "on"}


def environment_safety_blocks(config: CopierConfig) -> tuple[str, ...]:
    """Apply deployment flags as additional deny-only gates.

    Persisted, authenticated configuration remains authoritative for enabling
    execution. Environment values can only block it; they cannot turn it on.
    """
    blockers: list[str] = []
    enabled = os.getenv("COPIER_ENABLED")
    if enabled is not None and enabled.lower() not in _TRUE:
        blockers.append("COPIER_ENABLED deployment gate is off")
    mode = os.getenv("COPIER_MODE")
    if mode is not None and mode.lower() != config.mode.lower():
        blockers.append("COPIER_MODE does not match authenticated persisted config")
    kill = os.getenv("COPIER_GLOBAL_KILL_SWITCH")
    if kill is not None and kill.lower() in _TRUE:
        blockers.append("COPIER_GLOBAL_KILL_SWITCH deployment gate is on")
    return tuple(blockers)
