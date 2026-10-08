from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionConfigSnapshot:
    config_version: str
    broker_adapter: str = "rithmic-placeholder"
    submission_enabled: bool = False
    global_kill_switch: bool = True
    lease_name: str = "v2-futures-execution"

    def __post_init__(self) -> None:
        if self.submission_enabled:
            raise ValueError("broker submission cannot be enabled in the V2 foundation phase")
        if self.broker_adapter != "rithmic-placeholder":
            raise ValueError("only the disabled Rithmic placeholder is available")
