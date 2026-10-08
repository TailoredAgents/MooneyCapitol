from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Mapping

from app.v2.capture.contracts import CapturePlant


_TRUE = {"1", "true", "yes", "on"}
_REQUIRED_SECRET_NAMES = ("RITHMIC_USERNAME", "RITHMIC_PASSWORD")
_REQUIRED_SETTING_NAMES = ("RITHMIC_DISCOVERY_URI", "RITHMIC_SYSTEM_NAME")


def _enabled(value: str | None) -> bool:
    return (value or "").strip().lower() in _TRUE


def _items(value: str | None) -> frozenset[str]:
    return frozenset(item.strip() for item in (value or "").split(",") if item.strip())


def _plants(value: str | None) -> frozenset[CapturePlant]:
    parsed: set[CapturePlant] = set()
    for item in _items(value):
        try:
            parsed.add(CapturePlant(item.upper()))
        except ValueError:
            # An invalid plant is retained as a preflight blocker rather than
            # crashing the liveness process before it can explain the problem.
            continue
    return frozenset(parsed)


def _bounded_float(
    source: Mapping[str, str], name: str, default: float, minimum: float
) -> tuple[float, bool]:
    try:
        value = float(source.get(name, str(default)))
    except (TypeError, ValueError):
        return default, False
    valid = math.isfinite(value) and value >= minimum
    return (value if valid else default, valid)


def _bounded_int(
    source: Mapping[str, str], name: str, default: int, minimum: int
) -> tuple[int, bool]:
    try:
        value = int(source.get(name, str(default)))
    except (TypeError, ValueError):
        return default, False
    return (max(value, minimum), value >= minimum)


@dataclass(frozen=True)
class CaptureConfig:
    connectivity_enabled: bool = False
    environment: str = "DISABLED"
    account_allowlist: frozenset[str] = field(default_factory=frozenset, repr=False)
    enabled_plants: frozenset[CapturePlant] = field(
        default_factory=lambda: frozenset({CapturePlant.ORDER, CapturePlant.PNL})
    )
    required_plants: frozenset[CapturePlant] = field(
        default_factory=lambda: frozenset({CapturePlant.ORDER, CapturePlant.PNL})
    )
    observer_factory: str | None = None
    journal_factory: str | None = None
    poll_seconds: float = 1.0
    reconcile_timeout_seconds: float = 120.0
    max_buffered_events: int = 50_000
    missing_settings: tuple[str, ...] = ()
    invalid_plants: tuple[str, ...] = ()
    invalid_runtime_values: tuple[str, ...] = ()
    bindings_configured: bool = False
    bindings_archive_checksum_missing: bool = False

    @classmethod
    def from_mapping(cls, values: Mapping[str, str] | None = None) -> "CaptureConfig":
        source = os.environ if values is None else values
        raw_plants = _items(source.get("RITHMIC_ENABLED_PLANTS", "ORDER,PNL"))
        enabled_plants = _plants(source.get("RITHMIC_ENABLED_PLANTS", "ORDER,PNL"))
        invalid_plants = tuple(sorted(raw_plants - {plant.value for plant in CapturePlant}))
        missing = tuple(
            name
            for name in (*_REQUIRED_SETTING_NAMES, *_REQUIRED_SECRET_NAMES)
            if not source.get(name, "").strip()
        )
        archive_path = source.get("RITHMIC_GENERATED_BINDINGS_ARCHIVE", "").strip()
        archive_b64_path = source.get(
            "RITHMIC_GENERATED_BINDINGS_ARCHIVE_B64_FILE", ""
        ).strip()
        archive_configured = bool(archive_path or archive_b64_path)
        bindings_configured = bool(
            source.get("RITHMIC_GENERATED_BINDINGS_PATH", "").strip()
            or archive_configured
        )
        poll_seconds, poll_valid = _bounded_float(
            source, "RITHMIC_CAPTURE_POLL_SECONDS", 1.0, 0.01
        )
        reconcile_timeout, timeout_valid = _bounded_float(
            source, "RITHMIC_RECONCILIATION_TIMEOUT_SECONDS", 120.0, 1.0
        )
        max_buffered, buffer_valid = _bounded_int(
            source, "RITHMIC_MAX_BUFFERED_EVENTS", 50_000, 1
        )
        invalid_runtime_values = tuple(
            name
            for name, valid in (
                ("RITHMIC_CAPTURE_POLL_SECONDS", poll_valid),
                ("RITHMIC_RECONCILIATION_TIMEOUT_SECONDS", timeout_valid),
                ("RITHMIC_MAX_BUFFERED_EVENTS", buffer_valid),
            )
            if not valid
        )
        archive_sha256 = source.get("RITHMIC_GENERATED_BINDINGS_SHA256", "").strip().lower()
        archive_checksum_valid = len(archive_sha256) == 64 and all(
            character in "0123456789abcdef" for character in archive_sha256
        )
        return cls(
            connectivity_enabled=_enabled(source.get("RITHMIC_CAPTURE_CONNECTIVITY_ENABLED")),
            environment=source.get("RITHMIC_ENVIRONMENT", "DISABLED").strip().upper(),
            account_allowlist=_items(source.get("RITHMIC_ACCOUNT_ALLOWLIST")),
            enabled_plants=enabled_plants,
            observer_factory=source.get("RITHMIC_CAPTURE_OBSERVER_FACTORY", "").strip() or None,
            journal_factory=source.get("RITHMIC_CAPTURE_JOURNAL_FACTORY", "").strip() or None,
            poll_seconds=poll_seconds,
            reconcile_timeout_seconds=reconcile_timeout,
            max_buffered_events=max_buffered,
            missing_settings=missing,
            invalid_plants=invalid_plants,
            invalid_runtime_values=invalid_runtime_values,
            bindings_configured=bindings_configured,
            bindings_archive_checksum_missing=archive_configured and not archive_checksum_valid,
        )

    @property
    def preflight_blockers(self) -> tuple[str, ...]:
        if not self.connectivity_enabled:
            return ("external_connectivity_disabled",)

        blockers: list[str] = []
        if self.environment != "TEST":
            blockers.append("environment_must_be_test")
        if not self.account_allowlist:
            blockers.append("account_allowlist_required")
        if self.invalid_plants:
            blockers.append("invalid_enabled_plant")
        if self.invalid_runtime_values:
            blockers.append("invalid_capture_runtime_setting")
        if not self.required_plants.issubset(self.enabled_plants):
            blockers.append("order_and_pnl_plants_required")
        if self.missing_settings:
            blockers.append("required_connection_settings_missing")
        if not self.bindings_configured:
            blockers.append("external_protocol_bindings_required")
        if self.bindings_archive_checksum_missing:
            blockers.append("bindings_archive_checksum_required")
        if not self.observer_factory:
            blockers.append("observer_factory_required")
        if not self.journal_factory:
            blockers.append("durable_journal_factory_required")
        return tuple(blockers)

    @property
    def connectivity_permitted(self) -> bool:
        return not self.preflight_blockers
