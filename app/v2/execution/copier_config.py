from __future__ import annotations

import os
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Mapping

from app.v2.execution.copier import CopierMode, FollowerPolicy


_TRUE = {"1", "true", "yes", "on"}
_EXPECTED_FOLLOWERS = 4


def _items(value: str | None) -> tuple[str, ...]:
    return tuple(item.strip() for item in (value or "").split(",") if item.strip())


def _decimal(
    source: Mapping[str, str],
    name: str,
    default: str,
) -> tuple[Decimal, bool]:
    try:
        value = Decimal(source.get(name, default).strip())
    except (AttributeError, InvalidOperation):
        return Decimal(default), False
    return value, value.is_finite()


def _integer(
    source: Mapping[str, str],
    name: str,
    default: int,
) -> tuple[int, bool]:
    try:
        value = int(source.get(name, str(default)).strip())
    except (AttributeError, TypeError, ValueError):
        return default, False
    return value, True


@dataclass(frozen=True)
class ShadowCopierConfig:
    requested_mode: str = CopierMode.SHADOW.value
    master_account_id: str | None = field(default=None, repr=False)
    follower_account_ids: tuple[str, ...] = field(default_factory=tuple, repr=False)
    max_risk_fraction: Decimal = Decimal("0")
    max_contracts: int = 0
    max_concurrent_open_risk_fraction: Decimal = Decimal("0")
    daily_realized_loss_fraction: Decimal = Decimal("0")
    daily_total_loss_fraction: Decimal = Decimal("0")
    max_relative_parity_error: Decimal = Decimal("0")
    margin_headroom_fraction: Decimal = Decimal("0.20")
    allow_over_target: bool = False
    nq_initial_margin: Decimal = Decimal("0")
    mnq_initial_margin: Decimal = Decimal("0")
    nq_cost_buffer: Decimal = Decimal("0")
    mnq_cost_buffer: Decimal = Decimal("0")
    invalid_values: tuple[str, ...] = ()

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, str] | None = None,
    ) -> "ShadowCopierConfig":
        source = os.environ if values is None else values
        decimal_names = {
            "max_risk_fraction": ("V2_COPIER_MAX_RISK_FRACTION", "0"),
            "max_concurrent_open_risk_fraction": (
                "V2_COPIER_MAX_CONCURRENT_RISK_FRACTION",
                "0",
            ),
            "daily_realized_loss_fraction": (
                "V2_COPIER_DAILY_REALIZED_LOSS_FRACTION",
                "0",
            ),
            "daily_total_loss_fraction": (
                "V2_COPIER_DAILY_TOTAL_LOSS_FRACTION",
                "0",
            ),
            "max_relative_parity_error": (
                "V2_COPIER_MAX_RELATIVE_PARITY_ERROR",
                "0",
            ),
            "margin_headroom_fraction": (
                "V2_COPIER_MARGIN_HEADROOM_FRACTION",
                "0.20",
            ),
            "nq_initial_margin": ("V2_COPIER_NQ_INITIAL_MARGIN", "0"),
            "mnq_initial_margin": ("V2_COPIER_MNQ_INITIAL_MARGIN", "0"),
            "nq_cost_buffer": ("V2_COPIER_NQ_COST_BUFFER", "0"),
            "mnq_cost_buffer": ("V2_COPIER_MNQ_COST_BUFFER", "0"),
        }
        parsed: dict[str, Decimal] = {}
        invalid: list[str] = []
        for field_name, (env_name, default) in decimal_names.items():
            parsed[field_name], valid = _decimal(source, env_name, default)
            if not valid:
                invalid.append(env_name)
        max_contracts, contracts_valid = _integer(
            source,
            "V2_COPIER_MAX_CONTRACTS",
            0,
        )
        if not contracts_valid:
            invalid.append("V2_COPIER_MAX_CONTRACTS")
        return cls(
            requested_mode=source.get(
                "V2_COPIER_MODE",
                CopierMode.SHADOW.value,
            ).strip().upper(),
            master_account_id=source.get("V2_COPIER_MASTER_ACCOUNT_ID", "").strip()
            or None,
            follower_account_ids=_items(
                source.get("V2_COPIER_FOLLOWER_ACCOUNT_IDS")
            ),
            max_contracts=max_contracts,
            allow_over_target=(
                source.get("V2_COPIER_ALLOW_OVER_TARGET", "0").strip().lower()
                in _TRUE
            ),
            invalid_values=tuple(sorted(invalid)),
            **parsed,
        )

    @property
    def effective_mode(self) -> CopierMode:
        if self.requested_mode == CopierMode.SHADOW.value:
            return CopierMode.SHADOW
        return CopierMode.OFF

    @property
    def preflight_blockers(self) -> tuple[str, ...]:
        blockers: list[str] = []
        if self.requested_mode not in {mode.value for mode in CopierMode}:
            blockers.append("invalid_copier_mode")
        elif self.requested_mode in {CopierMode.TEST.value, CopierMode.LIVE.value}:
            blockers.append("broker_submission_not_implemented")
        elif self.requested_mode == CopierMode.OFF.value:
            blockers.append("copier_off")
        if not self.master_account_id:
            blockers.append("master_account_required")
        if len(self.follower_account_ids) != _EXPECTED_FOLLOWERS:
            blockers.append("exactly_four_followers_required")
        if len(set(self.follower_account_ids)) != len(self.follower_account_ids):
            blockers.append("follower_accounts_must_be_unique")
        if self.master_account_id in set(self.follower_account_ids):
            blockers.append("master_cannot_be_follower")
        if self.invalid_values:
            blockers.append("invalid_numeric_configuration")
        if not Decimal("0") < self.max_risk_fraction <= Decimal("1"):
            blockers.append("max_risk_fraction_required")
        if self.max_contracts <= 0:
            blockers.append("max_contracts_required")
        if not Decimal("0") < self.max_concurrent_open_risk_fraction <= Decimal("1"):
            blockers.append("max_concurrent_risk_fraction_required")
        if not Decimal("0") < self.daily_realized_loss_fraction <= Decimal("1"):
            blockers.append("daily_realized_loss_fraction_required")
        if not Decimal("0") < self.daily_total_loss_fraction <= Decimal("1"):
            blockers.append("daily_total_loss_fraction_required")
        if not Decimal("0") <= self.max_relative_parity_error <= Decimal("1"):
            blockers.append("invalid_parity_tolerance")
        if not Decimal("0") <= self.margin_headroom_fraction < Decimal("1"):
            blockers.append("invalid_margin_headroom")
        if self.nq_initial_margin <= Decimal("0") or self.mnq_initial_margin <= Decimal("0"):
            blockers.append("current_broker_margins_required")
        if self.nq_cost_buffer < Decimal("0") or self.mnq_cost_buffer < Decimal("0"):
            blockers.append("cost_buffers_cannot_be_negative")
        return tuple(dict.fromkeys(blockers))

    @property
    def ready_for_shadow(self) -> bool:
        return self.effective_mode == CopierMode.SHADOW and not self.preflight_blockers

    def follower_policy(self) -> FollowerPolicy:
        if self.preflight_blockers:
            raise ValueError("shadow copier configuration is not ready")
        return FollowerPolicy(
            max_risk_fraction=self.max_risk_fraction,
            max_contracts=self.max_contracts,
            max_concurrent_open_risk_fraction=self.max_concurrent_open_risk_fraction,
            daily_realized_loss_fraction=self.daily_realized_loss_fraction,
            daily_total_loss_fraction=self.daily_total_loss_fraction,
            max_relative_parity_error=self.max_relative_parity_error,
            margin_headroom_fraction=self.margin_headroom_fraction,
            allow_over_target=self.allow_over_target,
        )
