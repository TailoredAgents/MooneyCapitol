from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping
from urllib.parse import urlparse


_TRUE = {"1", "true", "yes", "on"}
_SECRET_NAME = re.compile(r"^[A-Z][A-Z0-9_]{0,127}$")


class CopierAccountRole(str, Enum):
    MASTER = "MASTER"
    FOLLOWER = "FOLLOWER"


@dataclass(frozen=True)
class SecretReference:
    environment_variable: str = field(repr=False)

    @property
    def valid(self) -> bool:
        return bool(_SECRET_NAME.fullmatch(self.environment_variable))

    def present_in(self, source: Mapping[str, str]) -> bool:
        return self.valid and bool(source.get(self.environment_variable, "").strip())

    def resolve(self, source: Mapping[str, str]) -> str:
        if not self.valid:
            raise ValueError("secret reference is invalid")
        value = source.get(self.environment_variable, "").strip()
        if not value:
            raise ValueError("referenced secret is unavailable")
        return value


@dataclass(frozen=True)
class AccountConnectionConfig:
    connection_id: str
    role: CopierAccountRole
    follower_number: int | None
    environment: str
    account_id_secret: SecretReference = field(repr=False)
    username_secret: SecretReference = field(repr=False)
    password_secret: SecretReference = field(repr=False)
    discovery_uri: str = field(default="", repr=False)
    system_name: str = field(default="", repr=False)
    gateway_name: str | None = field(default=None, repr=False)
    account_id_present: bool = False
    username_present: bool = False
    password_present: bool = False

    @property
    def credentials_present(self) -> bool:
        return self.username_present and self.password_present

    @property
    def secrets_present(self) -> bool:
        return self.account_id_present and self.credentials_present

    @property
    def connection_settings_present(self) -> bool:
        return bool(self.discovery_uri and self.system_name)

    @property
    def discovery_uri_valid(self) -> bool:
        if not self.discovery_uri:
            return False
        parsed = urlparse(self.discovery_uri)
        return bool(
            parsed.scheme == "wss"
            and parsed.hostname
            and parsed.username is None
            and parsed.password is None
            and not parsed.query
            and not parsed.fragment
        )

    @property
    def display_name(self) -> str:
        if self.role == CopierAccountRole.MASTER:
            return "Master"
        return f"Follower {self.follower_number}"


@dataclass(frozen=True)
class ResolvedAccountConnection:
    connection_id: str
    role: CopierAccountRole
    follower_number: int | None
    environment: str
    account_id: str = field(repr=False)
    username: str = field(repr=False)
    password: str = field(repr=False)
    discovery_uri: str = field(repr=False)
    system_name: str = field(repr=False)
    gateway_name: str | None = field(default=None, repr=False)


@dataclass(frozen=True)
class IndependentAccountTopology:
    connectivity_enabled: bool
    environment: str
    accounts: tuple[AccountConnectionConfig, ...] = field(repr=False)
    duplicate_account_ids: bool = False
    duplicate_usernames: bool = False

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, str] | None = None,
    ) -> "IndependentAccountTopology":
        source = os.environ if values is None else values
        accounts = tuple(
            _account_from_mapping(source, slot, role, follower_number)
            for slot, role, follower_number in _account_slots()
        )
        resolved_account_ids = [
            account.account_id_secret.resolve(source)
            for account in accounts
            if account.account_id_present
        ]
        resolved_usernames = [
            account.username_secret.resolve(source)
            for account in accounts
            if account.username_present
        ]
        return cls(
            connectivity_enabled=(
                source.get("V2_COPIER_CONNECTIVITY_ENABLED", "0").strip().lower()
                in _TRUE
            ),
            environment=source.get(
                "V2_COPIER_ENVIRONMENT",
                "DISABLED",
            ).strip().upper(),
            accounts=accounts,
            duplicate_account_ids=(
                len(resolved_account_ids) != len(set(resolved_account_ids))
            ),
            duplicate_usernames=(
                len(resolved_usernames) != len(set(resolved_usernames))
            ),
        )

    @property
    def master(self) -> AccountConnectionConfig:
        return next(
            account
            for account in self.accounts
            if account.role == CopierAccountRole.MASTER
        )

    @property
    def followers(self) -> tuple[AccountConnectionConfig, ...]:
        return tuple(
            account
            for account in self.accounts
            if account.role == CopierAccountRole.FOLLOWER
        )

    @property
    def simulation_blockers(self) -> tuple[str, ...]:
        blockers: list[str] = []
        if len(self.accounts) != 5:
            blockers.append("exactly_five_connection_slots_required")
        if sum(account.role == CopierAccountRole.MASTER for account in self.accounts) != 1:
            blockers.append("exactly_one_master_slot_required")
        if len(self.followers) != 4:
            blockers.append("exactly_four_follower_slots_required")
        if len({account.connection_id for account in self.accounts}) != len(self.accounts):
            blockers.append("connection_ids_must_be_unique")
        follower_numbers = {account.follower_number for account in self.followers}
        if follower_numbers != {1, 2, 3, 4}:
            blockers.append("follower_numbers_must_be_one_through_four")
        if any(
            not secret.valid
            for account in self.accounts
            for secret in (
                account.account_id_secret,
                account.username_secret,
                account.password_secret,
            )
        ):
            blockers.append("invalid_secret_reference")
        return tuple(blockers)

    @property
    def ready_for_simulation(self) -> bool:
        return not self.simulation_blockers

    @property
    def credential_complete_account_count(self) -> int:
        return sum(account.secrets_present for account in self.accounts)

    @property
    def connection_settings_complete_account_count(self) -> int:
        return sum(
            account.connection_settings_present and account.discovery_uri_valid
            for account in self.accounts
        )

    @property
    def connectivity_blockers(self) -> tuple[str, ...]:
        blockers = list(self.simulation_blockers)
        if not self.connectivity_enabled:
            blockers.append("external_connectivity_disabled")
        if self.environment not in {"DISABLED", "TEST", "LIVE"}:
            blockers.append("invalid_copier_environment")
        elif self.environment == "LIVE":
            blockers.append("live_connectivity_not_implemented")
        elif self.environment != "TEST":
            blockers.append("environment_must_be_test")
        account_environments = {account.environment for account in self.accounts}
        if any(
            environment not in {"DISABLED", "TEST", "LIVE"}
            for environment in account_environments
        ):
            blockers.append("invalid_account_environment")
        if len(account_environments) != 1 or account_environments != {self.environment}:
            blockers.append("account_environments_must_match")
        if any(not account.connection_settings_present for account in self.accounts):
            blockers.append("connection_settings_missing")
        if any(
            account.discovery_uri and not account.discovery_uri_valid
            for account in self.accounts
        ):
            blockers.append("invalid_discovery_uri")
        if any(not account.account_id_present for account in self.accounts):
            blockers.append("account_id_secrets_missing")
        if any(not account.credentials_present for account in self.accounts):
            blockers.append("credential_secrets_missing")
        if self.duplicate_account_ids:
            blockers.append("broker_account_ids_must_be_unique")
        if self.duplicate_usernames:
            blockers.append("rithmic_usernames_must_be_unique")
        return tuple(dict.fromkeys(blockers))

    @property
    def ready_for_test_connectivity(self) -> bool:
        return not self.connectivity_blockers

    def resolved_connections(
        self,
        values: Mapping[str, str] | None = None,
    ) -> tuple[ResolvedAccountConnection, ...]:
        source = os.environ if values is None else values
        current = IndependentAccountTopology.from_mapping(source)
        if current.connectivity_blockers:
            raise ValueError("independent account topology is not ready")
        return tuple(
            ResolvedAccountConnection(
                connection_id=account.connection_id,
                role=account.role,
                follower_number=account.follower_number,
                environment=account.environment,
                account_id=account.account_id_secret.resolve(source),
                username=account.username_secret.resolve(source),
                password=account.password_secret.resolve(source),
                discovery_uri=account.discovery_uri,
                system_name=account.system_name,
                gateway_name=account.gateway_name,
            )
            for account in current.accounts
        )


def _account_slots() -> tuple[tuple[str, CopierAccountRole, int | None], ...]:
    return (
        ("MASTER", CopierAccountRole.MASTER, None),
        ("FOLLOWER_1", CopierAccountRole.FOLLOWER, 1),
        ("FOLLOWER_2", CopierAccountRole.FOLLOWER, 2),
        ("FOLLOWER_3", CopierAccountRole.FOLLOWER, 3),
        ("FOLLOWER_4", CopierAccountRole.FOLLOWER, 4),
    )


def _account_from_mapping(
    source: Mapping[str, str],
    slot: str,
    role: CopierAccountRole,
    follower_number: int | None,
) -> AccountConnectionConfig:
    prefix = f"V2_COPIER_{slot}"
    default_secret_prefix = f"V2_RITHMIC_{slot}"
    topology_environment = source.get(
        "V2_COPIER_ENVIRONMENT",
        "DISABLED",
    ).strip().upper()
    account_id_secret = SecretReference(
        source.get(
            f"{prefix}_ACCOUNT_ID_SECRET_REF",
            f"{default_secret_prefix}_ACCOUNT_ID",
        ).strip()
    )
    username_secret = SecretReference(
        source.get(
            f"{prefix}_USERNAME_SECRET_REF",
            f"{default_secret_prefix}_USERNAME",
        ).strip()
    )
    password_secret = SecretReference(
        source.get(
            f"{prefix}_PASSWORD_SECRET_REF",
            f"{default_secret_prefix}_PASSWORD",
        ).strip()
    )
    return AccountConnectionConfig(
        connection_id=slot.lower().replace("_", "-"),
        role=role,
        follower_number=follower_number,
        environment=(
            source.get(f"{prefix}_ENVIRONMENT", "").strip().upper()
            or topology_environment
        ),
        account_id_secret=account_id_secret,
        username_secret=username_secret,
        password_secret=password_secret,
        discovery_uri=source.get(f"{prefix}_DISCOVERY_URI", "").strip(),
        system_name=source.get(f"{prefix}_SYSTEM_NAME", "").strip(),
        gateway_name=source.get(f"{prefix}_GATEWAY_NAME", "").strip() or None,
        account_id_present=account_id_secret.present_in(source),
        username_present=username_secret.present_in(source),
        password_present=password_secret.present_in(source),
    )
