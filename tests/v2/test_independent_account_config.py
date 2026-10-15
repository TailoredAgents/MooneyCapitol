from __future__ import annotations

import pytest

from app.v2.execution.account_config import (
    CopierAccountRole,
    IndependentAccountTopology,
)


SLOTS = ("MASTER", "FOLLOWER_1", "FOLLOWER_2", "FOLLOWER_3", "FOLLOWER_4")


def _ready_values() -> dict[str, str]:
    values = {
        "V2_COPIER_CONNECTIVITY_ENABLED": "1",
        "V2_COPIER_ENVIRONMENT": "TEST",
    }
    for number, slot in enumerate(SLOTS, start=1):
        values.update(
            {
                f"V2_RITHMIC_{slot}_ACCOUNT_ID": f"private-account-{number}",
                f"V2_RITHMIC_{slot}_USERNAME": f"private-user-{number}",
                f"V2_RITHMIC_{slot}_PASSWORD": f"private-password-{number}",
                f"V2_COPIER_{slot}_DISCOVERY_URI": (
                    f"wss://rithmic-{number}.example.invalid/discovery"
                ),
                f"V2_COPIER_{slot}_SYSTEM_NAME": f"Rithmic Test {number}",
            }
        )
    return values


def test_default_topology_has_one_master_and_four_independent_followers() -> None:
    topology = IndependentAccountTopology.from_mapping({})

    assert len(topology.accounts) == 5
    assert topology.master.role == CopierAccountRole.MASTER
    assert [account.follower_number for account in topology.followers] == [1, 2, 3, 4]
    assert len({account.connection_id for account in topology.accounts}) == 5
    assert topology.ready_for_simulation is True
    assert topology.credential_complete_account_count == 0
    assert topology.ready_for_test_connectivity is False
    assert "external_connectivity_disabled" in topology.connectivity_blockers
    assert "credential_secrets_missing" in topology.connectivity_blockers


def test_complete_test_topology_resolves_without_exposing_secrets() -> None:
    values = _ready_values()
    topology = IndependentAccountTopology.from_mapping(values)

    assert topology.ready_for_test_connectivity is True
    assert topology.credential_complete_account_count == 5
    assert topology.connection_settings_complete_account_count == 5
    resolved = topology.resolved_connections(values)
    assert len(resolved) == 5
    assert resolved[0].account_id == "private-account-1"
    assert resolved[0].username == "private-user-1"
    assert resolved[0].environment == "TEST"
    rendered = repr(topology) + repr(resolved)
    assert "private-account" not in rendered
    assert "private-user" not in rendered
    assert "private-password" not in rendered
    assert "wss://" not in rendered


def test_duplicate_account_ids_and_usernames_block_connectivity() -> None:
    values = _ready_values()
    values["V2_RITHMIC_FOLLOWER_4_ACCOUNT_ID"] = values[
        "V2_RITHMIC_FOLLOWER_3_ACCOUNT_ID"
    ]
    values["V2_RITHMIC_FOLLOWER_4_USERNAME"] = values[
        "V2_RITHMIC_FOLLOWER_3_USERNAME"
    ]
    topology = IndependentAccountTopology.from_mapping(values)

    assert "broker_account_ids_must_be_unique" in topology.connectivity_blockers
    assert "rithmic_usernames_must_be_unique" in topology.connectivity_blockers
    with pytest.raises(ValueError, match="not ready"):
        topology.resolved_connections(values)


@pytest.mark.parametrize(
    "discovery_uri",
    (
        "https://example.invalid/discovery",
        "wss://user:password@example.invalid/discovery",
        "wss://example.invalid/discovery?credential=secret",
    ),
)
def test_discovery_endpoint_must_be_secret_free_wss(discovery_uri: str) -> None:
    values = _ready_values()
    values["V2_COPIER_MASTER_DISCOVERY_URI"] = discovery_uri
    topology = IndependentAccountTopology.from_mapping(values)

    assert "invalid_discovery_uri" in topology.connectivity_blockers
    assert topology.ready_for_test_connectivity is False


def test_secret_references_and_environments_fail_closed() -> None:
    values = _ready_values()
    values["V2_COPIER_MASTER_PASSWORD_SECRET_REF"] = "invalid-lowercase-name"
    values["V2_COPIER_FOLLOWER_1_ENVIRONMENT"] = "LIVE"
    topology = IndependentAccountTopology.from_mapping(values)

    assert "invalid_secret_reference" in topology.simulation_blockers
    assert "account_environments_must_match" in topology.connectivity_blockers
    assert topology.ready_for_test_connectivity is False
