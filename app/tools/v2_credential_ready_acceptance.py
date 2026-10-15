from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = Path("artifacts/v2-credential-ready-junit.xml")


@dataclass(frozen=True)
class AcceptanceRequirement:
    requirement_id: str
    description: str
    evidence: tuple[str, ...]


ACCEPTANCE_REQUIREMENTS = (
    AcceptanceRequirement(
        "lifecycle_types",
        "All supported entry, order, and lifecycle command types",
        (
            "tests/v2/test_copy_lifecycle.py::test_position_lifecycle_compiles_add_reduce_exit_and_ordered_reversal",
            "tests/v2/test_copy_lifecycle.py::test_bracket_oco_replace_cancel_and_stop_limit_contracts",
            "tests/v2/test_rithmic_write_boundary.py::test_official_new_modify_cancel_and_exit_field_maps",
            "tests/v2/test_rithmic_write_boundary.py::test_official_bracket_and_oco_field_maps",
        ),
    ),
    AcceptanceRequirement(
        "independent_controls",
        "Independent enable, pause, flatten, and global-pause behavior",
        (
            "tests/v2/test_copy_control.py::test_four_followers_keep_independent_enable_pause_and_flatten_state",
            "tests/v2/test_copy_control.py::test_flatten_request_cannot_claim_disabled_until_system_confirms_flat",
            "tests/v2/test_copy_control.py::test_global_pause_uses_profile_revision_and_is_audited",
        ),
    ),
    AcceptanceRequirement(
        "protection",
        "Mandatory stop/bracket protection and protection reconciliation",
        (
            "tests/v2/test_copy_lifecycle.py::test_position_increase_cannot_be_compiled_without_attached_stop_policy",
            "tests/v2/test_execution_recovery.py::test_missing_protection_on_filled_position_is_a_locking_divergence",
        ),
    ),
    AcceptanceRequirement(
        "cumulative_sizing",
        "Cumulative target sizing across scale-ins, scale-outs, closes, and reversals",
        (
            "tests/v2/test_shadow_copier.py::test_cumulative_sizing_plans_total_position_deltas_for_add_and_reduction",
            "tests/v2/test_shadow_copier.py::test_cumulative_sizing_plans_reversal_from_total_not_event_delta",
            "tests/v2/test_shadow_copier.py::test_cumulative_sizing_closes_followers_when_master_is_flat",
        ),
    ),
    AcceptanceRequirement(
        "funding_and_margin",
        "Underfunded followers, contract granularity, and insufficient margin",
        (
            "tests/v2/test_shadow_copier.py::test_underfunded_follower_reports_untradeable_contract_granularity",
            "tests/v2/test_shadow_copier.py::test_funding_report_quantifies_one_mnq_equity_and_margin_shortfalls",
        ),
    ),
    AcceptanceRequirement(
        "rejections",
        "Master and follower rejection behavior",
        (
            "tests/v2/test_copy_lifecycle.py::test_master_rejection_can_only_cancel_an_existing_follower_order",
            "tests/v2/test_copy_lifecycle.py::test_follower_broker_rejection_is_terminal_and_isolated",
        ),
    ),
    AcceptanceRequirement(
        "event_ordering",
        "Partial fills plus duplicate, stale, and out-of-order events",
        (
            "tests/v2/test_copy_lifecycle.py::test_partial_fill_fill_and_duplicate_receipt_are_monotonic",
            "tests/v2/test_copy_lifecycle.py::test_store_rejects_stale_revision_and_duplicate_revision_identity",
        ),
    ),
    AcceptanceRequirement(
        "credential_account_mismatch",
        "Credential/account and adapter/account-slot mismatch",
        (
            "tests/v2/test_independent_account_config.py::test_duplicate_account_ids_and_usernames_block_connectivity",
            "tests/v2/test_copy_lifecycle.py::test_dispatcher_rejects_adapter_account_slot_mismatch",
        ),
    ),
    AcceptanceRequirement(
        "connection_and_recovery",
        "Disconnect, reconnect, crash recovery, and stale equity/P&L",
        (
            "tests/v2/test_connection_supervisor.py::test_failed_follower_is_isolated_from_healthy_followers",
            "tests/v2/test_connection_supervisor.py::test_old_generation_signals_cannot_corrupt_reconnected_session",
            "tests/v2/test_connection_supervisor.py::test_broker_neutral_master_and_follower_data_require_fresh_ready_generation",
            "tests/v2/test_execution_recovery.py::test_broker_snapshot_adopts_explained_order_after_crash_without_duplicate",
        ),
    ),
    AcceptanceRequirement(
        "execution_fencing",
        "Active-instance takeover and monotonic execution fencing",
        (
            "tests/v2/test_execution_recovery.py::test_fenced_takeover_marks_abandoned_claim_uncertain_and_never_retries",
        ),
    ),
    AcceptanceRequirement(
        "contract_expiry",
        "Contract-expiry mismatch is blocked with no automatic roll",
        (
            "tests/v2/test_shadow_copier.py::test_master_stop_and_same_expiry_are_mandatory",
            "tests/v2/test_shadow_copier.py::test_cumulative_sizing_blocks_expiry_changes_until_position_is_flat",
        ),
    ),
    AcceptanceRequirement(
        "redaction",
        "Secrets, broker accounts, and protocol identifiers are redacted",
        (
            "tests/v2/test_rithmic_security_redaction.py::test_rithmic_identifiers_pii_credentials_and_payloads_are_redacted",
            "tests/v2/test_rithmic_write_boundary.py::test_account_route_repr_does_not_leak_broker_selectors",
            "tests/test_v2_copier_api.py::test_v2_copier_status_requires_auth_and_redacts_configuration",
        ),
    ),
    AcceptanceRequirement(
        "operator_security",
        "Authorization, CSRF, optimistic revision conflict, and immutable audit behavior",
        (
            "tests/test_v2_copier_api.py::test_v2_copy_control_supports_revisions_global_pause_and_audit",
            "tests/test_v2_copier_api.py::test_v2_copy_control_cookie_mutation_requires_csrf",
            "tests/v2/test_copy_control.py::test_copy_control_migration_matches_models_and_protects_audit",
        ),
    ),
    AcceptanceRequirement(
        "concurrent_latency",
        "Latency instrumentation under concurrent four-follower load",
        (
            "tests/v2/test_copy_lifecycle.py::test_four_followers_dispatch_concurrently_with_latency_instrumentation",
            "tests/v2/test_execution_recovery.py::test_lifecycle_latency_records_event_decision_submit_ack_and_fill",
        ),
    ),
    AcceptanceRequirement(
        "live_submission_impossible",
        "Live order submission remains impossible at the credential-ready milestone",
        (
            "tests/v2/test_rithmic_write_boundary.py::test_boundary_has_no_live_mode_and_test_requires_all_explicit_gates",
            "tests/v2/test_learning_execution_security.py::test_execution_service_is_foundation_only_and_order_submission_is_impossible",
            "tests/test_v2_copier_api.py::test_v2_copier_status_requires_auth_and_redacts_configuration",
        ),
    ),
)


def acceptance_node_ids() -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            node_id
            for requirement in ACCEPTANCE_REQUIREMENTS
            for node_id in requirement.evidence
        )
    )


def manifest_payload() -> dict[str, object]:
    return {
        "milestone": "v2_credential_ready",
        "live_submission_possible": False,
        "requirement_count": len(ACCEPTANCE_REQUIREMENTS),
        "test_count": len(acceptance_node_ids()),
        "requirements": [
            {
                "id": item.requirement_id,
                "description": item.description,
                "evidence": list(item.evidence),
            }
            for item in ACCEPTANCE_REQUIREMENTS
        ],
    }


def _absolute_node_id(node_id: str) -> str:
    path, separator, selector = node_id.partition("::")
    resolved = str((REPO_ROOT / path).resolve())
    return f"{resolved}{separator}{selector}" if separator else resolved


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the credential-free V2 copier release gate and write an auditable "
            "JUnit report. This command cannot enable broker submission."
        )
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help=f"JUnit report path (default: {DEFAULT_REPORT.as_posix()})",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the requirement-to-test manifest without running tests.",
    )
    args = parser.parse_args(argv)

    if args.list:
        print(json.dumps(manifest_payload(), indent=2, sort_keys=True))
        return 0

    report = args.report if args.report.is_absolute() else REPO_ROOT / args.report
    report.parent.mkdir(parents=True, exist_ok=True)

    import pytest

    return int(
        pytest.main(
            [
                *(_absolute_node_id(item) for item in acceptance_node_ids()),
                "-q",
                f"--junitxml={report}",
            ]
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
