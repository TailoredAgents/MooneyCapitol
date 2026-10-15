from __future__ import annotations

import ast

from app.tools.v2_credential_ready_acceptance import (
    ACCEPTANCE_REQUIREMENTS,
    REPO_ROOT,
    acceptance_node_ids,
    manifest_payload,
)


EXPECTED_REQUIREMENTS = {
    "lifecycle_types",
    "independent_controls",
    "protection",
    "cumulative_sizing",
    "funding_and_margin",
    "rejections",
    "event_ordering",
    "credential_account_mismatch",
    "connection_and_recovery",
    "execution_fencing",
    "contract_expiry",
    "redaction",
    "operator_security",
    "concurrent_latency",
    "live_submission_impossible",
}


def test_acceptance_manifest_is_complete_and_targets_real_tests() -> None:
    requirement_ids = {item.requirement_id for item in ACCEPTANCE_REQUIREMENTS}
    assert requirement_ids == EXPECTED_REQUIREMENTS

    node_ids = acceptance_node_ids()
    assert len(node_ids) == len(set(node_ids))
    for node_id in node_ids:
        relative_path, separator, test_name = node_id.partition("::")
        assert separator == "::"
        path = REPO_ROOT / relative_path
        assert path.is_file()
        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        functions = {
            item.name
            for item in module.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert test_name in functions

    payload = manifest_payload()
    assert payload["milestone"] == "v2_credential_ready"
    assert payload["live_submission_possible"] is False
    assert payload["requirement_count"] == len(EXPECTED_REQUIREMENTS)
