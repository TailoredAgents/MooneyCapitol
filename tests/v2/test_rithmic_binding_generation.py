from pathlib import Path

import pytest

from app.tools.generate_rithmic_bindings import READ_ONLY_PROTO_FILES, _require_external
from app.v2.brokers.rithmic_protocol.constants import INBOUND_BINDINGS, OUTBOUND_BINDINGS


def test_binding_manifest_contains_observation_schemas_and_no_order_mutations():
    required = {
        "request_login.proto",
        "request_subscribe_for_order_updates.proto",
        "request_show_orders.proto",
        "request_replay_executions.proto",
        "request_pnl_position_snapshot.proto",
        "request_show_brackets.proto",
        "request_reference_data.proto",
    }
    forbidden = {
        "request_new_order.proto",
        "request_modify_order.proto",
        "request_cancel_order.proto",
        "request_cancel_all_orders.proto",
        "request_bracket_order.proto",
        "request_oco_order.proto",
        "request_link_orders.proto",
        "request_exit_position.proto",
    }
    assert required <= set(READ_ONLY_PROTO_FILES)
    assert forbidden.isdisjoint(READ_ONLY_PROTO_FILES)


def test_binding_manifest_covers_every_registered_read_only_message():
    registered = {
        f"{module.removesuffix('_pb2')}.proto"
        for module, _ in (*INBOUND_BINDINGS.values(), *OUTBOUND_BINDINGS.values())
    }
    assert registered <= set(READ_ONLY_PROTO_FILES)


def test_generator_refuses_to_write_generated_derivatives_inside_repository():
    with pytest.raises(ValueError, match="outside the Git repository"):
        _require_external(Path("generated/rithmic"), "generated bindings output")
