from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from app.tools.generate_rithmic_write_bindings import WRITE_PROTO_FILES
from app.v2.brokers.base import BrokerSubmissionDisabled
from app.v2.brokers.rithmic_protocol.constants import Template
from app.v2.brokers.rithmic_protocol.errors import BindingsUnavailable
from app.v2.brokers.rithmic_write import (
    WRITE_REQUEST_BINDINGS,
    WRITE_RESPONSE_BINDINGS,
    ExternalWriteCodec,
    RithmicAccountRoute,
    RithmicTestWriteAdapter,
    RithmicWriteMode,
    RithmicWriteRejected,
    RithmicWriteUnavailable,
)
from app.v2.domain.models import Side
from app.v2.execution.lifecycle import (
    BracketOperation,
    CopyAction,
    LifecycleCommand,
    OrderDuration,
    OrderType,
)


def _route() -> RithmicAccountRoute:
    return RithmicAccountRoute(
        connection_id="follower-1",
        fcm_id="fcm",
        ib_id="ib",
        account_id="account",
        trade_route="route",
    )


def _command(
    action: CopyAction = CopyAction.SUBMIT,
    *,
    order_type: OrderType = OrderType.MARKET,
    side: Side | None = Side.BUY,
    quantity: int = 2,
    **values: object,
) -> LifecycleCommand:
    return LifecycleCommand(
        command_id=f"command-{action.value.lower()}",
        idempotency_key=f"intent-{action.value.lower()}",
        intent_id="copy-intent-1",
        ordinal=1,
        action=action,
        contract_id="CME:MNQZ6",
        symbol="MNQZ6",
        exchange="CME",
        side=side,
        quantity=quantity,
        order_type=order_type,
        duration=OrderDuration.DAY,
        **values,
    )


def _shadow() -> RithmicTestWriteAdapter:
    return RithmicTestWriteAdapter(
        mode=RithmicWriteMode.SHADOW,
        environment="TEST",
        route=_route(),
    )


def test_boundary_has_no_live_mode_and_test_requires_all_explicit_gates() -> None:
    with pytest.raises(ValueError):
        RithmicWriteMode("LIVE")
    with pytest.raises(ValueError, match="explicit"):
        RithmicTestWriteAdapter(
            mode="LIVE",  # type: ignore[arg-type]
            environment="TEST",
            route=_route(),
            test_submission_authorized=True,
            codec=_Codec({"rp_code": ["0"]}),
            transport=_Transport(),
        )
    with pytest.raises(BrokerSubmissionDisabled, match="Rithmic Test"):
        RithmicTestWriteAdapter(
            mode=RithmicWriteMode.TEST,
            environment="PRODUCTION",
            route=_route(),
            test_submission_authorized=True,
        )
    with pytest.raises(BrokerSubmissionDisabled, match="authorization"):
        RithmicTestWriteAdapter(
            mode=RithmicWriteMode.TEST,
            environment="TEST",
            route=_route(),
        )
    with pytest.raises(RithmicWriteUnavailable, match="bindings"):
        RithmicTestWriteAdapter(
            mode=RithmicWriteMode.TEST,
            environment="TEST",
            route=_route(),
            test_submission_authorized=True,
        )


def test_off_never_transmits_and_shadow_only_builds_a_request() -> None:
    off = RithmicTestWriteAdapter(
        mode=RithmicWriteMode.OFF,
        environment="PRODUCTION",
        route=_route(),
    )
    with pytest.raises(BrokerSubmissionDisabled):
        asyncio.run(off.dispatch(_command()))

    result = asyncio.run(_shadow().dispatch(_command()))
    assert result.transmitted is False
    assert result.request.template_id == Template.NEW_ORDER_REQUEST
    assert result.request.fields["quantity_64"] == 2
    assert result.request.fields["manual_or_auto"] == 2
    assert result.request.fields["user_msg"] == ["intent-submit"]


def test_official_new_modify_cancel_and_exit_field_maps() -> None:
    adapter = _shadow()
    limit = adapter.build_request(
        _command(
            order_type=OrderType.LIMIT,
            limit_price=Decimal("20000.25"),
        )
    )
    assert limit.template_id == Template.NEW_ORDER_REQUEST
    assert limit.response_template_id == Template.NEW_ORDER_RESPONSE
    assert limit.fields["price_type"] == 1
    assert limit.fields["price"] == 20000.25
    assert limit.fields["transaction_type"] == 1

    stop = adapter.build_request(
        _command(
            CopyAction.MODIFY,
            order_type=OrderType.STOP_MARKET,
            side=Side.SELL,
            quantity=1,
            trigger_price=Decimal("19900"),
            existing_broker_order_id="basket-1",
        )
    )
    assert stop.template_id == Template.MODIFY_ORDER_REQUEST
    assert stop.fields["basket_id"] == "basket-1"
    assert stop.fields["trigger_price"] == 19900.0
    assert stop.fields["price_type"] == 4

    cancel = adapter.build_request(
        _command(
            CopyAction.CANCEL,
            side=None,
            quantity=0,
            existing_broker_order_id="basket-2",
        )
    )
    assert cancel.template_id == Template.CANCEL_ORDER_REQUEST
    assert cancel.fields["basket_id"] == "basket-2"

    exit_request = adapter.build_request(
        _command(CopyAction.EXIT_POSITION, side=Side.SELL, quantity=2)
    )
    assert exit_request.template_id == Template.EXIT_POSITION_REQUEST
    assert exit_request.fields["symbol"] == "MNQZ6"
    assert "quantity_64" not in exit_request.fields


def test_official_bracket_and_oco_field_maps() -> None:
    adapter = _shadow()
    bracket = adapter.build_request(
        _command(
            CopyAction.BRACKET,
            stop_ticks=8,
            target_ticks=16,
            bracket_operation=BracketOperation.AFOCCA,
        )
    )
    assert bracket.template_id == Template.BRACKET_ORDER_REQUEST
    assert bracket.fields["bracket_type"] == 3
    assert bracket.fields["stop_quantity_64"] == [2]
    assert bracket.fields["stop_ticks"] == [8]
    assert bracket.fields["target_ticks"] == [16]
    assert bracket.fields["order_operation_type"] == "AFOCCA"

    oco = adapter.build_request(
        _command(
            CopyAction.OCO,
            side=Side.SELL,
            quantity=1,
            limit_price=Decimal("20100"),
            trigger_price=Decimal("19900"),
        )
    )
    assert oco.template_id == Template.OCO_ORDER_REQUEST
    assert oco.fields["price_type"] == [1, 4]
    assert oco.fields["transaction_type"] == [2, 2]
    assert oco.fields["price"] == [20100.0, 0.0]
    assert oco.fields["trigger_price"] == [0.0, 19900.0]


class _Codec:
    def __init__(self, response: object) -> None:
        self.response = response
        self.requests = []

    def encode(self, request):
        self.requests.append(request)
        return b"encoded-request"

    def decode(self, expected_template_id, payload):
        assert payload == b"encoded-response"
        return self.response


class _Transport:
    def __init__(self) -> None:
        self.calls = []

    async def exchange(self, payload, **values):
        self.calls.append((payload, values))
        return b"encoded-response"


def test_test_dispatch_requires_exact_success_and_retains_all_oco_basket_ids() -> None:
    codec = _Codec(
        {
            "template_id": int(Template.OCO_ORDER_RESPONSE),
            "rq_handler_rp_code": ["0"],
            "rp_code": ["0"],
            "basket_id": ["target-basket", "stop-basket"],
        }
    )
    transport = _Transport()
    adapter = RithmicTestWriteAdapter(
        mode=RithmicWriteMode.TEST,
        environment="Rithmic Test",
        route=_route(),
        test_submission_authorized=True,
        codec=codec,
        transport=transport,
    )
    result = asyncio.run(
        adapter.dispatch(
            _command(
                CopyAction.OCO,
                side=Side.SELL,
                quantity=1,
                limit_price=Decimal("20100"),
                trigger_price=Decimal("19900"),
            )
        )
    )
    assert result.transmitted
    assert result.broker_order_ids == ("target-basket", "stop-basket")
    assert transport.calls[0][1]["expected_template_id"] == (
        Template.OCO_ORDER_RESPONSE
    )

    rejected = RithmicTestWriteAdapter(
        mode=RithmicWriteMode.TEST,
        environment="TEST",
        route=_route(),
        test_submission_authorized=True,
        codec=_Codec({"rp_code": ["7"]}),
        transport=_Transport(),
    )
    with pytest.raises(RithmicWriteRejected):
        asyncio.run(rejected.dispatch(_command()))


def test_write_binding_set_is_separate_exact_and_external(tmp_path: Path) -> None:
    proto_names = {
        f"{module.removesuffix('_pb2')}.proto"
        for module, _ in (
            *WRITE_REQUEST_BINDINGS.values(),
            *WRITE_RESPONSE_BINDINGS.values(),
        )
    }
    assert proto_names == set(WRITE_PROTO_FILES)
    assert "request_login.proto" not in WRITE_PROTO_FILES

    generated = tmp_path / "write-bindings"
    generated.mkdir()
    filenames = {
        f"{module}.py"
        for module, _ in (
            *WRITE_REQUEST_BINDINGS.values(),
            *WRITE_RESPONSE_BINDINGS.values(),
        )
    }
    checksums = {}
    for filename in filenames:
        payload = f"# {filename}\n".encode()
        (generated / filename).write_bytes(payload)
        checksums[filename] = hashlib.sha256(payload).hexdigest()
    (generated / "rithmic_write_bindings_manifest.json").write_text(
        json.dumps(
            {
                "protocol_package": "RProtocolAPI.0.90.0.0",
                "template_version": "5.55",
                "files": checksums,
            }
        ),
        encoding="utf-8",
    )
    codec = ExternalWriteCodec(generated, forbid_workspace_root=Path.cwd())
    assert codec.registry.binding_path == generated.resolve()
    with pytest.raises(BindingsUnavailable, match="outside Git"):
        ExternalWriteCodec(generated, forbid_workspace_root=tmp_path)


def test_account_route_repr_does_not_leak_broker_selectors() -> None:
    rendered = repr(_route())
    assert "account" not in rendered
    assert "fcm" not in rendered
    assert "route" not in rendered
    request_rendered = repr(_shadow().build_request(_command()))
    assert "account" not in request_rendered
    assert "fcm" not in request_rendered
    assert "route" not in request_rendered


def test_protocol_request_does_not_claim_rithmic_idempotency() -> None:
    request = _shadow().build_request(_command())
    assert request.fields["user_tag"] == "command-submit"
    assert request.fields["user_msg"] == ["intent-submit"]
    assert "client_order_id" not in request.fields
