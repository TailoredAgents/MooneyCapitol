from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Protocol

from app.v2.brokers.base import BrokerSubmissionDisabled
from app.v2.brokers.rithmic_protocol.bindings import ExternalBindingRegistry
from app.v2.brokers.rithmic_protocol.constants import (
    PROTOCOL_PACKAGE_VERSION,
    PROTOCOL_TEMPLATE_VERSION,
    Template,
)
from app.v2.brokers.rithmic_protocol.errors import BindingsUnavailable
from app.v2.brokers.rithmic_protocol.fields import FieldView, is_exact_success_response
from app.v2.brokers.rithmic_protocol.framing import extract_template_id
from app.v2.domain.models import Side
from app.v2.execution.lifecycle import (
    CopyAction,
    LifecycleCommand,
    OrderDuration,
    OrderType,
)


WRITE_REQUEST_BINDINGS: dict[int, tuple[str, str]] = {
    int(Template.NEW_ORDER_REQUEST): ("request_new_order_pb2", "RequestNewOrder"),
    int(Template.MODIFY_ORDER_REQUEST): (
        "request_modify_order_pb2",
        "RequestModifyOrder",
    ),
    int(Template.CANCEL_ORDER_REQUEST): (
        "request_cancel_order_pb2",
        "RequestCancelOrder",
    ),
    int(Template.OCO_ORDER_REQUEST): ("request_oco_order_pb2", "RequestOCOOrder"),
    int(Template.BRACKET_ORDER_REQUEST): (
        "request_bracket_order_pb2",
        "RequestBracketOrder",
    ),
    int(Template.EXIT_POSITION_REQUEST): (
        "request_exit_position_pb2",
        "RequestExitPosition",
    ),
}

WRITE_RESPONSE_BINDINGS: dict[int, tuple[str, str]] = {
    int(Template.NEW_ORDER_RESPONSE): ("response_new_order_pb2", "ResponseNewOrder"),
    int(Template.MODIFY_ORDER_RESPONSE): (
        "response_modify_order_pb2",
        "ResponseModifyOrder",
    ),
    int(Template.CANCEL_ORDER_RESPONSE): (
        "response_cancel_order_pb2",
        "ResponseCancelOrder",
    ),
    int(Template.OCO_ORDER_RESPONSE): (
        "response_oco_order_pb2",
        "ResponseOCOOrder",
    ),
    int(Template.BRACKET_ORDER_RESPONSE): (
        "response_bracket_order_pb2",
        "ResponseBracketOrder",
    ),
    int(Template.EXIT_POSITION_RESPONSE): (
        "response_exit_position_pb2",
        "ResponseExitPosition",
    ),
}


class RithmicWriteMode(str, Enum):
    OFF = "OFF"
    SHADOW = "SHADOW"
    TEST = "TEST"


class RithmicWriteRejected(RuntimeError):
    broker_rejection = True


class RithmicWriteUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class RithmicAccountRoute:
    connection_id: str
    fcm_id: str = field(repr=False)
    ib_id: str = field(repr=False)
    account_id: str = field(repr=False)
    trade_route: str = field(repr=False)

    def __post_init__(self) -> None:
        if self.connection_id not in {f"follower-{number}" for number in range(1, 5)}:
            raise ValueError("write routes are limited to the four follower connections")
        for name in ("fcm_id", "ib_id", "account_id", "trade_route"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or not value.strip()
                or len(value) > 256
                or any(ord(character) < 32 for character in value)
            ):
                raise ValueError(f"{name} is required")


@dataclass(frozen=True)
class RithmicProtocolRequest:
    template_id: int
    response_template_id: int
    message_name: str
    fields: Mapping[str, object] = field(repr=False)
    correlation_id: str


@dataclass(frozen=True)
class RithmicDispatchResult:
    transmitted: bool
    broker_order_ids: tuple[str, ...] = field(repr=False)
    request: RithmicProtocolRequest


class RithmicWriteCodec(Protocol):
    def encode(self, request: RithmicProtocolRequest) -> bytes: ...

    def decode(self, expected_template_id: int, payload: bytes) -> object: ...


class RithmicTestTransport(Protocol):
    async def exchange(
        self,
        payload: bytes,
        *,
        expected_template_id: int,
        correlation_id: str,
    ) -> bytes: ...


class ExternalWriteCodec:
    """Loads only checksum-pinned write bindings generated outside Git."""

    def __init__(
        self,
        binding_path: Path,
        *,
        forbid_workspace_root: Path,
    ) -> None:
        path = binding_path.expanduser().resolve(strict=True)
        root = forbid_workspace_root.resolve()
        try:
            path.relative_to(root)
        except ValueError:
            pass
        else:
            raise BindingsUnavailable("Rithmic write bindings must remain outside Git")
        _validate_write_manifest(path)
        self.registry = ExternalBindingRegistry(
            path,
            inbound=WRITE_RESPONSE_BINDINGS,
            outbound=WRITE_REQUEST_BINDINGS,
        )

    def encode(self, request: RithmicProtocolRequest) -> bytes:
        message = self.registry.build(request.template_id, **request.fields)
        serializer = getattr(message, "SerializeToString", None)
        if not callable(serializer):
            raise BindingsUnavailable("generated write request cannot serialize")
        return bytes(serializer())

    def decode(self, expected_template_id: int, payload: bytes) -> object:
        actual = extract_template_id(payload)
        if actual != expected_template_id:
            raise RithmicWriteRejected("Rithmic returned an unexpected write response")
        return self.registry.decode(payload)


class RithmicTestWriteAdapter:
    """Execution-only R|Protocol boundary with no production/live mode."""

    def __init__(
        self,
        *,
        mode: RithmicWriteMode,
        environment: str,
        route: RithmicAccountRoute,
        test_submission_authorized: bool = False,
        codec: RithmicWriteCodec | None = None,
        transport: RithmicTestTransport | None = None,
    ) -> None:
        if not isinstance(mode, RithmicWriteMode):
            raise ValueError("write mode must be an explicit RithmicWriteMode")
        self.mode = mode
        if not isinstance(environment, str):
            raise ValueError("write environment must be text")
        self.environment = environment.strip().upper().replace(" ", "_")
        if not self.environment:
            raise ValueError("write environment is required")
        self.route = route
        self.codec = codec
        self.transport = transport
        if mode == RithmicWriteMode.TEST:
            if self.environment not in {"TEST", "RITHMIC_TEST"}:
                raise BrokerSubmissionDisabled(
                    "the write adapter can transmit only in Rithmic Test"
                )
            if not test_submission_authorized:
                raise BrokerSubmissionDisabled(
                    "Rithmic Test submission requires an explicit authorization gate"
                )
            if codec is None or transport is None:
                raise RithmicWriteUnavailable(
                    "Rithmic Test requires external bindings and an injected transport"
                )

    @property
    def connection_id(self) -> str:
        return self.route.connection_id

    def build_request(self, command: LifecycleCommand) -> RithmicProtocolRequest:
        common = {
            "user_msg": [command.idempotency_key],
            "fcm_id": self.route.fcm_id,
            "ib_id": self.route.ib_id,
            "account_id": self.route.account_id,
        }
        request_template: Template
        response_template: Template
        message_name: str
        fields: dict[str, object]
        if command.action == CopyAction.SUBMIT:
            request_template = Template.NEW_ORDER_REQUEST
            response_template = Template.NEW_ORDER_RESPONSE
            message_name = "RequestNewOrder"
            fields = _new_order_fields(command, common, self.route.trade_route)
        elif command.action == CopyAction.MODIFY:
            request_template = Template.MODIFY_ORDER_REQUEST
            response_template = Template.MODIFY_ORDER_RESPONSE
            message_name = "RequestModifyOrder"
            fields = _modify_fields(command, common)
        elif command.action == CopyAction.CANCEL:
            request_template = Template.CANCEL_ORDER_REQUEST
            response_template = Template.CANCEL_ORDER_RESPONSE
            message_name = "RequestCancelOrder"
            fields = {
                **common,
                "basket_id": _broker_order_id(command),
                "manual_or_auto": 2,
            }
        elif command.action == CopyAction.BRACKET:
            request_template = Template.BRACKET_ORDER_REQUEST
            response_template = Template.BRACKET_ORDER_RESPONSE
            message_name = "RequestBracketOrder"
            fields = _bracket_fields(command, common, self.route.trade_route)
        elif command.action == CopyAction.OCO:
            request_template = Template.OCO_ORDER_REQUEST
            response_template = Template.OCO_ORDER_RESPONSE
            message_name = "RequestOCOOrder"
            fields = _oco_fields(command, common, self.route.trade_route)
        elif command.action == CopyAction.EXIT_POSITION:
            request_template = Template.EXIT_POSITION_REQUEST
            response_template = Template.EXIT_POSITION_RESPONSE
            message_name = "RequestExitPosition"
            fields = {
                **common,
                "symbol": command.symbol,
                "exchange": command.exchange,
                "manual_or_auto": 2,
            }
        else:  # pragma: no cover - enum exhaustiveness guard
            raise ValueError("unsupported lifecycle command")
        return RithmicProtocolRequest(
            template_id=int(request_template),
            response_template_id=int(response_template),
            message_name=message_name,
            fields=fields,
            correlation_id=command.idempotency_key,
        )

    async def dispatch(self, command: LifecycleCommand) -> RithmicDispatchResult:
        request = self.build_request(command)
        if self.mode == RithmicWriteMode.OFF:
            raise BrokerSubmissionDisabled("Rithmic write mode is OFF")
        if self.mode == RithmicWriteMode.SHADOW:
            return RithmicDispatchResult(False, (), request)
        if self.codec is None or self.transport is None:  # constructor enforces this
            raise RithmicWriteUnavailable("Rithmic Test transport is unavailable")
        payload = self.codec.encode(request)
        response_payload = await self.transport.exchange(
            payload,
            expected_template_id=request.response_template_id,
            correlation_id=request.correlation_id,
        )
        response = self.codec.decode(request.response_template_id, response_payload)
        view = FieldView(response)
        intermediate = view.strings("rq_handler_rp_code")
        terminal = view.strings("rp_code")
        if not intermediate and not terminal:
            raise RithmicWriteRejected("Rithmic write response omitted response codes")
        if intermediate and not is_exact_success_response(intermediate):
            raise RithmicWriteRejected("Rithmic request handler rejected the command")
        if terminal and not is_exact_success_response(terminal):
            raise RithmicWriteRejected("Rithmic rejected the command")
        basket_ids = tuple(
            item for item in view.strings("basket_id") if item.strip()
        )
        return RithmicDispatchResult(True, basket_ids, request)


def _new_order_fields(
    command: LifecycleCommand,
    common: Mapping[str, object],
    trade_route: str,
) -> dict[str, object]:
    fields = {
        **common,
        "user_tag": command.command_id,
        "symbol": command.symbol,
        "exchange": command.exchange,
        "quantity_64": command.quantity,
        "transaction_type": _transaction_type(command.side),
        "duration": _duration(command.duration),
        "price_type": _price_type(command.order_type),
        "trade_route": trade_route,
        "manual_or_auto": 2,
    }
    _add_prices(fields, command)
    return fields


def _modify_fields(
    command: LifecycleCommand,
    common: Mapping[str, object],
) -> dict[str, object]:
    fields = {
        **common,
        "basket_id": _broker_order_id(command),
        "symbol": command.symbol,
        "exchange": command.exchange,
        "quantity_64": command.quantity,
        "price_type": _price_type(command.order_type),
        "manual_or_auto": 2,
    }
    _add_prices(fields, command)
    return fields


def _bracket_fields(
    command: LifecycleCommand,
    common: Mapping[str, object],
    trade_route: str,
) -> dict[str, object]:
    if command.stop_ticks is None:
        raise ValueError("bracket command has no protective stop")
    if command.bracket_operation is None:
        raise ValueError("bracket command has no explicit operation policy")
    fields = _new_order_fields(command, common, trade_route)
    fields.update(
        {
            "user_type": 3,
            "bracket_type": (
                3 if command.stop_ticks and command.target_ticks else 1
                if command.stop_ticks
                else 2
            ),
            "order_operation_type": command.bracket_operation.value,
        }
    )
    if command.stop_ticks is not None:
        fields["stop_quantity_64"] = [command.quantity]
        fields["stop_ticks"] = [command.stop_ticks]
    if command.target_ticks is not None:
        fields["target_quantity_64"] = [command.quantity]
        fields["target_ticks"] = [command.target_ticks]
    return fields


def _oco_fields(
    command: LifecycleCommand,
    common: Mapping[str, object],
    trade_route: str,
) -> dict[str, object]:
    if command.side is None or command.limit_price is None or command.trigger_price is None:
        raise ValueError("OCO command requires side, target price, and stop price")
    side = _transaction_type(command.side)
    return {
        **common,
        "user_tag": [command.command_id, command.command_id],
        "symbol": [command.symbol, command.symbol],
        "exchange": [command.exchange, command.exchange],
        "quantity_64": [command.quantity, command.quantity],
        "price": [float(command.limit_price), 0.0],
        "trigger_price": [0.0, float(command.trigger_price)],
        "transaction_type": [side, side],
        "duration": [_duration(command.duration), _duration(command.duration)],
        "price_type": [1, 4],
        "trade_route": [trade_route, trade_route],
        "manual_or_auto": [2, 2],
    }


def _add_prices(fields: dict[str, object], command: LifecycleCommand) -> None:
    if command.limit_price is not None:
        fields["price"] = float(command.limit_price)
    if command.trigger_price is not None:
        fields["trigger_price"] = float(command.trigger_price)


def _transaction_type(side: Side | None) -> int:
    if side == Side.BUY:
        return 1
    if side == Side.SELL:
        return 2
    raise ValueError("order action requires a side")


def _duration(duration: OrderDuration) -> int:
    return {
        OrderDuration.DAY: 1,
        OrderDuration.GTC: 2,
        OrderDuration.IOC: 3,
        OrderDuration.FOK: 4,
    }[duration]


def _price_type(order_type: OrderType) -> int:
    return {
        OrderType.LIMIT: 1,
        OrderType.MARKET: 2,
        OrderType.STOP_LIMIT: 3,
        OrderType.STOP_MARKET: 4,
    }[order_type]


def _broker_order_id(command: LifecycleCommand) -> str:
    value = (command.existing_broker_order_id or "").strip()
    if not value:
        raise ValueError("broker order ID is required")
    return value


def _validate_write_manifest(path: Path) -> None:
    manifest_path = path / "rithmic_write_bindings_manifest.json"
    if not manifest_path.is_file() or manifest_path.stat().st_size > 1024 * 1024:
        raise BindingsUnavailable("generated write bindings require a bounded manifest")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BindingsUnavailable("write bindings manifest is invalid") from exc
    expected_files = {
        f"{module}.py"
        for module, _ in (
            *WRITE_REQUEST_BINDINGS.values(),
            *WRITE_RESPONSE_BINDINGS.values(),
        )
    }
    if (
        not isinstance(manifest, dict)
        or manifest.get("protocol_package")
        != f"RProtocolAPI.{PROTOCOL_PACKAGE_VERSION}"
        or manifest.get("template_version") != PROTOCOL_TEMPLATE_VERSION
        or not isinstance(manifest.get("files"), dict)
        or set(manifest["files"]) != expected_files
    ):
        raise BindingsUnavailable("write bindings manifest does not match runtime")
    actual_files = {item.name for item in path.glob("*_pb2.py") if item.is_file()}
    if actual_files != expected_files:
        raise BindingsUnavailable("write binding directory contains missing or extra modules")
    for filename, expected in manifest["files"].items():
        if not isinstance(expected, str) or len(expected) != 64:
            raise BindingsUnavailable("write bindings manifest checksum is invalid")
        digest = hashlib.sha256((path / filename).read_bytes()).hexdigest()
        if not hmac.compare_digest(digest, expected.lower()):
            raise BindingsUnavailable(f"write binding checksum mismatch: {filename}")
