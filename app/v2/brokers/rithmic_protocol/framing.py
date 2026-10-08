from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .constants import TEMPLATE_ID_FIELD_NUMBER
from .errors import InvalidFrame


_MAX_VARINT_BYTES = 10


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    for shift in range(0, 70, 7):
        if offset >= len(data):
            raise InvalidFrame("truncated protobuf varint")
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
    raise InvalidFrame(f"protobuf varint exceeds {_MAX_VARINT_BYTES} bytes")


def _skip_field(data: bytes, offset: int, wire_type: int) -> int:
    if wire_type == 0:
        _, offset = _read_varint(data, offset)
        return offset
    if wire_type == 1:
        offset += 8
    elif wire_type == 2:
        length, offset = _read_varint(data, offset)
        offset += length
    elif wire_type == 5:
        offset += 4
    else:
        # Current official schemas do not use deprecated protobuf groups.
        raise InvalidFrame(f"unsupported protobuf wire type {wire_type}")
    if offset > len(data):
        raise InvalidFrame("truncated protobuf field")
    return offset


def extract_template_id(frame: bytes | bytearray | memoryview) -> int:
    """Read field 154467 directly from one protobuf WebSocket message.

    R|Protocol WebSocket payload version 2 has no four-byte length prefix: one
    binary WebSocket message is exactly one serialized protobuf message.
    """

    if not isinstance(frame, (bytes, bytearray, memoryview)):
        raise InvalidFrame("R|Protocol frames must be binary")
    data = bytes(frame)
    if not data:
        raise InvalidFrame("empty R|Protocol frame")

    offset = 0
    template_id: int | None = None
    while offset < len(data):
        key, offset = _read_varint(data, offset)
        field_number, wire_type = key >> 3, key & 0x07
        if field_number == 0:
            raise InvalidFrame("protobuf field number zero is invalid")
        if field_number == TEMPLATE_ID_FIELD_NUMBER:
            if wire_type != 0:
                raise InvalidFrame("template_id must use protobuf varint encoding")
            if template_id is not None:
                # Protobuf parsers commonly apply last-value-wins semantics to
                # duplicate singular fields.  Reject duplicates outright so an
                # allowlisted first ID cannot smuggle a mutation ID later.
                raise InvalidFrame("protobuf frame contains duplicate template_id")
            template_id, offset = _read_varint(data, offset)
            if template_id <= 0:
                raise InvalidFrame("template_id must be positive")
            continue
        offset = _skip_field(data, offset, wire_type)
    if template_id is None:
        raise InvalidFrame("protobuf frame does not contain template_id")
    return template_id


@dataclass(frozen=True)
class EncodedFrame:
    template_id: int
    payload: bytes


def encode_message(message: Any) -> EncodedFrame:
    serializer = getattr(message, "SerializeToString", None)
    if not callable(serializer):
        raise InvalidFrame("message does not implement SerializeToString")
    try:
        payload = serializer()
    except Exception as exc:  # protobuf supplies the useful detail
        raise InvalidFrame("protobuf serialization failed") from exc
    if not isinstance(payload, bytes):
        raise InvalidFrame("SerializeToString must return bytes")
    template_id = extract_template_id(payload)
    declared = getattr(message, "template_id", None)
    if declared is not None and int(declared) != template_id:
        raise InvalidFrame("declared template_id differs from serialized frame")
    return EncodedFrame(template_id=template_id, payload=payload)
