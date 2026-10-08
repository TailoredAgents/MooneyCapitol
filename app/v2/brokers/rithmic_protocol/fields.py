from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from typing import Any

from .errors import NormalizationError


_MISSING = object()
_EXACT_SUCCESS_RESPONSE = ("0",)


class FieldView:
    """Presence-aware access to synthetic mappings or generated proto2 objects."""

    def __init__(self, message: Mapping[str, Any] | Any) -> None:
        self.message = message

    def has(self, name: str) -> bool:
        if isinstance(self.message, Mapping):
            return name in self.message
        descriptor = getattr(self.message, "DESCRIPTOR", None)
        fields = getattr(descriptor, "fields_by_name", {})
        field = fields.get(name) if fields else None
        if field is None:
            return hasattr(self.message, name)
        if getattr(field, "is_repeated", False) or getattr(field, "label", None) == 3:
            return len(getattr(self.message, name)) > 0
        checker = getattr(self.message, "HasField", None)
        if callable(checker):
            try:
                return bool(checker(name))
            except (ValueError, TypeError):
                pass
        return hasattr(self.message, name)

    def get(self, name: str, default: Any = None) -> Any:
        if isinstance(self.message, Mapping):
            return self.message.get(name, default)
        if not self.has(name):
            return default
        return getattr(self.message, name, default)

    def first(self, *names: str, default: Any = None) -> Any:
        for name in names:
            value = self.get(name, _MISSING)
            if value is not _MISSING and value is not None:
                return value
        return default

    def strings(self, name: str) -> tuple[str, ...]:
        value = self.get(name, ())
        if value is None:
            return ()
        if isinstance(value, str):
            return (value,)
        return tuple(str(item) for item in value)


def is_exact_success_response(codes: Sequence[str]) -> bool:
    """Return true only for the official single-code success response."""

    return tuple(codes) == _EXACT_SUCCESS_RESPONSE


def optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text != "" else None


def optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "enabled"}:
        return True
    if text in {"false", "0", "no", "disabled"}:
        return False
    raise NormalizationError("invalid boolean field")


def optional_int(value: Any, *, unsigned: bool = False) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise NormalizationError("boolean is not an integer field")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise NormalizationError("invalid integer field") from exc
    if unsigned and normalized < 0:
        raise NormalizationError("unsigned field may not be negative")
    return normalized


def optional_decimal(value: Any) -> Decimal | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise NormalizationError("boolean is not a decimal field")
    try:
        return value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise NormalizationError("invalid decimal field") from exc


def enum_name(value: Any, values: Mapping[int, str]) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        return value.strip().upper() or None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return str(value)
    return values.get(number, f"UNKNOWN_ENUM_{number}")


def tuple_text(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value)
    return (str(value),)
