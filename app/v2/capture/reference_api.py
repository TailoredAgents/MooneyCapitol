from __future__ import annotations

import asyncio
import hashlib
import hmac
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Mapping, Protocol, Sequence


_SEARCH_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._/-]{1,31}$")
_CODE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,31}$")
_TICK_TYPE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")
_SEARCH_FIELDS = (
    "symbol",
    "exchange",
    "symbol_name",
    "product_code",
    "instrument_type",
    "expiration_date",
)
_REFERENCE_FIELDS = (
    "symbol",
    "exchange",
    "exchange_symbol",
    "symbol_name",
    "trading_symbol",
    "trading_exchange",
    "product_code",
    "instrument_type",
    "underlying_symbol",
    "expiration_date",
    "currency",
    "tick_size_type",
    "price_display_format",
    "is_tradable",
    "minimum_quoted_price_change",
    "minimum_feed_price_change",
    "single_point_value",
    "quote_to_feed_price_factor",
    "feed_to_quote_price_factor",
    "presence_bits",
)
_TICK_FIELDS = (
    "tick_size_type",
    "minimum_feed_price_change",
    "first_price",
    "last_price",
    "first_price_operator",
    "last_price_operator",
    "presence_bits",
)


class ReferenceQueryUnavailable(RuntimeError):
    pass


class ReferenceQueryFailed(RuntimeError):
    pass


class ReferenceQueryTimeout(ReferenceQueryFailed):
    pass


class ReferenceObserverSurface(Protocol):
    @property
    def ticker_reference_available(self) -> bool: ...

    async def search_reference_symbols(
        self,
        search_text: str,
        *,
        exchange: str | None,
        product_code: str | None,
        timeout_seconds: float,
    ) -> Sequence[Mapping[str, Any]]: ...

    async def contract_reference(
        self,
        symbol: str,
        exchange: str,
        *,
        timeout_seconds: float,
    ) -> Any: ...

    async def reference_tick_sizes(
        self,
        tick_size_type: str,
        *,
        timeout_seconds: float,
    ) -> Sequence[Any]: ...


def _boolean(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _bounded_float(value: str | None, *, default: float) -> float:
    try:
        parsed = float(value or default)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) and 1.0 <= parsed <= 30.0 else default


def _bounded_int(value: str | None, *, default: int, maximum: int) -> int:
    try:
        parsed = int(value or default)
    except (TypeError, ValueError):
        return default
    return parsed if 1 <= parsed <= maximum else default


@dataclass(frozen=True)
class ReferenceQueryConfig:
    enabled: bool = False
    token_digest: bytes | None = field(default=None, repr=False)
    timeout_seconds: float = 15.0
    maximum_results: int = 50

    @classmethod
    def from_mapping(cls, values: Mapping[str, str]) -> "ReferenceQueryConfig":
        token = values.get("RITHMIC_REFERENCE_QUERY_TOKEN", "")
        # A missing or weak token disables the surface instead of opening an
        # unauthenticated fallback. The raw token is never retained.
        digest = hashlib.sha256(token.encode("utf-8")).digest() if 32 <= len(token) <= 512 else None
        return cls(
            enabled=_boolean(values.get("RITHMIC_REFERENCE_QUERY_ENABLED")),
            token_digest=digest,
            timeout_seconds=_bounded_float(
                values.get("RITHMIC_REFERENCE_QUERY_TIMEOUT_SECONDS"), default=15.0
            ),
            maximum_results=_bounded_int(
                values.get("RITHMIC_REFERENCE_QUERY_MAX_RESULTS"),
                default=50,
                maximum=100,
            ),
        )

    @property
    def available(self) -> bool:
        return self.enabled and self.token_digest is not None

    def authorize(self, candidate: str | None) -> bool:
        candidate_digest = hashlib.sha256((candidate or "").encode("utf-8")).digest()
        expected = self.token_digest or bytes(hashlib.sha256().digest_size)
        return self.available and hmac.compare_digest(candidate_digest, expected)


def validated_search_text(value: str) -> str:
    normalized = value.strip()
    if not _SEARCH_PATTERN.fullmatch(normalized):
        raise ValueError("invalid symbol search")
    return normalized


def validated_code(value: str | None, *, required: bool = False) -> str | None:
    normalized = (value or "").strip()
    if not normalized and not required:
        return None
    if not _CODE_PATTERN.fullmatch(normalized):
        raise ValueError("invalid reference code")
    return normalized


def validated_tick_size_type(value: str) -> str:
    normalized = value.strip()
    if not _TICK_TYPE_PATTERN.fullmatch(normalized):
        raise ValueError("invalid tick-size type")
    return normalized


def _field(value: Any, name: str) -> Any:
    return value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)


def _safe_value(value: Any) -> str | int | float | bool | None:
    if value is None or isinstance(value, (int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ReferenceQueryFailed("reference response was invalid")
        return value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ReferenceQueryFailed("reference response was invalid")
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, (Mapping, list, tuple, set, frozenset)):
        raise ReferenceQueryFailed("reference response was invalid")
    text = str(value)
    if len(text) > 256 or any(ord(character) < 32 for character in text):
        raise ReferenceQueryFailed("reference response was invalid")
    return text


def _public_record(value: Any, fields: Sequence[str]) -> dict[str, Any]:
    return {
        name: safe
        for name in fields
        if (safe := _safe_value(_field(value, name))) is not None
    }


class ReferenceQueryController:
    def __init__(self, service: ReferenceObserverSurface, config: ReferenceQueryConfig) -> None:
        self.service = service
        self.config = config

    def authorize(self, token: str | None) -> bool:
        return self.config.authorize(token)

    def require_available(self) -> None:
        if not self.config.available or not self.service.ticker_reference_available:
            raise ReferenceQueryUnavailable("reference query service is unavailable")

    async def search(
        self,
        search_text: str,
        *,
        exchange: str | None,
        product_code: str | None,
        limit: int,
    ) -> dict[str, Any]:
        self.require_available()
        if not 1 <= limit <= self.config.maximum_results:
            raise ValueError("invalid result limit")
        try:
            rows = await self.service.search_reference_symbols(
                validated_search_text(search_text),
                exchange=validated_code(exchange),
                product_code=validated_code(product_code),
                timeout_seconds=self.config.timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise ReferenceQueryTimeout("reference query timed out") from exc
        except ConnectionError as exc:
            raise ReferenceQueryUnavailable("reference query service is unavailable") from exc
        return {
            "count": min(len(rows), limit),
            "truncated": len(rows) > limit,
            "results": [_public_record(row, _SEARCH_FIELDS) for row in rows[:limit]],
        }

    async def reference(self, symbol: str, exchange: str) -> dict[str, Any]:
        self.require_available()
        try:
            result = await self.service.contract_reference(
                validated_code(symbol, required=True) or "",
                validated_code(exchange, required=True) or "",
                timeout_seconds=self.config.timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise ReferenceQueryTimeout("reference query timed out") from exc
        except ConnectionError as exc:
            raise ReferenceQueryUnavailable("reference query service is unavailable") from exc
        return _public_record(result, _REFERENCE_FIELDS)

    async def tick_sizes(self, tick_size_type: str) -> dict[str, Any]:
        self.require_available()
        try:
            rows = await self.service.reference_tick_sizes(
                validated_tick_size_type(tick_size_type),
                timeout_seconds=self.config.timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise ReferenceQueryTimeout("reference query timed out") from exc
        except ConnectionError as exc:
            raise ReferenceQueryUnavailable("reference query service is unavailable") from exc
        limited = rows[: self.config.maximum_results]
        return {
            "count": len(limited),
            "truncated": len(rows) > len(limited),
            "results": [_public_record(row, _TICK_FIELDS) for row in limited],
        }
